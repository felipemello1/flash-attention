"""Private, CPU-only per-call capability policy for the SM100 SwapAB backend, plus one speed rule
for short sliding windows."""

from dataclasses import dataclass
import math
from types import MappingProxyType


# Deliberately exhaustive: adding a forward option requires a capability review.
# Unknown options fail closed even when their value is None/default.
FORWARD_OPTION_POLICY = MappingProxyType(
    {
        "q": "canonical tensor",
        "k": "canonical tensor",
        "v": "canonical tensor",
        "qv": "unsupported",
        "cu_seqlens_q": "one query per sequence",
        "cu_seqlens_k": "unsupported",
        "seqused_q": "unsupported",
        "seqused_k": "supported",
        "max_seqlen_q": "hint",
        "max_seqlen_k": "hint",
        "min_seqlen_k": "hint",
        "page_table": "bounded paging",
        "softmax_scale": "supported",
        "causal": "supported",
        "softcap": "supported",
        "window_size_left": "supported",
        "window_size_right": "supported",
        "learnable_sink": "supported",
        "tile_mn": "unsupported",
        "mma_pv_is_rs": "unsupported",
        "intra_wg_overlap": "unsupported",
        "num_threads": "default only",
        "num_splits": "supported",
        "pack_gqa": "not False",
        "_arch": "resolved architecture",
        "score_mod": "supported",
        "mask_mod": "supported",
        "block_sparse_tensors": "unsupported",
        "return_lse": "supported",
        "out": "supported",
        "lse": "supported",
        "aux_tensors": "supported",
        "aux_scalars": "unsupported",
        "q_descale": "unsupported",
        "k_descale": "unsupported",
        "v_descale": "unsupported",
        "gather_kv_indices": "unsupported",
        "scheduler_metadata": "unsupported",
        "seqlen_k_per_split": "unsupported",
        "disable_scheduler_metadata": "default only",
        "gather_bwd_recompute_p": "default only",
    }
)

# Speed rule for sliding windows (auto mode only), measured on GB300 decode at windows 128 and 2048
# with 128-token pages. A window caps the keys per sequence; with few keys the standard kernel is
# faster.
SWAPAB_MIN_WINDOW = 1024  # window 128: SwapAB takes 1.0-2.2x the standard kernel's time
# TODO: windows between 128 and 2048 keys are not measured; 1024 is a midpoint.
# hd64, window 2048: SwapAB takes 1.02-1.18x the standard kernel's time once batch x kv heads
# >= 1024 (0.54-1.01x below). TODO: the crossover between 2k keys (loses) and 8k (wins) is not
# measured.
SWAPAB_HD64_SHORT_WINDOW = 2048
SWAPAB_HD64_STANDARD_MIN_SEQ_KV_HEADS = 1024  # smallest batch x kv heads on the standard kernel


@dataclass(frozen=True)
class SwapABDecision:
    use_swapab: bool
    reason: str
    logical_kv_capacity: int = 0

    @property
    def cache_token(self):
        # Bump the private backend version when its compiled ABI changes.
        return ("sm100_swapab", 1) if self.use_swapab else ("standard", 1)


def select_swapab_backend(
    request,
    *,
    q,
    k,
    v,
    page_table,
    arch,
    requires_grad,
    mode="auto",
    use_clc_scheduler=False,
    disable_s_ping_pong=False,
    disable_2cta=False,
):
    if mode not in ("auto", "standard", "swapab"):
        raise ValueError("FLASH_ATTENTION_SM100_BACKEND must be auto, standard, or swapab")
    if mode == "standard":
        return SwapABDecision(False, "standard backend requested")

    def reject(reason):
        if mode == "swapab":
            raise ValueError("SwapAB unsupported: " + reason)
        return SwapABDecision(False, reason)

    unknown = set(request) - FORWARD_OPTION_POLICY.keys()
    if unknown:
        return reject("unclassified forward options: " + ", ".join(sorted(unknown)))
    for name, policy in FORWARD_OPTION_POLICY.items():
        if policy == "unsupported" and request.get(name) is not None:
            return reject(name)
    for name, default in (
        ("num_threads", 384),
        ("disable_scheduler_metadata", False),
        ("gather_bwd_recompute_p", False),
    ):
        if request.get(name, default) != default:
            return reject(name)
    if request.get("pack_gqa") is False:
        return reject("pack_gqa")
    if use_clc_scheduler or disable_s_ping_pong:
        return reject("use_clc_scheduler / disable_s_ping_pong")
    scale = request.get("softmax_scale")
    # The transposed core scales the block maximum after reduction, which
    # requires a finite positive scale. Preserve the normal backend's behavior
    # for other values instead of silently changing score/max semantics.
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        return reject("nonpositive/nonfinite softmax_scale")
    if arch not in (100, 103):
        return reject("architecture must be SM100 or SM103")
    if requires_grad or any(t.requires_grad for t in request.get("aux_tensors") or ()):
        return reject("training")
    # Varlen Q is (total_q, h, d) with at most one token per sequence (host max_seqlen_q == 1).
    varlen_q = request.get("cu_seqlens_q") is not None
    max_seqlen_q = request.get("max_seqlen_q")
    if varlen_q and not (isinstance(max_seqlen_q, int) and max_seqlen_q == 1):
        return reject("cu_seqlens_q needs max_seqlen_q == 1")
    if (
        q is None
        or len(q.shape) != (3 if varlen_q else 4)
        or any(t is None or len(t.shape) != 4 for t in (k, v))
    ):
        return reject("rank-four K/V and Q ((total_q, h, d) with cu_seqlens_q) required")
    if str(q.dtype) not in ("torch.float16", "torch.bfloat16") or not q.dtype == k.dtype == v.dtype:
        return reject("dtype")
    if (
        (not varlen_q and q.shape[1] != 1)
        or q.shape[-1] not in (64, 96, 128)
        or v.shape[3] not in (64, 96, 128)
    ):
        return reject("single query / head dimensions")
    if k.shape[3] != q.shape[-1] or k.shape[:3] != v.shape[:3]:
        return reject("K/V shapes")
    if k.shape[2] <= 0 or q.shape[-2] % k.shape[2] or not 1 <= q.shape[-2] // k.shape[2] <= 32:
        return reject("GQA ratio")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        return reject("contiguous head dimension")
    batch_size = request["cu_seqlens_q"].shape[0] - 1 if varlen_q else q.shape[0]
    if page_table is not None:
        if k.shape[1] < 128 or k.shape[1] % 128:
            return reject("page size must be divisible by 128")
        capacity = page_table.shape[1] * k.shape[1]
    else:
        if batch_size != k.shape[0]:
            return reject("batch size")
        capacity = k.shape[1]
    window = request.get("window_size_left")
    if mode == "auto" and window is not None and window >= 0:
        short = window < SWAPAB_MIN_WINDOW or (
            q.shape[-1] == 64
            and window <= SWAPAB_HD64_SHORT_WINDOW
            and batch_size * k.shape[2] >= SWAPAB_HD64_STANDARD_MIN_SEQ_KV_HEADS
        )
        if short:
            return SwapABDecision(False, "short sliding window: the standard kernel is faster")
    return SwapABDecision(True, "eligible SM100/SM103 single-query inference", capacity)
