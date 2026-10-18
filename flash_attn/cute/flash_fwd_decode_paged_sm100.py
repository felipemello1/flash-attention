"""GQA decode with swapped MMA operands, head_dim 256, SM100 / SM103.

In decode each sequence has one query token, so a kv head's tile holds only its G q heads
(4 or 6 for Qwen3.5). The general forward puts them on the M axis of a 128-row tile and pays the
full tile for every key block. This kernel swaps the MMA operands, the layout TRT-LLM and
FlashInfer use for GQA decode ("swapsMmaAb"):

    S^T = K Q^T     M = 128 keys,            N = 16 (G q heads, padded), K = 256 dims
    O^T = V^T P^T   M = 128 dims (per half), N = 16 q heads,             K = 128 keys

so a key block costs two 128 x 16 MMAs instead of two 128 x 128 ones.

One CTA per (split, kv head, batch):
- warps 0-3: thread t owns TMEM lane t, i.e. key t of S^T and dims t, 128 + t of O^T. Softmax
  reduces over keys, i.e. across threads (warp redux + a 4-warp exchange). O stays in registers,
  rescaled per block, so no correction warpgroup is needed; the same warps write the epilogue.
- warp 4: MMA. S(i + 1) is issued before PV(i), so the softmax of block i overlaps S(i + 1).
- warp 5: TMA. Q once; K and V in 128-dim halves (32 KB) through a 6-stage ring.

Keys come from a paged cache with 128-token pages (one page per key block). With num_splits > 1
each split writes an fp32 partial O and LSE for flash_fwd_combine.
"""

import math
from functools import partial
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05

from flash_attn.cute import blackwell_helpers as fa_sm100_utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned


class FlashAttentionDecodeSwappedSm100:
    head_dim = 256
    n_block_size = 128  # keys per block, == page size
    n_heads_padded = 16  # MMA N; tcgen05 M = 128 needs N % 16 == 0
    kv_stage = 6  # ring of 128 x 128 half tiles (32 KB each)
    tmem_o_offset = 2 * 16  # S uses columns [0, 32), O uses [32, 96)

    compute_warp_ids = (0, 1, 2, 3)
    mma_warp_id = 4
    load_warp_id = 5
    threads_per_cta = 32 * 6

    def __init__(self, qhead_per_kvhead: int, is_split_kv: bool):
        assert 1 <= qhead_per_kvhead <= self.n_heads_padded
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv

    @staticmethod
    def can_implement(
        *,
        head_dim: int,
        head_dim_v: int,
        qhead_per_kvhead: int,
        page_size: Optional[int],
        max_seqlen_q: int,
        is_fp8: bool,
    ) -> bool:
        return (
            head_dim == 256
            and head_dim_v == 256
            and qhead_per_kvhead <= 16
            and page_size == 128
            and max_seqlen_q == 1
            and not is_fp8
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (rows, h, d)
        mK: cute.Tensor,  # (num_pages, page_size, h_k, d)
        mV: cute.Tensor,  # (num_pages, page_size, h_k, d)
        mO: cute.Tensor,  # (rows, h, d), or (num_splits, rows, h, d) fp32 for split-KV
        mLSE: Optional[cute.Tensor],  # (rows, h), or (num_splits, rows, h)
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor],  # (b + 1,): row of batch b; rows == b if None
        mSeqUsedK: Optional[cute.Tensor],  # (b,)
        mPageTable: cute.Tensor,  # (b, max_pages)
        num_splits: Int32,
        stream: cuda.CUstream = None,
    ):
        self.dtype = mQ.element_type
        mQ, mK, mV = [assume_tensor_aligned(t) for t in (mQ, mK, mV)]
        # Q^T is the B operand of S^T = K Q^T: (h, d, rows), K-major.
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 2, 0]))
        # K is the A operand: (page_size, d, h_k, num_pages), K-major.
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 3, 2, 0]))
        # V^T is the A operand of O^T = V^T P^T: (d, page_size, h_k, num_pages), MN-major.
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[3, 1, 2, 0]))

        self.mma_tiler = (128, self.n_heads_padded, 128)
        tiled_mma_s = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        tiled_mma_o = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        # K and V^T half tiles have the same physical layout, so a ring stage holds either.
        sK_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_s, self.mma_tiler, self.dtype, self.kv_stage
        )
        sV_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_o, self.mma_tiler, self.dtype, self.kv_stage
        )
        # Q^T: one stage per 128-dim half. P^T: double buffered.
        sQ_layout = sm100_utils.make_smem_layout_b(tiled_mma_s, self.mma_tiler, self.dtype, 2)
        sP_layout = sm100_utils.make_smem_layout_b(tiled_mma_o, self.mma_tiler, self.dtype, 2)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma_s.thr_id.shape,)
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler,
            tiled_mma_s,
            cta_layout_vmnk.shape,
        )
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler,
            tiled_mma_s,
            cta_layout_vmnk.shape,
        )
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler,
            tiled_mma_o,
            cta_layout_vmnk.shape,
        )
        self.tma_bytes_kv = cute.size_in_bytes(self.dtype, cute.select(sK_layout, mode=[0, 1, 2]))
        self.tma_bytes_q = 2 * cute.size_in_bytes(
            self.dtype, cute.select(sQ_layout, mode=[0, 1, 2])
        )

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[Int64, 2]
            mbar_kv: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_s: cute.struct.MemRange[Int64, 2 * 2]
            mbar_p: cute.struct.MemRange[Int64, 2 * 2]
            mbar_o: cute.struct.MemRange[Int64, 2 * 2]
            tmem_holding_buf: Int32
            # Per-warp, per-head block max (double buffered by block parity) and row sum.
            sMax: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 4 * 16], 16]
            sSum: cute.struct.Align[cute.struct.MemRange[Float32, 4 * 16], 16]
            sQ: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sQ_layout)], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sP_layout)], 1024]
            sKV: cute.struct.Align[
                cute.struct.MemRange[
                    self.dtype, cutlass.max(cute.cosize(sK_layout), cute.cosize(sV_layout))
                ],
                1024,
            ]

        self.shared_storage = SharedStorage
        softmax_scale_log2 = softmax_scale * math.log2(math.e)
        batch_size = mPageTable.shape[0]
        num_head_kv = mK.shape[2]
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            softmax_scale_log2,
            num_splits,
            sQ_layout,
            sK_layout,
            sV_layout,
            sP_layout,
            tiled_mma_s,
            tiled_mma_o,
        ).launch(
            grid=[num_splits, num_head_kv, batch_size],
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        softmax_scale_log2: Float32,
        num_splits: Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        tiled_mma_s: cute.TiledMma,
        tiled_mma_o: cute.TiledMma,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        split_idx, head_idx_kv, batch_idx = cute.arch.block_idx()
        if warp_idx == self.load_warp_id:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V):
                cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=cute.arch.WARP_SIZE * (len(self.compute_warp_ids) + 1)
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
        )

        thread_group = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        pipeline_q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_q.data_ptr(),
            num_stages=1,
            producer_group=thread_group(1),
            consumer_group=thread_group(1),
            tx_count=self.tma_bytes_q,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_kv = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_kv.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=thread_group(1),
            consumer_group=thread_group(1),
            tx_count=self.tma_bytes_kv,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # S^T ready (MMA -> compute); S buffer free (compute -> MMA).
        pipeline_s = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_s.data_ptr(),
            num_stages=2,
            producer_group=thread_group(1),
            consumer_group=thread_group(num_compute_threads),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # P^T written (compute -> MMA); P buffer free once PV completes (MMA -> compute).
        pipeline_p = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_p.data_ptr(),
            num_stages=2,
            producer_group=thread_group(num_compute_threads),
            consumer_group=thread_group(1),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # PV(i) ready in TMEM (MMA -> compute); O buffer free (compute -> MMA).
        pipeline_o = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_o.data_ptr(),
            num_stages=2,
            producer_group=thread_group(1),
            consumer_group=thread_group(num_compute_threads),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sK = storage.sKV.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)

        thr_mma_s = tiled_mma_s.get_slice(0)
        thr_mma_o = tiled_mma_o.get_slice(0)
        # TMEM is allocated whole (one CTA per SM), so it starts at column 0.
        tStS = thr_mma_s.make_fragment_C(
            cute.append(thr_mma_s.partition_shape_C(self.mma_tiler[:2]), 2)
        )
        # O^T buffers: index 2 * buffer + dim half.
        tOtO = thr_mma_o.make_fragment_C(
            cute.append(thr_mma_o.partition_shape_C(self.mma_tiler[:2]), 4)
        )
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO.layout)

        # The key range of this split.
        seqlen_k = (
            mSeqUsedK[batch_idx]
            if const_expr(mSeqUsedK is not None)
            else mPageTable.shape[1] * self.n_block_size
        )
        q_row = mCuSeqlensQ[batch_idx] if const_expr(mCuSeqlensQ is not None) else batch_idx
        has_query = (
            mCuSeqlensQ[batch_idx + 1] > q_row if const_expr(mCuSeqlensQ is not None) else True
        )
        num_n_blocks = (seqlen_k + self.n_block_size - 1) // self.n_block_size
        n_blocks_per_split = (num_n_blocks + num_splits - 1) // num_splits
        n_block_min = split_idx * n_blocks_per_split
        n_block_max = cutlass.min(n_block_min + n_blocks_per_split, num_n_blocks)
        num_iters = n_block_max - n_block_min if has_query else 0

        if warp_idx == self.load_warp_id:
            self.load(
                mQ,
                mK,
                mV,
                mPageTable,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                thr_mma_s,
                thr_mma_o,
                pipeline_q,
                pipeline_kv,
                head_idx_kv,
                batch_idx,
                q_row,
                n_block_min,
                num_iters,
            )

        if warp_idx == self.mma_warp_id:
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.mma(
                tiled_mma_s,
                tiled_mma_o,
                sQ,
                sK,
                sV,
                sP,
                tStS,
                tOtO,
                pipeline_q,
                pipeline_kv,
                pipeline_s,
                pipeline_p,
                pipeline_o,
                num_iters,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if warp_idx < self.mma_warp_id:
            tmem.wait_for_alloc()
            self.compute(
                mO,
                mLSE,
                sP,
                tStS,
                tOtO,
                storage,
                pipeline_s,
                pipeline_p,
                pipeline_o,
                softmax_scale_log2,
                seqlen_k,
                split_idx,
                head_idx_kv,
                q_row,
                has_query,
                n_block_min,
                num_iters,
            )
            tmem_alloc_barrier.arrive()

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPageTable: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        thr_mma_s: cute.ThrMma,
        thr_mma_o: cute.ThrMma,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        head_idx_kv: Int32,
        batch_idx: Int32,
        q_row: Int32,
        n_block_min: Int32,
        num_iters: Int32,
    ):
        kv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        if num_iters > 0:
            # Q^T: the kv head's G q heads; the other padded rows read the next heads or
            # zero-filled out-of-bounds rows, and only feed S columns that are never stored.
            mQ_cur = cute.domain_offset(
                (head_idx_kv * self.qhead_per_kvhead, 0), mQ[None, None, q_row]
            )
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler, mode=[1, 2]), (0, None))
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_Q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(thr_mma_s.partition_B(gQ), 0, 3),
            )
            q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            pipeline_q.producer_acquire(q_state)
            for half in cutlass.range_constexpr(2):
                cute.copy(
                    tma_atom_Q,
                    tQgQ[None, half],
                    tQsQ[None, half],
                    tma_bar_ptr=pipeline_q.producer_get_barrier(q_state),
                )

            # (128, 128, half, page)
            gK = cute.local_tile(
                mK[None, None, head_idx_kv, None],
                cute.select(self.mma_tiler, mode=[0, 2]),
                (0, None, None),
            )
            gV = cute.local_tile(
                mV[None, None, head_idx_kv, None],
                cute.select(self.mma_tiler, mode=[0, 2]),
                (None, 0, None),
            )
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(thr_mma_s.partition_A(gK), 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(thr_mma_o.partition_A(gV), 0, 3),
            )

            # Same order as the MMA warp: K(0), then K(i + 1), V(i), then V(last).
            kv_state = self.load_kv_block(
                tma_atom_K, tKgK, tKsK, mPageTable[batch_idx, n_block_min], pipeline_kv, kv_state
            )
            for i in cutlass.range(num_iters - 1, unroll=1):
                n_block = n_block_min + i
                kv_state = self.load_kv_block(
                    tma_atom_K,
                    tKgK,
                    tKsK,
                    mPageTable[batch_idx, n_block + 1],
                    pipeline_kv,
                    kv_state,
                )
                kv_state = self.load_kv_block(
                    tma_atom_V, tVgV, tVsV, mPageTable[batch_idx, n_block], pipeline_kv, kv_state
                )
            kv_state = self.load_kv_block(
                tma_atom_V,
                tVgV,
                tVsV,
                mPageTable[batch_idx, n_block_min + num_iters - 1],
                pipeline_kv,
                kv_state,
            )
            pipeline_kv.producer_tail(kv_state)

    @cute.jit
    def load_kv_block(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        page_idx: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        kv_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load one key block's K or V into two ring stages (one per 128-dim half)."""
        for half in cutlass.range_constexpr(2):
            pipeline_kv.producer_acquire(kv_state)
            cute.copy(
                tma_atom,
                tXgX[None, half, page_idx],
                tXsX[None, kv_state.index],
                tma_bar_ptr=pipeline_kv.producer_get_barrier(kv_state),
            )
            kv_state.advance()
        return kv_state

    @cute.jit
    def mma(
        self,
        tiled_mma_s: cute.TiledMma,
        tiled_mma_o: cute.TiledMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sP: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s: pipeline.PipelineAsync,
        pipeline_p: pipeline.PipelineAsync,
        pipeline_o: pipeline.PipelineAsync,
        num_iters: Int32,
    ):
        tSrK = tiled_mma_s.make_fragment_A(sK)
        tSrQ = tiled_mma_s.make_fragment_B(sQ)
        tOrV = tiled_mma_o.make_fragment_A(sV)
        tOrP = tiled_mma_o.make_fragment_B(sP)
        kv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
        s_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
        p_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
        o_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)

        if num_iters > 0:
            q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            pipeline_q.consumer_wait(q_state)
            mma_s = partial(self.mma_s, tiled_mma_s, tStS, tSrK, tSrQ, pipeline_kv, pipeline_s)
            mma_pv = partial(
                self.mma_pv, tiled_mma_o, tOtO, tOrV, tOrP, pipeline_kv, pipeline_p, pipeline_o
            )
            kv_state, s_state = mma_s(kv_state, s_state)
            for i in cutlass.range(num_iters - 1, unroll=1):
                kv_state, s_state = mma_s(kv_state, s_state)
                kv_state, p_state, o_state = mma_pv(kv_state, p_state, o_state)
            kv_state, p_state, o_state = mma_pv(kv_state, p_state, o_state)

    @cute.jit
    def mma_s(
        self,
        tiled_mma_s: cute.TiledMma,
        tStS: cute.Tensor,
        tSrK: cute.Tensor,
        tSrQ: cute.Tensor,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s: pipeline.PipelineAsync,
        kv_state: pipeline.PipelineState,
        s_state: pipeline.PipelineState,
    ):
        """S^T = K Q^T for one key block, summed over the two 128-dim halves."""
        pipeline_s.producer_acquire(s_state)
        for half in cutlass.range_constexpr(2):
            pipeline_kv.consumer_wait(kv_state)
            fa_sm100_utils.gemm(
                tiled_mma_s,
                tStS[None, None, None, s_state.index],
                tSrK[None, None, None, kv_state.index],
                tSrQ[None, None, None, half],
                zero_init=half == 0,
            )
            pipeline_kv.consumer_release(kv_state)
            kv_state.advance()
        pipeline_s.producer_commit(s_state)
        s_state.advance()
        return kv_state, s_state

    @cute.jit
    def mma_pv(
        self,
        tiled_mma_o: cute.TiledMma,
        tOtO: cute.Tensor,
        tOrV: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_p: pipeline.PipelineAsync,
        pipeline_o: pipeline.PipelineAsync,
        kv_state: pipeline.PipelineState,
        p_state: pipeline.PipelineState,
        o_state: pipeline.PipelineState,
    ):
        """O^T = V^T P^T for one key block, one 128-dim half per MMA, into a fresh buffer."""
        pipeline_p.consumer_wait(p_state)
        pipeline_o.producer_acquire(o_state)
        for half in cutlass.range_constexpr(2):
            pipeline_kv.consumer_wait(kv_state)
            fa_sm100_utils.gemm(
                tiled_mma_o,
                tOtO[None, None, None, 2 * o_state.index + half],
                tOrV[None, None, None, kv_state.index],
                tOrP[None, None, None, p_state.index],
                zero_init=True,
            )
            pipeline_kv.consumer_release(kv_state)
            kv_state.advance()
        pipeline_p.consumer_release(p_state)
        pipeline_o.producer_commit(o_state)
        p_state.advance()
        o_state.advance()
        return kv_state, p_state, o_state

    @cute.jit
    def compute(
        self,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sP: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        storage,
        pipeline_s: pipeline.PipelineAsync,
        pipeline_p: pipeline.PipelineAsync,
        pipeline_o: pipeline.PipelineAsync,
        softmax_scale_log2: Float32,
        seqlen_k: Int32,
        split_idx: Int32,
        head_idx_kv: Int32,
        q_row: Int32,
        has_query: cutlass.Boolean,
        n_block_min: Int32,
        num_iters: Int32,
    ):
        H = self.n_heads_padded
        num_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_threads
        warp = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        # Lane t of every accumulator is key t (S^T) or dim t of a half (O^T).
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(H)), Float32
        )
        tSAcc = tStS[(None, None), 0, 0, 0]
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO[(None, None), 0, 0, 0])
        acc_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(self.mma_tiler[:2])).shape
        tSrS = cute.make_rmem_tensor(acc_shape, Float32)
        tOrO_t2r = cute.make_rmem_tensor(acc_shape, Float32)

        sMax = storage.sMax.get_tensor(cute.make_layout((2, 4, H), stride=(4 * H, H, 1)))
        sSum = storage.sSum.get_tensor(cute.make_layout((4, H), stride=(H, 1)))
        compute_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=num_threads)

        row_max = cute.make_rmem_tensor((H,), Float32)  # running max, log2-scaled
        row_sum = cute.make_rmem_tensor((H,), Float32)  # this thread's keys only
        scale = cute.make_rmem_tensor((H,), Float32)  # rescale of O for the current block
        scale_prev = cute.make_rmem_tensor((H,), Float32)
        acc_O = cute.make_rmem_tensor((2, H), Float32)  # dims tidx and 128 + tidx
        tSrP = cute.make_rmem_tensor((H,), self.dtype)
        row_max.fill(-Float32.inf)
        row_sum.fill(0.0)
        scale_prev.fill(0.0)
        acc_O.fill(0.0)

        s_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
        p_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
        o_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)

        for i in cutlass.range(num_iters, unroll=1):
            parity = i % 2
            pipeline_s.consumer_wait(s_state)
            tStS_cur = cute.make_tensor(tStS_t2r.iterator + H * s_state.index, tStS_t2r.layout)
            cute.copy(thr_tmem_load, tStS_cur, tSrS)
            cute.arch.fence_view_async_tmem_load()
            pipeline_s.consumer_release(s_state)
            s_state.advance()

            # Keys past seqlen_k (the tail of the last page) get -inf.
            in_bound = (n_block_min + i) * self.n_block_size + tidx < seqlen_k
            for h in cutlass.range_constexpr(H):
                tSrS[h] = tSrS[h] * softmax_scale_log2 if in_bound else -Float32.inf
                block_max = cute.arch.warp_redux_sync(tSrS[h], "fmax")
                if lane == 0:
                    sMax[parity, warp, h] = block_max
            compute_barrier.arrive_and_wait()
            for h in cutlass.range_constexpr(H):
                block_max = cutlass.max(
                    cutlass.max(sMax[parity, 0, h], sMax[parity, 1, h]),
                    cutlass.max(sMax[parity, 2, h], sMax[parity, 3, h]),
                )
                row_max_new = cutlass.max(row_max[h], block_max)
                # No finite key yet: keep exponents finite (exp2(-inf - 0) == 0).
                row_max_safe = row_max_new if row_max_new != -Float32.inf else 0.0
                p = cute.math.exp2(tSrS[h] - row_max_safe, fastmath=True)
                scale[h] = cute.math.exp2(row_max[h] - row_max_safe, fastmath=True)
                row_sum[h] = row_sum[h] * scale[h] + p
                row_max[h] = row_max_new
                tSrP[h] = p.to(self.dtype)

            # P^T row of key tidx: 16 contiguous heads.
            pipeline_p.producer_acquire(p_state)
            sP_row = sP[((None, tidx % 16), 0, tidx // 16, p_state.index)]
            cute.autovec_copy(tSrP, sP_row)
            cute.arch.fence_proxy("async.shared", space="cta")
            pipeline_p.producer_commit(p_state)
            p_state.advance()

            if i > 0:
                o_state = self.accumulate_pv(
                    thr_tmem_load, tOtO_t2r, tOrO_t2r, acc_O, scale_prev, pipeline_o, o_state
                )
            for h in cutlass.range_constexpr(H):
                scale_prev[h] = scale[h]
        if num_iters > 0:
            o_state = self.accumulate_pv(
                thr_tmem_load, tOtO_t2r, tOrO_t2r, acc_O, scale_prev, pipeline_o, o_state
            )

        # Row sums over all keys: warp shuffle, then across the 4 warps.
        for h in cutlass.range_constexpr(H):
            total = row_sum[h]
            for k in cutlass.range_constexpr(5):
                total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << k)
            if lane == 0:
                sSum[warp, h] = total
        compute_barrier.arrive_and_wait()
        for h in cutlass.range_constexpr(H):
            row_sum[h] = (sSum[0, h] + sSum[1, h]) + (sSum[2, h] + sSum[3, h])

        if has_query:
            G = self.qhead_per_kvhead
            head_base = head_idx_kv * G
            mO_cur = (
                mO[split_idx, q_row, None, None]
                if const_expr(self.is_split_kv)
                else mO[q_row, None, None]
            )
            for h in cutlass.range_constexpr(G):
                inv_sum = 1.0 / row_sum[h] if row_sum[h] > 0.0 else 0.0
                for half in cutlass.range_constexpr(2):
                    mO_cur[head_base + h, half * 128 + tidx] = (acc_O[half, h] * inv_sum).to(
                        mO.element_type
                    )
            if const_expr(mLSE is not None):
                mLSE_cur = (
                    mLSE[split_idx, q_row, None]
                    if const_expr(self.is_split_kv)
                    else mLSE[q_row, None]
                )
                for h in cutlass.range_constexpr(G):
                    if tidx == h:
                        lse = (
                            (row_max[h] + cute.math.log2(row_sum[h], fastmath=True)) * math.log(2.0)
                            if row_sum[h] > 0.0
                            else -Float32.inf
                        )
                        mLSE_cur[head_base + h] = lse

    @cute.jit
    def accumulate_pv(
        self,
        thr_tmem_load: cute.TiledCopy,
        tOtO_t2r: cute.Tensor,
        tOrO_t2r: cute.Tensor,
        acc_O: cute.Tensor,
        scale_prev: cute.Tensor,
        pipeline_o: pipeline.PipelineAsync,
        o_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """acc_O = acc_O * scale_prev + PV of the previous block, one head per column."""
        H = self.n_heads_padded
        pipeline_o.consumer_wait(o_state)
        for half in cutlass.range_constexpr(2):
            tOtO_cur = cute.make_tensor(
                tOtO_t2r.iterator + H * (2 * o_state.index + half), tOtO_t2r.layout
            )
            cute.copy(thr_tmem_load, tOtO_cur, tOrO_t2r)
            for h in cutlass.range_constexpr(H):
                acc_O[half, h] = acc_O[half, h] * scale_prev[h] + tOrO_t2r[h]
        cute.arch.fence_view_async_tmem_load()
        pipeline_o.consumer_release(o_state)
        o_state.advance()
        return o_state
