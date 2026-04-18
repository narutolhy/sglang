"""NVSHMEM-based MoE expert-parallel token dispatcher.

Uses `torch.ops.symm_mem.all_to_all_vdev` (1-D, no NUM_TILES16 cap) as the
underlying collective. Token layout: per-(token, k) slots grouped by target
rank. Output is wrapped into a `NvshmemDispatchOutput` consumed by
`fused_moe_triton.layer.FusedMoE.run_moe_core` via a bridge path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional, List

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    BaseDispatcherConfig,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


logger = logging.getLogger(__name__)


def is_nvshmem_available() -> bool:
    try:
        import torch.distributed._symmetric_memory as sm

        return bool(sm.is_nvshmem_available())
    except Exception:
        return False


class NvshmemDispatchOutput(NamedTuple):
    """Dispatch output for NVSHMEM backend.

    Layout: per-(token, k) slot mode. Each received row is a single token's
    hidden state tagged with one destination local expert and the source
    (source_rank, source_slot_idx) so combine can scatter back correctly.
    """

    hidden_states: torch.Tensor          # [recv_slots, H]
    topk_ids: torch.Tensor                # [recv_slots, 1] — local-expert idx
    topk_weights: torch.Tensor            # [recv_slots, 1]
    num_recv_tokens_per_expert: List[int] # [num_local_experts]
    # Bookkeeping for combine (typing.NamedTuple rejects leading-underscore names):
    sort_idx: torch.Tensor                # [M * top_k] permutation used on send
    per_rank_send: torch.Tensor           # [world_size] counts we sent per rank
    per_rank_recv: torch.Tensor           # [world_size] counts we received per rank
    orig_tokens: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.NVSHMEM


@dataclass
class NvshmemDispatcherConfig(BaseDispatcherConfig):
    num_experts: int
    num_local_experts: int
    hidden_size: int
    top_k: int
    max_tokens_per_rank: int = 16384
    dtype: torch.dtype = torch.bfloat16


class NvshmemDispatcher(BaseDispatcher):
    """1-D NVSHMEM all_to_all_vdev dispatcher for MoE EP."""

    def __init__(self, config: NvshmemDispatcherConfig):
        super().__init__()
        if not is_nvshmem_available():
            raise RuntimeError(
                "NVSHMEM backend is not available in this torch build. "
                "Need torch >= 2.11 with NVSHMEM support."
            )

        import torch.distributed._symmetric_memory as sm

        self.config = config
        self.group = dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.ne_local = config.num_local_experts
        self.ne_global = config.num_experts
        self.top_k = config.top_k
        self.H = config.hidden_size
        self.dtype = config.dtype

        sm.set_backend("NVSHMEM")
        self.group_name = self.group.group_name

        # Send side: this rank ships at most max_tokens_per_rank * top_k slots.
        # Recv side: under balanced routing ≈ same as send side; worst-case
        # routing (all top_k picks land on one rank) could need world_size ×
        # this, but sizing for that quadruples buffer memory. We keep the
        # balanced sizing and document the skew risk.
        max_slots = config.max_tokens_per_rank * config.top_k
        self.max_slots = max_slots
        # Packed slot layout: [H bf16 hidden | 2 bf16 reinterpreted as int32 id
        # | 2 bf16 reinterpreted as float32 weight | padding]. Carrying
        # id+weight inside the same symm_mem buffer cuts dispatch from 3 a2a
        # ops to 1. NVSHMEM's allToAllV asserts peer_size % block_size == 0;
        # 512-element alignment on stride keeps that true for any per-rank
        # slot count (empirically block_size is 512 for bf16 on B200).
        _raw_stride = config.hidden_size + 4
        _align = 512
        self._slot_stride_bf16 = (
            (_raw_stride + _align - 1) // _align
        ) * _align
        self.max_numel = max_slots * self._slot_stride_bf16

        dev = f"cuda:{torch.cuda.current_device()}"
        self._send_buf = sm.empty(self.max_numel, dtype=config.dtype, device=dev)
        self._recv_buf = sm.empty(self.max_numel, dtype=config.dtype, device=dev)
        # Combine's reverse a2a reuses the stride-aligned packed layout
        # (hidden goes in [0:H), tail [H:stride) is zero-padding). Matching
        # the dispatch stride avoids re-hitting the NVSHMEM block-alignment
        # assertion on peer_size.
        self._combine_send_buf = sm.empty(self.max_numel, dtype=config.dtype, device=dev)
        self._combine_recv_buf = sm.empty(self.max_numel, dtype=config.dtype, device=dev)
        # For 1-D all_to_all_vdev: in_splits [ws], out_splits_offsets [2, ws].
        self._in_splits = sm.empty(self.world_size, dtype=torch.int64, device=dev)
        self._out_splits_offsets = sm.empty(
            2 * self.world_size, dtype=torch.int64, device=dev
        ).view(2, self.world_size)
        # Second pair of splits for combine's a2a.
        self._combine_in_splits = sm.empty(
            self.world_size, dtype=torch.int64, device=dev
        )
        self._combine_out_splits_offsets = sm.empty(
            2 * self.world_size, dtype=torch.int64, device=dev
        ).view(2, self.world_size)

        for t in (
            self._send_buf, self._recv_buf,
            self._combine_send_buf, self._combine_recv_buf,
            self._in_splits, self._out_splits_offsets,
            self._combine_in_splits, self._combine_out_splits_offsets,
        ):
            sm.rendezvous(t.view(-1), self.group_name)

        # Pre-materialized GPU scalars — `tensor * python_int` triggers a
        # CPU→GPU scalar copy during CUDA-graph capture, so keep them
        # on-device.
        self._stride_t = torch.full(
            (1,), self._slot_stride_bf16, dtype=torch.int64, device=dev
        )
        self._H_t = torch.full((1,), self.H, dtype=torch.int64, device=dev)

        logger.info(
            "NvshmemDispatcher ready: world=%d ne_local=%d top_k=%d H=%d "
            "max_slots=%d, buf=%.1fMB each (packed)",
            self.world_size, self.ne_local, self.top_k, self.H, max_slots,
            self.max_numel * self.dtype.itemsize / (1 << 20),
        )
        # LayerCommunicator puts any non-none a2a backend into SCATTERED mlp
        # mode, which inserts a reduce_scatter/all_gather around every MoE
        # layer when attention is still in FULL mode. Pairing NVSHMEM with
        # --enable-attn-tp-input-scattered keeps the whole model SCATTERED
        # and cuts those per-layer NCCL ops (observed: 27→2ms ReduceScatter
        # on 1024-token prefill, ~21% concurrent-batch throughput).
        try:
            from sglang.srt.server_args import get_global_server_args
            if self.rank == 0 and not get_global_server_args().enable_attn_tp_input_scattered:
                logger.info(
                    "NvshmemDispatcher hint: consider "
                    "--enable-attn-tp-input-scattered for ~20-30%% throughput "
                    "win at serving-size batches (keeps model in SCATTERED "
                    "mode end-to-end, eliminates per-MoE NCCL rs/ag)."
                )
        except Exception:
            pass

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: "TopKOutput"
    ) -> DispatchOutput:
        """Per-(token, k) dispatch via a single packed NVSHMEM a2a.

        Slot layout in the packed bf16 buffer (stride = H + 4):
          [0 : H)       hidden_states (bf16)
          [H : H+2)     id (int32, viewed through 2 bf16 slots)
          [H+2 : H+4)   weight (float32, viewed through 2 bf16 slots)

        CUDA-graph-safe: returns FULL max-sized views instead of dynamic
        slices. Padding rows have weight=0 (pre-zeroed recv buffer) so their
        contribution to combine's sum is exactly zero.
        """
        M, H = hidden_states.shape
        assert H == self.H
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        slots = M * self.top_k
        assert slots <= self.max_slots, f"slots={slots} > max_slots={self.max_slots}"

        flat_ids = topk_ids.flatten()
        dst_rank = (flat_ids // self.ne_local).to(torch.int64)
        sort_idx = torch.argsort(dst_rank, stable=True)
        # Graph-safe per-rank count (bincount output shape is data-dependent).
        per_rank_send = torch.zeros(
            self.world_size, dtype=torch.int64, device=dst_rank.device
        )
        per_rank_send.scatter_add_(0, dst_rank, torch.ones_like(dst_rank))

        flat_hs = hidden_states.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, H)
        sorted_hs = flat_hs[sort_idx].contiguous()
        local_exp_id = (flat_ids % self.ne_local).to(torch.int32)[sort_idx].contiguous()
        sorted_weights = topk_weights.flatten()[sort_idx].to(torch.float32).contiguous()

        # Pre-zero recv so padding slots surface as weight=0 for the runner's
        # filter_expert mask + combine's weighted-sum. A full-buffer zero
        # coalesces better than a strided zero of just the weight region
        # (tried, lost ~4ms on 1024-token prefill).
        self._recv_buf.zero_()

        # Pack into a single buffer. Slot layout (byte view, stride bytes
        # per slot = self._slot_stride_bf16 * 2):
        #   [0 : 2*H)          hidden (bf16)
        #   [2*H : 2*H+4)      int32 id
        #   [2*H+4 : 2*H+8)    float32 weight
        #   [2*H+8 : stride*2) padding
        stride = self._slot_stride_bf16
        send_view = self._send_buf.view(self.max_slots, stride)
        send_view[:slots, :H].copy_(sorted_hs)
        # Strided byte-level writes into the meta region. copy_ handles
        # non-contiguous destinations just fine.
        slot_size_bytes = stride * 2
        meta_off = H * 2
        buf_bytes = self._send_buf.view(torch.uint8).view(
            self.max_slots, slot_size_bytes
        )
        buf_bytes[:slots, meta_off : meta_off + 4].copy_(
            local_exp_id.view(torch.uint8).view(slots, 4)
        )
        buf_bytes[:slots, meta_off + 4 : meta_off + 8].copy_(
            sorted_weights.view(torch.uint8).view(slots, 4)
        )

        # in_splits are ELEMENT counts (bf16) per destination rank.
        self._in_splits.copy_(per_rank_send * self._stride_t)

        torch.ops.symm_mem.all_to_all_vdev(
            self._send_buf, self._recv_buf, self._in_splits, self._out_splits_offsets,
            self.group_name,
        )

        # Unpack views over the max-sized recv buffer. Use a byte view for
        # id/weight and a bf16 view for hidden; the hidden slice has stride
        # (stride, 1) so it's not contiguous — fused_experts asserts
        # contiguity, so copy once (one layer ≈ 4MB, <5us on B200 HBM).
        recv_view = self._recv_buf.view(self.max_slots, stride)
        recv_hs = recv_view[:, :H].contiguous()  # [max_slots, H] bf16
        recv_bytes = self._recv_buf.view(torch.uint8).view(
            self.max_slots, slot_size_bytes
        )
        recv_ids = (
            recv_bytes[:, meta_off : meta_off + 4]
            .contiguous()
            .view(-1)
            .view(torch.int32)
            .view(self.max_slots, 1)
        )
        recv_w = (
            recv_bytes[:, meta_off + 4 : meta_off + 8]
            .contiguous()
            .view(-1)
            .view(torch.float32)
            .view(self.max_slots, 1)
        )
        # per_rank_recv in slot-rows: elements received / stride.
        per_rank_recv = (self._out_splits_offsets[0] // self._stride_t)

        return NvshmemDispatchOutput(
            hidden_states=recv_hs,
            topk_ids=recv_ids,
            topk_weights=recv_w,
            num_recv_tokens_per_expert=[],
            sort_idx=sort_idx,
            per_rank_send=per_rank_send,
            per_rank_recv=per_rank_recv,
            orig_tokens=M,
        )

    def combine(self, combine_input: "NvshmemCombineInput") -> torch.Tensor:
        """Reverse dispatch: weighted expert outputs → scatter back to [M, H].

        Graph-safe: derives all shapes from orig_tokens (Python int, known at
        capture time) instead of summing a GPU tensor. Padding rows in the
        max-sized expert_out have weights=0 so weighted is exactly 0 there.
        """
        expert_out = combine_input.hidden_states     # [max_slots, H]
        weights = combine_input.weights              # [max_slots, 1], pad=0
        sort_idx = combine_input.sort_idx            # [M * top_k]
        per_rank_recv = combine_input.per_rank_recv  # on-device int64
        orig_M = combine_input.orig_tokens           # Python int
        H = self.H
        total_sent = orig_M * self.top_k             # Python int

        weighted = expert_out * weights
        # Pack weighted hidden into stride-aligned slot layout. The tail
        # [H:stride) bytes are never read by the receiver (it only reads
        # [:, :H] on each slot), so skip zeroing them; the stride alignment
        # is purely to satisfy NVSHMEM's peer_size % block_size check.
        stride = self._slot_stride_bf16
        combine_send_view = self._combine_recv_buf.view(self.max_slots, stride)
        combine_send_view[:, :H].copy_(weighted)
        self._combine_in_splits.copy_(per_rank_recv * self._stride_t)

        torch.ops.symm_mem.all_to_all_vdev(
            self._combine_recv_buf, self._combine_send_buf,
            self._combine_in_splits, self._combine_out_splits_offsets,
            self.group_name,
        )

        # Strip tail padding: sorted_out[i] = send_buf_view[i, :H].
        send_out_view = self._combine_send_buf.view(self.max_slots, stride)
        sorted_out = send_out_view[:total_sent, :H]

        # Inverse permute back to (M, top_k) order, then sum over top_k.
        inv_sort = torch.empty_like(sort_idx)
        inv_sort[sort_idx] = torch.arange(total_sent, device=sort_idx.device)
        unsorted = sorted_out[inv_sort]
        return unsorted.view(orig_M, self.top_k, H).sum(dim=1)


class NvshmemCombineInput(NamedTuple):
    """Combine-side input that pairs expert outputs with dispatch bookkeeping."""

    hidden_states: torch.Tensor      # [total_recv, H] — expert outputs
    weights: torch.Tensor             # [total_recv, 1] — topk weights
    sort_idx: torch.Tensor            # dispatch bookkeeping
    per_rank_send: torch.Tensor
    per_rank_recv: torch.Tensor
    orig_tokens: int

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.STANDARD


def run_nvshmem_moe(
    layer, dispatch_output: "NvshmemDispatchOutput"
) -> "NvshmemCombineInput":
    """Unquantized grouped GEMM for the NVSHMEM dispatch layout.

    Each received row is one (token, k) slot already routed to the rank that
    owns its target expert. Feed the rows to the triton fused-moe kernel with
    top_k=1 (each slot → one local expert); topk_weights are passed as ones
    here because combine() still has to weight and sum across the original
    top_k slots from the sender side.

    Weights layout (matches unquant.UnquantizedFusedMoEMethod.create_weights):
      w13_weight[ne_local, 2*I, H]  — fused gate+up
      w2_weight[ne_local, H, I]     — down
    """
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_impl

    x = dispatch_output.hidden_states          # [max_slots, H], bf16, contig
    if x.shape[0] == 0:
        out = torch.empty_like(x)
    else:
        # Each received row is already routed to one local expert. Padding
        # rows (weight==0) get their topk_id rewritten to ne_local (an
        # out-of-range id) so fused_experts' filter_expert=True path
        # skips them at moe_align_block_size time instead of lumping them
        # into expert 0's grouped GEMM as padding.
        ne_local = layer.w13_weight.shape[0]
        topk_ids_2d = dispatch_output.topk_ids.to(torch.int32)
        weights_raw = dispatch_output.topk_weights             # [max_slots, 1]
        padding_mask = weights_raw == 0
        # Use where() — graph-safe, no .item(), creates a new tensor.
        topk_ids_2d = torch.where(
            padding_mask.to(torch.bool),
            torch.full_like(topk_ids_2d, ne_local),
            topk_ids_2d,
        )
        ones_w = torch.ones_like(topk_ids_2d, dtype=torch.float32)
        cfg = layer.moe_runner_config
        out = fused_experts_impl(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=ones_w,
            topk_ids=topk_ids_2d,
            activation=cfg.activation,
            is_gated=cfg.is_gated,
            apply_router_weight_on_input=False,
            inplace=False,
            filter_expert=True,
        )  # [max_slots, H] — out rows for padding slots are garbage but
           # their combine weight is 0 so they don't contribute to the sum.

    return NvshmemCombineInput(
        hidden_states=out,
        weights=dispatch_output.topk_weights.to(x.dtype),
        sort_idx=dispatch_output.sort_idx,
        per_rank_send=dispatch_output.per_rank_send,
        per_rank_recv=dispatch_output.per_rank_recv,
        orig_tokens=dispatch_output.orig_tokens,
    )
