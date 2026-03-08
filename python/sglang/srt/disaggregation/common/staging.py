"""
GPU Staging Buffer for heterogeneous TP KV cache transfer.

When prefill attn_tp_size != decode attn_tp_size, the per-token RDMA approach
generates O(tokens * layers) small RDMA requests. This module provides a staging
buffer mechanism that gathers scattered head slices into contiguous GPU memory,
enabling bulk RDMA transfers that reduce request count to O(layers) or O(1).

Usage:
    Activated by setting SGLANG_DISAGG_STAGING_BUFFER=1.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class StagingBuffer:
    """Pre-allocated GPU staging buffer for bulk KV transfer.

    When a custom_mem_pool is provided (e.g., mooncake NVLink allocator),
    the buffer is allocated within that pool so it's compatible with
    NVLink/MNNVL transport (requires cuMemCreate-backed memory).
    """

    def __init__(
        self,
        size_bytes: int,
        device: str,
        gpu_id: int,
        custom_mem_pool=None,
    ):
        self.size_bytes = size_bytes
        self.device = device
        self.gpu_id = gpu_id

        torch.cuda.set_device(gpu_id)
        if custom_mem_pool is not None:
            with torch.cuda.use_mem_pool(custom_mem_pool):
                self.buffer = torch.empty(size_bytes, dtype=torch.uint8, device=device)
            alloc_method = "custom_mem_pool (cuMemCreate)"
        else:
            self.buffer = torch.empty(size_bytes, dtype=torch.uint8, device=device)
            alloc_method = "cudaMalloc (NVLink incompatible!)"
        self.data_ptr = self.buffer.data_ptr()

        logger.info(
            f"StagingBuffer allocated: {size_bytes / (1024*1024):.1f} MB "
            f"on {device}, method={alloc_method}, ptr=0x{self.data_ptr:x}"
        )

    def get_ptr(self) -> int:
        return self.data_ptr

    def get_size(self) -> int:
        return self.size_bytes

    def fits(self, required_bytes: int) -> bool:
        return required_bytes <= self.size_bytes


class RecvSlotPool:
    """Decode-side pre-allocated recv staging slot pool.

    Each slot is a StagingBuffer that can hold one request's KV data
    from all prefill ranks. Slots are assigned per-request and freed
    after scatter completes. Thread-safe via mutex.
    """

    def __init__(
        self,
        num_slots: int,
        slot_size_bytes: int,
        device: str,
        gpu_id: int,
        custom_mem_pool=None,
    ):
        self.num_slots = num_slots
        self.slot_size_bytes = slot_size_bytes
        self.slots: List[StagingBuffer] = []
        self.flags: List[int] = [0] * num_slots  # 0=free, 1=occupied
        self.lock = threading.Lock()

        for i in range(num_slots):
            buf = StagingBuffer(slot_size_bytes, device, gpu_id, custom_mem_pool)
            self.slots.append(buf)

        logger.info(
            f"RecvSlotPool: {num_slots} slots x "
            f"{slot_size_bytes / (1024*1024):.1f} MB = "
            f"{num_slots * slot_size_bytes / (1024*1024):.1f} MB total"
        )

    def assign(self) -> Optional[int]:
        """Assign a free slot. Returns slot_id or None if all occupied."""
        with self.lock:
            for i, flag in enumerate(self.flags):
                if flag == 0:
                    self.flags[i] = 1
                    return i
            return None

    def free(self, slot_id: int):
        """Release a slot back to the pool."""
        with self.lock:
            self.flags[slot_id] = 0

    def get_slot(self, slot_id: int) -> StagingBuffer:
        return self.slots[slot_id]

    def get_all_ptrs(self) -> List[int]:
        """Return all slot GPU pointers for bootstrap registration."""
        return [slot.get_ptr() for slot in self.slots]

    def get_all_sizes(self) -> List[int]:
        return [slot.get_size() for slot in self.slots]


class StagingAllocator:
    """Decode-side dynamic staging ring buffer allocator with overcommit.

    One large pre-allocated GPU buffer used as a ring buffer. Each request
    gets a (alloc_id, offset, round) triple based on its actual byte
    requirement. Allocation (assign) is overcommit — it always succeeds
    as long as the request fits in the buffer. Overlap safety is enforced
    on the prefill side before RDMA, using a watermark that tracks the
    oldest un-freed allocation.

    The watermark (round, tail_offset) is periodically sent to prefill.
    Prefill transfer workers wait before writing if their target region
    overlaps with not-yet-freed data from a previous round.
    """

    def __init__(
        self,
        total_size_bytes: int,
        device: str,
        gpu_id: int,
        custom_mem_pool=None,
    ):
        self.buffer = StagingBuffer(total_size_bytes, device, gpu_id, custom_mem_pool)
        self.total_size = total_size_bytes
        self.base_ptr = self.buffer.data_ptr
        self.head = 0
        self.round = 0
        self.allocations: dict = {}  # alloc_id -> (offset, size, round)
        self.alloc_order: List[int] = []
        self.next_alloc_id = 0
        self.watermark_round = 0
        self.watermark_tail = 0
        self.lock = threading.Lock()

        logger.info(
            f"StagingAllocator (ring+overcommit): "
            f"{total_size_bytes / (1024*1024):.1f} MB "
            f"on {device}, ptr=0x{self.base_ptr:x}"
        )

    def assign(self, required_bytes: int) -> Optional[Tuple[int, int, int]]:
        """Allocate a region. Returns (alloc_id, offset, round) or None.

        Overcommit: does not check overlap. Prefill side checks watermark
        before RDMA to ensure the region is safe to write.
        """
        with self.lock:
            if required_bytes > self.total_size:
                return None

            space_at_end = self.total_size - self.head
            if required_bytes <= space_at_end:
                offset = self.head
                self.head += required_bytes
            else:
                self.round += 1
                offset = 0
                self.head = required_bytes

            alloc_id = self.next_alloc_id
            self.next_alloc_id += 1
            self.allocations[alloc_id] = (offset, required_bytes, self.round)
            self.alloc_order.append(alloc_id)
            return (alloc_id, offset, self.round)

    def free(self, alloc_id: int):
        """Free an allocation and advance watermark past consecutive freed entries."""
        with self.lock:
            if alloc_id not in self.allocations:
                return
            self.allocations.pop(alloc_id)

            while self.alloc_order and self.alloc_order[0] not in self.allocations:
                self.alloc_order.pop(0)

            if not self.allocations:
                self.watermark_round = self.round
                self.watermark_tail = self.head
            elif self.alloc_order:
                off, _, rnd = self.allocations[self.alloc_order[0]]
                self.watermark_round = rnd
                self.watermark_tail = off

    def get_watermark(self) -> Tuple[int, int]:
        """Return (round, tail_offset). Everything before this is safe to write."""
        with self.lock:
            return (self.watermark_round, self.watermark_tail)

    def get_ptr(self, alloc_id: int) -> int:
        offset, _, _ = self.allocations[alloc_id]
        return self.base_ptr + offset

    def get_offset(self, alloc_id: int) -> int:
        offset, _, _ = self.allocations[alloc_id]
        return offset

    def get_round(self, alloc_id: int) -> int:
        _, _, rnd = self.allocations[alloc_id]
        return rnd

    def get_base_ptr(self) -> int:
        return self.base_ptr

    def get_total_size(self) -> int:
        return self.total_size


def compute_staging_layout(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    dtype_size: int,
    num_layers: int,
) -> Tuple[int, int]:
    """Compute per-layer and total staging buffer size in bytes.

    Returns:
        (per_layer_bytes, total_bytes)
    """
    per_layer_bytes = num_tokens * num_heads * head_dim * dtype_size
    total_bytes = per_layer_bytes * num_layers * 2  # K + V
    return per_layer_bytes, total_bytes


def gather_kv_head_slices(
    kv_buffer_tensor: torch.Tensor,
    page_indices: torch.Tensor,
    head_start: int,
    num_heads: int,
    staging_tensor: torch.Tensor,
    page_size: int = 1,
):
    """Gather KV head slices from scattered pages into contiguous staging buffer.

    Args:
        kv_buffer_tensor: The KV buffer for one layer, shape [pool_size, head_num, head_dim].
            The buffer is always 3D. When page_size > 1, each page occupies
            page_size consecutive slots in the first dimension.
        page_indices: [num_pages] int32/int64 tensor of page indices.
        head_start: Starting head index for the slice.
        num_heads: Number of heads to gather.
        staging_tensor: Output tensor, contiguous, matching the gathered shape.
        page_size: Number of tokens per page.
    """
    if page_size == 1:
        selected = kv_buffer_tensor[
            page_indices, head_start : head_start + num_heads, :
        ]
        staging_tensor.copy_(selected.reshape(staging_tensor.shape))
    else:
        offsets = torch.arange(page_size, device=page_indices.device)
        token_indices = (page_indices.unsqueeze(1) * page_size + offsets).reshape(-1)
        selected = kv_buffer_tensor[
            token_indices, head_start : head_start + num_heads, :
        ]
        staging_tensor.copy_(selected.reshape(staging_tensor.shape))


def scatter_kv_head_slices(
    staging_tensor: torch.Tensor,
    kv_buffer_tensor: torch.Tensor,
    page_indices: torch.Tensor,
    head_start: int,
    num_heads: int,
    page_size: int = 1,
):
    """Scatter KV head slices from contiguous staging buffer to KV cache.

    Args:
        staging_tensor: Input tensor from staging buffer (contiguous packed data).
        kv_buffer_tensor: The KV buffer for one layer, shape [pool_size, head_num, head_dim].
        page_indices: [num_pages] int32/int64 tensor of page indices.
        head_start: Starting head index for the slice.
        num_heads: Number of heads to scatter.
        page_size: Number of tokens per page.
    """
    head_dim = kv_buffer_tensor.shape[-1]
    if page_size == 1:
        num_tokens = page_indices.shape[0]
        data = staging_tensor.reshape(num_tokens, num_heads, head_dim)
        kv_buffer_tensor[page_indices, head_start : head_start + num_heads, :] = data
    else:
        num_tokens = page_indices.shape[0] * page_size
        offsets = torch.arange(page_size, device=page_indices.device)
        token_indices = (page_indices.unsqueeze(1) * page_size + offsets).reshape(-1)
        data = staging_tensor.reshape(num_tokens, num_heads, head_dim)
        kv_buffer_tensor[token_indices, head_start : head_start + num_heads, :] = data


def gather_all_layers_to_staging(
    k_buffers: list,
    v_buffers: list,
    page_indices_np,
    staging_buffer: StagingBuffer,
    src_head_start: int,
    num_heads: int,
    page_size: int,
    gpu_id: int,
) -> int:
    """Gather all layers' K and V head slices into a staging buffer.

    All GPU operations (set_device, tensor creation, gather, synchronize)
    are encapsulated here so callers don't need to import torch.

    Returns:
        Total bytes written to staging buffer.
    """
    import numpy as np

    num_layers = len(k_buffers)
    head_dim = k_buffers[0].shape[-1]
    dtype_size = k_buffers[0].element_size()
    num_tokens = len(page_indices_np) * page_size
    per_layer_bytes = num_tokens * num_heads * head_dim * dtype_size

    torch.cuda.set_device(gpu_id)
    page_idx_tensor = torch.from_numpy(page_indices_np.astype(np.int64)).to(
        f"cuda:{gpu_id}"
    )

    if not hasattr(staging_buffer, "_gather_stream"):
        staging_buffer._gather_stream = torch.cuda.Stream(device=f"cuda:{gpu_id}")

    # Ensure gather doesn't read KV data that forward hasn't finished writing
    staging_buffer._gather_stream.wait_stream(
        torch.cuda.default_stream(torch.device(f"cuda:{gpu_id}"))
    )

    staging_view = staging_buffer.buffer
    offset = 0
    with torch.cuda.stream(staging_buffer._gather_stream):
        for layer_id in range(num_layers):
            lv = (
                staging_view[offset : offset + per_layer_bytes]
                .view(k_buffers[layer_id].dtype)
                .reshape(num_tokens, num_heads, head_dim)
            )
            gather_kv_head_slices(
                k_buffers[layer_id],
                page_idx_tensor,
                src_head_start,
                num_heads,
                lv,
                page_size,
            )
            offset += per_layer_bytes
        for layer_id in range(num_layers):
            lv = (
                staging_view[offset : offset + per_layer_bytes]
                .view(v_buffers[layer_id].dtype)
                .reshape(num_tokens, num_heads, head_dim)
            )
            gather_kv_head_slices(
                v_buffers[layer_id],
                page_idx_tensor,
                src_head_start,
                num_heads,
                lv,
                page_size,
            )
            offset += per_layer_bytes

    staging_buffer._gather_stream.synchronize()
    return offset


def scatter_staging_to_kv(
    staging_buffer_view: torch.Tensor,
    k_buffers: list,
    v_buffers: list,
    page_idx_tensor: torch.Tensor,
    page_size: int,
    prefill_attn_tp_size: int,
    decode_attn_tp_size: int,
    dst_tp_rank: int,
    total_kv_heads: int,
) -> None:
    """Scatter data from a contiguous staging region into KV cache buffers.

    This is backend-agnostic: it only performs GPU scatter operations.
    The caller is responsible for resolving page indices, ensuring
    the staging data is visible on GPU, and selecting the CUDA stream.
    """
    num_layers = len(k_buffers)
    head_dim = k_buffers[0].shape[-1]
    dtype_size = k_buffers[0].element_size()
    num_tokens = page_idx_tensor.shape[0] * page_size

    if prefill_attn_tp_size > decode_attn_tp_size:
        num_writers = prefill_attn_tp_size // max(1, decode_attn_tp_size)
    else:
        num_writers = 1

    for writer_rank in range(num_writers):
        _, num_heads, dst_head_start, _ = compute_head_slice_params(
            prefill_attn_tp_size,
            decode_attn_tp_size,
            writer_rank,
            dst_tp_rank,
            total_kv_heads,
        )
        per_layer_bytes = num_tokens * num_heads * head_dim * dtype_size
        per_rank_bytes = per_layer_bytes * num_layers * 2
        rank_base = writer_rank * per_rank_bytes

        offset = rank_base
        for layer_id in range(num_layers):
            layer_data = (
                staging_buffer_view[offset : offset + per_layer_bytes]
                .view(k_buffers[layer_id].dtype)
                .reshape(num_tokens, num_heads, head_dim)
            )
            scatter_kv_head_slices(
                layer_data,
                k_buffers[layer_id],
                page_idx_tensor,
                dst_head_start,
                num_heads,
                page_size,
            )
            offset += per_layer_bytes
        for layer_id in range(num_layers):
            layer_data = (
                staging_buffer_view[offset : offset + per_layer_bytes]
                .view(v_buffers[layer_id].dtype)
                .reshape(num_tokens, num_heads, head_dim)
            )
            scatter_kv_head_slices(
                layer_data,
                v_buffers[layer_id],
                page_idx_tensor,
                dst_head_start,
                num_heads,
                page_size,
            )
            offset += per_layer_bytes


def compute_head_slice_params(
    src_attn_tp_size: int,
    dst_attn_tp_size: int,
    src_tp_rank: int,
    dst_tp_rank: int,
    total_kv_heads: int,
) -> Tuple[int, int, int, int]:
    """Compute head slicing parameters for heterogeneous TP transfer.

    Returns:
        (src_head_start, num_heads_to_send, dst_head_start, num_heads_to_send)
    """
    src_heads_per_rank = max(1, total_kv_heads // src_attn_tp_size)
    dst_heads_per_rank = max(1, total_kv_heads // dst_attn_tp_size)

    local_tp_rank = src_tp_rank % src_attn_tp_size
    dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size

    if src_attn_tp_size > dst_attn_tp_size:
        src_head_start = 0
        num_heads_to_send = src_heads_per_rank
        src_replication = max(1, src_attn_tp_size // total_kv_heads)
        unique_head_idx = local_tp_rank // src_replication
        dst_head_start = (unique_head_idx * src_heads_per_rank) % dst_heads_per_rank
    else:
        src_head_start = (
            dst_tp_rank_in_group * dst_heads_per_rank
        ) % src_heads_per_rank
        num_heads_to_send = dst_heads_per_rank
        dst_head_start = 0

    return src_head_start, num_heads_to_send, dst_head_start, num_heads_to_send


def compute_staging_layout(
    src_attn_tp_size: int,
    dst_attn_tp_size: int,
    dst_tp_rank: int,
    total_kv_heads: int,
    num_tokens: int,
    bytes_per_head_token: int,
    num_layers: int,
) -> Tuple[int, List[int], int]:
    """Compute per-writer byte layout for a staging region.

    Returns:
        (num_writers, writer_bytes_list, total_bytes)
        where writer_bytes_list[i] = bytes for writer i covering all layers (K+V).
    """
    if src_attn_tp_size > dst_attn_tp_size:
        num_writers = src_attn_tp_size // max(1, dst_attn_tp_size)
    else:
        num_writers = 1

    writer_bytes = []
    for wr in range(num_writers):
        _, nh, _, _ = compute_head_slice_params(
            src_attn_tp_size,
            dst_attn_tp_size,
            wr,
            dst_tp_rank,
            total_kv_heads,
        )
        writer_bytes.append(num_tokens * nh * bytes_per_head_token * num_layers * 2)
    return num_writers, writer_bytes, sum(writer_bytes)


def allocate_chunk_staging(
    allocator,
    num_pages: int,
    page_size: int,
    chunk_pages: int,
    prefill_attn_tp: int,
    decode_attn_tp: int,
    dst_tp_rank: int,
    total_kv_heads: int,
    bytes_per_head_per_token: int,
    num_kv_layers: int,
) -> List[Tuple[int, int, int, int]]:
    """Allocate per-chunk staging regions from a StagingAllocator.

    Splits total_pages into chunks of chunk_pages, computes the required staging
    bytes per chunk via compute_staging_layout, and assigns from the allocator.

    Returns list of (alloc_id, offset, round, end) per chunk.
    Failed allocations are recorded as (-1, -1, 0, -1).
    """
    infos: List[Tuple[int, int, int, int]] = []
    remaining = num_pages
    while remaining > 0:
        cp = min(remaining, chunk_pages)
        chunk_tokens = cp * page_size
        _, _, required = compute_staging_layout(
            prefill_attn_tp,
            decode_attn_tp,
            dst_tp_rank,
            total_kv_heads,
            chunk_tokens,
            bytes_per_head_per_token,
            num_kv_layers,
        )
        result = allocator.assign(required)
        if result is not None:
            alloc_id, offset, rnd = result
            infos.append((alloc_id, offset, rnd, offset + required))
        else:
            logger.warning_once(
                f"[Staging] allocator returned None for chunk: need {required} bytes"
            )
            infos.append((-1, -1, 0, -1))
        remaining -= cp
    return infos


def resolve_total_kv_heads(
    kv_args,
    attn_tp_size: int,
    kv_buffer_tensors=None,
) -> int:
    """Resolve the global total KV head count from available metadata.

    Tries in order: kv_args.total_kv_head_num, kv_args.kv_head_num * attn_tp_size,
    kv_buffer_tensors shape.  Raises if none are available.
    """
    total = getattr(kv_args, "total_kv_head_num", 0)
    if total > 0:
        return total
    per_rank = getattr(kv_args, "kv_head_num", 0)
    if per_rank > 0:
        return per_rank * attn_tp_size
    if kv_buffer_tensors is not None:
        k_bufs = (
            kv_buffer_tensors.get("k_buffers")
            if isinstance(kv_buffer_tensors, dict)
            else None
        )
        if k_bufs and len(k_bufs) > 0:
            return int(k_bufs[0].shape[1]) * max(1, attn_tp_size)
    raise ValueError(
        "Cannot resolve total_kv_heads: kv_args has neither total_kv_head_num "
        "nor kv_head_num, and no kv_buffer_tensors provided. "
        "Ensure DecodePreallocQueue._init_kv_manager sets kv_args.kv_head_num."
    )
