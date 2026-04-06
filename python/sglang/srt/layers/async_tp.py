"""
Async Tensor Parallelism backends.

This module supports two experimental implementations:

- `symm_mem`: PyTorch private fused ops from `torch.distributed._symmetric_memory`
- `nccl`: an explicit NCCL stream-overlap prototype using pynccl collectives

The NCCL prototype keeps the current model-side "scattered" layout contract but
replaces the hidden symmetric-memory pipeline with explicit chunk scheduling on
top of standard collectives.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group

logger = logging.getLogger(__name__)

_async_tp_available: Optional[bool] = None
_fused_mrs = None  # fused_matmul_reduce_scatter
_fused_agm = None  # fused_all_gather_matmul
_restride_mrs = None  # restride_A_for_fused_matmul_reduce_scatter
_restride_agm = None  # restride_A_shard_for_fused_all_gather_matmul
_tp_world_sizes: Dict[str, int] = {}
_comm_streams: Dict[int, torch.cuda.Stream] = {}


def _get_async_tp_backend() -> str:
    return os.getenv("SGLANG_ASYNC_TP_BACKEND", "symm_mem").strip().lower()


def _get_async_tp_chunk_size() -> int:
    value = os.getenv("SGLANG_ASYNC_TP_CHUNK_SIZE", "1024")
    try:
        return max(int(value), 0)
    except ValueError:
        logger.warning("Invalid SGLANG_ASYNC_TP_CHUNK_SIZE=%s; using 1024", value)
        return 1024


def _get_comm_stream(device: torch.device) -> torch.cuda.Stream:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    if device_index not in _comm_streams:
        with torch.cuda.device(device_index):
            _comm_streams[device_index] = torch.cuda.Stream(priority=0)
    return _comm_streams[device_index]


def _normalize_chunk_rows(total_rows: int, tp_size: int) -> int:
    configured = _get_async_tp_chunk_size()
    if configured <= 0 or configured >= total_rows:
        return total_rows
    chunk_rows = max(configured // tp_size * tp_size, tp_size)
    return min(chunk_rows, total_rows)


def _init_fused_ops():
    """Lazily import and cache the fused op references."""
    global _fused_mrs, _fused_agm, _restride_mrs, _restride_agm, _async_tp_available

    if _async_tp_available is not None:
        return _async_tp_available

    try:
        from torch.distributed._symmetric_memory import (
            _fused_all_gather_matmul,
            _fused_matmul_reduce_scatter,
            restride_A_for_fused_matmul_reduce_scatter,
            restride_A_shard_for_fused_all_gather_matmul,
        )

        _fused_mrs = _fused_matmul_reduce_scatter
        _fused_agm = _fused_all_gather_matmul
        _restride_mrs = restride_A_for_fused_matmul_reduce_scatter
        _restride_agm = restride_A_shard_for_fused_all_gather_matmul
        _async_tp_available = True
        logger.info("Async TP fused ops loaded from torch.distributed._symmetric_memory")
    except ImportError:
        _async_tp_available = False
        logger.info(
            "torch.distributed._symmetric_memory fused ops not available. "
            "Async TP requires PyTorch >= 2.6 with CUDA."
        )

    return _async_tp_available


def _get_tp_world_size(group_name: str) -> int:
    """Get TP world size from group name, cached."""
    if group_name not in _tp_world_sizes:
        group = dist.distributed_c10d._resolve_process_group(group_name)
        _tp_world_sizes[group_name] = dist.get_world_size(group)
    return _tp_world_sizes[group_name]


def is_async_tp_available() -> bool:
    """Check if the selected backend is available."""
    if _get_async_tp_backend() == "nccl":
        tp_group = get_tp_group()
        return tp_group.world_size > 1 and tp_group.pynccl_comm is not None
    return _init_fused_ops()


def _get_transposed_weight(weight: torch.Tensor) -> torch.Tensor:
    """Cache the contiguous transposed view for static inference weights."""
    version = getattr(weight, "_version", None)
    cache_key = (
        weight.data_ptr(),
        tuple(weight.shape),
        tuple(weight.stride()),
        version,
    )
    cached = getattr(weight, "_sglang_async_tp_weight_t_cache", None)
    if cached is not None:
        cached_key, cached_weight_t = cached
        if cached_key == cache_key:
            return cached_weight_t

    weight_t = weight.T.contiguous()
    try:
        weight._sglang_async_tp_weight_t_cache = (cache_key, weight_t)
    except (AttributeError, RuntimeError):
        pass
    return weight_t


def _run_gemm(layer, input_chunk: torch.Tensor, bias: Optional[torch.Tensor]):
    assert layer.quant_method is not None
    return layer.quant_method.apply(layer, input_chunk, bias)


def _fused_all_gather_matmul_symm_mem(
    input_shard: torch.Tensor,
    weights: List[torch.Tensor],
    group_name: str,
    gather_dim: int = 0,
) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
    _init_fused_ops()
    A_shard = _restride_agm(input_shard, gather_dim)
    Bs = [_get_transposed_weight(w) for w in weights]
    gathered, mm_outputs = _fused_agm(
        A_shard,
        Bs,
        gather_dim=gather_dim,
        group_name=group_name,
        return_A=False,
    )
    return gathered, mm_outputs


def _fused_matmul_reduce_scatter_symm_mem(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_name: str,
    bias: Optional[torch.Tensor] = None,
    scatter_dim: int = 0,
) -> torch.Tensor:
    _init_fused_ops()

    M = input.shape[scatter_dim]
    tp_size = _get_tp_world_size(group_name)
    if M % tp_size != 0:
        raise ValueError(
            f"fused_matmul_reduce_scatter requires {scatter_dim=}-dim size {M} "
            f"to be divisible by tp_size={tp_size}"
        )

    A = _restride_mrs(input, scatter_dim)
    output = _fused_mrs(
        A,
        _get_transposed_weight(weight),
        "sum",
        scatter_dim=scatter_dim,
        group_name=group_name,
    )
    if bias is not None:
        output = output + bias
    return output


def _chunked_all_gather_matmul_nccl(
    layer,
    input_shard: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_shard.dim() == 2, "NCCL async TP prototype only supports 2D inputs"
    tp_group = get_tp_group()
    tp_size = tp_group.world_size
    total_local_rows = input_shard.shape[0]
    total_rows = total_local_rows * tp_size
    local_chunk_rows = _normalize_chunk_rows(total_rows, tp_size) // tp_size

    if local_chunk_rows == total_local_rows:
        gathered = input_shard.new_empty(total_rows, input_shard.shape[1])
        tp_group.all_gather_into_tensor(gathered, input_shard)
        return _run_gemm(layer, gathered, bias)

    comm_stream = _get_comm_stream(input_shard.device)
    current_stream = torch.cuda.current_stream(input_shard.device)

    gather_buffers = [
        input_shard.new_empty(local_chunk_rows * tp_size, input_shard.shape[1])
        for _ in range(2)
    ]
    gather_ready = [torch.cuda.Event() for _ in range(2)]
    compute_done = [None, None]
    outputs: List[torch.Tensor] = []
    num_chunks = (total_local_rows + local_chunk_rows - 1) // local_chunk_rows

    def launch_gather(chunk_idx: int, buf_idx: int):
        start = chunk_idx * local_chunk_rows
        rows = min(local_chunk_rows, total_local_rows - start)
        if rows <= 0:
            return
        if compute_done[buf_idx] is not None:
            comm_stream.wait_event(compute_done[buf_idx])
        local_chunk = input_shard.narrow(0, start, rows)
        gather_buf = gather_buffers[buf_idx][: rows * tp_size]
        with torch.cuda.stream(comm_stream):
            tp_group.all_gather_into_tensor(gather_buf, local_chunk)
            gather_ready[buf_idx].record(comm_stream)

    launch_gather(0, 0)
    for chunk_idx in range(num_chunks):
        curr_idx = chunk_idx % 2
        next_idx = (chunk_idx + 1) % 2
        current_stream.wait_event(gather_ready[curr_idx])

        start = chunk_idx * local_chunk_rows
        rows = min(local_chunk_rows, total_local_rows - start)
        gathered_rows = rows * tp_size
        gathered = gather_buffers[curr_idx][:gathered_rows]
        outputs.append(_run_gemm(layer, gathered, bias))
        compute_done[curr_idx] = torch.cuda.Event()
        compute_done[curr_idx].record(current_stream)

        if chunk_idx + 1 < num_chunks:
            launch_gather(chunk_idx + 1, next_idx)

    current_stream.wait_stream(comm_stream)
    return torch.cat(outputs, dim=0)


def _chunked_matmul_reduce_scatter_nccl(
    layer,
    input: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scatter_dim: int = 0,
) -> torch.Tensor:
    assert scatter_dim == 0, "NCCL async TP prototype only supports scatter_dim=0"
    assert input.dim() == 2, "NCCL async TP prototype only supports 2D inputs"
    tp_group = get_tp_group()
    tp_size = tp_group.world_size
    total_rows = input.shape[0]
    if total_rows % tp_size != 0:
        raise ValueError(
            f"NCCL async TP prototype requires total_rows={total_rows} divisible by tp_size={tp_size}"
        )

    chunk_rows = _normalize_chunk_rows(total_rows, tp_size)
    local_rows = total_rows // tp_size
    if chunk_rows == total_rows:
        partial = _run_gemm(layer, input, None)
        output = partial.new_empty(local_rows, partial.shape[1])
        tp_group.reduce_scatter_tensor(output, partial)
        if bias is not None:
            output = output + bias
        return output

    comm_stream = _get_comm_stream(input.device)
    current_stream = torch.cuda.current_stream(input.device)
    num_chunks = (total_rows + chunk_rows - 1) // chunk_rows
    partial_chunks: List[Optional[torch.Tensor]] = [None, None]
    compute_ready = [None, None]
    rs_done = [None, None]
    outputs: List[torch.Tensor] = [None] * num_chunks  # type: ignore[assignment]

    def launch_compute(chunk_idx: int, buf_idx: int):
        start = chunk_idx * chunk_rows
        rows = min(chunk_rows, total_rows - start)
        if rows <= 0:
            return
        if rs_done[buf_idx] is not None:
            current_stream.wait_event(rs_done[buf_idx])
        input_chunk = input.narrow(0, start, rows)
        partial_chunks[buf_idx] = _run_gemm(layer, input_chunk, None)
        compute_ready[buf_idx] = torch.cuda.Event()
        compute_ready[buf_idx].record(current_stream)

    def launch_reduce_scatter(chunk_idx: int, buf_idx: int):
        start = chunk_idx * chunk_rows
        rows = min(chunk_rows, total_rows - start)
        if rows <= 0:
            return
        output_rows = rows // tp_size
        out_chunk = input.new_empty(output_rows, layer.output_size)
        ready_event = compute_ready[buf_idx]
        assert ready_event is not None
        partial = partial_chunks[buf_idx]
        assert partial is not None
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(ready_event)
            tp_group.reduce_scatter_tensor(out_chunk, partial)
            if bias is not None:
                out_chunk.add_(bias)
            rs_done[buf_idx] = torch.cuda.Event()
            rs_done[buf_idx].record(comm_stream)
        outputs[chunk_idx] = out_chunk

    launch_compute(0, 0)
    for chunk_idx in range(num_chunks):
        curr_idx = chunk_idx % 2
        next_idx = (chunk_idx + 1) % 2
        launch_reduce_scatter(chunk_idx, curr_idx)
        if chunk_idx + 1 < num_chunks:
            launch_compute(chunk_idx + 1, next_idx)

    current_stream.wait_stream(comm_stream)
    return torch.cat(outputs, dim=0)


def fused_matmul_reduce_scatter(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_name: str,
    bias: Optional[torch.Tensor] = None,
    scatter_dim: int = 0,
    layer=None,
) -> torch.Tensor:
    """
    Fused GEMM + ReduceScatter with backend selection.

    `layer` is optional and only used by the NCCL prototype backend so it can
    reuse the layer's existing quantization/GEMM path.
    """
    backend = _get_async_tp_backend()
    if backend == "nccl":
        if layer is None:
            raise ValueError("NCCL async TP backend requires the layer instance")
        return _chunked_matmul_reduce_scatter_nccl(
            layer, input, bias=bias, scatter_dim=scatter_dim
        )
    return _fused_matmul_reduce_scatter_symm_mem(
        input, weight, group_name, bias=bias, scatter_dim=scatter_dim
    )


def fused_all_gather_matmul(
    input_shard: torch.Tensor,
    weights: List[torch.Tensor],
    group_name: str,
    gather_dim: int = 0,
    layer=None,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
    """
    Fused AllGather + GEMM with backend selection.

    `layer` is optional and only used by the NCCL prototype backend.
    """
    backend = _get_async_tp_backend()
    if backend == "nccl":
        if layer is None:
            raise ValueError("NCCL async TP backend requires the layer instance")
        return None, [_chunked_all_gather_matmul_nccl(layer, input_shard, bias=bias)]
    return _fused_all_gather_matmul_symm_mem(
        input_shard, weights, group_name, gather_dim=gather_dim
    )
