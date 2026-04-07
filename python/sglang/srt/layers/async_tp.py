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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group

logger = logging.getLogger(__name__)

ASYNC_TP_LAYOUT_ATTR = "_async_tp_layout"
ASYNC_TP_LOGICAL_TOKENS_ATTR = "_async_tp_logical_num_tokens"
ASYNC_TP_PADDED_TOKENS_ATTR = "_async_tp_padded_num_tokens"
ASYNC_TP_TP_SIZE_ATTR = "_async_tp_tp_size"

_async_tp_available: Optional[bool] = None
_fused_mrs = None  # fused_matmul_reduce_scatter
_fused_agm = None  # fused_all_gather_matmul
_restride_mrs = None  # restride_A_for_fused_matmul_reduce_scatter
_restride_agm = None  # restride_A_shard_for_fused_all_gather_matmul
_tp_world_sizes: Dict[str, int] = {}
_comm_streams: Dict[int, torch.cuda.Stream] = {}
_decision_log_keys: set[str] = set()


@dataclass
class AsyncTpTensor:
    tensor: torch.Tensor
    layout: str
    logical_num_tokens: int
    padded_num_tokens: int
    tp_size: int

    @property
    def shard_num_tokens(self) -> int:
        return self.tensor.shape[0]

    def is_scattered(self) -> bool:
        return self.layout == "scattered"

    def with_tensor(self, tensor: torch.Tensor) -> "AsyncTpTensor":
        return AsyncTpTensor(
            tensor=tensor,
            layout=self.layout,
            logical_num_tokens=self.logical_num_tokens,
            padded_num_tokens=self.padded_num_tokens,
            tp_size=self.tp_size,
        )

    def mark_scattered(
        self, tensor: torch.Tensor, padded_num_tokens: Optional[int] = None
    ) -> "AsyncTpTensor":
        return AsyncTpTensor(
            tensor=tensor,
            layout="scattered",
            logical_num_tokens=self.logical_num_tokens,
            padded_num_tokens=(
                padded_num_tokens
                if padded_num_tokens is not None
                else tensor.shape[0] * self.tp_size
            ),
            tp_size=self.tp_size,
        )

    def attach(self) -> torch.Tensor:
        setattr(self.tensor, ASYNC_TP_LAYOUT_ATTR, self.layout)
        setattr(
            self.tensor, ASYNC_TP_LOGICAL_TOKENS_ATTR, self.logical_num_tokens
        )
        setattr(self.tensor, ASYNC_TP_PADDED_TOKENS_ATTR, self.padded_num_tokens)
        setattr(self.tensor, ASYNC_TP_TP_SIZE_ATTR, self.tp_size)
        return self.tensor

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, logical_num_tokens: int, tp_size: int
    ) -> "AsyncTpTensor":
        layout = getattr(
            tensor,
            ASYNC_TP_LAYOUT_ATTR,
            "scattered" if tensor.shape[0] != logical_num_tokens else "full",
        )
        return cls(
            tensor=tensor,
            layout=layout,
            logical_num_tokens=getattr(
                tensor, ASYNC_TP_LOGICAL_TOKENS_ATTR, logical_num_tokens
            ),
            padded_num_tokens=getattr(
                tensor,
                ASYNC_TP_PADDED_TOKENS_ATTR,
                tensor.shape[0] * tp_size if layout == "scattered" else tensor.shape[0],
            ),
            tp_size=getattr(tensor, ASYNC_TP_TP_SIZE_ATTR, tp_size),
        )


def _get_async_tp_backend() -> str:
    return os.getenv("SGLANG_ASYNC_TP_BACKEND", "symm_mem").strip().lower()


def _get_async_tp_chunk_size() -> int:
    value = os.getenv("SGLANG_ASYNC_TP_CHUNK_SIZE", "1024")
    try:
        return max(int(value), 0)
    except ValueError:
        logger.warning("Invalid SGLANG_ASYNC_TP_CHUNK_SIZE=%s; using 1024", value)
        return 1024


def _get_async_tp_pipeline_stages() -> int:
    value = os.getenv("SGLANG_ASYNC_TP_PIPELINE_STAGES", "3")
    try:
        return max(int(value), 2)
    except ValueError:
        logger.warning(
            "Invalid SGLANG_ASYNC_TP_PIPELINE_STAGES=%s; using 3", value
        )
        return 3


def _should_log_decisions() -> bool:
    value = os.getenv("SGLANG_ASYNC_TP_LOG_DECISIONS", "")
    return value.lower() in ("1", "true", "yes", "y")


def log_async_tp_decision_once(key: str, message: str) -> None:
    if not _should_log_decisions() or key in _decision_log_keys:
        return
    _decision_log_keys.add(key)
    logger.info("[async_tp] %s", message)
    # Also print to ensure visibility in all process configurations
    import sys
    print(f"[async_tp] {message}", file=sys.stderr, flush=True)


def log_async_tp_backend_once() -> None:
    backend = _get_async_tp_backend()
    if backend == "nccl":
        log_async_tp_decision_once(
            f"backend:{backend}:{_get_async_tp_chunk_size()}:{_get_async_tp_pipeline_stages()}",
            f"using backend={backend} chunk_size={_get_async_tp_chunk_size()} pipeline_stages={_get_async_tp_pipeline_stages()}",
        )
    else:
        log_async_tp_decision_once(
            f"backend:{backend}",
            f"using backend={backend}",
        )


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
    log_async_tp_backend_once()
    if _get_async_tp_backend() == "nccl":
        tp_group = get_tp_group()
        return tp_group.world_size > 1 and tp_group.pynccl_comm is not None
    return _init_fused_ops()


def get_async_tp_matmul_unavailable_reason(
    input: torch.Tensor, weight: torch.Tensor
) -> Optional[str]:
    backend = _get_async_tp_backend()
    if backend == "nccl":
        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            return "tp_world_size<=1"
        if tp_group.pynccl_comm is None:
            return "missing_pynccl_comm"
        if not input.is_cuda or not weight.is_cuda:
            return "non_cuda_tensor"
        return None

    if not _init_fused_ops():
        return "missing_symm_mem_fused_ops"
    if not input.is_cuda or not weight.is_cuda:
        return "non_cuda_tensor"
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return f"unsupported_dtype(input={input.dtype},weight={weight.dtype})"
    return None


def can_run_async_tp_matmul(input: torch.Tensor, weight: torch.Tensor) -> bool:
    return get_async_tp_matmul_unavailable_reason(input, weight) is None


def _get_transposed_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return contiguous transposed weight. No caching to avoid doubling
    GPU memory for large models (e.g. 70B TP=4 needs ~33GB extra)."""
    weight_t = weight.T.contiguous()
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
    pipeline_stages = min(_get_async_tp_pipeline_stages(), num_chunks if (num_chunks := (total_local_rows + local_chunk_rows - 1) // local_chunk_rows) > 0 else 1)

    gather_buffers = [
        input_shard.new_empty(local_chunk_rows * tp_size, input_shard.shape[1])
        for _ in range(pipeline_stages)
    ]
    gather_ready = [torch.cuda.Event() for _ in range(pipeline_stages)]
    compute_done = [None] * pipeline_stages
    outputs: List[torch.Tensor] = []

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

    for chunk_idx in range(min(num_chunks, pipeline_stages)):
        launch_gather(chunk_idx, chunk_idx)

    for chunk_idx in range(num_chunks):
        curr_idx = chunk_idx % pipeline_stages
        current_stream.wait_event(gather_ready[curr_idx])

        start = chunk_idx * local_chunk_rows
        rows = min(local_chunk_rows, total_local_rows - start)
        gathered_rows = rows * tp_size
        gathered = gather_buffers[curr_idx][:gathered_rows]
        outputs.append(_run_gemm(layer, gathered, bias))
        compute_done[curr_idx] = torch.cuda.Event()
        compute_done[curr_idx].record(current_stream)

        next_chunk_idx = chunk_idx + pipeline_stages
        if next_chunk_idx < num_chunks:
            launch_gather(next_chunk_idx, curr_idx)

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
    pipeline_stages = min(_get_async_tp_pipeline_stages(), num_chunks if num_chunks > 0 else 1)
    partial_chunks: List[Optional[torch.Tensor]] = [None] * pipeline_stages
    compute_ready = [None] * pipeline_stages
    rs_done = [None] * pipeline_stages
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

    for chunk_idx in range(min(num_chunks, pipeline_stages)):
        launch_compute(chunk_idx, chunk_idx)

    for chunk_idx in range(num_chunks):
        curr_idx = chunk_idx % pipeline_stages
        launch_reduce_scatter(chunk_idx, curr_idx)
        next_chunk_idx = chunk_idx + pipeline_stages
        if next_chunk_idx < num_chunks:
            launch_compute(next_chunk_idx, curr_idx)

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
