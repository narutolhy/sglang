"""
Async Tensor Parallelism: GEMM + ReduceScatter / AllGather + GEMM overlap.

Uses torch.distributed._symmetric_memory fused ops to overlap GEMM computation
with TP communication. Instead of sequential GEMM → AllReduce, decomposes into:
  - RowParallel: GEMM + ReduceScatter (fused, overlapped)
  - ColumnParallel: AllGather + GEMM (fused, overlapped)

Between RowParallel and ColumnParallel, tensors are in "scattered" state
[M/tp, hidden_dim], and element-wise ops (residual add, layernorm) operate
on the scattered tensors.

Requires:
  - PyTorch >= 2.6 with CUDA
  - SM >= 9.0 (H100+) with symmetric memory / multicast support
  - bfloat16 tensors
"""

import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_async_tp_available: Optional[bool] = None
_fused_mrs = None  # fused_matmul_reduce_scatter
_fused_agm = None  # fused_all_gather_matmul
_restride_mrs = None  # restride_A_for_fused_matmul_reduce_scatter
_restride_agm = None  # restride_A_shard_for_fused_all_gather_matmul


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


def is_async_tp_available() -> bool:
    """Check if fused GEMM+communication ops are available."""
    return _init_fused_ops()


def fused_matmul_reduce_scatter(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_name: str,
    bias: Optional[torch.Tensor] = None,
    scatter_dim: int = 0,
) -> torch.Tensor:
    """
    Fused GEMM + ReduceScatter.

    Computes output = ReduceScatter(input @ weight.T) with internal pipelining
    that overlaps GEMM chunks with ReduceScatter communication.

    Args:
        input: [M, K] input tensor (each TP rank has partial K)
        weight: [N, K] weight matrix (full output, partial input)
        group_name: TP process group name
        bias: Optional bias tensor
        scatter_dim: Dimension to scatter along (default 0, token dim)

    Returns:
        [M/tp, N] scattered reduced output
    """
    _init_fused_ops()

    # Restride input for optimal memory access pattern
    A = _restride_mrs(input, scatter_dim)

    # _fused_matmul_reduce_scatter(A, B, ...) computes ReduceScatter(A @ B)
    # weight is [N, K], we need B=[K, N], so pass weight.T
    output = _fused_mrs(
        A,
        weight.T.contiguous(),
        "sum",
        scatter_dim=scatter_dim,
        group_name=group_name,
    )
    if bias is not None:
        output = output + bias
    return output


def fused_all_gather_matmul(
    input_shard: torch.Tensor,
    weights: List[torch.Tensor],
    group_name: str,
    gather_dim: int = 0,
) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
    """
    Fused AllGather + GEMM.

    Gathers input shards across TP ranks, then computes matmul with each weight.
    AllGather communication overlaps with GEMM computation internally.

    Args:
        input_shard: [M/tp, K] scattered input tensor
        weights: List of [N_i, K] weight matrices (transposed stored)
        group_name: TP process group name
        gather_dim: Dimension to gather along (default 0, token dim)

    Returns:
        Tuple of (gathered_input_or_None, [matmul_results])
        gathered_input: [M, K] or None
        matmul_results: list of [M, N_i] tensors
    """
    _init_fused_ops()

    # Restride input shard for optimal memory access
    A_shard = _restride_agm(input_shard, gather_dim)

    # B matrices: weight is [N, K], need [K, N]
    Bs = [w.T.contiguous() for w in weights]
    gathered, mm_outputs = _fused_agm(
        A_shard,
        Bs,
        gather_dim=gather_dim,
        group_name=group_name,
        return_A=False,
    )
    return gathered, mm_outputs
