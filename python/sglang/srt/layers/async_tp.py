"""
Async Tensor Parallelism: GEMM + ReduceScatter / AllGather + GEMM overlap.

Uses torch.ops.symm_mem fused ops to overlap GEMM computation with TP
communication. Instead of sequential GEMM → AllReduce, decomposes into:
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


def is_async_tp_available() -> bool:
    """Check if torch.ops.symm_mem fused ops are available."""
    global _async_tp_available
    if _async_tp_available is not None:
        return _async_tp_available

    try:
        _async_tp_available = (
            hasattr(torch.ops, "symm_mem")
            and hasattr(torch.ops.symm_mem, "fused_matmul_reduce_scatter")
            and hasattr(torch.ops.symm_mem, "fused_all_gather_matmul")
        )
    except Exception:
        _async_tp_available = False

    if not _async_tp_available:
        logger.info(
            "torch.ops.symm_mem fused ops not available. "
            "Async TP requires PyTorch >= 2.6 with CUDA."
        )
    return _async_tp_available


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
    # torch.ops.symm_mem.fused_matmul_reduce_scatter expects:
    #   A @ B where A=[M,K], B=[K,N] -> matmul -> [M,N] -> reduce_scatter -> [M/tp, N]
    # But weight is stored as [N, K] (transposed), so we pass weight.T
    output = torch.ops.symm_mem.fused_matmul_reduce_scatter(
        input,
        weight.T,
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
    # torch.ops.symm_mem.fused_all_gather_matmul expects B matrices as [K, N]
    # but our weights are [N, K], so we transpose
    Bs = [w.T for w in weights]
    gathered, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
        input_shard,
        Bs,
        gather_dim=gather_dim,
        group_name=group_name,
        return_A=False,  # Don't need the gathered A tensor
    )
    return gathered, mm_outputs
