"""Fused SiLU-and-Mul + FP8 per-group quantization JIT kernel.

Replaces two separate kernels:
  1. silu_and_mul(gateup_output, down_input)
  2. per_token_group_quant_fp8(down_input, group_size=128)

with a single kernel that computes silu(gate)*up and quantizes to FP8
in one pass, avoiding the intermediate BF16 tensor allocation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_fused_silu_mul_quant_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "fused_silu_mul_quant",
        *args,
        cuda_files=["elementwise/fused_silu_mul_quant.cuh"],
        cuda_wrappers=[
            ("fused_silu_mul_quant", f"fused_silu_mul_quant<{args}>"),
        ],
    )


def fused_silu_mul_quant(
    input: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-10,
    column_major_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused SiLU+Mul+FP8 quantization.

    Input: [M, 2*N] bf16 — gate_up projection output
    Output: ([M, N] fp8_e4m3, [M, N/group_size] or [N/group_size, M] float32 scales)

    Parameters
    ----------
    input         : CUDA tensor [M, 2*N] in bf16
    group_size    : quantization group size (default 128)
    eps           : epsilon for scale computation
    column_major_scales : if True, scales are [N/group_size, M] (column-major)
    """
    assert input.is_cuda, "input must be CUDA tensor"
    assert input.dtype in (torch.bfloat16,), f"Unsupported dtype {input.dtype}"
    assert input.ndim == 2, f"Expected 2D tensor, got {input.ndim}D"
    assert input.shape[1] % 2 == 0, "Last dim must be even (gate+up)"

    M = input.shape[0]
    N = input.shape[1] // 2
    assert N % group_size == 0, f"N={N} not divisible by group_size={group_size}"

    num_groups_per_row = N // group_size
    fp8_max = 448.0  # E4M3 max

    output_q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=input.device)
    if column_major_scales:
        output_s = torch.empty(
            (num_groups_per_row, M), dtype=torch.float32, device=input.device
        )
    else:
        output_s = torch.empty(
            (M, num_groups_per_row), dtype=torch.float32, device=input.device
        )

    if M > 0:
        module = _jit_fused_silu_mul_quant_module(input.dtype)
        module.fused_silu_mul_quant(
            input.contiguous(),
            output_q,
            output_s,
            group_size,
            eps,
            fp8_max,
            column_major_scales,
        )

    return output_q, output_s
