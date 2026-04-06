# Async TP Experimental Status

This note captures the current state of the experimental async tensor-parallel
implementation on branch `codex/mini-syncopate-moe-pipeline`.

## Scope

The current implementation uses PyTorch's private
`torch.distributed._symmetric_memory` fused kernels:

- `fused_matmul_reduce_scatter`
- `fused_all_gather_matmul`

The goal is to overlap TP communication with GEMM by keeping intermediate
tensors in a scattered `[M / tp, H]` layout between row-parallel and
column-parallel layers.

## What Was Fixed

The following issues were fixed while validating the feature:

1. Non-divisible token counts no longer silently miscompute.
   - The fused symmetric-memory RS path only supports token counts divisible by
     `tp_size`.
   - Async TP now falls back to the standard path for non-divisible `M`.

2. TP world size caching is now correct per process group.
   - The earlier implementation cached a single global world size.
   - It now caches by `group_name`.

3. Residual scattering no longer forces a materializing copy.
   - The residual slice now uses `narrow(...)` instead of `.contiguous()`.

4. Transposed weights are cached.
   - The earlier wrapper recomputed `weight.T.contiguous()` on every forward.
   - Async TP now caches the contiguous transposed weight using tensor identity
     and version metadata.

5. Runtime safety checks were added for fused matmul usage.
   - Async TP fused paths now automatically fall back unless:
     - fused symmetric-memory kernels are available
     - tensors are CUDA tensors
     - tensors are `torch.bfloat16`

6. Final epilogue all-gather sizing is now safe.
   - The final gather in model epilogues now allocates `shard_size * tp_size`
     and slices back to `num_tokens`.

## Validation Summary

### Correctness / runtime

- Service startup succeeded on H100 with `--enable-async-tp`
- Decode correctness was preserved on smoke tests
- Non-divisible token counts now take the standard fallback path instead of
  silently dropping tokens

### Performance findings

After the correctness and copy-path fixes, the large `elementwise+reduce`
regression was mostly removed, but the async TP backend remained slower.

Representative result on `Llama-3.1-70B`, `TP=4`, `4x H100`, no PCG:

- Baseline total GPU kernel time: `4197 ms`
- Async TP total GPU kernel time: `12061 ms`
- Async TP remained about `2.9x` slower

Dominant async TP costs:

- `symm_mem_barrier`: about `54-60%`
- `fused_all_gather`: about `18%`
- GEMM time increased because the fused path chunks GEMMs into smaller pieces

Improved compared with earlier experiments:

- `elementwise+reduce` dropped from a pathological multi-second hotspot to a
  small fraction of total time after fixing copy-heavy wrapper behavior

## Main Conclusion

The high-level async TP idea is still reasonable, but the current
`_symmetric_memory` backend does not perform well for the tested decode
configuration. After fixing wrapper-level issues, the main bottleneck is no
longer extra Python-side copies. It is the backend protocol itself:

- heavy `symm_mem_barrier` cost
- communication pipelining overhead
- smaller, less efficient GEMM chunks

The environment variable `TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP=1` did not
activate any new native kernel path in profiling and did not materially change
performance.

## Recommended Next Step

Do not invest further in the current `_symmetric_memory` backend for this
serving configuration. If async TP overlap is still a priority, prototype a new
backend that uses:

- NCCL collectives
- dedicated CUDA streams
- double buffering
- explicit chunk scheduling in SGLang

See `async_tp_nccl_overlap_sketch.md` for a concrete prototype direction.
