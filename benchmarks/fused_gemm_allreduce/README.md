# Fused GEMM + AllReduce Prototype

Producer-consumer megakernel: GEMM tiles notify AllReduce consumer CTAs
via per-tile flags, enabling tile-level overlap without global barriers.

Uses PyTorch symmetric_memory for cross-GPU buffers + standard Triton
(no fork needed).

## Usage

```bash
torchrun --nproc_per_node=4 benchmark.py
```
