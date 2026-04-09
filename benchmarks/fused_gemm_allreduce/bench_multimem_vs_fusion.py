"""
Micro-benchmark: multimem_all_reduce_ vs flashinfer allreduce_fusion
for the same data sizes.

Usage:
    torchrun --nproc_per_node=4 --master_port=39600 bench_multimem_vs_fusion.py
"""
import argparse
import os
import time

import torch
import torch.distributed as dist


def init_dist():
    rank = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    torch.cuda.set_device(rank)
    return rank, ws


def _bench(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    rank, ws = init_dist()
    group = dist.group.WORLD
    group_name = group.group_name

    # Try to import multimem
    try:
        import torch.distributed._symmetric_memory as symm_mem
        has_multimem = True
    except ImportError:
        has_multimem = False

    # Try to import flashinfer allreduce fusion
    try:
        from flashinfer.comm import allreduce_with_rmsnorm
        has_flashinfer_fusion = True
    except ImportError:
        has_flashinfer_fusion = False
        # Try alternative import
        try:
            from sglang.srt.layers.layernorm import RMSNorm
            has_rmsnorm = True
        except:
            has_rmsnorm = False

    if rank == 0:
        props = torch.cuda.get_device_properties(0)
        print("=" * 80)
        print(f"multimem vs flashinfer fusion Micro-benchmark")
        print(f"GPU: {props.name}, TP={ws}")
        print(f"multimem available: {has_multimem}")
        print(f"flashinfer fusion available: {has_flashinfer_fusion}")
        print("=" * 80)

    # Test shapes matching Qwen3.5-35B-A3B TP=4
    # hidden_size=3072
    H = 3072
    test_configs = [
        # (M, label) - M is number of tokens
        (1, "decode bs=1"),
        (32, "decode bs=32"),
        (128, "decode bs=128"),
        (512, "prefill 512"),
        (1024, "prefill 1K"),
        (2048, "prefill 2K"),
        (4096, "prefill 4K"),
        (8192, "prefill 8K"),
    ]

    # Setup multimem buffer
    max_numel = max(m * H for m, _ in test_configs)
    if has_multimem:
        symm_buf = symm_mem.empty(max_numel, dtype=torch.bfloat16, device=f"cuda:{rank}")
        handle = symm_mem.rendezvous(symm_buf, group_name)
        if handle.multicast_ptr == 0:
            if rank == 0:
                print("WARNING: multicast not supported")
            has_multimem = False

    # Setup RMSNorm weight (for fusion comparison)
    rmsnorm_weight = torch.ones(H, dtype=torch.bfloat16, device="cuda")
    eps = 1e-6

    # Try to get flashinfer fusion kernel
    try:
        # The actual kernel used in SGLang
        from flashinfer.trtllm_allreduce_fusion import (
            allreduce_fusion_oneshot_lamport,
            init_custom_allreduce,
        )

        # Initialize custom allreduce
        init_custom_allreduce(rank, ws)
        has_flashinfer_ar = True
    except ImportError:
        has_flashinfer_ar = False

    if rank == 0 and not has_flashinfer_ar:
        print("flashinfer trtllm_allreduce_fusion not available, will use NCCL as reference")

    for M, label in test_configs:
        numel = M * H
        size_mb = numel * 2 / 1e6

        x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")

        if rank == 0:
            print(f"\n--- {label}: M={M}, shape=[{M},{H}], {size_mb:.1f}MB ---")

        # 1. NCCL AllReduce baseline
        def bench_nccl():
            dist.all_reduce(x, group=group)
        t_nccl = _bench(bench_nccl, args.warmup, args.iters)

        # 2. multimem AllReduce (with copy-in)
        if has_multimem:
            def bench_multimem_copy():
                symm_buf[:numel].copy_(x.view(-1))
                torch.ops.symm_mem.multimem_all_reduce_(symm_buf[:numel], "sum", group_name)
            t_multimem_copy = _bench(bench_multimem_copy, args.warmup, args.iters)
        else:
            t_multimem_copy = float('nan')

        # 3. multimem AllReduce (zero-copy, data already in buffer)
        if has_multimem:
            # Pre-fill buffer
            symm_buf[:numel].copy_(x.view(-1))
            def bench_multimem_zc():
                torch.ops.symm_mem.multimem_all_reduce_(symm_buf[:numel], "sum", group_name)
            t_multimem_zc = _bench(bench_multimem_zc, args.warmup, args.iters)
        else:
            t_multimem_zc = float('nan')

        # 4. multimem AllReduce + separate RMSNorm
        if has_multimem:
            def bench_multimem_plus_norm():
                torch.ops.symm_mem.multimem_all_reduce_(symm_buf[:numel], "sum", group_name)
                buf_view = symm_buf[:numel].view(M, H)
                # Simulate fused_add_rmsnorm: residual += buf_view; out = rmsnorm(residual)
                residual.add_(buf_view)
                torch.nn.functional.rms_norm(residual, (H,), rmsnorm_weight, eps)
            t_multimem_norm = _bench(bench_multimem_plus_norm, args.warmup, args.iters)
        else:
            t_multimem_norm = float('nan')

        # 5. NCCL AllReduce + separate RMSNorm (simulate what non-fusion does)
        def bench_nccl_plus_norm():
            dist.all_reduce(x, group=group)
            residual.add_(x)
            torch.nn.functional.rms_norm(residual, (H,), rmsnorm_weight, eps)
        t_nccl_norm = _bench(bench_nccl_plus_norm, args.warmup, args.iters)

        if rank == 0:
            print(f"  NCCL AR only:              {t_nccl:>8.1f}us")
            print(f"  multimem (copy-in):        {t_multimem_copy:>8.1f}us")
            print(f"  multimem (zero-copy):      {t_multimem_zc:>8.1f}us")
            print(f"  multimem ZC + RMSNorm:     {t_multimem_norm:>8.1f}us")
            print(f"  NCCL + RMSNorm:            {t_nccl_norm:>8.1f}us")
            if not (t_multimem_zc != t_multimem_zc):  # not nan
                print(f"  --- multimem ZC vs NCCL: {t_nccl/t_multimem_zc:.2f}x ---")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
