"""
Benchmark v4: Optimized Triton GEMM megakernel + AllReduce.

Compares:
  1. cuBLAS GEMM only (torch.mm)
  2. Triton GEMM only (standalone, no AllReduce)
  3. Sequential: cuBLAS GEMM + NCCL AllReduce
  4. Fused megakernel: Triton GEMM + flag-based AllReduce

Usage:
    torchrun --nproc_per_node=4 benchmark.py
"""
import argparse, os, time
import torch
import torch.distributed as dist
from fused_gemm_ar import FusedGemmAllReduce, triton_gemm


def init_dist():
    rank = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    torch.cuda.set_device(rank)
    return rank, ws


def _bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--intermediate-size", type=int, default=28672)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    rank, ws = init_dist()
    group = dist.group.WORLD
    H, I = args.hidden_size, args.intermediate_size
    H_tp, I_tp = H // ws, I // ws

    if rank == 0:
        props = torch.cuda.get_device_properties(0)
        print("=" * 80)
        print(f"v4 Benchmark: Optimized Triton GEMM Megakernel")
        print(f"GPU: {props.name}, SMs: {props.multi_processor_count}, TP={ws}")
        print("=" * 80)

    shapes = [("o_proj", H_tp, H), ("down_proj", I_tp, H)]
    M_values = [1024, 2048, 4096, 8192]

    # --- Test 1: Triton GEMM standalone vs cuBLAS ---
    if rank == 0:
        print(f"\n{'='*80}")
        print("Part 1: Triton GEMM vs cuBLAS GEMM (no AllReduce)")
        print(f"{'='*80}")
        print(f"{'Layer':<12} {'M':>5} | {'cuBLAS':>8} | {'Triton':>8} | {'Ratio':>7}")
        print(f"{'-'*12} {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")

    for name, K, N in shapes:
        for M in M_values:
            a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            w_kn = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
            w_nk = w_kn.t().contiguous()

            t_cublas = _bench(lambda: torch.mm(a, w_kn), args.warmup, args.iters)
            t_triton = _bench(lambda: triton_gemm(a, w_kn), args.warmup, args.iters)

            if rank == 0:
                ratio = t_triton / t_cublas
                print(f"{name:<12} {M:>5} | {t_cublas:>6.3f}ms | {t_triton:>6.3f}ms | {ratio:>6.2f}x")
            del a, w_kn, w_nk

    # --- Test 2: Fused megakernel vs sequential ---
    if rank == 0:
        print(f"\n{'='*80}")
        print("Part 2: Fused Megakernel vs Sequential (GEMM + AllReduce)")
        print(f"{'='*80}")
        print(f"{'Layer':<12} {'M':>5} | {'Seq':>8} | {'Fused':>8} | {'Speedup':>8} | {'Correct'}")
        print(f"{'-'*12} {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")

    for name, K, N in shapes:
        for M in M_values:
            a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            w_nk = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
            w_kn = w_nk.t().contiguous()

            try:
                fused = FusedGemmAllReduce(
                    group, max_M=M, N=N,
                    device=torch.device(f"cuda:{rank}"),
                    num_comm_sms=16, block_m=128, block_n=128, block_k=64,
                    num_stages=3, num_warps=8,
                )
            except Exception as e:
                if rank == 0:
                    print(f"{name:<12} {M:>5} | INIT FAIL: {e}")
                continue

            # Correctness
            ref = torch.nn.functional.linear(a, w_nk)
            dist.all_reduce(ref, group=group)
            out = fused(a, w_kn)
            torch.cuda.synchronize()
            cos = torch.nn.functional.cosine_similarity(
                ref.flatten().float(), out.flatten().float(), dim=0
            ).item()
            correct = "OK" if cos > 0.999 else f"FAIL({cos:.4f})"

            # Benchmark
            t_seq = _bench(
                lambda: (dist.all_reduce(torch.nn.functional.linear(a, w_nk), group=group)),
                args.warmup, args.iters
            )
            t_fused = _bench(lambda: fused(a, w_kn), args.warmup, args.iters)

            if rank == 0:
                sp = t_seq / t_fused
                print(f"{name:<12} {M:>5} | {t_seq:>6.3f}ms | {t_fused:>6.3f}ms | {sp:>7.2f}x | {correct}")

            fused.cleanup()
            del a, w_nk, w_kn
            torch.cuda.empty_cache()
            dist.barrier()

    if rank == 0:
        print(f"\n{'='*80}\nDone.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
