"""Sweep megakernel configs to find optimal parameters."""
import argparse, os, time, itertools
import torch
import torch.distributed as dist
from fused_gemm_ar import FusedGemmAllReduce


def init_dist():
    rank = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    torch.cuda.set_device(rank)
    return rank, ws


def _bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    rank, ws = init_dist()
    group = dist.group.WORLD
    H = 8192
    H_tp = H // ws
    I = 28672
    I_tp = I // ws

    # Test shapes
    test_cases = [
        ("o_proj",    H_tp, H,  4096),
        ("o_proj",    H_tp, H,  8192),
        ("down_proj", I_tp, H,  4096),
        ("down_proj", I_tp, H,  8192),
    ]

    # Config sweep
    configs = list(itertools.product(
        [4, 8, 16],       # NUM_COMM_SMS
        [128, 256],        # BLOCK_N
        [64],              # BLOCK_K
        [3],               # NUM_STAGES
        [8],               # NUM_WARPS
    ))

    if rank == 0:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, SMs: {props.multi_processor_count}, TP={ws}")
        print(f"Sweeping {len(configs)} configs x {len(test_cases)} shapes\n")

    # Baseline: sequential
    if rank == 0:
        print("Baselines (Sequential: F.linear + NCCL AllReduce):")
    baselines = {}
    for name, K, N, M in test_cases:
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w_nk = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        t = _bench(lambda: dist.all_reduce(torch.nn.functional.linear(a, w_nk), group=group))
        baselines[(name, M)] = t
        if rank == 0:
            print(f"  {name} M={M}: {t:.3f}ms")
        del a, w_nk
    torch.cuda.empty_cache()

    if rank == 0:
        print(f"\n{'Config':>30} | {'Shape':>18} | {'Fused':>8} | {'vs Seq':>7} | {'Correct'}")
        print("-" * 90)

    best_results = {}

    for num_comm, block_n, block_k, num_stages, num_warps in configs:
        config_str = f"comm={num_comm},BN={block_n},BK={block_k},st={num_stages}"

        for name, K, N, M in test_cases:
            a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            w_nk = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
            w_kn = w_nk.t().contiguous()

            shape_str = f"{name} M={M}"

            try:
                fused = FusedGemmAllReduce(
                    group, max_M=M, N=N,
                    device=torch.device(f"cuda:{rank}"),
                    num_comm_sms=num_comm,
                    block_m=128, block_n=block_n, block_k=block_k,
                    num_stages=num_stages, num_warps=num_warps,
                )
            except Exception as e:
                if rank == 0:
                    print(f"{config_str:>30} | {shape_str:>18} | {'FAIL':>8} | {'-':>7} | {str(e)[:30]}")
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
            t_fused = _bench(lambda: fused(a, w_kn))
            t_seq = baselines[(name, M)]
            ratio = t_seq / t_fused

            if rank == 0:
                print(f"{config_str:>30} | {shape_str:>18} | {t_fused:>6.3f}ms | {ratio:>6.2f}x | {correct}")

            # Track best
            key = (name, M)
            if key not in best_results or t_fused < best_results[key][0]:
                best_results[key] = (t_fused, config_str, ratio)

            fused.cleanup()
            del a, w_nk, w_kn
            torch.cuda.empty_cache()
            dist.barrier()

    # Summary
    if rank == 0:
        print(f"\n{'='*80}")
        print("Best configs per shape:")
        for (name, M), (t, cfg, ratio) in sorted(best_results.items()):
            t_seq = baselines[(name, M)]
            print(f"  {name} M={M}: {cfg} → {t:.3f}ms (seq={t_seq:.3f}ms, ratio={ratio:.2f}x)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
