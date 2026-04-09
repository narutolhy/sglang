"""
Simulate Llama-70B TP=4 prefill AllReduce pattern.

80 layers × 2 AllReduces per layer (after o_proj + after down_proj) = 160 calls.
Compares: NCCL vs symm_mem.multimem_all_reduce.

Usage:
    torchrun --nproc_per_node=4 bench_prefill_ar.py
"""
import argparse, os, time
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def init_dist():
    rank = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    torch.cuda.set_device(rank)
    return rank, ws


def bench_nccl_prefill(tensors, group, warmup=5, iters=20):
    """Simulate full prefill AllReduce sequence with NCCL."""
    for _ in range(warmup):
        for t in tensors:
            buf = t.clone()
            dist.all_reduce(buf, group=group)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        for t in tensors:
            buf = t.clone()
            dist.all_reduce(buf, group=group)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def bench_multimem_prefill(tensors, symm_buf, group_name, warmup=5, iters=20):
    """Simulate full prefill AllReduce sequence with symm_mem.multimem."""
    max_numel = symm_buf.numel()

    for _ in range(warmup):
        for t in tensors:
            n = t.numel()
            symm_buf[:n].copy_(t.view(-1))
            torch.ops.symm_mem.multimem_all_reduce_(symm_buf[:n], "sum", group_name)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        for t in tensors:
            n = t.numel()
            symm_buf[:n].copy_(t.view(-1))
            torch.ops.symm_mem.multimem_all_reduce_(symm_buf[:n], "sum", group_name)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=80)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    rank, ws = init_dist()
    group = dist.group.WORLD

    H = args.hidden_size
    num_layers = args.num_layers

    if rank == 0:
        props = torch.cuda.get_device_properties(0)
        print("=" * 70)
        print(f"Llama-70B Prefill AllReduce Simulation")
        print(f"GPU: {props.name}, TP={ws}")
        print(f"Layers: {num_layers}, Hidden: {H}")
        print(f"AllReduces per forward: {num_layers * 2}")
        print("=" * 70)

    for M in [1024, 2048, 4096, 8192]:
        if rank == 0:
            print(f"\n--- M = {M} tokens ---")

        # Create tensors matching Llama-70B AllReduce sizes
        # Each layer: o_proj AllReduce [M, H] + down_proj AllReduce [M, H]
        tensors = []
        for _ in range(num_layers):
            tensors.append(torch.randn(M, H, dtype=torch.bfloat16, device="cuda"))  # o_proj
            tensors.append(torch.randn(M, H, dtype=torch.bfloat16, device="cuda"))  # down_proj

        max_numel = M * H

        # Setup symm_mem
        symm_buf = symm_mem.empty(max_numel, dtype=torch.bfloat16, device=f"cuda:{rank}")
        handle = symm_mem.rendezvous(symm_buf, group.group_name)

        # Correctness check
        ref = tensors[0].clone()
        dist.all_reduce(ref, group=group)

        n = tensors[0].numel()
        symm_buf[:n].copy_(tensors[0].view(-1))
        torch.ops.symm_mem.multimem_all_reduce_(symm_buf[:n], "sum", group.group_name)
        out = symm_buf[:n].view(tensors[0].shape).clone()

        cos = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), out.flatten().float(), dim=0
        ).item()
        if rank == 0:
            print(f"  Correctness: cos_sim={cos:.8f}")

        # Benchmark NCCL
        t_nccl = bench_nccl_prefill(tensors, group, args.warmup, args.iters)

        # Benchmark multimem
        t_mm = bench_multimem_prefill(tensors, symm_buf, group.group_name, args.warmup, args.iters)

        if rank == 0:
            num_ar = len(tensors)
            nccl_per = t_nccl / num_ar
            mm_per = t_mm / num_ar
            speedup = t_nccl / t_mm
            saved = t_nccl - t_mm
            print(f"  NCCL total:     {t_nccl:8.2f}ms  ({nccl_per:.3f}ms/call)")
            print(f"  multimem total: {t_mm:8.2f}ms  ({mm_per:.3f}ms/call)")
            print(f"  Speedup: {speedup:.2f}x  (saved {saved:.2f}ms)")

        del tensors, symm_buf
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        print(f"\n{'='*70}\nDone.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
