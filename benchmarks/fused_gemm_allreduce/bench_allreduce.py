"""
Benchmark: P2P AllReduce (symmetric_memory) vs NCCL AllReduce.

Tests multiple AllReduce implementations:
  1. NCCL all_reduce (baseline)
  2. P2P direct: read from all peers via symmetric_memory, sum locally (Triton kernel)
  3. P2P direct: same but with torch ops (no Triton)

Usage:
    torchrun --nproc_per_node=4 bench_allreduce.py
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl


def init_dist():
    rank = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    torch.cuda.set_device(rank)
    return rank, ws


# ---------------------------------------------------------------------------
# Triton P2P AllReduce kernel
# ---------------------------------------------------------------------------

@triton.jit
def _p2p_allreduce_kernel(
    peer_addrs_ptr,  # int64 tensor [WORLD_SIZE] — peer buffer data_ptrs
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Read from all peers via P2P/NVLink, sum, write to output."""
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)

    for start in range(pid * BLOCK_SIZE, numel, num_blocks * BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for r in range(WORLD_SIZE):
            buf_addr = tl.load(peer_addrs_ptr + r)
            buf_ptr = buf_addr.to(tl.pointer_type(tl.bfloat16))
            data = tl.load(buf_ptr + offs, mask=mask, other=0.0)
            acc += data.to(tl.float32)

        tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


class P2PAllReduce:
    """P2P AllReduce using symmetric_memory + Triton kernel."""

    def __init__(self, group, max_numel, device, num_sms=132, block_size=1024):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.device = device
        self.num_sms = num_sms
        self.block_size = block_size

        # Symmetric memory buffer
        self.symm_buf = symm_mem.empty(max_numel, dtype=torch.bfloat16, device=device)
        self.symm_handle = symm_mem.rendezvous(self.symm_buf, group.group_name)

        # Collect peer addresses
        peer_addrs = []
        for r in range(self.world_size):
            peer = self.symm_handle.get_buffer(
                r, sizes=(max_numel,), dtype=torch.bfloat16
            )
            peer_addrs.append(peer.data_ptr())
        self.peer_addrs = torch.tensor(peer_addrs, dtype=torch.int64, device=device)

        # Keep peer tensor references alive
        self._peer_tensors = []
        for r in range(self.world_size):
            self._peer_tensors.append(
                self.symm_handle.get_buffer(r, sizes=(max_numel,), dtype=torch.bfloat16)
            )

    def allreduce(self, inp, out=None):
        """
        P2P AllReduce: copy input to symm buffer, read all peers, sum.

        Args:
            inp: [numel] or [M, N] bfloat16 tensor
            out: optional output tensor (same shape as inp)
        Returns:
            all-reduced tensor
        """
        numel = inp.numel()
        if out is None:
            out = torch.empty_like(inp)

        # Copy input to symmetric buffer (makes it visible to peers)
        self.symm_buf[:numel].copy_(inp.view(-1))

        # Barrier: ensure all ranks have written before reading
        # Use NCCL barrier (lightweight, ~10μs)
        dist.barrier(self.group)

        # Triton kernel: read from all peers and sum
        grid = (min(self.num_sms, triton.cdiv(numel, self.block_size)),)
        _p2p_allreduce_kernel[grid](
            self.peer_addrs,
            out.view(-1),
            numel,
            BLOCK_SIZE=self.block_size,
            WORLD_SIZE=self.world_size,
        )
        return out

    def allreduce_torch(self, inp, out=None):
        """Same but using torch ops instead of Triton (for comparison)."""
        numel = inp.numel()
        if out is None:
            out = torch.zeros_like(inp)
        else:
            out.zero_()

        self.symm_buf[:numel].copy_(inp.view(-1))
        dist.barrier(self.group)

        # Sum from all peers using torch
        for r in range(self.world_size):
            peer = self._peer_tensors[r][:numel].view(inp.shape)
            out.add_(peer)
        return out

    def cleanup(self):
        self.symm_buf = None
        self.symm_handle = None
        self._peer_tensors = None


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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    rank, ws = init_dist()
    group = dist.group.WORLD

    if rank == 0:
        props = torch.cuda.get_device_properties(0)
        print("=" * 80)
        print(f"AllReduce Benchmark: NCCL vs P2P (symmetric_memory)")
        print(f"GPU: {props.name}, SMs: {props.multi_processor_count}, TP={ws}")
        print("=" * 80)

    # Test shapes matching Llama-70B TP=4 AllReduce sizes
    H = 8192
    shapes = []
    for M in [512, 1024, 2048, 4096, 8192]:
        shapes.append((M, H, M * H))  # (M, N, numel)

    max_numel = max(s[2] for s in shapes)

    # Setup P2P AllReduce
    p2p = P2PAllReduce(group, max_numel, torch.device(f"cuda:{rank}"))

    if rank == 0:
        print(f"\n{'M':>6} x {'N':>5} | {'numel':>10} | {'NCCL':>8} | {'P2P-Triton':>10} | {'P2P-Torch':>10} | {'NCCL/P2P-T':>10}")
        print("-" * 80)

    for M, N, numel in shapes:
        inp = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        out_nccl = torch.empty_like(inp)
        out_p2p = torch.empty_like(inp)
        out_torch = torch.empty_like(inp)

        # Correctness check
        out_nccl.copy_(inp)
        dist.all_reduce(out_nccl, group=group)

        out_p2p = p2p.allreduce(inp)
        torch.cuda.synchronize()

        out_torch = p2p.allreduce_torch(inp)
        torch.cuda.synchronize()

        cos_triton = torch.nn.functional.cosine_similarity(
            out_nccl.flatten().float(), out_p2p.flatten().float(), dim=0
        ).item()
        cos_torch = torch.nn.functional.cosine_similarity(
            out_nccl.flatten().float(), out_torch.flatten().float(), dim=0
        ).item()

        if rank == 0 and M == shapes[0][0]:
            print(f"  Correctness: P2P-Triton cos_sim={cos_triton:.8f}, P2P-Torch cos_sim={cos_torch:.8f}")

        # Benchmark NCCL
        def nccl_fn():
            buf = inp.clone()
            dist.all_reduce(buf, group=group)
        t_nccl = _bench(nccl_fn, args.warmup, args.iters)

        # Benchmark P2P Triton
        t_p2p = _bench(lambda: p2p.allreduce(inp, out_p2p), args.warmup, args.iters)

        # Benchmark P2P Torch
        t_torch = _bench(lambda: p2p.allreduce_torch(inp, out_torch), args.warmup, args.iters)

        if rank == 0:
            ratio = t_nccl / t_p2p
            print(
                f"{M:>6} x {N:>5} | {numel:>10} | {t_nccl:>6.3f}ms | "
                f"{t_p2p:>8.3f}ms | {t_torch:>8.3f}ms | {ratio:>9.2f}x"
            )

        del inp, out_nccl, out_p2p, out_torch

    # Also test with torch.ops.symm_mem if available
    if rank == 0:
        print(f"\n--- PyTorch symm_mem built-in AllReduce ---")

    try:
        symm_buf2 = symm_mem.empty(max_numel, dtype=torch.bfloat16, device=f"cuda:{rank}")
        handle2 = symm_mem.rendezvous(symm_buf2, group.group_name)

        for M, N, numel in shapes:
            inp = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

            def symm_ar():
                symm_buf2[:numel].copy_(inp.view(-1))
                torch.ops.symm_mem.multimem_all_reduce_(
                    symm_buf2[:numel], "sum", group.group_name
                )

            try:
                t_symm = _bench(symm_ar, args.warmup, args.iters)
                if rank == 0:
                    print(f"  {M:>6} x {N:>5}: symm_mem.multimem_all_reduce = {t_symm:.3f}ms")
            except Exception as e:
                if rank == 0:
                    print(f"  {M:>6} x {N:>5}: symm_mem FAILED: {e}")
            del inp
    except Exception as e:
        if rank == 0:
            print(f"  symm_mem AllReduce not available: {e}")

    p2p.cleanup()

    if rank == 0:
        print(f"\n{'='*80}\nDone.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
