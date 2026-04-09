"""Minimal test: verify cuBLAS + flag setter + consumer kernel work together."""
import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import time

def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    M, K, N = 1024, 2048, 8192
    num_chunks = 4
    chunk_m = M // num_chunks

    if rank == 0:
        print(f"Test: M={M}, K={K}, N={N}, TP={world_size}, chunks={num_chunks}")

    # Inputs
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")  # [N, K]

    # --- Test 1: Sequential baseline ---
    ref = torch.nn.functional.linear(a, w)
    dist.all_reduce(ref)
    if rank == 0:
        print(f"Test 1 (sequential): ref_norm={ref.float().norm().item():.2f}")

    # --- Test 2: cuBLAS → symm buffer → manual all_reduce ---
    symm_buf = symm_mem.empty(M * N, dtype=torch.bfloat16, device="cuda")
    symm_handle = symm_mem.rendezvous(symm_buf, dist.group.WORLD.group_name)
    symm_buf_2d = symm_buf.view(M, N)

    # cuBLAS GEMM into symm buffer
    torch.mm(a, w.t(), out=symm_buf_2d)
    torch.cuda.synchronize()
    dist.barrier()

    # Read from all peers and reduce
    out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    for r in range(world_size):
        peer_buf = symm_handle.get_buffer(r, sizes=(M * N,), dtype=torch.bfloat16)
        out += peer_buf.view(M, N)
    torch.cuda.synchronize()

    max_diff = (ref - out).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref.flatten().float(), out.flatten().float(), dim=0
    ).item()
    if rank == 0:
        print(f"Test 2 (cuBLAS→symm→manual reduce): max_diff={max_diff:.4f}, cos_sim={cos_sim:.8f}")

    # --- Test 3: Two-stream overlap test ---
    if rank == 0:
        print("\nTest 3: Two-stream concurrent kernel test...")

    # Allocate flag buffer
    symm_flags = symm_mem.empty(num_chunks, dtype=torch.int32, device="cuda")
    symm_flags_handle = symm_mem.rendezvous(symm_flags, dist.group.WORLD.group_name)
    symm_flags.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    from fused_gemm_ar import _set_chunk_flag_kernel, _allreduce_consumer_kernel

    # Collect peer addresses
    peer_out_addrs = []
    peer_flag_addrs = []
    for r in range(world_size):
        peer_out = symm_handle.get_buffer(r, sizes=(M * N,), dtype=torch.bfloat16)
        peer_flag = symm_flags_handle.get_buffer(r, sizes=(num_chunks,), dtype=torch.int32)
        peer_out_addrs.append(peer_out.data_ptr())
        peer_flag_addrs.append(peer_flag.data_ptr())

    peer_out_addrs_t = torch.tensor(peer_out_addrs, dtype=torch.int64, device="cuda")
    peer_flag_addrs_t = torch.tensor(peer_flag_addrs, dtype=torch.int64, device="cuda")
    output_buf = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    compute_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream()

    iter_count = 1

    if rank == 0:
        print("  Launching consumer on comm_stream...")

    # Launch consumer on comm_stream
    with torch.cuda.stream(comm_stream):
        _allreduce_consumer_kernel[(16,)](
            peer_out_addrs_t,
            peer_flag_addrs_t,
            output_buf,
            M * N,        # total_elems
            chunk_m * N,   # chunk_elems
            num_chunks,
            iter_count,
            BLOCK_SIZE=1024,
            NUM_COMM_SMS=16,
            WORLD_SIZE=world_size,
        )

    if rank == 0:
        print("  Consumer launched. Now doing chunked GEMM + flag setting...")

    # Chunked GEMM + flag setting on compute_stream
    w_t = w.t()
    for c in range(num_chunks):
        start = c * chunk_m
        end = start + chunk_m
        torch.mm(a[start:end], w_t, out=symm_buf_2d[start:end])
        _set_chunk_flag_kernel[(1,)](symm_flags, c, iter_count)
        if rank == 0:
            print(f"    Chunk {c} GEMM done, flag set")

    if rank == 0:
        print("  Waiting for comm_stream...")

    # Wait with timeout
    start_time = time.time()
    while not comm_stream.query():
        if time.time() - start_time > 30:
            if rank == 0:
                print("  TIMEOUT! Consumer kernel didn't finish in 30s")
                print("  This means two-stream overlap is NOT working on this GPU")
            dist.destroy_process_group()
            return
        time.sleep(0.01)

    if rank == 0:
        print("  Consumer finished!")

    torch.cuda.synchronize()

    max_diff = (ref - output_buf).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref.flatten().float(), output_buf.flatten().float(), dim=0
    ).item()
    if rank == 0:
        print(f"Test 3 result: max_diff={max_diff:.4f}, cos_sim={cos_sim:.8f}")

    dist.destroy_process_group()
    if rank == 0:
        print("\nAll tests done!")


if __name__ == "__main__":
    main()
