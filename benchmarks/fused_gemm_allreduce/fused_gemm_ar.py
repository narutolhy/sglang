"""
Fused GEMM + AllReduce: Producer-Consumer Megakernel (v4)

v4 optimizations:
  - BLOCK_K=64 (was 32), halves K-loop iterations
  - num_warps=8 for higher throughput
  - Software pipelining via tl.range(num_stages=3)
  - Inline PTX for flag operations (ld.acquire.sys / st.release.gpu)
  - Monotonic iteration counter (no flag reset between calls)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Inline PTX primitives
# ---------------------------------------------------------------------------

@triton.jit
def _ld_acquire_sys(ptr):
    return tl.inline_asm_elementwise(
        "ld.acquire.sys.global.b32 $0, [$1];",
        "=r, l", [ptr], dtype=tl.int32, is_pure=False, pack=1,
    )

@triton.jit
def _st_release_gpu(ptr, val):
    tl.inline_asm_elementwise(
        "st.release.gpu.global.b32 [$1], $2;",
        "=r, l, r", [ptr, val], dtype=tl.int32, is_pure=False, pack=1,
    )

@triton.jit
def _membar_sys():
    tl.inline_asm_elementwise(
        "membar.sys; mov.b32 $0, 0;",
        "=r", [], dtype=tl.int32, is_pure=False, pack=1,
    )


# ---------------------------------------------------------------------------
# Megakernel: producer-consumer with optimized GEMM
# ---------------------------------------------------------------------------

@triton.jit
def _fused_gemm_allreduce_kernel(
    a_ptr, b_ptr,
    my_out_ptr, final_out_ptr,
    peer_out_addrs_ptr, peer_flag_addrs_ptr,
    my_flag_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    num_pid_n, num_tiles,
    iter_count,
    NUM_COMM_SMS: tl.constexpr,
    TOTAL_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid < NUM_COMM_SMS:
        # =====================================================
        # Consumer: AllReduce
        # =====================================================
        for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n

            for r in range(WORLD_SIZE):
                flag_base_int = tl.load(peer_flag_addrs_ptr + r)
                flag_base = flag_base_int.to(tl.pointer_type(tl.int32))
                while _ld_acquire_sys(flag_base + tile_id) != iter_count:
                    pass

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            linear = offs_m[:, None] * N + offs_n[None, :]

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for r in range(WORLD_SIZE):
                buf_base_int = tl.load(peer_out_addrs_ptr + r)
                buf_base = buf_base_int.to(tl.pointer_type(tl.bfloat16))
                data = tl.load(buf_base + linear, mask=mask, other=0.0)
                acc += data.to(tl.float32)

            tl.store(final_out_ptr + linear, acc.to(tl.bfloat16), mask=mask)

    else:
        # =====================================================
        # Producer: Optimized persistent GEMM
        # =====================================================
        num_gemm_sms = TOTAL_SMS - NUM_COMM_SMS
        gemm_pid = pid - NUM_COMM_SMS

        for tile_id in range(gemm_pid, num_tiles, num_gemm_sms):
            # L2 cache swizzle for better locality
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Pipelined K-loop with software prefetch
            num_k_iters = tl.cdiv(K, BLOCK_K)
            for k in tl.range(0, num_k_iters, num_stages=NUM_STAGES):
                k_remaining = K - k * BLOCK_K
                a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
                acc = tl.dot(a, b, acc)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk

            # Store tile
            c = acc.to(tl.bfloat16)
            offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(
                my_out_ptr + offs_cm[:, None] * N + offs_cn[None, :],
                c, mask=c_mask,
            )

            # Signal using swizzled tile coordinates (must match consumer)
            flag_id = pid_m * num_pid_n + pid_n
            _membar_sys()
            _st_release_gpu(my_flag_ptr + flag_id, iter_count)


# ---------------------------------------------------------------------------
# Standalone Triton GEMM (for baseline comparison)
# ---------------------------------------------------------------------------

@triton.jit
def _triton_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * N + offs_cn[None, :], c, mask=mask)


def triton_gemm(a, b, block_m=128, block_n=128, block_k=64, num_stages=3, num_warps=8):
    """Standalone Triton GEMM for comparison. a:[M,K] b:[K,N] -> [M,N]"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _triton_gemm_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_SIZE_M=8, NUM_STAGES=num_stages,
        num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class FusedGemmAllReduce:
    def __init__(
        self, group, max_M, N, device,
        num_comm_sms=16, block_m=128, block_n=128, block_k=64,
        num_stages=3, num_warps=8,
    ):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.device = device
        self.max_M = max_M
        self.N = N
        self.num_comm_sms = num_comm_sms
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_stages = num_stages
        self.num_warps = num_warps
        self._iter_count = 0

        max_m_tiles = (max_M + block_m - 1) // block_m
        max_n_tiles = (N + block_n - 1) // block_n
        self.max_tiles = max_m_tiles * max_n_tiles

        self.symm_out = symm_mem.empty(max_M * N, dtype=torch.bfloat16, device=device)
        self.symm_out_handle = symm_mem.rendezvous(self.symm_out, group.group_name)

        self.symm_flags = symm_mem.empty(self.max_tiles, dtype=torch.int32, device=device)
        self.symm_flags_handle = symm_mem.rendezvous(self.symm_flags, group.group_name)

        self.symm_flags.zero_()
        torch.cuda.synchronize()
        dist.barrier(group)

        peer_out_addrs, peer_flag_addrs = [], []
        for r in range(self.world_size):
            peer_out_addrs.append(
                self.symm_out_handle.get_buffer(r, sizes=(max_M * N,), dtype=torch.bfloat16).data_ptr()
            )
            peer_flag_addrs.append(
                self.symm_flags_handle.get_buffer(r, sizes=(self.max_tiles,), dtype=torch.int32).data_ptr()
            )

        self.peer_out_addrs = torch.tensor(peer_out_addrs, dtype=torch.int64, device=device)
        self.peer_flag_addrs = torch.tensor(peer_flag_addrs, dtype=torch.int64, device=device)
        self.output_buf = torch.empty(max_M, N, dtype=torch.bfloat16, device=device)

        props = torch.cuda.get_device_properties(device)
        self.total_sms = props.multi_processor_count

    def __call__(self, a, b):
        return self.forward(a, b)

    def forward(self, a, b):
        """a: [M, K], b: [K, N] → [M, N] all-reduced."""
        M, K = a.shape
        num_pid_n = (self.N + self.block_n - 1) // self.block_n
        num_pid_m = (M + self.block_m - 1) // self.block_m
        num_tiles = num_pid_m * num_pid_n

        self._iter_count += 1

        _fused_gemm_allreduce_kernel[(self.total_sms,)](
            a, b, self.symm_out, self.output_buf,
            self.peer_out_addrs, self.peer_flag_addrs, self.symm_flags,
            M, self.N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            num_pid_n, num_tiles, self._iter_count,
            NUM_COMM_SMS=self.num_comm_sms,
            TOTAL_SMS=self.total_sms,
            BLOCK_M=self.block_m, BLOCK_N=self.block_n, BLOCK_K=self.block_k,
            GROUP_SIZE_M=8, WORLD_SIZE=self.world_size,
            NUM_STAGES=self.num_stages,
            num_warps=self.num_warps,
        )
        return self.output_buf[:M, :self.N]

    def cleanup(self):
        self.symm_out = self.symm_flags = None
        self.symm_out_handle = self.symm_flags_handle = None
