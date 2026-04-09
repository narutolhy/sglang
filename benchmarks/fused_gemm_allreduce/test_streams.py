"""Test: verify two non-default streams can run kernels concurrently on H100."""
import torch
import triton
import triton.language as tl
import time


@triton.jit
def _spin_kernel(flag_ptr, expected_val):
    """Spin until flag == expected_val."""
    while tl.atomic_add(flag_ptr, 0, sem='acquire', scope='gpu') != expected_val:
        pass


@triton.jit
def _set_flag_kernel(flag_ptr, val):
    """Set flag = val."""
    tl.atomic_xchg(flag_ptr, val, sem='release', scope='gpu')


def test_default_stream():
    """Test: consumer on stream B, producer on DEFAULT stream → expect deadlock."""
    print("Test 1: consumer on non-default, producer on DEFAULT stream")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    comm_stream = torch.cuda.Stream()

    # Launch spinner on comm_stream
    with torch.cuda.stream(comm_stream):
        _spin_kernel[(1,)](flag, 1)

    # Set flag on DEFAULT stream
    # DEFAULT stream serializes with comm_stream → deadlock!
    print("  Launching flag setter on default stream...")
    start = time.time()
    _set_flag_kernel[(1,)](flag, 1)

    # Wait with timeout
    while not comm_stream.query():
        if time.time() - start > 5:
            print("  DEADLOCK after 5s (as expected with default stream)")
            # Force cleanup
            torch.cuda._sleep(0)
            return False
        time.sleep(0.01)
    print("  OK (no deadlock)")
    return True


def test_two_nondefault_streams():
    """Test: consumer on stream A, producer on stream B → should work."""
    print("\nTest 2: consumer on stream A, producer on stream B (both non-default)")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    # Launch spinner on stream A
    with torch.cuda.stream(stream_a):
        _spin_kernel[(1,)](flag, 1)

    # Set flag on stream B (non-default)
    with torch.cuda.stream(stream_b):
        _set_flag_kernel[(1,)](flag, 1)

    # Wait with timeout
    start = time.time()
    while not stream_a.query():
        if time.time() - start > 5:
            print("  DEADLOCK after 5s — concurrent kernel execution not working")
            return False
        time.sleep(0.01)

    elapsed = (time.time() - start) * 1000
    print(f"  OK! Completed in {elapsed:.1f}ms — two-stream concurrency works!")
    return True


def test_cublas_with_spinner():
    """Test: persistent consumer on stream A, cuBLAS on stream B."""
    print("\nTest 3: persistent consumer + cuBLAS on two non-default streams")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    M, K, N = 1024, 2048, 8192
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    stream_a = torch.cuda.Stream()  # consumer (spin-wait)
    stream_b = torch.cuda.Stream()  # producer (cuBLAS + flag)

    # Launch spinner on stream A (1 CTA, spins until flag=1)
    with torch.cuda.stream(stream_a):
        _spin_kernel[(1,)](flag, 1)

    # Launch cuBLAS + flag setter on stream B
    with torch.cuda.stream(stream_b):
        torch.mm(a, w, out=out)  # cuBLAS GEMM
        _set_flag_kernel[(1,)](flag, 1)  # set flag after GEMM

    start = time.time()
    while not stream_a.query():
        if time.time() - start > 10:
            print("  DEADLOCK after 10s — cuBLAS can't run alongside spinner")
            return False
        time.sleep(0.01)

    elapsed = (time.time() - start) * 1000
    ref = torch.mm(a, w)
    diff = (ref - out).abs().max().item()
    print(f"  OK! Completed in {elapsed:.1f}ms, max_diff={diff:.4f}")
    return True


def test_cublas_with_16sm_spinner():
    """Test: 16-CTA persistent consumer + cuBLAS on two non-default streams."""
    print("\nTest 4: 16-CTA spinner (simulating AllReduce consumer) + cuBLAS")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    M, K, N = 4096, 2048, 8192
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    # 16 CTAs spinning (like our AllReduce consumer)
    with torch.cuda.stream(stream_a):
        _spin_kernel[(16,)](flag, 1)

    # cuBLAS on stream B
    with torch.cuda.stream(stream_b):
        torch.mm(a, w, out=out)
        _set_flag_kernel[(1,)](flag, 1)

    start = time.time()
    while not stream_a.query():
        if time.time() - start > 10:
            print("  DEADLOCK after 10s")
            return False
        time.sleep(0.01)

    elapsed = (time.time() - start) * 1000
    ref = torch.mm(a, w)
    diff = (ref - out).abs().max().item()
    print(f"  OK! {elapsed:.1f}ms, max_diff={diff:.4f}")
    return True


if __name__ == "__main__":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, SMs: {props.multi_processor_count}\n")

    # Skip test_default_stream() — it causes unrecoverable deadlock
    print("Test 1: SKIPPED (default stream deadlock is unrecoverable)\n")
    test_two_nondefault_streams()
    test_cublas_with_spinner()
    test_cublas_with_16sm_spinner()
    print("\nAll tests done!")
