#!/bin/bash
# E2E benchmark: Llama-3.1-70B-Instruct TP=4
# Compares baseline AllReduce vs symm_mem multimem AllReduce
#
# Usage: bash bench_e2e.sh

export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export PATH="$HOME/.local/bin:$PATH"

MODEL="meta-llama/Llama-3.1-70B-Instruct"
PORT=30100
TP=4
INPUT_LENS="1024,2048,4096"
NUM_PROMPTS=10
MAX_NEW_TOKENS=1  # prefill-only

run_bench() {
    local label=$1
    shift
    local extra_args="$@"

    echo ""
    echo "============================================================"
    echo "  $label"
    echo "============================================================"

    # Start server
    echo "Starting server with: $extra_args"
    python3 -m sglang.launch_server \
        --model $MODEL \
        --tp $TP \
        --port $PORT \
        --mem-fraction-static 0.85 \
        --disable-radix-cache \
        $extra_args \
        > /tmp/sglang_server_${label}.log 2>&1 &
    SERVER_PID=$!

    # Wait for server
    echo "Waiting for server (PID=$SERVER_PID)..."
    for i in $(seq 1 120); do
        if curl -sf -m 2 http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server died! Check /tmp/sglang_server_${label}.log"
            return 1
        fi
        sleep 1
    done

    if ! curl -sf -m 2 http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server didn't start in 120s"
        kill $SERVER_PID 2>/dev/null
        return 1
    fi

    # Run benchmark for each input length
    for INPUT_LEN in $(echo $INPUT_LENS | tr ',' ' '); do
        echo ""
        echo "--- Input length: $INPUT_LEN ---"
        python3 -m sglang.bench_serving \
            --backend sglang \
            --port $PORT \
            --model $MODEL \
            --dataset-name random \
            --random-input $INPUT_LEN \
            --random-output $MAX_NEW_TOKENS \
            --num-prompts $NUM_PROMPTS \
            --request-rate 999 \
            2>&1 | grep -E "mean_ttft|median_ttft|mean_e2e|median_e2e|throughput|Benchmark"
    done

    # Shutdown
    echo ""
    echo "Shutting down server..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 3
}

echo "============================================================"
echo "  Llama-3.1-70B-Instruct TP=$TP Prefill Benchmark"
echo "  Comparing: Baseline vs symm_mem multimem AllReduce"
echo "============================================================"

# Baseline: default AllReduce (custom_AR + NCCL)
run_bench "baseline" ""

# Optimized: symm_mem multimem AllReduce
run_bench "multimem" "--enable-torch-symm-mem --disable-custom-all-reduce"

echo ""
echo "============================================================"
echo "  Done! Compare TTFT numbers above."
echo "============================================================"
