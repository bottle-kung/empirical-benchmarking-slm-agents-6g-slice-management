#!/bin/bash
# SLM 6G Eval v2 — Resource Re-Run with CPU% + CI (v3)
# lab-slm-01: 0.5B, 1.5B, 3B × Q2_K, Q4_K_M, Q8_0 (9 runs)
# lab-slm-02: 7B × Q2_K, Q4_K_M, Q8_0 (3 runs)
set -e

SCRIPT=/tmp/benchmark_runner_v2.py
RESULTS=/tmp/resource_results
mkdir -p $RESULTS /tmp/benchmark_results
rm -f $RESULTS/*.json

run_one() {
    local model=$1 family=$2 size=$3 q=$4
    local stamp=$(date +%H:%M:%S)
    echo "[$stamp] === $model ($size $q) ==="
    python3 $SCRIPT --model "$model" --family "$family" --phase 2
    local safe=$(echo "$model" | tr ':' '_' | tr '.' '_')
    cp /tmp/benchmark_results/benchmark_${safe}.json $RESULTS/${size}_${q}.json
    echo "[$(date +%H:%M:%S)] Done: $model"
}

echo "[$(date +%H:%M:%S)] === RESOURCE RE-RUN v3 (CPU%+CI) ==="
echo "Host: $(hostname) | CPUs: $(nproc)"
echo ""

if [ "$(hostname)" = "lab-slm-01" ]; then
    # Q4_K_M (standard Ollama models)
    run_one "qwen2.5:0.5b" "Qwen" "0.5B" "Q4_K_M"
    run_one "qwen2.5:1.5b" "Qwen" "1.5B" "Q4_K_M"
    run_one "qwen2.5:3b" "Qwen" "3B" "Q4_K_M"
    
    # Q2_K (custom)
    run_one "qwen2_5-0_5b-q2k:latest" "Qwen" "0.5B" "Q2_K"
    run_one "qwen2_5-1_5b-q2k:latest" "Qwen" "1.5B" "Q2_K"
    run_one "qwen2_5-3b-q2k:latest" "Qwen" "3B" "Q2_K"
    
    # Q8_0 (custom)
    run_one "qwen2_5-0_5b-q8_0:latest" "Qwen" "0.5B" "Q8_0"
    run_one "qwen2_5-1_5b-q8_0:latest" "Qwen" "1.5B" "Q8_0"
    run_one "qwen2_5-3b-q8_0:latest" "Qwen" "3B" "Q8_0"
fi

if [ "$(hostname)" = "lab-slm-02" ]; then
    # Q4_K_M (standard)
    run_one "qwen2.5:7b" "Qwen" "7B" "Q4_K_M"
    # Q2_K (custom)
    run_one "qwen2.5-7b-q2k:latest" "Qwen" "7B" "Q2_K"
    # Q8_0 (custom)
    run_one "qwen2.5-7b-q8_0:latest" "Qwen" "7B" "Q8_0"
fi

echo ""
echo "[$(date +%H:%M:%S)] === ALL DONE ==="
ls -la $RESULTS/
