#!/bin/bash
# Phase 3 — Re-run all 4 benchmarks with proper result preservation
# Run on lab-slm-01

set -e

BENCH=/tmp/benchmark_runner.py
SAVE=/tmp/phase3_results
GGUF=/tmp/gguf_models
mkdir -p $SAVE

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 0: Re-create 3B Q8_0 (it failed before) ──
log "Creating qwen2.5-3b-q8_0..."
ollama rm qwen2.5-3b-q8_0 2>/dev/null || true
ollama create qwen2.5-3b-q8_0 -f $GGUF/Modelfile.qwen2.5_3b_q8_0
ollama run qwen2.5-3b-q8_0 "Hello" 2>&1 | head -1
log "qwen2.5-3b-q8_0 created OK"

# ── Run benchmarks one by one, save each ──
declare -A MODELS=(
    ["0.5B_q2k"]="qwen2.5-0.5b-q2_k"
    ["0.5B_q8_0"]="qwen2.5-0.5b-q8_0"
    ["3B_q2k"]="qwen2.5-3b-q2_k"
    ["3B_q8_0"]="qwen2.5-3b-q8_0"
)

for KEY in "0.5B_q2k" "0.5B_q8_0" "3B_q2k" "3B_q8_0"; do
    MODEL="${MODELS[$KEY]}"
    log "=== Running $MODEL ($KEY) ==="
    
    rm -rf /tmp/benchmark_results
    mkdir -p /tmp/benchmark_results
    
    python3 $BENCH --model $MODEL --family Qwen --phase 3
    
    # Find result file and copy to safe location
    RESULT=$(ls /tmp/benchmark_results/benchmark_*.json 2>/dev/null | head -1)
    if [ -n "$RESULT" ]; then
        cp "$RESULT" "$SAVE/${KEY}.json"
        log "Saved: $SAVE/${KEY}.json ($(du -h "$SAVE/${KEY}.json" | cut -f1))"
    else
        log "ERROR: No result file for $MODEL!"
    fi
done

log "=== ALL DONE ==="
ls -lh $SAVE/
