#!/bin/bash
# Phase 2 Re-run — lab-slm-02
# Models: 7B(Q2,Q4,Q8) = 3 runs
set -e
BENCH=/tmp/benchmark_runner_v2.py
SAVE=/tmp/resource_results
GGUF=/tmp/gguf_models
mkdir -p $SAVE $GGUF

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Pulling qwen2.5:7b..."
ollama pull qwen2.5:7b 2>&1 | tail -1

run_one() {
    local TAG=$1 SIZE=$2 QLEVEL=$3
    log "=== $TAG ($SIZE $QLEVEL) ==="
    rm -rf /tmp/benchmark_results && mkdir -p /tmp/benchmark_results
    python3 $BENCH --model $TAG --family Qwen --phase 3
    cp /tmp/benchmark_results/benchmark_*.json $SAVE/${SIZE}_${QLEVEL}.json 2>/dev/null
    log "Done: $TAG"
}

# Q4_K_M
run_one "qwen2.5:7b" "7B" "Q4_K_M"

# Q2_K
SIZE="7B"
TAG="qwen2.5-7b-q2k"
if ! ollama list | grep -q "$TAG"; then
    log "Creating $TAG..."
    [ ! -f $GGUF/qwen2.5-7B-Q2_K.gguf ] && \
        wget -q --show-progress -O $GGUF/qwen2.5-7B-Q2_K.gguf \
        "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q2_K.gguf"
    cat > $GGUF/MF_7B_q2k << MFEOF
FROM $GGUF/qwen2.5-7B-Q2_K.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
MFEOF
    ollama rm $TAG 2>/dev/null || true
    ollama create $TAG -f $GGUF/MF_7B_q2k
fi
run_one "$TAG" "7B" "Q2_K"

# Q8_0
TAG="qwen2.5-7b-q8_0"
if ! ollama list | grep -q "$TAG"; then
    log "Creating $TAG..."
    [ ! -f $GGUF/qwen2.5-7B-Q8_0.gguf ] && \
        wget -q --show-progress -O $GGUF/qwen2.5-7B-Q8_0.gguf \
        "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q8_0.gguf"
    cat > $GGUF/MF_7B_q8_0 << MFEOF
FROM $GGUF/qwen2.5-7B-Q8_0.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
MFEOF
    ollama rm $TAG 2>/dev/null || true
    ollama create $TAG -f $GGUF/MF_7B_q8_0
fi
run_one "$TAG" "7B" "Q8_0"

log "=== ALL DONE ==="
ls -lh $SAVE/
