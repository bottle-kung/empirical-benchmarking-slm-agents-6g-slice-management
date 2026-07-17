#!/bin/bash
# Phase 2+3 Re-run — lab-slm-01
# Models: 0.5B(Q2,Q4,Q8), 1.5B(Q2,Q4,Q8), 3B(Q2,Q4,Q8) = 9 runs
# Output: /tmp/resource_results/
set -e
BENCH=/tmp/benchmark_runner_v2.py
SAVE=/tmp/resource_results
GGUF=/tmp/gguf_models
mkdir -p $SAVE $GGUF

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Ensure default models available
log "Pulling default models..."
ollama pull qwen2.5:0.5b 2>&1 | tail -1
ollama pull qwen2.5:1.5b 2>&1 | tail -1
ollama pull qwen2.5:3b 2>&1 | tail -1

run_one() {
    local TAG=$1 PHASE=$2 SIZE=$3 QLEVEL=$4
    log "=== $TAG ($SIZE $QLEVEL) ==="
    rm -rf /tmp/benchmark_results && mkdir -p /tmp/benchmark_results
    python3 $BENCH --model $TAG --family Qwen --phase $PHASE
    cp /tmp/benchmark_results/benchmark_*.json $SAVE/${SIZE}_${QLEVEL}.json 2>/dev/null
    log "Done: $TAG"
}

# ── Q4_K_M (default) ──
run_one "qwen2.5:0.5b" 2 "0.5B" "Q4_K_M"
run_one "qwen2.5:1.5b" 2 "1.5B" "Q4_K_M"
run_one "qwen2.5:3b"   2 "3B"   "Q4_K_M"

# ── Q2_K custom (if built) or create ──
for SIZE in "0.5B" "1.5B" "3B"; do
    TAG="qwen2.5-${SIZE,,}-q2k"
    TAG=${TAG//./_}
    if ! ollama list | grep -q "$TAG"; then
        log "Creating $TAG..."
        # Download GGUF if needed
        REPO_SIZE="${SIZE%-*}"  # remove trailing B? Actually keep as is
        [ ! -f $GGUF/qwen2.5-${SIZE}-Q2_K.gguf ] && \
            wget -q --show-progress -O $GGUF/qwen2.5-${SIZE}-Q2_K.gguf \
            "https://huggingface.co/bartowski/Qwen2.5-${SIZE}-Instruct-GGUF/resolve/main/Qwen2.5-${SIZE}-Instruct-Q2_K.gguf"
        cat > $GGUF/MF_${SIZE}_q2k << MFEOF
FROM $GGUF/qwen2.5-${SIZE}-Q2_K.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
MFEOF
        ollama rm $TAG 2>/dev/null || true
        ollama create $TAG -f $GGUF/MF_${SIZE}_q2k
    fi
    run_one "$TAG" 3 "${SIZE}" "Q2_K"
done

# ── Q8_0 custom ──
for SIZE in "0.5B" "1.5B" "3B"; do
    TAG="qwen2.5-${SIZE,,}-q8_0"
    TAG=${TAG//./_}
    if ! ollama list | grep -q "$TAG"; then
        log "Creating $TAG..."
        [ ! -f $GGUF/qwen2.5-${SIZE}-Q8_0.gguf ] && \
            wget -q --show-progress -O $GGUF/qwen2.5-${SIZE}-Q8_0.gguf \
            "https://huggingface.co/bartowski/Qwen2.5-${SIZE}-Instruct-GGUF/resolve/main/Qwen2.5-${SIZE}-Instruct-Q8_0.gguf"
        cat > $GGUF/MF_${SIZE}_q8_0 << MFEOF
FROM $GGUF/qwen2.5-${SIZE}-Q8_0.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
MFEOF
        ollama rm $TAG 2>/dev/null || true
        ollama create $TAG -f $GGUF/MF_${SIZE}_q8_0
    fi
    run_one "$TAG" 3 "${SIZE}" "Q8_0"
done

log "=== ALL DONE ==="
ls -lh $SAVE/
