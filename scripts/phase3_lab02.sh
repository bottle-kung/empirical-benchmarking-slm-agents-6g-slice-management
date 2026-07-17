#!/bin/bash
# Phase 3 — 7B Q2_K + Q8_0 on lab-slm-02
set -e
BENCH=/tmp/benchmark_runner.py
SAVE=/tmp/phase3b_results
GGUF=/tmp/gguf_models
mkdir -p $SAVE $GGUF

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 7B Q2_K ──
log "Downloading 7B Q2_K..."
wget -q --show-progress -O $GGUF/qwen2.5-7b-Q2_K.gguf \
  'https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q2_K.gguf'

cat > $GGUF/Modelfile.7b_q2k << 'MFEOF'
FROM /tmp/gguf_models/qwen2.5-7b-Q2_K.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
MFEOF
ollama rm qwen2.5-7b-q2k 2>/dev/null || true
ollama create qwen2.5-7b-q2k -f $GGUF/Modelfile.7b_q2k

log "Running 7B Q2_K..."
rm -rf /tmp/benchmark_results && mkdir -p /tmp/benchmark_results
python3 $BENCH --model qwen2.5-7b-q2k --family Qwen --phase 3
cp /tmp/benchmark_results/benchmark_*.json $SAVE/7B_q2k.json 2>/dev/null
log "7B Q2_K done"

# ── 7B Q8_0 ──
log "Downloading 7B Q8_0..."
wget -q --show-progress -O $GGUF/qwen2.5-7b-Q8_0.gguf \
  'https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q8_0.gguf'

cat > $GGUF/Modelfile.7b_q8_0 << 'MFEOF'
FROM /tmp/gguf_models/qwen2.5-7b-Q8_0.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
MFEOF
ollama rm qwen2.5-7b-q8_0 2>/dev/null || true
ollama create qwen2.5-7b-q8_0 -f $GGUF/Modelfile.7b_q8_0

log "Running 7B Q8_0..."
rm -rf /tmp/benchmark_results && mkdir -p /tmp/benchmark_results
python3 $BENCH --model qwen2.5-7b-q8_0 --family Qwen --phase 3
cp /tmp/benchmark_results/benchmark_*.json $SAVE/7B_q8_0.json 2>/dev/null

log "=== ALL DONE ==="
ls -lh $SAVE/
