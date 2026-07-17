#!/usr/bin/env python3
"""
Phase 3 Runner — Pull Q2_K and Q8_0 GGUF files, create Ollama models, run benchmarks.
Runs on VM: lab-slm-01 (172.16.1.213)
"""
import subprocess, json, os, sys, time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
MODELS = [
    {"size": "0.5B", "repo": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
     "gguf_q2": "Qwen2.5-0.5B-Instruct-Q2_K.gguf",
     "gguf_q8": "Qwen2.5-0.5B-Instruct-Q8_0.gguf"},
    {"size": "3B", "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
     "gguf_q2": "Qwen2.5-3B-Instruct-Q2_K.gguf",
     "gguf_q8": "Qwen2.5-3B-Instruct-Q8_0.gguf"},
]

RESULTS_DIR = Path("/tmp/benchmark_results")
GGUF_DIR = Path("/tmp/gguf_models")
BENCHMARK_SCRIPT = Path("/tmp/benchmark_runner.py")

HOSTNAME = subprocess.check_output("hostname").decode().strip()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)

def run(cmd, timeout=600):
    """Run shell command, return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"  [WARN] exit={result.returncode}: {result.stderr[:200]}")
    return result.stdout.strip()

def download_gguf(repo, filename):
    """Download a GGUF file from HuggingFace."""
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    dest = GGUF_DIR / filename
    if dest.exists():
        print(f"  Already cached: {dest.name} ({dest.stat().st_size/1024/1024:.0f} MB)")
        return dest
    print(f"  Downloading {filename}...")
    run(f"wget -q --show-progress -O {dest} {url}", timeout=600)
    if dest.exists():
        print(f"  Done: {dest.stat().st_size/1024/1024:.0f} MB")
    return dest

def create_ollama_model(gguf_path, model_name, q_level):
    """Create Ollama model from GGUF file."""
    # Check if already exists
    existing = run(f"ollama list 2>/dev/null | grep '{model_name}'")
    if model_name in existing:
        print(f"  Model {model_name} already exists, deleting and recreating...")
        run(f"ollama rm {model_name}", timeout=30)
    
    # Create Modelfile
    template = '{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n'
    modelfile_content = f"FROM {gguf_path}\nTEMPLATE \"\"\"{template}\"\"\"\n"
    modelfile_path = GGUF_DIR / f"Modelfile.{model_name.replace(':', '_')}"
    modelfile_path.write_text(modelfile_content)
    
    print(f"  Creating Ollama model {model_name}...")
    out = run(f"ollama create {model_name} -f {modelfile_path}", timeout=120)
    print(f"  {out[:200]}")
    
    # Verify
    show = run(f"ollama show {model_name} 2>/dev/null")
    print(f"  Verified: {show[:200]}")

# ── Main Flow ─────────────────────────────────────────────────
print("="*60)
print(f"Phase 3 Runner — {HOSTNAME}")
print("="*60)

model_tags = []

for m in MODELS:
    size = m["size"]
    print(f"\n{'─'*60}")
    print(f"Processing Qwen2.5 {size}")
    print(f"{'─'*60}")
    
    for q_level, gguf_fn in [("Q2_K", m["gguf_q2"]), ("Q8_0", m["gguf_q8"])]:
        print(f"\n  [{q_level}] Start...")
        model_tag = f"qwen2.5-{size.lower().replace('.','_')}-{q_level.lower()}"
        
        # Step 1: Download GGUF
        gguf_path = download_gguf(m["repo"], gguf_fn)
        
        # Step 2: Create Ollama model
        create_ollama_model(gguf_path, model_tag, q_level)
        
        # Step 3: Run benchmark
        print(f"\n  [{q_level}] Running benchmark...")
        model_tags.append(model_tag)
        
        run(f"python3 {BENCHMARK_SCRIPT} --model {model_tag} --family Qwen --phase 3 --q-level {q_level}", timeout=36000)
        
        print(f"  [{q_level}] Benchmark complete!")

# ── Summary ────────────────────────────────────────────────────
print("\n" + "="*60)
print("Phase 3 Complete!")
print("="*60)
for tag in model_tags:
    result_file = RESULTS_DIR / f"benchmark_{tag}.json"
    if result_file.exists():
        size_kb = result_file.stat().st_size / 1024
        print(f"  ✓ {result_file.name} ({size_kb:.0f} KB)")
    else:
        print(f"  ✗ {result_file.name} NOT FOUND")
print(f"\nResults in: {RESULTS_DIR}")
print("DONE")
