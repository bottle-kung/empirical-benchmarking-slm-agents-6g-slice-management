#!/usr/bin/env python3
"""
SLM Agent Evaluation Benchmark v2
=================================
Cross-family comparison of open-weight ~8B models for 6G slice management.
Features:
  - Ollama HTTP API inference
  - Enriched system prompt (same as v1 Synonym experiment)
  - Resource monitoring (CPU, RAM via Prometheus, Energy via RAPL)
  - Supports parallel execution on 2 VMs

Usage:
  # Phase 1: Test all 5 models @ ~8B
  python3 scripts/benchmark.py --phase 1

  # Phase 2: Size scaling on best model
  python3 scripts/benchmark.py --phase 2 --model qwen2.5

  # Test single model
  python3 scripts/benchmark.py --model llama3.1:8b --vm 172.16.1.213
"""

import json
import re
import time
import argparse
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
PROJECT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT / "dataset" / "intents.jsonl"
RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# SSH access
SSH_PASS = "cpeadmin"
SSH_OPTS = "-o StrictHostKeyChecking=no -o PasswordAuthentication=yes"
PROXMOX_HOST = "172.16.1.206"

# Prometheus
PROMETHEUS_URL = "http://localhost:9090"

# Model → VM mapping for Phase 1 parallel execution
PHASE1_MODELS = [
    {"model": "llama3.1:8b",   "vm": "172.16.1.213", "family": "Llama"},
    {"model": "deepseek-r1:8b", "vm": "172.16.1.214", "family": "DeepSeek"},
    {"model": "glm4:9b",       "vm": "172.16.1.214", "family": "GLM"},
    {"model": "qwen2.5:7b",    "vm": "172.16.1.213", "family": "Qwen"},
    {"model": "gemma2:9b",     "vm": "172.16.1.213", "family": "Gemma"},
]

# Phase 2: Qwen family size variants (if Qwen wins)
QWEN_SIZES = [
    {"model": "qwen2.5:0.5b", "params_b": 0.5},
    {"model": "qwen2.5:1.5b", "params_b": 1.5},
    {"model": "qwen2.5:3b",   "params_b": 3.0},
    {"model": "qwen2.5:7b",   "params_b": 7.0},
]

# Other family size variants for Phase 2 (if not Qwen)
SIZE_VARIANTS = {
    "Llama":    [{"model": "llama3.2:1b", "params_b": 1.0}, {"model": "llama3.2:3b", "params_b": 3.0}, {"model": "llama3.1:8b", "params_b": 8.0}],
    "DeepSeek": [{"model": "deepseek-r1:1.5b", "params_b": 1.5}, {"model": "deepseek-r1:7b", "params_b": 7.0}, {"model": "deepseek-r1:8b", "params_b": 8.0}],
    "Gemma":    [{"model": "gemma2:2b", "params_b": 2.0}, {"model": "gemma2:9b", "params_b": 9.0}],
}

# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPT (Enriched — same as v1 Synonym experiment)
# ═══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a 6G Core Network Slice Management Agent. 
Your job is to translate natural language intents into JSON commands for network orchestration.

Rules:
1. Respond ONLY with a valid JSON object — no markdown, no explanations, no code fences
2. The JSON MUST contain an "action" field
3. Use these slice types when appropriate: "eMBB" (enhanced Mobile Broadband), "URLLC" (Ultra-Reliable Low Latency), "mMTC" (massive Machine Type Communications)
4. For ambiguous requests, infer the best configuration based on context
5. Include all relevant parameters: latency_ms, bandwidth_gbps, device_density, priority, etc.

Example:
Input: "Create a URLLC slice with 1ms latency for robot control"
Output: {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 1}"""


# ═══════════════════════════════════════════════════════════════
# JSON PARSER
# ═══════════════════════════════════════════════════════════════
def parse_json(text: str) -> Optional[dict]:
    """Extract JSON from model output, handling markdown fences."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("\n")
        cleaned = "\n".join(parts[1:-1]) if len(parts) > 2 else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for m in re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            pass
    return None


def exact_score(parsed: dict, truth: dict) -> dict:
    """Compute exact match score against ground truth."""
    if not parsed:
        return {"format_valid": False, "action_correct": False, "match_pct": 0, "exact_keys": 0, "total_keys": len(truth)}
    action_ok = str(parsed.get("action", "")).lower() == truth["action"].lower()
    exact = sum(1 for k in truth if k in parsed and str(parsed[k]).lower() == str(truth[k]).lower())
    total = len(truth)
    match_pct = round(exact / total * 100, 1) if total > 0 else 0
    return {"format_valid": True, "action_correct": action_ok, "match_pct": match_pct, "exact_keys": exact, "total_keys": total}


# ═══════════════════════════════════════════════════════════════
# OLLAMA CLIENT
# ═══════════════════════════════════════════════════════════════
def ollama_query(vm_ip: str, model: str, prompt: str, retries: int = 3) -> dict:
    """Call Ollama API on a remote VM."""
    url = f"http://{vm_ip}:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 256}
    }
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = requests.post(url, json=payload, timeout=120)
            t_inf = (time.time() - t0) * 1000
            data = resp.json()
            response_text = data.get("response", "")
            return {
                "response": response_text,
                "t_inference_ms": round(t_inf, 1),
                "eval_count": data.get("eval_count", 0),
                "eval_duration_ns": data.get("eval_duration", 0),
                "tokens_per_sec": round(data.get("eval_count", 0) / (data.get("eval_duration", 1) / 1e9), 1) if data.get("eval_duration") else 0,
                "error": None,
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 ** attempt)
            else:
                return {"response": "", "t_inference_ms": 0, "error": str(e)}
    return {"response": "", "t_inference_ms": 0, "error": "max retries"}


# ═══════════════════════════════════════════════════════════════
# RESOURCE MONITORING
# ═══════════════════════════════════════════════════════════════
def prometheus_query(query: str) -> float:
    """Query Prometheus for a metric value."""
    try:
        resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=5)
        data = resp.json()
        if data["status"] == "success" and data["data"]["result"]:
            return float(data["data"]["result"][0]["value"][1])
    except Exception:
        pass
    return 0.0


def get_vm_metrics(vm_ip: str, vm_label: str) -> dict:
    """Collect current CPU and memory metrics from Prometheus for a VM."""
    cpu = prometheus_query(f'100 - (avg by(instance) (rate(node_cpu_seconds_total{{mode="idle",instance=~"{vm_ip}:9100"}}[30s])) * 100)')
    mem_used = prometheus_query(f'node_memory_MemTotal_bytes{{instance=~"{vm_ip}:9100"}} - node_memory_MemAvailable_bytes{{instance=~"{vm_ip}:9100"}}')
    mem_total = prometheus_query(f'node_memory_MemTotal_bytes{{instance=~"{vm_ip}:9100"}}')
    return {
        "cpu_percent": round(cpu, 1),
        "mem_used_gb": round(mem_used / 1e9, 2),
        "mem_total_gb": round(mem_total / 1e9, 2),
    }


def get_rapl_energy() -> dict:
    """Read RAPL energy from Proxmox host via SSH."""
    try:
        result = subprocess.run(
            ["sshpass", "-p", SSH_PASS, "ssh"] + SSH_OPTS.split() +
            [f"root@{PROXMOX_HOST}", "/usr/local/bin/rapl_snapshot.sh"],
            capture_output=True, text=True, timeout=10
        )
        return json.loads(result.stdout.strip())
    except Exception as e:
        return {"total_uj": 0, "error": str(e)}


def collect_resource_snapshot(vm_ip: str, vm_label: str) -> dict:
    """Take a snapshot of all resource metrics."""
    metrics = get_vm_metrics(vm_ip, vm_label)
    energy = get_rapl_energy()
    metrics["energy_uj"] = energy.get("total_uj", 0)
    metrics["energy_joules"] = round(energy.get("total_uj", 0) / 1e6, 2)
    return metrics


# ═══════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════
def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_benchmark(model_name: str, vm_ip: str, family: str, phase: int = 1) -> dict:
    """Run full benchmark for one model on one VM."""
    intents = load_dataset()
    print(f"\n{'='*60}")
    print(f"🔬 Benchmark: {model_name} ({family}) on {vm_ip}")
    print(f"   Phase: {phase} | Intents: {len(intents)}")
    print(f"{'='*60}")

    # Pre-inference resource snapshot
    print("\n📊 Collecting pre-benchmark resource metrics...")
    pre_metrics = collect_resource_snapshot(vm_ip, vm_ip)
    pre_energy_uj = pre_metrics.get("energy_uj", 0)
    print(f"   CPU: {pre_metrics['cpu_percent']}% | Mem: {pre_metrics['mem_used_gb']}GB | Energy: {pre_metrics['energy_joules']}J")

    # Warmup
    print("\n🔥 Warmup (2 intents, discarded)...")
    for intent in intents[:2]:
        ollama_query(vm_ip, model_name, intent["input_text"])
        sys.stdout.write("."); sys.stdout.flush()
    print()

    # Benchmark loop
    results = []
    total_failures = 0
    total_tokens = 0
    total_time_ms = 0

    print(f"\n🎯 Running {len(intents)} intents...")
    t_start = time.time()

    for i, intent in enumerate(intents):
        result = ollama_query(vm_ip, model_name, intent["input_text"])

        if result.get("error"):
            total_failures += 1

        parsed = parse_json(result.get("response", "")) if not result.get("error") else None
        score = exact_score(parsed, intent["ground_truth"])

        record = {
            "model": model_name,
            "family": family,
            "intent_id": intent["intent_id"],
            "domain": intent["domain"],
            "complexity": intent["complexity"],
            "input_text": intent["input_text"],
            "ground_truth": intent["ground_truth"],
            "response": result.get("response", ""),
            "parsed": parsed,
            "duration_s": round(result["t_inference_ms"] / 1000, 2),
            "tokens": result.get("eval_count", 0),
            "tokens_per_sec": result.get("tokens_per_sec", 0),
            "format_valid": score["format_valid"],
            "action_correct": score["action_correct"],
            "match_pct": score["match_pct"],
            "exact_keys": score["exact_keys"],
            "total_keys": score["total_keys"],
            "error": result.get("error"),
        }
        results.append(record)
        total_tokens += result.get("eval_count", 0)
        total_time_ms += result["t_inference_ms"]

        # Progress
        if (i + 1) % 20 == 0 or i == len(intents) - 1:
            pct = (i + 1) / len(intents) * 100
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(intents) - i - 1) / rate if rate > 0 else 0
            valid_count = sum(1 for r in results if r["format_valid"])
            print(f"   [{i+1:3d}/{len(intents)}] {pct:5.1f}% | Valid: {valid_count}/{i+1} | "
                  f"Rate: {rate:.1f} int/s | ETA: {eta:.0f}s | Failures: {total_failures}")

    total_elapsed_s = time.time() - t_start

    # Post-inference resource snapshot
    print("\n📊 Collecting post-benchmark resource metrics...")
    time.sleep(5)  # Let Prometheus catch up
    post_metrics = collect_resource_snapshot(vm_ip, vm_ip)
    post_energy_uj = post_metrics.get("energy_uj", 0)
    energy_used_j = round((post_energy_uj - pre_energy_uj) / 1e6, 2) if pre_energy_uj > 0 else 0

    # Summary
    format_valid_count = sum(1 for r in results if r["format_valid"])
    action_correct_count = sum(1 for r in results if r["action_correct"])
    avg_match_pct = round(sum(r["match_pct"] for r in results) / len(results), 1)
    avg_time_s = round(total_time_ms / len(results) / 1000, 2)
    avg_tokens = round(total_tokens / len(results), 1)

    # By domain
    by_domain = defaultdict(lambda: {"count": 0, "format_valid": 0, "action_correct": 0, "sum_match_pct": 0.0})
    for r in results:
        d = by_domain[r["domain"]]
        d["count"] += 1
        d["format_valid"] += 1 if r["format_valid"] else 0
        d["action_correct"] += 1 if r["action_correct"] else 0
        d["sum_match_pct"] += r["match_pct"]

    by_domain_summary = {}
    for domain, stats in sorted(by_domain.items()):
        by_domain_summary[domain] = {
            "count": stats["count"],
            "format_valid": stats["format_valid"],
            "action_correct": stats["action_correct"],
            "avg_match_pct": round(stats["sum_match_pct"] / stats["count"], 1),
        }

    summary = {
        "model": model_name,
        "family": family,
        "vm": vm_ip,
        "phase": phase,
        "total": len(results),
        "format_valid": format_valid_count,
        "format_valid_pct": round(format_valid_count / len(results) * 100, 1),
        "action_correct": action_correct_count,
        "action_correct_pct": round(action_correct_count / len(results) * 100, 1),
        "avg_match_pct": avg_match_pct,
        "avg_time_s": avg_time_s,
        "total_time_s": round(total_elapsed_s, 1),
        "avg_tokens": avg_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_sec": round(total_tokens / total_elapsed_s, 1) if total_elapsed_s > 0 else 0,
        "failures": total_failures,
        "by_domain": by_domain_summary,
        "resources": {
            "pre_cpu_pct": pre_metrics["cpu_percent"],
            "post_cpu_pct": post_metrics["cpu_percent"],
            "avg_cpu_pct": round((pre_metrics["cpu_percent"] + post_metrics["cpu_percent"]) / 2, 1),
            "pre_mem_gb": pre_metrics["mem_used_gb"],
            "post_mem_gb": post_metrics["mem_used_gb"],
            "peak_mem_gb": max(pre_metrics["mem_used_gb"], post_metrics["mem_used_gb"]),
            "energy_joules": energy_used_j,
            "energy_per_intent_j": round(energy_used_j / len(results), 3) if energy_used_j else 0,
        },
    }

    print(f"\n{'='*60}")
    print(f"📊 {model_name} Summary:")
    print(f"   Format Valid:  {summary['format_valid_pct']:.1f}% ({format_valid_count}/{len(results)})")
    print(f"   Action Correct: {summary['action_correct_pct']:.1f}% ({action_correct_count}/{len(results)})")
    print(f"   Avg Match:     {avg_match_pct:.1f}%")
    print(f"   Avg Time:      {avg_time_s}s/intent")
    print(f"   Avg Tokens/s:  {summary['avg_tokens_per_sec']:.1f}")
    print(f"   Total Time:    {total_elapsed_s:.0f}s")
    print(f"   Energy:        {energy_used_j}J ({summary['resources']['energy_per_intent_j']}J/intent)")
    print(f"   CPU Avg:       {summary['resources']['avg_cpu_pct']:.1f}%")
    print(f"   Peak Mem:      {summary['resources']['peak_mem_gb']:.1f}GB")
    print(f"   Failures:      {total_failures}")
    print(f"{'='*60}")

    return {"summary": summary, "results": results}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SLM Agent Benchmark v2")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1, help="Phase 1 or 2")
    parser.add_argument("--model", type=str, help="Single model to test")
    parser.add_argument("--vm", type=str, help="VM IP for single model test")
    parser.add_argument("--family", type=str, help="Model family for single test")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.model and args.vm:
        # Single model mode
        result = run_benchmark(args.model, args.vm, args.family or args.model, args.phase)
        output_path = RESULTS_DIR / f"benchmark_{args.model.replace(':','_')}_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved: {output_path}")
        return

    if args.phase == 1:
        models = PHASE1_MODELS
    else:
        # Phase 2: determine best model from Phase 1 results, then test sizes
        print("Phase 2 requires Phase 1 results first. Run --phase 1 first.")
        return

    # Run benchmarks
    all_results = []
    for i, m in enumerate(models):
        print(f"\n{'#'*60}")
        print(f"## PHASE {args.phase} — Model {i+1}/{len(models)}: {m['model']} ({m['family']}) on {m['vm']}")
        print(f"{'#'*60}")

        result = run_benchmark(m["model"], m["vm"], m["family"], args.phase)
        all_results.append(result)

        # Save intermediate
        inter_path = RESULTS_DIR / f"phase{args.phase}_intermediate_{timestamp}.json"
        with open(inter_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Final save
    final_path = RESULTS_DIR / f"phase{args.phase}_benchmark_{timestamp}.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  PHASE {args.phase} — FINAL RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<22} {'Family':<12} {'Format':>8} {'Action':>8} {'Exact':>8} {'Time':>8} {'Energy':>10} {'CPU':>8}")
    print(f"{'-'*22} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

    for r in all_results:
        s = r["summary"]
        resources = s.get("resources", {})
        print(f"{s['model']:<22} {s['family']:<12} {s['format_valid_pct']:>7.1f}% {s['action_correct_pct']:>7.1f}% "
              f"{s['avg_match_pct']:>7.1f}% {s['avg_time_s']:>7.1f}s "
              f"{resources.get('energy_joules',0):>9.0f}J {resources.get('avg_cpu_pct',0):>7.1f}%")

    print(f"\n💾 Saved: {final_path}")
    print("✅ Phase 1 complete!")


if __name__ == "__main__":
    main()
