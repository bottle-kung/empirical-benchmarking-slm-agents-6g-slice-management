#!/usr/bin/env python3
"""
SLM Agent Benchmark — VM-local runner
======================================
Runs directly on the VM, calling Ollama on localhost:11434.
Saves results as JSON to be collected later.

Usage (run on VM):
  python3 /tmp/benchmark_runner.py --model llama3.1:8b --family Llama --phase 1
"""

import json
import re
import time
import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional

import requests

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
OLLAMA_URL = "http://localhost:11434/api/generate"
DATASET_PATH = "/tmp/intents.jsonl"
OUTPUT_DIR = "/tmp/benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enriched System Prompt (same as v1 Synonym experiment)
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


def parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("\n")
        cleaned = "\n".join(parts[1:-1]) if len(parts) > 2 else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    # Strip DeepSeek-R1 <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    for m in re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned):
        try:
            result = json.loads(m)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return None


def exact_score(parsed: dict, truth: dict) -> dict:
    if not parsed:
        return {"format_valid": False, "action_correct": False, "match_pct": 0, "exact_keys": 0, "total_keys": len(truth)}
    action_ok = str(parsed.get("action", "")).lower() == truth["action"].lower()
    exact = sum(1 for k in truth if k in parsed and str(parsed[k]).lower() == str(truth[k]).lower())
    total = len(truth)
    match_pct = round(exact / total * 100, 1) if total > 0 else 0
    return {"format_valid": True, "action_correct": action_ok, "match_pct": match_pct, "exact_keys": exact, "total_keys": total}


def ollama_query(model: str, prompt: str, retries: int = 3) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024}
    }
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            t_inf = (time.time() - t0) * 1000
            data = resp.json()
            response_text = data.get("response", "")
            # DeepSeek-R1 puts everything in "thinking" field
            thinking = data.get("thinking", "")
            if thinking and not response_text.strip():
                response_text = thinking
            eval_count = data.get("eval_count", 0)
            eval_duration = data.get("eval_duration", 1)
            return {
                "response": response_text,
                "t_inference_ms": round(t_inf, 1),
                "eval_count": eval_count,
                "tokens_per_sec": round(eval_count / (eval_duration / 1e9), 1) if eval_duration else 0,
                "error": None,
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 ** attempt)
            else:
                return {"response": "", "t_inference_ms": 0, "error": str(e)}
    return {"response": "", "t_inference_ms": 0, "error": "max retries"}


def load_dataset():
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_benchmark(model_name: str, family: str, phase: int = 1) -> dict:
    intents = load_dataset()
    
    hostname = os.popen("hostname").read().strip()
    cpu_info = os.popen("nproc").read().strip()
    mem_info = os.popen("free -h | grep Mem | awk '{print $2}'").read().strip()
    
    print(f"\n{'='*60}", flush=True)
    print(f"🔬 Benchmark: {model_name} ({family}) — Phase {phase}", flush=True)
    print(f"   Host: {hostname} | CPU: {cpu_info} cores | RAM: {mem_info}", flush=True)
    print(f"   Dataset: {len(intents)} intents", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Warmup
    print("🔥 Warmup (2 intents)...", flush=True)
    for intent in intents[:2]:
        ollama_query(model_name, intent["input_text"])
        sys.stdout.write("."); sys.stdout.flush()
    print(" Done\n", flush=True)

    # Benchmark
    results = []
    total_failures = 0
    total_tokens = 0
    total_time_ms = 0
    t_start = time.time()

    print(f"🎯 Running {len(intents)} intents...\n", flush=True)

    for i, intent in enumerate(intents):
        result = ollama_query(model_name, intent["input_text"])
        
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
            "parsed_keys": list(parsed.keys()) if parsed else [],
            "error": result.get("error"),
        }
        results.append(record)
        total_tokens += result.get("eval_count", 0)
        total_time_ms += result["t_inference_ms"]

        if (i + 1) % 20 == 0 or i == len(intents) - 1:
            pct = (i + 1) / len(intents) * 100
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(intents) - i - 1) / rate if rate > 0 else 0
            valid_count = sum(1 for r in results if r["format_valid"])
            print(f"   [{i+1:3d}/{len(intents)}] {pct:5.1f}% | Valid: {valid_count}/{i+1} | "
                  f"Rate: {rate:.1f} int/s | ETA: {eta:.0f}s | Err: {total_failures}", flush=True)

    total_elapsed_s = time.time() - t_start

    # Summary
    format_valid_count = sum(1 for r in results if r["format_valid"])
    action_correct_count = sum(1 for r in results if r["action_correct"])
    avg_match_pct = round(sum(r["match_pct"] for r in results) / len(results), 1)
    avg_time_s = round(total_time_ms / len(results) / 1000, 2)
    avg_tokens = round(total_tokens / len(results), 1)

    by_domain = defaultdict(lambda: {"count": 0, "format_valid": 0, "action_correct": 0, "sum_match_pct": 0.0})
    for r in results:
        d = by_domain[r["domain"]]
        d["count"] += 1
        d["format_valid"] += 1 if r["format_valid"] else 0
        d["action_correct"] += 1 if r["action_correct"] else 0
        d["sum_match_pct"] += r["match_pct"]

    summary = {
        "model": model_name,
        "family": family,
        "vm_hostname": hostname,
        "vm_cpu_cores": int(cpu_info),
        "vm_ram": mem_info,
        "phase": phase,
        "total": len(results),
        "format_valid": format_valid_count,
        "format_valid_pct": round(format_valid_count / len(results) * 100, 1),
        "action_correct": action_correct_count,
        "action_correct_pct": round(action_correct_count / len(results) * 100, 1),
        "avg_match_pct": avg_match_pct,
        "avg_time_s": avg_time_s,
        "total_time_s": round(total_elapsed_s, 1),
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "avg_tokens_per_sec": round(total_tokens / total_elapsed_s, 1) if total_elapsed_s > 0 else 0,
        "failures": total_failures,
        "by_domain": {domain: {
            "count": s["count"],
            "format_valid": s["format_valid"],
            "action_correct": s["action_correct"],
            "avg_match_pct": round(s["sum_match_pct"] / s["count"], 1),
        } for domain, s in sorted(by_domain.items())},
    }

    print(f"\n{'='*60}", flush=True)
    print(f"📊 {model_name} Summary:", flush=True)
    print(f"   Format Valid:  {summary['format_valid_pct']:.1f}%", flush=True)
    print(f"   Action Correct: {summary['action_correct_pct']:.1f}%", flush=True)
    print(f"   Avg Match:     {avg_match_pct:.1f}%", flush=True)
    print(f"   Avg Time:      {avg_time_s}s/intent", flush=True)
    print(f"   Tokens/s:      {summary['avg_tokens_per_sec']:.1f}", flush=True)
    print(f"   Total Time:    {total_elapsed_s:.0f}s", flush=True)
    print(f"{'='*60}\n", flush=True)

    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--family", type=str, required=True)
    parser.add_argument("--phase", type=int, default=1)
    args = parser.parse_args()

    result = run_benchmark(args.model, args.family, args.phase)
    
    safe_name = args.model.replace(":", "_").replace(".", "_")
    output_path = os.path.join(OUTPUT_DIR, f"benchmark_{safe_name}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved: {output_path}", flush=True)
    print(f"✅ Done: {args.model}", flush=True)


if __name__ == "__main__":
    main()
