#!/usr/bin/env python3
"""
SLM Agent Benchmark v2 — VM-local runner with RESOURCE MONITORING (+CI)
=====================================================================
Runs on VM, calling Ollama on localhost:11434.
Collects: CPU%, Memory MB, Energy (J) per intent via psutil sampling.
Computes 95% CI (mean, SD, CI_lower, CI_upper) for all numeric metrics.

Usage (run on VM):
  python3 /tmp/benchmark_runner.py --model qwen2.5:7b --family Qwen --phase 1
"""
import json, re, time, argparse, sys, os, threading, math
from pathlib import Path
from collections import defaultdict
from typing import Optional
import requests
import psutil


def t_value_95(n):
    """Approximate t-value for 95% CI with n-1 degrees of freedom."""
    df = n - 1
    if df < 1: return float('inf')
    if df > 100: return 1.984
    if df > 50: return 2.009
    if df > 30: return 2.042
    return 2.0 + 2.0 / math.sqrt(df)


def ci95(values):
    """Compute (mean, sd, ci_lower, ci_upper) for 95% confidence interval."""
    n = len(values)
    if n < 2: return (values[0] if n else 0, 0, 0, 0)
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    t = t_value_95(n)
    margin = t * sd / math.sqrt(n)
    return (round(mean, 2), round(sd, 2), round(mean - margin, 2), round(mean + margin, 2))


# ═══════════════════════════════════════════════════════════════
OLLAMA_URL = "http://localhost:11434/api/generate"
DATASET_PATH = "/tmp/intents.jsonl"
OUTPUT_DIR = "/tmp/benchmark_results"
TDP_WATTS = 135  # estimated TDP for 16-core VM CPU
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    if not text: return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("\n")
        cleaned = "\n".join(parts[1:-1]) if len(parts) > 2 else cleaned[3:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict): return result
    except json.JSONDecodeError: pass
    for m in re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned):
        try:
            result = json.loads(m)
            if isinstance(result, dict): return result
        except json.JSONDecodeError: pass
    return None


def exact_score(parsed: dict, truth: dict) -> dict:
    if not parsed:
        return {"format_valid": False, "action_correct": False, "match_pct": 0, "exact_keys": 0, "total_keys": len(truth)}
    action_ok = str(parsed.get("action", "")).lower() == truth["action"].lower()
    exact = sum(1 for k in truth if k in parsed and str(parsed[k]).lower() == str(truth[k]).lower())
    total = len(truth)
    match_pct = round(exact / total * 100, 1) if total > 0 else 0
    return {"format_valid": True, "action_correct": action_ok, "match_pct": match_pct, "exact_keys": exact, "total_keys": total}


def find_ollama_pids():
    """Find all llama-server process PIDs (Ollama inference workers)."""
    pids = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'llama' in proc.info['name'].lower():
                pids.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    if not pids:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmd = ' '.join(proc.info['cmdline'] or [])
                if 'ollama' in cmd.lower() and 'serve' in cmd.lower():
                    pids.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    return pids


class ResourceSampler:
    """Samples system-wide CPU% and process Memory in a background thread."""
    def __init__(self, pids, interval=0.5):
        self.pids = pids
        self.interval = interval
        self.samples = []  # list of (cpu_pct, mem_rss_sum_mb)
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.samples = []
        # Prime psutil cpu_percent (first call always returns 0)
        psutil.cpu_percent(interval=None)
        for pid in self.pids:
            try:
                psutil.Process(pid).cpu_percent(interval=None)
            except: pass
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        return self.summarize()
    
    def _sample_loop(self):
        while self.running:
            # System-wide CPU% (all cores)
            cpu_sys = psutil.cpu_percent(interval=0)
            # Per-process memory
            mem_total = 0
            count = 0
            for pid in self.pids:
                try:
                    p = psutil.Process(pid)
                    mem_total += p.memory_info().rss
                    count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied): pass
            if count > 0:
                self.samples.append((cpu_sys, mem_total / (1024*1024)))
            time.sleep(self.interval)
    
    def summarize(self):
        if not self.samples:
            return {"avg_cpu_pct": 0, "peak_memory_mb": 0, "avg_memory_mb": 0}
        cpus = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        return {
            "avg_cpu_pct": round(sum(cpus) / len(cpus), 1),
            "peak_memory_mb": round(max(mems), 1),
            "avg_memory_mb": round(sum(mems) / len(mems), 1),
        }


def ollama_query(model: str, prompt: str, retries: int = 3) -> dict:
    payload = {
        "model": model, "prompt": prompt, "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024}
    }
    pids = find_ollama_pids()
    
    for attempt in range(retries):
        try:
            sampler = ResourceSampler(pids)
            sampler.start()
            
            t0 = time.time()
            resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
            t_inf = (time.time() - t0) * 1000
            
            res_info = sampler.stop()
            
            data = resp.json()
            response_text = data.get("response", "")
            thinking = data.get("thinking", "")
            if thinking and not response_text.strip():
                response_text = thinking
            
            eval_count = data.get("eval_count", 0)
            eval_duration = data.get("eval_duration", 1)
            
            # Energy estimate: avg_cpu_fraction * TDP * time
            cpu_frac = res_info["avg_cpu_pct"] / 100.0 if res_info["avg_cpu_pct"] > 0 else 0.95
            energy_j = round(cpu_frac * TDP_WATTS * (t_inf / 1000), 1)
            
            return {
                "response": response_text,
                "t_inference_ms": round(t_inf, 1),
                "eval_count": eval_count,
                "tokens_per_sec": round(eval_count / (eval_duration / 1e9), 1) if eval_duration else 0,
                "cpu_pct": res_info["avg_cpu_pct"],
                "peak_memory_mb": res_info["peak_memory_mb"],
                "avg_memory_mb": res_info["avg_memory_mb"],
                "energy_j": energy_j,
                "error": None,
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 ** attempt)
            else:
                return {"response": "", "t_inference_ms": 0, "cpu_pct": 0, "peak_memory_mb": 0, "avg_memory_mb": 0, "energy_j": 0, "error": str(e)}
    return {"response": "", "t_inference_ms": 0, "cpu_pct": 0, "peak_memory_mb": 0, "avg_memory_mb": 0, "energy_j": 0, "error": "max retries"}


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
    print(f"   Dataset: {len(intents)} intents | Monitoring: CPU/Mem/Energy", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Warmup
    print("🔥 Warmup (2 intents)...", flush=True)
    for intent in intents[:2]:
        ollama_query(model_name, intent["input_text"])
        sys.stdout.write("."); sys.stdout.flush()
    print(" Done\n", flush=True)

    results = []
    total_failures = 0
    total_tokens = 0
    total_time_ms = 0
    total_energy_j = 0.0
    t_start = time.time()

    print(f"🎯 Running {len(intents)} intents...\n", flush=True)
    for i, intent in enumerate(intents):
        result = ollama_query(model_name, intent["input_text"])
        
        if result.get("error"):
            total_failures += 1

        parsed = parse_json(result.get("response", "")) if not result.get("error") else None
        score = exact_score(parsed, intent["ground_truth"])

        record = {
            "model": model_name, "family": family,
            "intent_id": intent["intent_id"],
            "domain": intent["domain"], "complexity": intent["complexity"],
            "input_text": intent["input_text"],
            "ground_truth": intent["ground_truth"],
            "response": result.get("response", ""),
            "parsed": parsed,
            "duration_s": round(result["t_inference_ms"] / 1000, 2),
            "tokens": result.get("eval_count", 0),
            "tokens_per_sec": result.get("tokens_per_sec", 0),
            "cpu_pct": result.get("cpu_pct", 0),
            "peak_memory_mb": result.get("peak_memory_mb", 0),
            "avg_memory_mb": result.get("avg_memory_mb", 0),
            "energy_j": result.get("energy_j", 0),
            "format_valid": score["format_valid"],
            "action_correct": score["action_correct"],
            "match_pct": score["match_pct"],
            "exact_keys": score["exact_keys"], "total_keys": score["total_keys"],
            "parsed_keys": list(parsed.keys()) if parsed else [],
            "error": result.get("error"),
        }
        results.append(record)
        total_tokens += result.get("eval_count", 0)
        total_time_ms += result["t_inference_ms"]
        total_energy_j += result.get("energy_j", 0)

        if (i + 1) % 20 == 0 or i == len(intents) - 1:
            pct = (i+1)/len(intents)*100
            elapsed = time.time() - t_start
            rate = (i+1)/elapsed if elapsed > 0 else 0
            eta = (len(intents)-i-1)/rate if rate > 0 else 0
            valid_count = sum(1 for r in results if r["format_valid"])
            print(f"   [{i+1:3d}/{len(intents)}] {pct:5.1f}% | Valid: {valid_count}/{i+1} | "
                  f"Rate: {rate:.1f} int/s | ETA: {eta:.0f}s | Err: {total_failures}", flush=True)

    total_elapsed_s = time.time() - t_start
    n = len(results)
    
    # Basic counts
    format_valid_count = sum(1 for r in results if r["format_valid"])
    action_correct_count = sum(1 for r in results if r["action_correct"])
    
    # CI for continuous metrics across all intents
    match_pcts = [r["match_pct"] for r in results]
    durations = [r["duration_s"] for r in results]
    energies = [r["energy_j"] for r in results]
    cpu_pcts = [r["cpu_pct"] for r in results if r["cpu_pct"] > 0]
    mem_avgs = [r["avg_memory_mb"] for r in results if r["avg_memory_mb"] > 0]
    mem_peaks = [r["peak_memory_mb"] for r in results if r["peak_memory_mb"] > 0]
    tps_vals = [r["tokens_per_sec"] for r in results]
    
    ci = {
        "match_pct": ci95(match_pcts),
        "duration_s": ci95(durations),
        "energy_j": ci95(energies),
        "cpu_pct": ci95(cpu_pcts) if cpu_pcts else (0, 0, 0, 0),
        "avg_memory_mb": ci95(mem_avgs) if mem_avgs else (0, 0, 0, 0),
        "peak_memory_mb": ci95(mem_peaks) if mem_peaks else (0, 0, 0, 0),
        "tokens_per_sec": ci95(tps_vals),
    }

    # Domain breakdown with CI
    by_domain_data = defaultdict(lambda: {
        "count": 0, "format_valid": 0, "action_correct": 0,
        "match_list": [], "energy_list": [], "cpu_list": [], "time_list": [],
    })
    for r in results:
        d = by_domain_data[r["domain"]]
        d["count"] += 1
        d["format_valid"] += 1 if r["format_valid"] else 0
        d["action_correct"] += 1 if r["action_correct"] else 0
        d["match_list"].append(r["match_pct"])
        d["energy_list"].append(r.get("energy_j", 0))
        d["cpu_list"].append(r.get("cpu_pct", 0))
        d["time_list"].append(r["duration_s"])
    
    by_domain = {}
    for domain, s in sorted(by_domain_data.items()):
        by_domain[domain] = {
            "count": s["count"],
            "format_valid": s["format_valid"],
            "action_correct": s["action_correct"],
            "avg_match_pct": round(sum(s["match_list"])/s["count"], 1),
            "ci_match_pct": ci95(s["match_list"]),
            "avg_energy_j": round(sum(s["energy_list"])/s["count"], 1),
            "ci_energy_j": ci95(s["energy_list"]),
            "avg_cpu_pct": round(sum(s["cpu_list"])/s["count"], 1),
            "ci_cpu_pct": ci95(s["cpu_list"]) if any(x > 0 for x in s["cpu_list"]) else None,
            "avg_time_s": round(sum(s["time_list"])/s["count"], 2),
            "ci_time_s": ci95(s["time_list"]),
        }

    summary = {
        "model": model_name, "family": family,
        "vm_hostname": hostname, "vm_cpu_cores": int(cpu_info), "vm_ram": mem_info,
        "phase": phase, "total": n,
        "format_valid": format_valid_count, "format_valid_pct": round(format_valid_count/n*100, 1),
        "action_correct": action_correct_count, "action_correct_pct": round(action_correct_count/n*100, 1),
        "avg_match_pct": round(sum(match_pcts)/n, 1),
        "avg_time_s": round(total_time_ms/n/1000, 2),
        "total_time_s": round(total_elapsed_s, 1),
        "total_tokens": total_tokens, "avg_tokens": round(total_tokens/n, 1),
        "avg_tokens_per_sec": round(total_tokens/total_elapsed_s, 1) if total_elapsed_s > 0 else 0,
        "failures": total_failures,
        # Resource metrics
        "avg_cpu_pct": ci["cpu_pct"][0] if cpu_pcts else 95.0,
        "avg_peak_memory_mb": ci["peak_memory_mb"][0],
        "avg_runtime_memory_mb": ci["avg_memory_mb"][0],
        "total_energy_j": round(total_energy_j, 1),
        "avg_energy_j_per_intent": round(total_energy_j/n, 1) if n else 0,
        "energy_efficiency_j_per_token": round(total_energy_j/total_tokens, 3) if total_tokens else 0,
        "tdp_watts": TDP_WATTS,
        # CIs (mean, sd, ci_lower, ci_upper)
        "ci_95": ci,
        "by_domain": by_domain,
    }

    print(f"\n{'='*60}", flush=True)
    print(f"📊 {model_name} Summary:", flush=True)
    print(f"   Format: {summary['format_valid_pct']:.1f}% | Action: {summary['action_correct_pct']:.1f}% | Match: {summary['avg_match_pct']:.1f}%", flush=True)
    print(f"   Time: {summary['avg_time_s']}s/intent | Tok/s: {summary['avg_tokens_per_sec']:.1f}", flush=True)
    print(f"   CPU: {summary['avg_cpu_pct']:.1f}% | Mem: {summary['avg_runtime_memory_mb']:.0f} MB | Energy: {int(total_energy_j)} J", flush=True)
    print(f"   CI(Match): [{ci['match_pct'][2]:.1f}, {ci['match_pct'][3]:.1f}] | CI(Time): [{ci['duration_s'][2]:.2f}, {ci['duration_s'][3]:.2f}]", flush=True)
    print(f"   CI(CPU): [{ci['cpu_pct'][2]:.1f}, {ci['cpu_pct'][3]:.1f}]% | CI(Energy): [{ci['energy_j'][2]:.1f}, {ci['energy_j'][3]:.1f}]J", flush=True)
    print(f"   Total: {total_elapsed_s:.0f}s | {total_tokens} tokens", flush=True)
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
