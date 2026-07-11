#!/usr/bin/env python3
"""
DeepSeek-V4 Pro Baseline — Upper Bound for SLM Comparison
==========================================================
Runs ALL 200 intents through DeepSeek API with the EXACT same
system prompt and schema used for Qwen/Llama evaluation.

Output: deepseek_baseline_results.json
"""

import os
import json, re, time, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE)

DATASET = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval/dataset/intents.jsonl")
OUTPUT = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval/results/deepseek_baseline_results.json")

# ── EXACT SAME System Prompt ─────────────────────────────────
SYSTEM_PROMPT = """You are a 6G Core Network Slice Manager.
Output ONLY valid JSON. Use these EXACT field names:
action, slice_type, bandwidth_gbps, latency_ms, device_density, 
priority, target, replicas, strategy, step_percent, duration_hours, preempt, slice_id
No markdown, no code fences, no explanation."""


# ── JSON Parser ──────────────────────────────────────────────
def parse_json(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("\n")
        if len(parts) > 2:
            cleaned = "\n".join(parts[1:-1])
        else:
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except:
        for m in re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned):
            try: return json.loads(m)
            except: pass
    return None


# ── Exact Match Scorer ───────────────────────────────────────
def exact_score(parsed: dict, truth: dict) -> dict:
    if not parsed:
        return {"format_valid": False, "action_correct": False, "match_pct": 0, "exact_keys": 0}

    action_ok = str(parsed.get("action", "")).lower() == truth["action"].lower()
    exact = sum(1 for k in truth if k in parsed and str(parsed[k]).lower() == str(truth[k]).lower())
    semantic_bonus = 0
    if "slice_type" in truth and "slice_type" in parsed:
        if truth["slice_type"].lower() == str(parsed["slice_type"]).lower():
            semantic_bonus = 1
    total = len(truth) + 1
    match_pct = min(100, (exact + semantic_bonus) / total * 100)

    return {
        "format_valid": True,
        "action_correct": action_ok,
        "match_pct": round(match_pct, 1),
        "exact_keys": exact,
        "total_keys": len(truth),
    }


# ── DeepSeek Call ────────────────────────────────────────────
def call_deepseek(prompt: str, retries=3) -> dict:
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            t_inference = (time.time() - t0) * 1000  # ms
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            return {
                "response": content,
                "t_inference_ms": round(t_inference, 1),
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 ** attempt)
            else:
                return {"response": "", "t_inference_ms": 0, "error": str(e)}
    return {"response": "", "t_inference_ms": 0, "error": "max retries"}


# ── MAIN ─────────────────────────────────────────────────────
def main():
    # Load dataset
    intents = []
    with open(DATASET, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                intents.append(json.loads(line))

    total = len(intents)
    print(f"📦 {total} intents loaded")
    print(f"🤖 Model: {DEEPSEEK_MODEL}")
    print(f"📅 Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    results = []
    ok = 0
    fail = 0
    total_time = 0.0

    for i, intent in enumerate(intents):
        iid = intent.get("intent_id", f"INT-{i}")
        prompt = intent.get("input_text", "")
        truth = intent.get("ground_truth", {})
        domain = intent.get("domain", "Unknown")
        complexity = intent.get("complexity", "Unknown")

        # Progress
        if i % 20 == 0 or i == total - 1:
            pct = (i + 1) / total * 100
            elapsed = total_time / 60 if total_time > 0 else 0
            eta = (elapsed / max(i, 1)) * (total - i) if i > 0 else 0
            print(f"  [{i+1:>3}/{total} {pct:5.1f}%] {iid} | {ok}✅ {fail}❌ | "
                  f"elapsed: {elapsed:.1f}m | ETA: {eta:.1f}m", end="\r" if i < total-1 else "\n")

        # Call DeepSeek
        resp = call_deepseek(prompt)
        total_time += resp.get("t_inference_ms", 0) / 1000

        parsed = parse_json(resp.get("response", ""))
        score = exact_score(parsed, truth)

        if score["format_valid"]:
            ok += 1
        else:
            fail += 1

        results.append({
            "model": DEEPSEEK_MODEL,
            "intent_id": iid,
            "domain": domain,
            "complexity": complexity,
            "input_text": prompt,
            "ground_truth": truth,
            "response": resp.get("response", ""),
            "parsed": parsed,
            "t_inference_ms": resp.get("t_inference_ms", 0),
            "prompt_tokens": resp.get("prompt_tokens", 0),
            "completion_tokens": resp.get("completion_tokens", 0),
            **score,
        })

        # Rate limit safety
        if i < total - 1:
            time.sleep(0.3)

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    valid = sum(1 for r in results if r["format_valid"])
    action_ok = sum(1 for r in results if r["action_correct"])
    avg_match = sum(r["match_pct"] for r in results) / total
    avg_inf = sum(r["t_inference_ms"] for r in results) / total

    by_domain = defaultdict(lambda: {"count": 0, "match_sum": 0, "action_ok": 0, "format_ok": 0, "time_sum": 0})
    for r in results:
        d = r["domain"]
        by_domain[d]["count"] += 1
        by_domain[d]["match_sum"] += r["match_pct"]
        by_domain[d]["action_ok"] += 1 if r["action_correct"] else 0
        by_domain[d]["format_ok"] += 1 if r["format_valid"] else 0
        by_domain[d]["time_sum"] += r["t_inference_ms"]

    by_complexity = defaultdict(lambda: {"count": 0, "match_sum": 0, "action_ok": 0, "format_ok": 0})
    for r in results:
        c = r["complexity"]
        by_complexity[c]["count"] += 1
        by_complexity[c]["match_sum"] += r["match_pct"]
        by_complexity[c]["action_ok"] += 1 if r["action_correct"] else 0
        by_complexity[c]["format_ok"] += 1 if r["format_valid"] else 0

    print(f"\n{'='*60}")
    print(f"  DEEPSEEK BASELINE — Summary")
    print(f"{'='*60}")
    print(f"  Format Valid:    {valid}/{total} ({valid/total*100:.1f}%)")
    print(f"  Action Correct:  {action_ok}/{total} ({action_ok/total*100:.1f}%)")
    print(f"  Avg Exact Match: {avg_match:.1f}%")
    print(f"  Avg T_inference: {avg_inf:.0f} ms")
    print(f"  Total Time:      {total_time/60:.1f} min")
    print(f"\n  By Domain:")
    for d, s in sorted(by_domain.items()):
        print(f"    {d:<22} Format: {s['format_ok']}/{s['count']} | Action: {s['action_ok']}/{s['count']} | "
              f"Match: {s['match_sum']/s['count']:.1f}% | Time: {s['time_sum']/s['count']:.0f}ms")
    print(f"\n  By Complexity:")
    for c, s in sorted(by_complexity.items()):
        print(f"    {c:<12} Format: {s['format_ok']}/{s['count']} | Action: {s['action_ok']}/{s['count']} | "
              f"Match: {s['match_sum']/s['count']:.1f}%")

    # Save
    final = {
        "model": DEEPSEEK_MODEL,
        "total_intents": total,
        "summary": {
            "format_accuracy": round(valid / total * 100, 1),
            "action_accuracy": round(action_ok / total * 100, 1),
            "avg_exact_match": round(avg_match, 1),
            "avg_inference_ms": round(avg_inf, 1),
            "total_time_s": round(total_time, 1),
            "by_domain": {d: {
                "count": s["count"],
                "format_accuracy": round(s["format_ok"] / s["count"] * 100, 1),
                "action_accuracy": round(s["action_ok"] / s["count"] * 100, 1),
                "avg_match_pct": round(s["match_sum"] / s["count"], 1),
                "avg_time_ms": round(s["time_sum"] / s["count"], 1),
            } for d, s in by_domain.items()},
            "by_complexity": {c: {
                "count": s["count"],
                "format_accuracy": round(s["format_ok"] / s["count"] * 100, 1),
                "action_accuracy": round(s["action_ok"] / s["count"] * 100, 1),
                "avg_match_pct": round(s["match_sum"] / s["count"], 1),
            } for c, s in by_complexity.items()},
        },
        "results": results,
    }

    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved: {OUTPUT}")
    print(f"📅 Finished: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
