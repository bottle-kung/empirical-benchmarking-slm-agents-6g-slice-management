#!/usr/bin/env python3
"""DeepSeek-V4 Pro — Enriched Prompt Baseline (same prompt as Synonym experiment)"""
import os
import json, re, time
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from openai import OpenAI

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE)

DATASET = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval/dataset/intents.jsonl")
OUTPUT = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval/results/deepseek_enriched_results.json")

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

def parse_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("\n")
        cleaned = "\n".join(parts[1:-1]) if len(parts) > 2 else cleaned[3:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try: return json.loads(cleaned)
    except:
        for m in re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned):
            try: return json.loads(m)
            except: pass
    return None

def score_response(parsed, truth):
    if not parsed:
        return {"format_valid": False, "action_correct": False, "match_pct": 0, "exact_keys": 0, "total_keys": len(truth)}
    action_ok = str(parsed.get("action", "")).lower() == truth["action"].lower()
    exact = sum(1 for k in truth if k in parsed and str(parsed[k]).lower() == str(truth[k]).lower())
    total = len(truth)
    match_pct = round(exact / total * 100, 1) if total > 0 else 0
    return {"format_valid": True, "action_correct": action_ok, "match_pct": match_pct, "exact_keys": exact, "total_keys": total}

def call_deepseek(prompt, retries=3):
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=256,
            )
            t_inf = (time.time() - t0) * 1000
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            return {
                "response": content, "t_inference_ms": round(t_inf, 1),
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            }
        except Exception as e:
            if attempt < retries - 1: time.sleep(3 ** attempt)
            else: return {"response": "", "t_inference_ms": 0, "error": str(e)}
    return {"response": "", "t_inference_ms": 0, "error": "max retries"}

def main():
    intents = []
    with open(DATASET, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: intents.append(json.loads(line))
    total = len(intents)
    print(f"Total: {total} intents | Model: {DEEPSEEK_MODEL} (Enriched Prompt)")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    
    results = []
    ok = fail = 0
    total_time = 0.0
    
    for i, intent in enumerate(intents):
        iid = intent.get("intent_id", f"INT-{i}")
        prompt = intent.get("input_text", "")
        truth = intent.get("ground_truth", {})
        domain = intent.get("domain", "Unknown")
        complexity = intent.get("complexity", "Unknown")
        
        if i % 20 == 0:
            pct = (i + 1) / total * 100
            print(f"  [{i+1:>3}/{total} {pct:5.1f}%] {iid} | {ok} ok {fail} fail")
        
        resp = call_deepseek(prompt)
        total_time += resp.get("t_inference_ms", 0) / 1000
        parsed = parse_json(resp.get("response", ""))
        score = score_response(parsed, truth)
        
        if score["format_valid"]: ok += 1
        else: fail += 1
        
        results.append({
            "model": DEEPSEEK_MODEL, "intent_id": iid, "domain": domain, "complexity": complexity,
            "input_text": prompt, "ground_truth": truth, "response": resp.get("response", ""),
            "parsed": parsed, "t_inference_ms": resp.get("t_inference_ms", 0),
            "prompt_tokens": resp.get("prompt_tokens", 0),
            "completion_tokens": resp.get("completion_tokens", 0), **score,
        })
        
        if i < total - 1: time.sleep(0.3)
    
    # Summary
    valid = sum(1 for r in results if r["format_valid"])
    act_ok = sum(1 for r in results if r["action_correct"])
    avg_match = sum(r["match_pct"] for r in results) / total
    avg_inf = sum(r["t_inference_ms"] for r in results) / total
    
    by_domain = defaultdict(lambda: {"count": 0, "match_sum": 0, "act_ok": 0})
    for r in results:
        d = r["domain"]
        by_domain[d]["count"] += 1
        by_domain[d]["match_sum"] += r["match_pct"]
        by_domain[d]["act_ok"] += 1 if r["action_correct"] else 0
    
    print(f"\n===== DEEPSEEK + ENRICHED PROMPT =====")
    print(f"Format Valid:   {valid}/{total} ({valid/total*100:.1f}%)")
    print(f"Action Correct: {act_ok}/{total} ({act_ok/total*100:.1f}%)")
    print(f"Avg Match:      {avg_match:.1f}%")
    print(f"Avg Inference:  {avg_inf:.0f} ms")
    print(f"Total Time:     {total_time/60:.1f} min")
    print(f"\nBy Domain:")
    for d, s in sorted(by_domain.items()):
        print(f"  {d:<22} Match: {s['match_sum']/s['count']:.1f}% | Action: {s['act_ok']}/{s['count']} ({s['act_ok']/s['count']*100:.1f}%)")
    
    final = {
        "model": DEEPSEEK_MODEL, "prompt_type": "enriched",
        "total_intents": total,
        "summary": {
            "format_accuracy": round(valid/total*100, 1),
            "action_accuracy": round(act_ok/total*100, 1),
            "avg_exact_match": round(avg_match, 1),
            "avg_inference_ms": round(avg_inf, 1),
            "total_time_s": round(total_time, 1),
            "by_domain": {d: {"count": s["count"], "action_accuracy": round(s["act_ok"]/s["count"]*100, 1),
                                  "avg_match_pct": round(s["match_sum"]/s["count"], 1)}
                          for d, s in by_domain.items()},
        },
        "results": results,
    }
    
    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved: {OUTPUT}")
    print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
