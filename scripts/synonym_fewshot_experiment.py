#!/usr/bin/env python3
"""
Experiment 3: Synonym Mapper + Few-Shot Prompting
==================================================
Proves that vocabulary errors (60.8% of failures) can be fixed by:
  1. Synonym Mapper: Post-process model output → canonical schema terms
  2. Few-Shot Prompting: 3 examples in system prompt teach vocabulary

Runs on Qwen2.5-1.5B (the Pareto-optimal sweet spot) via Proxmox host.
Tests 4 conditions from 2 actual runs:
  - Baseline (original prompt, no mapper)
  - +Synonym Mapper (original prompt + post-process mapping)
  - +Few-Shot (enhanced prompt, no mapper)
  - +Both (enhanced prompt + post-process mapping)

Output: results/synonym_fewshot_results.json
"""

import json, re, time, requests, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
MODEL = "qwen2.5:1.5b"
OLLAMA_URL = "http://172.16.1.206:11434/api/generate"

DATASET = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval/dataset/intents.jsonl")
OUTPUT = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval/results/synonym_fewshot_results.json")
LOG = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval/results/synonym_fewshot_log.txt")

# Open log file for dual output
log_f = open(LOG, 'w', buffering=1)

# Save original print
_original_print = print

def dual_print(*args, **kwargs):
    """Print to both stdout and log file simultaneously."""
    _original_print(*args, **kwargs)
    _original_print(*args, **kwargs, file=log_f)
    log_f.flush()

# ═══════════════════════════════════════════════════════════════
# SYNONYM MAPPER — maps model vocabulary to canonical schema
# ═══════════════════════════════════════════════════════════════
SYNONYM_MAP = {
    # ── Actions ──
    "action": {
        "create": "CREATE",
        "deploy": "CREATE",
        "provision": "CREATE",
        "initialize": "CREATE",
        "setup": "CREATE",
        "set up": "CREATE",
        "add": "CREATE",
        "create_slice": "CREATE",
        "new": "CREATE",
        "scale": "SCALE",
        "scale_up": "SCALE_UP",
        "scale down": "SCALE_DOWN",
        "scale_down": "SCALE_DOWN",
        "increase": "SCALE",
        "increase bandwidth": "SCALE",
        "decrease": "SCALE_DOWN",
        "reduce": "SCALE_DOWN",
        "optimize": "OPTIMIZE",
        "resolve": "RESOLVE",
        "preempt": "PREEMPT",
        "terminate": "TERMINATE",
        "reallocate": "REALLOCATE",
        "configure": "CONFIGURE",
        "config": "CONFIGURE",
        "modify": "CONFIGURE",
        "update": "CONFIGURE",
    },

    # ── Slice Types ──
    "slice_type": {
        "embb": "eMBB",
        "enhanced mobile broadband": "eMBB",
        "broadband": "eMBB",
        "high bandwidth": "eMBB",
        "urllc": "URLLC",
        "ultra reliable low latency": "URLLC",
        "ultra-reliable low-latency": "URLLC",
        "low latency": "URLLC",
        "mmtc": "mMTC",
        "massive iot": "mMTC",
        "massive machine type": "mMTC",
        "iot": "mMTC",
        "internet of things": "mMTC",
    },

    # ── Targets ──
    "target": {
        "bandwidth": "bandwidth",
        "throughput": "bandwidth",
        "speed": "bandwidth",
        "data rate": "bandwidth",
        "latency": "latency",
        "delay": "latency",
        "upf": "upf",
        "user plane": "upf",
        "compute": "compute",
        "cpu": "compute",
    },

    # ── Strategies ──
    "strategy": {
        "gradual": "gradual",
        "step": "gradual",
        "incremental": "gradual",
        "slowly": "gradual",
        "immediate": "immediate",
        "instant": "immediate",
        "now": "immediate",
    },

    # ── Priority values ──
    "priority": {
        "high": "critical",
        "highest": "critical",
        "urgent": "critical",
        "emergency": "critical",
        "low": "low",
        "normal": "normal",
        "medium": "normal",
    },
}

def apply_synonym_mapper(parsed: dict) -> dict:
    """Map model output vocabulary to canonical schema terms."""
    if not parsed:
        return parsed

    result = {}
    for key, value in parsed.items():
        key_lower = key.lower().strip()

        # Normalize key name (remove spaces, lowercase)
        norm_key = key_lower.replace(" ", "_").replace("-", "_")

        # Map values if the field has a synonym map
        mapped_value = value
        if isinstance(value, str) and norm_key in SYNONYM_MAP:
            val_lower = value.lower().strip()
            if val_lower in SYNONYM_MAP[norm_key]:
                mapped_value = SYNONYM_MAP[norm_key][val_lower]

        result[key] = mapped_value

    return result


# ═══════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════

BASELINE_PROMPT = """You are a 6G Core Network Slice Management Agent. 
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

FEWSHOT_PROMPT = """You are a 6G Core Network Slice Management Agent. 
Your job is to translate natural language intents into JSON commands for network orchestration.

Rules:
1. Respond ONLY with a valid JSON object — no markdown, no explanations, no code fences
2. The JSON MUST contain an "action" field
3. Use EXACT canonical action names: CREATE, SCALE, SCALE_UP, SCALE_DOWN, OPTIMIZE, RESOLVE, PREEMPT, TERMINATE, REALLOCATE, CONFIGURE
4. Use EXACT slice types: "eMBB", "URLLC", "mMTC"
5. For ambiguous requests, infer the best configuration based on context

Examples:
Input: "Create a URLLC slice with 1ms latency for robot control"
Output: {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 1}

Input: "Gradually increase bandwidth by 20% every hour for 4 hours"
Output: {"action": "SCALE", "target": "bandwidth", "strategy": "gradual", "step_percent": 20, "duration_hours": 4}

Input: "Emergency responders need absolute priority over regular users"
Output: {"action": "RESOLVE", "priority": "critical", "preempt": "regular_users", "slice_type": "URLLC"}"""


# ═══════════════════════════════════════════════════════════════
# JSON PARSER
# ═══════════════════════════════════════════════════════════════
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
            try:
                return json.loads(m)
            except:
                pass
    return None


# ═══════════════════════════════════════════════════════════════
# SCORER
# ═══════════════════════════════════════════════════════════════
def score_response(parsed: dict, truth: dict) -> dict:
    if not parsed:
        return {
            "format_valid": False, "action_correct": False,
            "match_pct": 0, "exact_keys": 0, "total_keys": len(truth),
        }

    action_ok = str(parsed.get("action", "")).lower() == truth["action"].lower()
    exact = sum(
        1 for k in truth
        if k in parsed and str(parsed[k]).lower() == str(truth[k]).lower()
    )
    total = len(truth)
    match_pct = round(exact / total * 100, 1) if total > 0 else 0

    return {
        "format_valid": True,
        "action_correct": action_ok,
        "match_pct": match_pct,
        "exact_keys": exact,
        "total_keys": total,
    }


# ═══════════════════════════════════════════════════════════════
# OLLAMA CALL (direct HTTP to Proxmox)
# ═══════════════════════════════════════════════════════════════
def call_ollama(prompt: str, system_prompt: str, retries=2) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.1}  # No num_predict limit
    }

    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
            elapsed = time.time() - t0

            data = resp.json()
            response_text = data.get("response", "")
            duration_s = data.get("total_duration", 0) / 1e9
            tokens = data.get("eval_count", 0)

            return {
                "response": response_text,
                "wall_time_s": round(elapsed, 1),
                "duration_s": round(duration_s, 1),
                "tokens": tokens,
                "speed_tok_s": round(tokens / duration_s, 2) if duration_s > 0 else 0,
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                return {"response": "", "error": str(e)}

    return {"response": "", "error": "max retries"}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    # Monkey-patch print for dual output (stdout + log file)
    import builtins
    builtins.print = dual_print
    
    # Load dataset
    intents = []
    with open(DATASET, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                intents.append(json.loads(line))

    total = len(intents)
    print(f"📦 {total} intents loaded")
    print(f"🤖 Model: {MODEL}")
    print(f"📅 Started: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")
    print(f"{'='*70}")

    # ═══════════════════════════════════════════════════════════
    # RUN 1: BASELINE PROMPT
    # ═══════════════════════════════════════════════════════════
    print(f"\n🔵 RUN 1/2: BASELINE Prompt ({total} intents)")
    print(f"{'─'*70}")

    raw_results = []  # Store raw responses for post-processing

    for i, intent in enumerate(intents):
        iid = intent.get("intent_id", f"INT-{i}")
        prompt = intent.get("input_text", "")
        truth = intent.get("ground_truth", {})
        domain = intent.get("domain", "Unknown")
        complexity = intent.get("complexity", "Unknown")

        resp = call_ollama(prompt, BASELINE_PROMPT)
        parsed = parse_json(resp.get("response", ""))
        score = score_response(parsed, truth)

        raw_results.append({
            "intent_id": iid,
            "domain": domain,
            "complexity": complexity,
            "input_text": prompt,
            "ground_truth": truth,
            "raw_response": resp.get("response", ""),
            "raw_parsed": parsed,
            "run": "baseline",
            **resp,
            **score,
        })

        if (i + 1) % 20 == 0 or i == total - 1:
            ok = sum(1 for r in raw_results if r["action_correct"])
            print(f"  [{i+1:>3}/{total}] Action: {ok}/{i+1} ({ok/(i+1)*100:.1f}%)", flush=True)

        if i < total - 1:
            time.sleep(0.2)

    print(f"\n  ✅ Baseline run complete: {len(raw_results)} responses")

    # ═══════════════════════════════════════════════════════════
    # RUN 2: FEW-SHOT PROMPT
    # ═══════════════════════════════════════════════════════════
    print(f"\n🟢 RUN 2/2: FEW-SHOT Prompt ({total} intents)")
    print(f"{'─'*70}")

    fewshot_results = []

    for i, intent in enumerate(intents):
        iid = intent.get("intent_id", f"INT-{i}")
        prompt = intent.get("input_text", "")
        truth = intent.get("ground_truth", {})

        resp = call_ollama(prompt, FEWSHOT_PROMPT)
        parsed = parse_json(resp.get("response", ""))
        score = score_response(parsed, truth)

        fewshot_results.append({
            "intent_id": iid,
            "domain": intent.get("domain", "Unknown"),
            "complexity": intent.get("complexity", "Unknown"),
            "input_text": prompt,
            "ground_truth": truth,
            "raw_response": resp.get("response", ""),
            "raw_parsed": parsed,
            "run": "fewshot",
            **resp,
            **score,
        })

        if (i + 1) % 20 == 0 or i == total - 1:
            ok = sum(1 for r in fewshot_results if r["action_correct"])
            print(f"  [{i+1:>3}/{total}] Action: {ok}/{i+1} ({ok/(i+1)*100:.1f}%)", flush=True)

        if i < total - 1:
            time.sleep(0.2)

    print(f"\n  ✅ Few-Shot run complete: {len(fewshot_results)} responses")

    # ═══════════════════════════════════════════════════════════
    # POST-PROCESS: Apply Synonym Mapper to both runs
    # ═══════════════════════════════════════════════════════════
    print(f"\n🔄 Applying Synonym Mapper post-processing...")
    print(f"{'─'*70}")

    all_results = []

    # Baseline + Synonym Mapper
    for r in raw_results:
        syn_parsed = apply_synonym_mapper(r.get("raw_parsed"))
        syn_score = score_response(syn_parsed, r["ground_truth"])
        all_results.append({
            **{k: v for k, v in r.items() if k not in ("raw_parsed", "format_valid", "action_correct", "match_pct", "exact_keys", "total_keys")},
            "condition": "Baseline",
            "synonym_applied": False,
            "parsed": r.get("raw_parsed"),
            **score_response(r.get("raw_parsed"), r["ground_truth"]),
        })
        all_results.append({
            **{k: v for k, v in r.items() if k not in ("raw_parsed", "format_valid", "action_correct", "match_pct", "exact_keys", "total_keys")},
            "condition": "Baseline + Synonym",
            "synonym_applied": True,
            "parsed": syn_parsed,
            **syn_score,
        })

    # Few-Shot + Synonym Mapper
    for r in fewshot_results:
        syn_parsed = apply_synonym_mapper(r.get("raw_parsed"))
        syn_score = score_response(syn_parsed, r["ground_truth"])
        all_results.append({
            **{k: v for k, v in r.items() if k not in ("raw_parsed", "format_valid", "action_correct", "match_pct", "exact_keys", "total_keys")},
            "condition": "Few-Shot",
            "synonym_applied": False,
            "parsed": r.get("raw_parsed"),
            **score_response(r.get("raw_parsed"), r["ground_truth"]),
        })
        all_results.append({
            **{k: v for k, v in r.items() if k not in ("raw_parsed", "format_valid", "action_correct", "match_pct", "exact_keys", "total_keys")},
            "condition": "Few-Shot + Synonym",
            "synonym_applied": True,
            "parsed": syn_parsed,
            **syn_score,
        })

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  📊 RESULTS SUMMARY — Synonym Mapper + Few-Shot Experiment")
    print(f"{'='*70}")

    conditions = ["Baseline", "Baseline + Synonym", "Few-Shot", "Few-Shot + Synonym"]
    summary = {}

    for cond in conditions:
        subset = [r for r in all_results if r["condition"] == cond]
        n = len(subset)
        fmt_ok = sum(1 for r in subset if r["format_valid"])
        act_ok = sum(1 for r in subset if r["action_correct"])
        avg_match = sum(r["match_pct"] for r in subset) / n if n > 0 else 0
        avg_time = sum(r.get("wall_time_s", 0) for r in subset) / n if n > 0 else 0

        summary[cond] = {
            "count": n,
            "format_accuracy": round(fmt_ok / n * 100, 1) if n > 0 else 0,
            "action_accuracy": round(act_ok / n * 100, 1) if n > 0 else 0,
            "avg_match_pct": round(avg_match, 1),
            "avg_wall_time_s": round(avg_time, 1),
        }

        print(f"\n  📌 {cond}")
        print(f"     Format: {fmt_ok}/{n} ({summary[cond]['format_accuracy']}%)")
        print(f"     Action: {act_ok}/{n} ({summary[cond]['action_accuracy']}%)")
        print(f"     Avg Match: {summary[cond]['avg_match_pct']}%")
        print(f"     Avg Time: {summary[cond]['avg_wall_time_s']}s")

    # ═══════════════════════════════════════════════════════════
    # ADDITIONAL: By Domain breakdown
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  📊 By Domain Breakdown")
    print(f"{'─'*70}")

    by_domain = defaultdict(lambda: defaultdict(lambda: {"count": 0, "act_ok": 0, "match_sum": 0}))
    for r in all_results:
        cond = r["condition"]
        dom = r["domain"]
        by_domain[cond][dom]["count"] += 1
        by_domain[cond][dom]["act_ok"] += 1 if r["action_correct"] else 0
        by_domain[cond][dom]["match_sum"] += r["match_pct"]

    domains_order = ["Slicing Provisioning", "Scaling Request", "Conflict Resolution"]
    print(f"\n  {'Condition':<24} {'Provisioning':>16} {'Scaling':>16} {'Conflict':>16}")
    print(f"  {'─'*24} {'─'*16} {'─'*16} {'─'*16}")
    for cond in conditions:
        prov = by_domain[cond].get("Slicing Provisioning", {"count": 0, "act_ok": 0, "match_sum": 0})
        scal = by_domain[cond].get("Scaling Request", {"count": 0, "act_ok": 0, "match_sum": 0})
        conf = by_domain[cond].get("Conflict Resolution", {"count": 0, "act_ok": 0, "match_sum": 0})

        p_match = round(prov["match_sum"] / prov["count"], 1) if prov["count"] > 0 else 0
        s_match = round(scal["match_sum"] / scal["count"], 1) if scal["count"] > 0 else 0
        c_match = round(conf["match_sum"] / conf["count"], 1) if conf["count"] > 0 else 0

        print(f"  {cond:<24} {p_match:>6.1f}% ({prov['act_ok']:>2}/{prov['count']:<2}) "
              f"{s_match:>6.1f}% ({scal['act_ok']:>2}/{scal['count']:<2}) "
              f"{c_match:>6.1f}% ({conf['act_ok']:>2}/{conf['count']:<2})")

    # ═══════════════════════════════════════════════════════════
    # IMPROVEMENT ANALYSIS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  📈 IMPROVEMENT ANALYSIS (vs Baseline)")
    print(f"{'─'*70}")

    base = summary["Baseline"]
    base_act = base["action_accuracy"]
    base_match = base["avg_match_pct"]

    for cond in ["Baseline + Synonym", "Few-Shot", "Few-Shot + Synonym"]:
        s = summary[cond]
        act_delta = s["action_accuracy"] - base_act
        match_delta = s["avg_match_pct"] - base_match
        print(f"  {cond:<24} Action: +{act_delta:+.1f}% | Match: +{match_delta:+.1f}%")

    # Also show how many vocabulary errors were fixed
    base_errors = [r for r in all_results if r["condition"] == "Baseline" and not r["action_correct"]]
    syn_fixed = 0
    for r in [x for x in all_results if x["condition"] == "Baseline + Synonym"]:
        base_r = next((b for b in base_errors if b["intent_id"] == r["intent_id"]), None)
        if base_r and r["action_correct"]:
            syn_fixed += 1

    print(f"\n  🔧 Synonym Mapper fixed {syn_fixed}/{len(base_errors)} previously-failed intents")
    if base_errors:
        print(f"     ({syn_fixed/len(base_errors)*100:.1f}% of baseline errors resolved by vocabulary mapping alone)")

    # Save
    final = {
        "model": MODEL,
        "total_intents": total,
        "prompts": {
            "baseline": BASELINE_PROMPT,
            "fewshot": FEWSHOT_PROMPT,
        },
        "synonym_map_keys": list(SYNONYM_MAP.keys()),
        "summary": summary,
        "by_domain": {cond: {dom: dict(s) for dom, s in doms.items()} 
                       for cond, doms in by_domain.items()},
        "results": all_results,
    }

    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"💾 Saved: {OUTPUT}")
    print(f"📅 Finished: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")
    print(f"⏱️  Total wall time: ~{summary['Baseline']['avg_wall_time_s'] * total / 60:.0f} min × 2 runs")


if __name__ == "__main__":
    main()
