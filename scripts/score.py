#!/usr/bin/env python3
"""
Phase 1 Semantic Scoring — DeepSeek Judge
==========================================
Uses DeepSeek-V4 Pro as LLM Judge for ambiguous intents.
Semantic Key Matching for non-ambiguous intents.

Usage:
  python3 scripts/score.py results/phase1_benchmark_*.json
"""

import json
import os
import re
import time
import sys
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

# Config
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE)

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "results"

# ═══════════════════════════════════════════════════════════════
# JUDGE PROMPT (same as v1)
# ═══════════════════════════════════════════════════════════════
JUDGE_PROMPT = """You are evaluating an AI agent for 6G Core Network Slice Management.
Score how well the agent handled this AMBIGUOUS intent (1-5).

**Intent**: {intent}
**Reference**: {truth}
**Agent Response**: {response}

Scoring Guide:
- 1: Completely wrong action — would cause system failure
- 2: Mostly wrong — some awareness but critically bad decisions
- 3: Partially correct — right direction but wrong specific action/params
- 4: Good — correct action with minor issues OR different but reasonable approach
- 5: Excellent — optimal response given ambiguity

LENIENT RULES:
- Accept synonyms (e.g. "set"/"provision" = CREATE, "increase"/"boost" = SCALE)
- Accept reasonable parameter differences (e.g. 10 Gbps vs 1 Gbps for video)
- Focus on INTENT comprehension, not exact vocabulary

Return ONLY valid JSON:
{{"score": <1-5>, "action_ok": <true/false>, "reasoning": "<1 sentence>", "vocab_issue": <true/false>}}"""

# ═══════════════════════════════════════════════════════════════
# SEMANTIC KEY MATCHER
# ═══════════════════════════════════════════════════════════════
ACTION_SYNONYMS = {
    "create": ["create", "set", "setup", "provision", "deploy", "initialize", "add", "new", "configure", "establish", "allocate"],
    "scale": ["scale", "increase", "boost", "expand", "grow", "upscale", "scale_up", "upgrade", "enhance", "raise"],
    "scale_down": ["scale_down", "decrease", "reduce", "shrink", "downsize", "lower", "cut", "trim"],
    "resolve": ["resolve", "prioritize", "preempt", "ensure", "handle", "manage", "guarantee", "protect", "allocate_priority"],
    "terminate": ["terminate", "stop", "disable", "remove", "delete", "kill", "shutdown", "decommission"],
    "reallocate": ["reallocate", "redistribute", "rebalance", "shift", "move", "migrate", "transfer"],
}

SLICE_SYNONYMS = {
    "embb": ["embb", "enhanced mobile broadband", "broadband", "high bandwidth", "video", "streaming", "multimedia"],
    "urllc": ["urllc", "ultra reliable", "low latency", "ultra-reliable", "mission critical", "real-time", "safety"],
    "mmtc": ["mmtc", "massive iot", "massive machine", "iot", "sensor", "massive", "low power"],
}


def semantic_key_match(parsed: dict, truth: dict) -> dict:
    """Score non-ambiguous intent with semantic key matching."""
    if not parsed:
        return {"score": 1, "match_pct": 0, "action_match": False, "details": "No valid JSON"}

    parsed_action = str(parsed.get("action", "")).lower()
    truth_action = str(truth.get("action", "")).lower()

    action_score = 0
    for canonical, synonyms in ACTION_SYNONYMS.items():
        if truth_action in synonyms or truth_action == canonical:
            if any(s in parsed_action for s in synonyms) or parsed_action == canonical:
                action_score = 1.0
            elif any(s in parsed_action[:4] for s in [s[:4] for s in synonyms]):
                action_score = 0.5
            break

    slice_score = 0
    if "slice_type" in truth and "slice_type" in parsed:
        truth_slice = str(truth["slice_type"]).lower()
        parsed_slice = str(parsed["slice_type"]).lower().replace(" ", "_")
        for canonical, synonyms in SLICE_SYNONYMS.items():
            if truth_slice == canonical or truth_slice in synonyms:
                if parsed_slice == canonical or any(s in parsed_slice for s in synonyms):
                    slice_score = 1.0
                elif parsed_slice in ["embb", "urllc", "mmtc", "hybrid"]:
                    slice_score = 0.3 if parsed_slice == truth_slice else 0
                break

    param_scores = []
    for key, expected in truth.items():
        if key in ("action", "slice_type"):
            continue
        if key not in parsed:
            continue
        p_val = str(parsed[key]).lower().strip('"\'')
        e_val = str(expected).lower()
        if p_val == e_val:
            param_scores.append(1.0)
        elif key in ("bandwidth_gbps", "latency_ms", "device_density", "replicas", "step_percent", "duration_hours", "buffer_gb"):
            try:
                pn, en = float(p_val), float(e_val)
                if en > 0:
                    ratio = min(pn, en) / max(pn, en)
                    if ratio >= 0.8:
                        param_scores.append(0.9)
                    elif ratio >= 0.5:
                        param_scores.append(0.5)
                    elif ratio >= 0.2:
                        param_scores.append(0.3)
            except ValueError:
                pass
        elif key in ("priority", "strategy", "target", "preempt", "slice_id", "optimize", "trigger"):
            if p_val in e_val or e_val in p_val:
                param_scores.append(0.7)
            elif any(w in p_val for w in e_val.split("_")):
                param_scores.append(0.4)

    weights = []
    values = []
    if action_score > 0 or truth_action in str(parsed.get("action", "")):
        weights.append(0.35)
        values.append(action_score)
    if "slice_type" in truth:
        weights.append(0.25)
        values.append(slice_score)
    for ps in param_scores:
        weights.append(0.4 / max(len(param_scores), 1))
        values.append(ps)

    if not weights:
        w_score = max(action_score, slice_score)
    else:
        total_w = sum(weights)
        w_score = sum(w * v for w, v in zip(weights, values)) / total_w if total_w > 0 else 0

    semantic_score = 1 + w_score * 4
    match_pct = w_score * 100

    return {
        "score": round(semantic_score, 1),
        "match_pct": round(match_pct, 1),
        "action_match": action_score >= 0.5,
        "action_score": round(action_score, 2),
        "slice_score": round(slice_score, 2),
        "params_matched": len(param_scores),
        "params_total": len([k for k in truth if k not in ("action", "slice_type")]),
    }


def deepseek_judge(intent_text: str, truth: dict, response: dict, retries=3) -> dict:
    """Call DeepSeek to judge an ambiguous intent response."""
    prompt = JUDGE_PROMPT.format(
        intent=intent_text,
        truth=json.dumps(truth),
        response=json.dumps(response) if response else "INVALID JSON",
    )

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "You are a 6G network evaluation expert. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
            result = json.loads(content)
            result["judge_model"] = DEEPSEEK_MODEL
            return result
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"score": None, "action_ok": False, "reasoning": str(e), "vocab_issue": False, "judge_model": DEEPSEEK_MODEL}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/score.py results/benchmark_*.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    print(f"Loading benchmark data: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    all_scored = []

    for model_data in data:
        model = model_data["summary"]["model"]
        family = model_data["summary"]["family"]
        results_list = model_data["results"]

        print(f"\n{'='*60}")
        print(f"📊 Scoring: {model} ({family})")
        print(f"{'='*60}")

        scored = []
        ambiguous_idx = 0
        total_ambiguous = sum(1 for r in results_list if r.get("complexity") == "Ambiguous")

        for i, r in enumerate(results_list):
            is_ambiguous = r.get("complexity") == "Ambiguous"

            if r.get("error"):
                scored.append({**r, "semantic_score": 1.0, "scoring_method": "error"})
                continue

            parsed = r.get("parsed")
            truth = r.get("ground_truth", {})
            intent_text = r.get("input_text", "")

            if is_ambiguous:
                ambiguous_idx += 1
                judgment = deepseek_judge(intent_text, truth, parsed)
                semantic_score = judgment.get("score", 2)

                scored.append({
                    **r,
                    "semantic_score": semantic_score,
                    "scoring_method": "deepseek_judge",
                    "judge_feedback": judgment.get("reasoning", ""),
                    "action_understood": judgment.get("action_ok", False),
                    "vocab_issue": judgment.get("vocab_issue", False),
                })

                if ambiguous_idx % 10 == 0:
                    print(f"  [{ambiguous_idx}/{total_ambiguous}] ambiguous judged...")
            else:
                match = semantic_key_match(parsed, truth)
                scored.append({
                    **r,
                    "semantic_score": match["score"],
                    "semantic_match_pct": match["match_pct"],
                    "scoring_method": "semantic_key",
                    "action_match": match["action_match"],
                })

        # Summary
        scores = [s["semantic_score"] for s in scored if "semantic_score" in s]
        avg_semantic = sum(scores) / len(scores) if scores else 0

        by_domain = defaultdict(list)
        for s in scored:
            by_domain[s.get("domain", "?")].append(s.get("semantic_score", 1))

        print(f"\n  📊 {model} Scored:")
        print(f"     Semantic Avg: {avg_semantic:.2f}/5 ({avg_semantic/5*100:.1f}%)")

        all_scored.append({
            "model": model,
            "family": family,
            "summary": {
                **model_data.get("summary", {}),
                "semantic_avg_score": round(avg_semantic, 2),
                "semantic_avg_pct": round(avg_semantic / 5 * 100, 1),
            },
            "results": scored,
        })

    # Comparison table
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON — Exact vs Semantic")
    print(f"{'='*70}")
    print(f"{'Model':<22} {'Family':<12} {'Exact':>8} {'Semantic':>10} {'Improve':>9}")
    print(f"{'-'*22} {'-'*12} {'-'*8} {'-'*10} {'-'*9}")

    for mr in all_scored:
        model = mr["model"]
        family = mr["family"]
        exact = mr["summary"]["avg_match_pct"]
        semantic = mr["summary"]["semantic_avg_pct"]
        improve = semantic - exact
        print(f"{model:<22} {family:<12} {exact:>7.1f}% {semantic:>9.1f}% {improve:>+8.1f}%")

    # Save
    output_path = RESULTS_DIR / f"scored_{input_path.name}"
    with open(output_path, "w") as f:
        json.dump(all_scored, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved: {output_path}")
    print("✅ Scoring complete!")


if __name__ == "__main__":
    main()
