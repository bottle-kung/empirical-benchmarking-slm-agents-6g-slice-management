#!/usr/bin/env python3
"""
Phase 1 Complete: Semantic Scoring Pipeline
============================================
- DeepSeek-V4 Pro as LLM Judge: ALL ambiguous intents (65 per model)
- Semantic Key Matching: 135 non-ambiguous intents per model
- Output: comparison_data.json (Exact + Semantic + Lenient scores)

Total: 3 models × 65 ambiguous = 195 DeepSeek calls
"""

import os
import json, time, re
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

# Config
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE)

BENCHMARK = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval/results/benchmark_200_full.json")
OUTPUT = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval/results/comparison_data_v2.json")

# ═══════════════════════════════════════════════════════════════
# JUDGE PROMPT
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

    # Action match (with synonyms)
    action_score = 0
    for canonical, synonyms in ACTION_SYNONYMS.items():
        if truth_action in synonyms or truth_action == canonical:
            if any(s in parsed_action for s in synonyms) or parsed_action == canonical:
                action_score = 1.0
            elif any(s in parsed_action[:4] for s in [s[:4] for s in synonyms]):
                action_score = 0.5  # Partial match
            break

    # Slice type match (with synonyms)
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

    # Key-value matching for remaining params
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

    # Composite score
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

    # Convert to 1-5 scale
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


# ═══════════════════════════════════════════════════════════════
# DEEPSEEK JUDGE
# ═══════════════════════════════════════════════════════════════

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
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading benchmark data...")
    with open(BENCHMARK) as f:
        data = json.load(f)

    all_scored = []

    for model_data in data:
        model = model_data["model"]
        results = model_data["results"]

        print(f"\n{'='*60}")
        print(f"📊 Scoring: {model}")
        print(f"{'='*60}")

        scored = []
        ambiguous_idx = 0
        total_ambiguous = sum(1 for r in results if r.get("complexity") == "Ambiguous")

        for i, r in enumerate(results):
            is_ambiguous = r.get("complexity") == "Ambiguous"

            if "error" in r:
                scored.append({**r, "semantic_score": 1.0, "scoring_method": "error"})
                continue

            parsed = r.get("parsed")
            truth = r.get("ground_truth", {})
            intent_text = r.get("input_text", "")

            if is_ambiguous:
                # DeepSeek Judge for ALL ambiguous intents
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

                if ambiguous_idx % 20 == 0:
                    print(f"  [{ambiguous_idx}/{total_ambiguous}] ambiguous judged...")
            else:
                # Semantic Key Match for non-ambiguous
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

        exact_avg = sum(s.get("match_pct", 0) for s in scored) / len(scored)

        by_domain = defaultdict(list)
        by_method = defaultdict(list)
        for s in scored:
            by_domain[s.get("domain", "?")].append(s.get("semantic_score", 1))
            by_method[s.get("scoring_method", "?")].append(s.get("semantic_score", 1))

        print(f"\n  📊 {model} Summary:")
        print(f"     Exact Match:      {model_data['summary']['avg_match_pct']:.1f}%")
        print(f"     Semantic Score:   {avg_semantic:.2f}/5.00 ({avg_semantic/5*100:.1f}%)")
        print(f"     By Domain:")
        for domain, dscores in sorted(by_domain.items()):
            print(f"       {domain:<22} Semantic: {sum(dscores)/len(dscores):.2f}/5")
        print(f"     By Method:")
        for method, mscores in sorted(by_method.items()):
            print(f"       {method:<22} Avg: {sum(mscores)/len(mscores):.2f}/5 (n={len(mscores)})")

        all_scored.append({
            "model": model,
            "summary": {
                **model_data.get("summary", {}),
                "semantic_avg_score": round(avg_semantic, 2),
                "semantic_avg_pct": round(avg_semantic / 5 * 100, 1),
                "exact_avg_pct": model_data["summary"]["avg_match_pct"],
                "by_domain_semantic": {d: round(sum(s)/len(s), 2) for d, s in by_domain.items()},
            },
            "results": scored,
        })

    # ═══════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON — Exact vs Semantic")
    print(f"{'='*70}")
    print(f"{'Model':<18} {'Exact':>7} {'Semantic':>9} {'Improve':>9} {'DeepSeek':>10} {'Best For'}")
    print(f"{'-'*18} {'-'*7} {'-'*9} {'-'*9} {'-'*10} {'-'*20}")

    for mr in all_scored:
        model = mr["model"]
        exact = mr["summary"]["exact_avg_pct"]
        semantic = mr["summary"]["semantic_avg_pct"]
        improve = semantic - exact

        # Get DeepSeek-specific average
        judge_scores = [s["semantic_score"] for s in mr["results"] if s.get("scoring_method") == "deepseek_judge"]
        judge_avg = sum(judge_scores) / len(judge_scores) if judge_scores else 0

        if "1.5" in model:
            best = "Speed + Efficiency ⚡"
        elif "3b" in model:
            best = "Balance 🎯"
        else:
            best = "Accuracy 📊"

        print(f"{model:<18} {exact:>5.1f}% {semantic:>7.1f}% {improve:>+8.1f}% "
              f"{judge_avg:>7.2f}/5 {best}")

    # Save
    with open(OUTPUT, "w") as f:
        json.dump(all_scored, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n💾 Saved: {OUTPUT}")
    print(f"✅ Phase 1 Complete!")


if __name__ == "__main__":
    main()
