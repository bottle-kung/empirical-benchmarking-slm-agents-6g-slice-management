#!/usr/bin/env python3
"""
Lenient Semantic Scoring using DeepSeek-V4 Pro (LLM-as-a-Judge)
================================================================
Evaluates model responses with semantic understanding instead of exact match.
- Ambiguous intents: 5-point rubric with reasoning feedback
- Non-ambiguous intents: semantic key-value similarity

API: DeepSeek (OpenAI-compatible)
"""

import json, time, os
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

# DeepSeek API config
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

BENCHMARK_PATH = "/jupyter_workspace/local/ai_agent/slm_6g_eval/results/benchmark_200_full.json"
OUTPUT_PATH = "/jupyter_workspace/local/ai_agent/slm_6g_eval/results/benchmark_lenient_scored.json"

JUDGE_PROMPT = """You are an expert evaluator for a 6G Core Network Slice Management system.
Score how well an SLM agent handled an intent on a 1-5 scale.

**Original Intent**: {intent}
**Reference (Ground Truth)**: {ground_truth}
**Agent Response**: {agent_response}

Scoring Rubric:
- 1: Completely wrong — wrong action, wrong parameters, would break the system
- 2: Mostly wrong — some awareness but critically wrong decisions
- 3: Partially correct — right idea but wrong specific action or parameters
- 4: Good — correct action with minor parameter issues OR different but reasonable approach
- 5: Excellent — optimal response, all parameters appropriate for the context

CRITICAL FOR LENIENT SCORING:
- If the agent used a SYNONYM for the action (e.g., "set" instead of "CREATE", "increase" instead of "SCALE", "prioritize" instead of "RESOLVE"), DO NOT penalize — score based on INTENT
- If the agent chose different but REASONABLE parameters (e.g., 10 Gbps instead of 1 Gbps for 4K video), give partial credit
- Focus on whether the agent UNDERSTOOD the intent, not whether it matched the exact vocabulary

Return ONLY valid JSON:
{{"score": <1-5>, "feedback": "<brief>", "action_understood": <true/false>, "reasonable_params": <true/false>}}"""


def load_benchmark(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def judge_response(intent_text: str, ground_truth: dict, agent_response: dict, retries=3) -> dict:
    """Call DeepSeek to judge a single response."""
    prompt = JUDGE_PROMPT.format(
        intent=intent_text,
        ground_truth=json.dumps(ground_truth),
        agent_response=json.dumps(agent_response) if agent_response else "INVALID JSON",
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
                max_tokens=256,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content[:-3]
            return json.loads(content)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"score": None, "feedback": str(e), "action_understood": False, "reasonable_params": False}


def semantic_key_match(parsed: dict, truth: dict) -> float:
    """Simple semantic key matching for non-ambiguous intents."""
    if not parsed:
        return 0.0
    
    # Action synonym mapping
    action_synonyms = {
        "create": ["create", "set", "setup", "provision", "deploy", "initialize", "add", "new"],
        "scale": ["scale", "increase", "boost", "expand", "grow", "upscale", "scale_up"],
        "scale_down": ["scale_down", "decrease", "reduce", "shrink", "downsize"],
        "resolve": ["resolve", "prioritize", "preempt", "ensure", "handle", "manage"],
        "terminate": ["terminate", "stop", "disable", "remove", "delete", "kill"],
        "reallocate": ["reallocate", "redistribute", "rebalance", "shift", "move"],
    }
    
    # Slice type synonyms
    slice_synonyms = {
        "embb": ["embb", "enhanced mobile broadband", "broadband", "high bandwidth"],
        "urllc": ["urllc", "ultra reliable", "low latency", "ultra-reliable"],
        "mmtc": ["mmtc", "massive iot", "massive machine", "iot"],
    }
    
    score = 0.0
    total = len(truth)
    
    for key, expected_val in truth.items():
        if key not in parsed:
            continue
        
        parsed_val = str(parsed[key]).lower()
        expected_str = str(expected_val).lower()
        
        # Exact match
        if parsed_val == expected_str:
            score += 1.0
            continue
        
        # Synonym match for action
        if key == "action":
            for canonical, synonyms in action_synonyms.items():
                if expected_str in synonyms and parsed_val in synonyms:
                    score += 0.7  # Partial credit for synonym
                    break
        
        # Synonym match for slice_type
        elif key == "slice_type":
            for canonical, synonyms in slice_synonyms.items():
                if expected_str in synonyms or expected_str == canonical:
                    if any(s in parsed_val for s in synonyms):
                        score += 0.8
                        break
        
        # Numeric proximity (within 2x)
        elif key in ("bandwidth_gbps", "latency_ms", "device_density", "replicas", "step_percent"):
            try:
                pv = float(parsed_val)
                ev = float(expected_str)
                if ev > 0:
                    ratio = min(pv, ev) / max(pv, ev)
                    if ratio >= 0.5:  # Within 2x
                        score += 0.6
                    elif ratio >= 0.2:  # Within 5x
                        score += 0.3
            except ValueError:
                pass
    
    return min(1.0, score / total) if total > 0 else 0.0


def main():
    print("Loading benchmark results...")
    data = load_benchmark(BENCHMARK_PATH)
    
    # Count ambiguous intents
    ambiguous_count = 65  # From dataset design: 23 + 22 + 20
    
    all_scored = []
    
    for model_data in data:
        model = model_data["model"]
        results = model_data["results"]
        
        print(f"\n{'='*60}")
        print(f"📊 Lenient Scoring: {model}")
        print(f"{'='*60}")
        
        scored_results = []
        semantic_scores = []
        judge_scores = []
        
        for i, r in enumerate(results):
            if "error" in r:
                scored_results.append({**r, "lenient_match": 0, "lenient_score": 1})
                continue
            
            is_ambiguous = r.get("complexity") == "Ambiguous"
            
            if is_ambiguous and i % 2 == 0:  # Sample every other ambiguous for cost
                # Use DeepSeek LLM-as-a-Judge
                judgment = judge_response(
                    r.get("input_text", ""),
                    r.get("ground_truth", {}),
                    r.get("parsed"),
                )
                score = judgment.get("score", 1)
                judge_scores.append(score)
                scored_results.append({
                    **r,
                    "lenient_score": score,
                    "lenient_type": "deepseek_judge",
                    "judge_feedback": judgment.get("feedback", ""),
                    "action_understood": judgment.get("action_understood", False),
                })
            else:
                # Semantic key matching
                match = semantic_key_match(r.get("parsed"), r.get("ground_truth", {}))
                # Convert 0-1 match to 1-5 scale
                score = 1 + match * 4  # 0→1, 0.5→3, 1.0→5
                semantic_scores.append(match)
                scored_results.append({
                    **r,
                    "lenient_match": round(match, 2),
                    "lenient_score": round(score, 1),
                    "lenient_type": "semantic_key",
                })
            
            if (i + 1) % 40 == 0:
                print(f"  [{i+1}/{len(results)}] scored...")
        
        # Summary
        all_lenient = [s["lenient_score"] for s in scored_results if "lenient_score" in s]
        avg_lenient = sum(all_lenient) / len(all_lenient) if all_lenient else 0
        
        # By domain
        by_domain = defaultdict(list)
        for s in scored_results:
            by_domain[s.get("domain", "?")].append(s.get("lenient_score", 1))
        
        print(f"\n  📊 {model} Lenient Score Summary:")
        print(f"     Avg Lenient Score: {avg_lenient:.2f}/5.00")
        print(f"     Semantic Matches:  {len(semantic_scores)} (avg: {sum(semantic_scores)/len(semantic_scores):.2f})" if semantic_scores else "")
        print(f"     DeepSeek Judgments: {len(judge_scores)} (avg: {sum(judge_scores)/len(judge_scores):.2f})" if judge_scores else "")
        print(f"     By Domain:")
        for domain, scores in sorted(by_domain.items()):
            print(f"       {domain:<22} Avg: {sum(scores)/len(scores):.2f}/5.00")
        
        all_scored.append({
            "model": model,
            "summary": {
                **model_data.get("summary", {}),
                "lenient_avg_score": round(avg_lenient, 2),
                "by_domain_lenient": {d: round(sum(s)/len(s), 2) for d, s in by_domain.items()},
            },
            "results": scored_results,
        })
    
    # Final comparison
    print(f"\n{'='*70}")
    print(f"  LENIENT SCORING COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<18} {'Exact':>7} {'Lenient':>8} {'Improvement':>12}")
    print(f"{'-'*18} {'-'*7} {'-'*8} {'-'*12}")
    
    for mr, ms in zip(data, all_scored):
        exact_avg = mr["summary"]["avg_match_pct"]
        lenient_avg = ms["summary"]["lenient_avg_score"]
        lenient_pct = lenient_avg / 5 * 100  # Convert 5-scale to %
        improvement = lenient_pct - exact_avg
        print(f"{mr['model']:<18} {exact_avg:>5.1f}% {lenient_pct:>6.1f}% {improvement:>+10.1f}%")
    
    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_scored, f, indent=2, default=str)
    print(f"\n💾 Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
