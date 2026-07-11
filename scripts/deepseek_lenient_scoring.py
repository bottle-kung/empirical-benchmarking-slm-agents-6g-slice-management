#!/usr/bin/env python3
"""
Run Lenient Semantic Scoring on DeepSeek enriched results.
Same methodology as SLM scoring (semantic match + LLM judge for ambiguous).
"""
import json, time, os
from collections import defaultdict, OrderedDict
from openai import OpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

INPUT = "/jupyter_workspace/local/ai_agent/slm_6g_eval/results/deepseek_enriched_results.json"
OUTPUT = "/jupyter_workspace/local/ai_agent/slm_6g_eval/results/deepseek_lenient_scored.json"

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

ACTION_SYNONYMS = {
    "create": ["create", "set", "setup", "provision", "deploy", "initialize", "add", "new"],
    "scale": ["scale", "increase", "boost", "expand", "grow", "upscale", "scale_up"],
    "scale_down": ["scale_down", "decrease", "reduce", "shrink", "downsize"],
    "resolve": ["resolve", "prioritize", "preempt", "ensure", "handle", "manage"],
    "terminate": ["terminate", "stop", "disable", "remove", "delete", "kill"],
    "reallocate": ["reallocate", "redistribute", "rebalance", "shift", "move"],
}

SLICE_SYNONYMS = {
    "embb": ["embb", "enhanced mobile broadband", "broadband", "high bandwidth"],
    "urllc": ["urllc", "ultra reliable", "low latency", "ultra-reliable"],
    "mmtc": ["mmtc", "massive iot", "massive machine", "iot"],
}


def judge_response(intent_text, ground_truth, agent_parsed, retries=3):
    """Call DeepSeek to judge a single response."""
    prompt = JUDGE_PROMPT.format(
        intent=intent_text,
        ground_truth=json.dumps(ground_truth),
        agent_response=json.dumps(agent_parsed) if agent_parsed else "INVALID JSON",
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
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


def semantic_key_match(parsed, truth):
    """Simple semantic key matching for non-ambiguous intents."""
    if not parsed:
        return 0.0
    score = 0.0
    total = len(truth)
    for key, expected_val in truth.items():
        if key not in parsed:
            continue
        parsed_val = str(parsed[key]).lower()
        expected_str = str(expected_val).lower()
        if parsed_val == expected_str:
            score += 1.0
            continue
        if key == "action":
            for canonical, synonyms in ACTION_SYNONYMS.items():
                if expected_str in synonyms and parsed_val in synonyms:
                    score += 0.7
                    break
        elif key == "slice_type":
            for canonical, synonyms in SLICE_SYNONYMS.items():
                if expected_str in synonyms or expected_str == canonical:
                    if any(s in parsed_val for s in synonyms):
                        score += 0.8
                        break
        elif key in ("bandwidth_gbps", "latency_ms", "device_density", "replicas", "step_percent"):
            try:
                pv = float(parsed_val)
                ev = float(expected_str)
                if ev > 0:
                    ratio = min(pv, ev) / max(pv, ev)
                    if ratio >= 0.5:
                        score += 0.6
                    elif ratio >= 0.2:
                        score += 0.3
            except ValueError:
                pass
    return min(1.0, score / total) if total > 0 else 0.0


def main():
    print("Loading DeepSeek enriched results...")
    with open(INPUT) as f:
        data = json.load(f)

    results = data["results"]
    domain_short = {
        "Slicing Provisioning": "Provisioning",
        "Scaling Request": "Scaling",
        "Conflict Resolution": "Conflict",
    }

    scored_results = []
    by_domain = defaultdict(list)
    judge_count = 0
    semantic_count = 0

    print(f"Scoring {len(results)} intents...")

    for i, r in enumerate(results):
        if "error" in r:
            scored_results.append({**r, "lenient_score": 1, "lenient_type": "error"})
            by_domain[r.get("domain", "?")].append(1)
            continue

        is_ambiguous = r.get("complexity") == "Ambiguous"
        if is_ambiguous:
            # Use LLM judge for all ambiguous (65 intents — cost ~$0.02)
            judgment = judge_response(
                r.get("input_text", ""),
                r.get("ground_truth", {}),
                r.get("parsed"),
            )
            score = judgment.get("score", 1)
            judge_count += 1
            scored_results.append({
                **r,
                "lenient_score": score,
                "lenient_type": "deepseek_judge",
                "judge_feedback": judgment.get("feedback", ""),
            })
        else:
            match = semantic_key_match(r.get("parsed"), r.get("ground_truth", {}))
            score = 1 + match * 4
            semantic_count += 1
            scored_results.append({
                **r,
                "lenient_match": round(match, 2),
                "lenient_score": round(score, 1),
                "lenient_type": "semantic_key",
            })

        dom = r.get("domain", "?")
        by_domain[dom].append(score)

        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(results)}] scored... (judge: {judge_count}, semantic: {semantic_count})")

    # Summarize
    all_scores = [s["lenient_score"] for s in scored_results if s.get("lenient_score") is not None]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

    print(f"\n{'='*60}")
    print(f"  DeepSeek-V4 Pro Lenient Scoring Results")
    print(f"{'='*60}")
    print(f"  Overall Avg: {avg_score:.2f}/5.00")
    print(f"  Judge calls: {judge_count}, Semantic: {semantic_count}")
    print(f"  By Domain:")

    domain_avgs = {}
    for dom, scores in sorted(by_domain.items()):
        short = domain_short.get(dom, dom)
        valid_scores = [s for s in scores if s is not None]
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        domain_avgs[short] = round(avg, 2)
        print(f"    {short:<20} Avg: {avg:.2f}/5.00  (n={len(valid_scores)})")

    output = {
        "model": "DeepSeek-V4 Pro",
        "summary": {
            **data.get("summary", {}),
            "lenient_avg_score": round(avg_score, 2),
            "by_domain_lenient": domain_avgs,
        },
        "results": scored_results,
    }

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n💾 Saved: {OUTPUT}")

    # Print the table-ready data
    print(f"\n{'='*60}")
    print(f"  TABLE-READY: Domain Semantic Scores (1-5)")
    print(f"{'='*60}")
    for dom in ["Provisioning", "Scaling", "Conflict"]:
        print(f"    {dom:<20} {domain_avgs.get(dom, 'N/A'):.2f}")


if __name__ == "__main__":
    main()
