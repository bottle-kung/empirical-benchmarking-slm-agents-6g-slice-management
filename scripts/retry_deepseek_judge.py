#!/usr/bin/env python3
"""Retry failed judge calls from deepseek_lenient_scored.json"""
import json, time, os
from collections import defaultdict
from openai import OpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

INPUT = "/jupyter_workspace/local/ai_agent/slm_6g_eval/results/deepseek_lenient_scored.json"

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

Return ONLY valid JSON:
{{"score": <1-5>, "feedback": "<brief>", "action_understood": <true/false>, "reasonable_params": <true/false>}}"""


def judge_response(intent_text, ground_truth, agent_parsed, retries=5):
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
            print(f"    Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(3 ** attempt)
    return None


ACTION_SYNONYMS_FB = {
    "create": ["create", "set", "setup", "provision", "deploy", "initialize", "add", "new"],
    "scale": ["scale", "increase", "boost", "expand", "grow", "upscale", "scale_up"],
    "scale_down": ["scale_down", "decrease", "reduce", "shrink", "downsize"],
    "resolve": ["resolve", "prioritize", "preempt", "ensure", "handle", "manage"],
    "terminate": ["terminate", "stop", "disable", "remove", "delete", "kill"],
    "reallocate": ["reallocate", "redistribute", "rebalance", "shift", "move"],
}
SLICE_SYNONYMS_FB = {
    "embb": ["embb", "enhanced mobile broadband", "broadband", "high bandwidth"],
    "urllc": ["urllc", "ultra reliable", "low latency", "ultra-reliable"],
    "mmtc": ["mmtc", "massive iot", "massive machine", "iot"],
}


def semantic_key_match_fallback(parsed, truth):
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
            for canonical, synonyms in ACTION_SYNONYMS_FB.items():
                if expected_str in synonyms and parsed_val in synonyms:
                    score += 0.7
                    break
        elif key == "slice_type":
            for canonical, synonyms in SLICE_SYNONYMS_FB.items():
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
    with open(INPUT) as f:
        data = json.load(f)

    results = data["results"]
    domain_short = {
        "Slicing Provisioning": "Provisioning",
        "Scaling Request": "Scaling",
        "Conflict Resolution": "Conflict",
    }

    # Find entries with None scores
    failed = [(i, r) for i, r in enumerate(results) if r.get("lenient_score") is None]
    print(f"Found {len(failed)} failed judge calls\n")

    fixed = 0
    for idx, r in failed:
        intent_id = r.get("intent_id", "?")
        domain = r.get("domain", "?")
        print(f"  [{idx}] {intent_id} ({domain_short.get(domain, domain)})...", end=" ")

        judgment = judge_response(
            r.get("input_text", ""),
            r.get("ground_truth", {}),
            r.get("parsed"),
        )

        if judgment and judgment.get("score") is not None:
            results[idx]["lenient_score"] = judgment["score"]
            results[idx]["judge_feedback"] = judgment.get("feedback", "")
            results[idx]["action_understood"] = judgment.get("action_understood", False)
            fixed += 1
            print(f"→ {judgment['score']}/5 ✅")
        else:
            # Fallback: use semantic key match (inline)
            parsed = r.get("parsed")
            truth = r.get("ground_truth", {})
            if not parsed:
                match = 0.0
            else:
                match = semantic_key_match_fallback(parsed, truth)
            score = 1 + match * 4
            results[idx]["lenient_score"] = round(score, 1)
            results[idx]["lenient_type"] = "semantic_key_fallback"
            print(f"→ fallback {score:.1f}/5 ⚠️")

        time.sleep(0.3)  # Rate limit

    print(f"\nFixed: {fixed}/{len(failed)}")

    # Recompute summary
    by_domain = defaultdict(list)
    for r in results:
        score = r.get("lenient_score")
        if score is not None:
            by_domain[r.get("domain", "?")].append(score)

    all_scores = [s for s in [r.get("lenient_score") for r in results] if s is not None]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

    domain_avgs = {}
    print(f"\n{'='*60}")
    print(f"  UPDATED DeepSeek-V4 Pro Lenient Scores")
    print(f"{'='*60}")
    print(f"  Overall Avg: {avg_score:.2f}/5.00")
    print(f"  By Domain:")
    for dom, scores in sorted(by_domain.items()):
        short = domain_short.get(dom, dom)
        avg = sum(scores) / len(scores)
        domain_avgs[short] = round(avg, 2)
        print(f"    {short:<20} Avg: {avg:.2f}/5.00  (n={len(scores)})")

    data["summary"]["lenient_avg_score"] = round(avg_score, 2)
    data["summary"]["by_domain_lenient"] = domain_avgs

    with open(INPUT, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n  TABLE-READY:")
    for dom in ["Provisioning", "Scaling", "Conflict"]:
        print(f"    {dom:<20} {domain_avgs.get(dom, 'N/A'):.2f}")


if __name__ == "__main__":
    main()
