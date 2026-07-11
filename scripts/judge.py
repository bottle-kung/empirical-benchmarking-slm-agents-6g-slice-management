#!/usr/bin/env python3
"""
LLM-as-a-Judge: Reasoning Capability Evaluation
================================================
For ambiguous intents where there's no single correct answer,
use a strong LLM (GPT-4o) as a judge to score the quality
of the model's reasoning and configuration choices.

Usage:
  python scripts/judge.py \
    --results results/qwen2.5-1.5b_results.json \
    --judge-model gpt-4o \
    --output results/qwen2.5-1.5b_judged.json
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Try to import from evaluate.py
try:
    from scripts.evaluate import LLMJudge as _BaseJudge
except ImportError:
    _BaseJudge = None

JUDGE_PROMPT = """You are an expert evaluator for a 6G Core Network Slice Management system.
Your task is to judge how well an AI agent handled an AMBIGUOUS natural-language intent.

**Context**: The agent was given a network management request where the user did NOT 
specify exact technical parameters. The agent had to infer the best configuration.

**Scoring Rubric (1-5):**
- 1: Completely wrong — would cause system failure or severe misconfiguration
- 2: Mostly wrong — incorrect action or critically wrong parameters
- 3: Partially correct — right intention but key parameters missing/wrong
- 4: Good — appropriate action with minor parameter issues
- 5: Excellent — optimal response given the ambiguity, all parameters appropriate

**Key evaluation dimensions:**
1. Did the agent choose the correct slice type (eMBB / URLLC / mMTC)?
2. Are the latency/bandwidth/device_density parameters reasonable for the context?
3. Did the agent handle edge cases (e.g., emergency situations, mixed requirements)?
4. Is the priority assignment appropriate?
5. For composite requests, did the agent break down requirements correctly?

---

**Original User Intent:**
{intent_text}

**AI Agent's JSON Response:**
{agent_response}

**Reference Ground Truth (ONE possible correct answer — the agent's answer may differ and still be correct):**
{ground_truth}

---

Evaluate the agent's response on a scale of 1-5. 
Return ONLY a valid JSON object (no markdown, no explanation outside JSON):

{{
  "score": <integer 1-5>,
  "feedback": "<2-3 sentence explanation of the score>",
  "strengths": ["<what was good>"],
  "weaknesses": ["<what could be improved>"],
  "suggested_correction": "<if score < 4, what the agent should have done differently>"
}}

Example:
{{
  "score": 4,
  "feedback": "Agent correctly identified URLLC slice type for surgical robots. Latency target of 5ms is appropriate. However, it missed the reliability requirement (should be 99.9999%).",
  "strengths": ["Correct slice type selection", "Appropriate latency target"],
  "weaknesses": ["Missing reliability parameter"],
  "suggested_correction": "Add reliability: '99.9999%' to the response for surgical applications."
}}"""


class LLMJudgeClient:
    """Client for calling the judging LLM."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider
        self.model = model
    
    def evaluate(self, intent_text: str, agent_response: dict, ground_truth: dict) -> dict:
        """
        Evaluate an agent's response to an ambiguous intent.
        
        In production, this calls the actual LLM API.
        For now, returns the prompt ready for sending.
        """
        prompt = JUDGE_PROMPT.format(
            intent_text=intent_text,
            agent_response=json.dumps(agent_response, indent=2, ensure_ascii=False),
            ground_truth=json.dumps(ground_truth, indent=2, ensure_ascii=False),
        )
        
        # Stub — in production, call the API and parse the response
        return {
            "status": "ready_for_api",
            "judge_model": self.model,
            "judge_provider": self.provider,
            "prompt": prompt,
            "score": None,
            "feedback": None,
            "strengths": [],
            "weaknesses": [],
            "suggested_correction": None,
        }


class BatchJudge:
    """
    Process evaluation results and judge ONLY ambiguous intents.
    """
    
    def __init__(self, results_path: str, judge_client: LLMJudgeClient):
        self.results_path = Path(results_path)
        self.judge = judge_client
        self.judgments = []
    
    def load_results(self) -> dict:
        with open(self.results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def filter_ambiguous(self, results_data: dict) -> list:
        """Filter to only ambiguous intents with valid parsed output."""
        ambiguous = []
        for r in results_data.get("results", []):
            if r["complexity"] == "Ambiguous" and r["model_output_parsed"] is not None:
                ambiguous.append(r)
        return ambiguous
    
    def run(self) -> list:
        """Judge all ambiguous intent responses."""
        data = self.load_results()
        ambiguous = self.filter_ambiguous(data)
        
        print(f"\n{'='*60}")
        print(f"LLM-as-a-Judge Evaluation")
        print(f"File: {self.results_path.name}")
        print(f"Judge Model: {self.judge.model}")
        print(f"Ambiguous intents to judge: {len(ambiguous)}")
        print(f"{'='*60}\n")
        
        for i, result in enumerate(ambiguous):
            intent_id = result["intent_id"]
            input_text = result["input_text"]
            ground_truth = result["ground_truth"]
            agent_response = result["model_output_parsed"]
            
            print(f"[{i+1}/{len(ambiguous)}] Judging {intent_id}...")
            
            judgment = self.judge.evaluate(
                intent_text=input_text,
                agent_response=agent_response,
                ground_truth=ground_truth,
            )
            
            judgment["intent_id"] = intent_id
            judgment["input_text"] = input_text
            judgment["ground_truth"] = ground_truth
            judgment["agent_response"] = agent_response
            
            self.judgments.append(judgment)
            
            score_str = f"Score: {judgment['score']}" if judgment['score'] else "Score: [API call required]"
            print(f"  {score_str}")
        
        return self.judgments
    
    def summary(self) -> dict:
        """Generate summary of judgments."""
        if not self.judgments:
            return {"error": "No judgments to summarize", "count": 0}
        
        judged = [j for j in self.judgments if j.get("score") is not None]
        total = len(self.judgments)
        
        if not judged:
            return {
                "total_ambiguous": total,
                "judged": 0,
                "note": "All judgments are stubs — implement API call to judge model",
            }
        
        scores = [j["score"] for j in judged]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        score_dist = {}
        for s in range(1, 6):
            score_dist[f"score_{s}"] = scores.count(s)
        
        return {
            "total_ambiguous": total,
            "judged": len(judged),
            "avg_reasoning_score": round(avg_score, 2),
            "score_distribution": score_dist,
            "by_domain": self._by_domain_summary(judged),
        }
    
    def _by_domain_summary(self, judged: list) -> dict:
        from collections import defaultdict
        by_domain = defaultdict(list)
        for j in judged:
            # Find domain from original result
            by_domain[j.get("domain", "Unknown")].append(j.get("score", 0))
        
        return {
            domain: {
                "count": len(scores),
                "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
            }
            for domain, scores in sorted(by_domain.items())
        }
    
    def save(self, output_path: str) -> None:
        output = {
            "summary": self.summary(),
            "judgments": self.judgments,
        }
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Judgments saved to: {path}")
    
    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n{'='*60}")
        print(f"JUDGMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total ambiguous intents: {s.get('total_ambiguous', 0)}")
        
        if s.get("judged", 0) == 0:
            print("⚠️  No scores yet — implement API call to judge model")
            return
        
        print(f"Judged: {s['judged']}")
        print(f"Avg Reasoning Score: {s['avg_reasoning_score']:.2f}/5.00")
        print(f"\nScore Distribution:")
        for key, count in sorted(s.get("score_distribution", {}).items()):
            bar = "█" * count
            print(f"  {key}: {count:3d} {bar}")
        
        if "by_domain" in s:
            print(f"\nBy Domain:")
            for domain, stats in s["by_domain"].items():
                print(f"  {domain}: avg {stats['avg_score']:.2f} (n={stats['count']})")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge for Ambiguous Intents")
    parser.add_argument("--results", required=True, help="Path to evaluation results JSON")
    parser.add_argument("--judge-model", default="gpt-4o", help="Judge model name")
    parser.add_argument("--judge-provider", default="openai", help="Judge model provider")
    parser.add_argument("--output", default=None, help="Output path for judgments JSON")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    
    results_path = args.results
    if not Path(results_path).is_absolute():
        results_path = str(project_root / results_path)
    
    output_path = args.output
    if output_path:
        if not Path(output_path).is_absolute():
            output_path = str(project_root / output_path)
    else:
        stem = Path(results_path).stem
        output_path = str(project_root / "results" / f"{stem}_judged.json")
    
    judge_client = LLMJudgeClient(
        provider=args.judge_provider,
        model=args.judge_model,
    )
    
    batch = BatchJudge(results_path, judge_client)
    batch.run()
    batch.print_summary()
    batch.save(output_path)


if __name__ == "__main__":
    main()
