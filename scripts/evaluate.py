#!/usr/bin/env python3
"""
SLM Agent Evaluation Engine
============================
Evaluates SLM/LLM responses against the ground truth benchmark.

Metrics:
  1. Format Accuracy: Can the model produce valid JSON?
  2. Parameter Exact Match: Do the key-values match ground truth?
  3. LLM-as-a-Judge: For ambiguous intents (no single correct answer)

Usage:
  python scripts/evaluate.py \
    --dataset dataset/intents.jsonl \
    --model gpt-4o \
    --provider openai \
    --output results/gpt-4o_results.json \
    --iterations 5
"""

import json
import time
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class EvaluationResult:
    intent_id: str
    domain: str
    complexity: str
    model_name: str
    iteration: int
    
    # Input / Output
    input_text: str
    ground_truth: dict
    model_output_raw: str
    model_output_parsed: Optional[dict] = None
    
    # Metrics
    format_valid: bool = False
    format_error: Optional[str] = None
    parameter_match: float = 0.0  # 0.0 - 1.0
    matched_keys: list = field(default_factory=list)
    mismatched_keys: list = field(default_factory=list)
    missing_keys: list = field(default_factory=list)
    extra_keys: list = field(default_factory=list)
    
    # Timing
    inference_time_ms: float = 0.0
    
    # LLM-as-Judge score (for ambiguous intents)
    reasoning_score: Optional[float] = None
    reasoning_feedback: Optional[str] = None


class FormatValidator:
    """Check if model output is valid JSON/YAML."""
    
    @staticmethod
    def validate(text: str) -> tuple[bool, Optional[dict], Optional[str]]:
        """
        Try to parse model output as JSON.
        Handles common issues: markdown code blocks, trailing commas, etc.
        """
        if not text or not text.strip():
            return False, None, "Empty output"
        
        # Strip markdown code blocks if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence
            cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
            # Remove closing fence
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        
        # Try direct JSON parse
        try:
            parsed = json.loads(cleaned)
            return True, parsed, None
        except json.JSONDecodeError as e:
            pass
        
        # Try extracting JSON from text (model might wrap in explanation)
        import re
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, cleaned, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                return True, parsed, None
            except json.JSONDecodeError:
                continue
        
        return False, None, f"Could not parse JSON: {str(e) if 'e' in dir() else 'no valid JSON object found'}"


class ParameterMatcher:
    """Compare model output parameters against ground truth."""
    
    @staticmethod
    def flatten_dict(d: dict, parent_key: str = "") -> dict:
        """Flatten nested dict to dot-notation keys."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(ParameterMatcher.flatten_dict(v, new_key))
            elif isinstance(v, list):
                # For lists, compare as JSON strings
                items[new_key] = json.dumps(v, sort_keys=True)
            else:
                items[new_key] = v
        return items
    
    @staticmethod
    def compare(parsed: dict, ground_truth: dict) -> dict:
        """
        Compare model output against ground truth.
        Returns detailed comparison results.
        """
        flat_parsed = ParameterMatcher.flatten_dict(parsed)
        flat_truth = ParameterMatcher.flatten_dict(ground_truth)
        
        all_truth_keys = set(flat_truth.keys())
        all_parsed_keys = set(flat_parsed.keys())
        
        matched = []
        mismatched = []
        
        for key in all_truth_keys:
            if key in flat_parsed:
                parsed_val = flat_parsed[key]
                truth_val = flat_truth[key]
                
                # Normalize comparison
                if isinstance(parsed_val, str) and isinstance(truth_val, str):
                    # Case-insensitive string compare
                    if parsed_val.lower() == truth_val.lower():
                        matched.append(key)
                    else:
                        mismatched.append({
                            "key": key,
                            "expected": truth_val,
                            "got": parsed_val
                        })
                elif str(parsed_val) == str(truth_val):
                    matched.append(key)
                else:
                    mismatched.append({
                        "key": key,
                        "expected": truth_val,
                        "got": parsed_val
                    })
            else:
                mismatched.append({
                    "key": key,
                    "expected": flat_truth[key],
                    "got": "MISSING"
                })
        
        # Keys in parsed but not in truth
        extra = [k for k in all_parsed_keys if k not in all_truth_keys]
        
        total_keys = len(all_truth_keys)
        match_count = len(matched)
        match_ratio = match_count / total_keys if total_keys > 0 else 0.0
        
        return {
            "match_ratio": match_ratio,
            "matched_keys": matched,
            "mismatched_keys": mismatched,
            "missing_keys": [k for k in all_truth_keys if k not in all_parsed_keys],
            "extra_keys": extra,
        }


class LLMJudge:
    """
    Use a strong LLM (GPT-4o) as judge for ambiguous intents.
    Scores 1-5 based on how well the model handled the ambiguous request.
    """
    
    JUDGE_PROMPT = """You are an expert evaluator for a 6G Network Slice Management system. 
You are judging how well an AI agent handled an ambiguous natural-language intent.

**Context**: The agent was given an ambiguous network management request and had to 
produce a JSON response with the appropriate action and parameters.

**Scoring Rubric (1-5):**
- 1: Completely wrong — would cause system failure or severe misconfiguration
- 2: Mostly wrong — incorrect action or critically wrong parameters
- 3: Partially correct — right direction but key parameters missing/wrong
- 4: Good — appropriate action with minor parameter issues
- 5: Excellent — optimal response for the ambiguous situation, all parameters appropriate

**Original Intent:**
{intent_text}

**Agent's JSON Response:**
{agent_response}

**Ground Truth Reference (not the only correct answer):**
{ground_truth}

Evaluate the agent's response. Return ONLY a JSON object with:
- "score": integer 1-5
- "feedback": brief explanation (2-3 sentences)

Example response:
{{"score": 4, "feedback": "Agent correctly identified URLLC slice type but latency target of 10ms is too relaxed for surgical robots — 5ms would be more appropriate."}}"""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider
        self.model = model
    
    def evaluate(self, intent_text: str, agent_response: dict, ground_truth: dict) -> dict:
        """Score an ambiguous intent response. Stub — requires actual API call."""
        # This is a stub. In production, call the LLM API.
        # For now, return a placeholder indicating the method is ready.
        prompt = self.JUDGE_PROMPT.format(
            intent_text=intent_text,
            agent_response=json.dumps(agent_response, indent=2),
            ground_truth=json.dumps(ground_truth, indent=2),
        )
        
        return {
            "status": "stub",
            "message": "LLM-as-a-Judge requires API call to judgment model. Implement _call_judge_api() with your provider.",
            "prompt": prompt,
            "score": None,
            "feedback": None,
        }


class ModelClient:
    """
    Abstract interface for calling different models.
    Supports: Ollama (local), OpenAI API, Anthropic API, and rule-based stub.
    """
    
    SYSTEM_PROMPT = """You are a 6G Core Network Slice Management Agent. 
Your job is to translate natural language intents into JSON commands for network orchestration.

Rules:
1. Respond ONLY with a valid JSON object — no markdown, no explanations, no code fences
2. The JSON MUST contain an "action" field
3. Use these slice types when appropriate: "eMBB" (enhanced Mobile Broadband), "URLLC" (Ultra-Reliable Low Latency), "mMTC" (massive Machine Type Communications)
4. For ambiguous requests, infer the best configuration based on context
5. Include all relevant parameters: latency_ms, bandwidth_gbps, device_density, priority, etc.

Example input: "Create a URLLC slice with 1ms latency for robot control"
Example output: {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 1}"""

    def __init__(self, provider: str, model: str, endpoint: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.endpoint = endpoint or "http://localhost:11434"
    
    def query(self, user_text: str) -> tuple[str, float]:
        """
        Send a prompt to the model. Returns (response_text, inference_time_ms).
        Stub implementation — replace with actual API calls.
        """
        # Stub: return a placeholder
        start = time.time()
        
        if self.provider == "rule_based":
            response, elapsed = self._rule_based(user_text, start)
        elif self.provider == "ollama":
            response, elapsed = self._ollama_query(user_text, start)
        elif self.provider == "openai":
            response, elapsed = self._openai_query(user_text, start)
        elif self.provider == "anthropic":
            response, elapsed = self._anthropic_query(user_text, start)
        else:
            response = json.dumps({"action": "UNSUPPORTED_PROVIDER", "error": f"Provider {self.provider} not implemented"})
            elapsed = (time.time() - start) * 1000
        
        return response, elapsed
    
    def _rule_based(self, text: str, start: float) -> tuple[str, float]:
        """Simple regex-based parser (baseline)."""
        import re
        text_lower = text.lower()
        
        if "embb" in text_lower or "video" in text_lower or "streaming" in text_lower:
            result = {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 1}
        elif "urllc" in text_lower or "latency" in text_lower or "robot" in text_lower:
            result = {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 5}
        elif "mmtc" in text_lower or "iot" in text_lower or "sensor" in text_lower:
            result = {"action": "CREATE", "slice_type": "mMTC", "device_density": 10000}
        elif "scale" in text_lower:
            result = {"action": "SCALE", "target": "upf", "replicas": 3}
        elif "conflict" in text_lower or "preempt" in text_lower:
            result = {"action": "RESOLVE", "terminate_slice": "unknown"}
        else:
            result = {"action": "UNKNOWN"}
        
        elapsed = (time.time() - start) * 1000
        return json.dumps(result), elapsed
    
    def _ollama_query(self, user_text: str, start: float) -> tuple[str, float]:
        """Call Ollama API (stub — implement with requests)."""
        # Stub for Ollama
        result = {
            "action": "STUB",
            "note": f"Ollama call to {self.model} — implement with requests.post()",
            "endpoint": f"{self.endpoint}/api/generate"
        }
        elapsed = (time.time() - start) * 1000
        return json.dumps(result), elapsed
    
    def _openai_query(self, user_text: str, start: float) -> tuple[str, float]:
        """Call OpenAI API (stub — implement with openai library)."""
        result = {
            "action": "STUB",
            "note": f"OpenAI call to {self.model} — implement with openai client",
        }
        elapsed = (time.time() - start) * 1000
        return json.dumps(result), elapsed
    
    def _anthropic_query(self, user_text: str, start: float) -> tuple[str, float]:
        """Call Anthropic API (stub — implement with anthropic library)."""
        result = {
            "action": "STUB",
            "note": f"Anthropic call to {self.model} — implement with anthropic client",
        }
        elapsed = (time.time() - start) * 1000
        return json.dumps(result), elapsed


class EvaluationRunner:
    """Main evaluation loop."""
    
    def __init__(
        self,
        dataset_path: str,
        model_client: ModelClient,
        model_name: str,
        iterations: int = 5,
        warmup: int = 1,
    ):
        self.dataset_path = Path(dataset_path)
        self.client = model_client
        self.model_name = model_name
        self.iterations = iterations
        self.warmup = warmup
        self.results: list[EvaluationResult] = []
    
    def load_dataset(self) -> list[dict]:
        """Load intents from JSONL file."""
        intents = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    intents.append(json.loads(line))
        return intents
    
    def run(self) -> list[EvaluationResult]:
        """Execute the full evaluation loop."""
        intents = self.load_dataset()
        total_runs = len(intents) * (self.warmup + self.iterations)
        
        print(f"\n{'='*60}")
        print(f"Evaluation: {self.model_name}")
        print(f"Dataset: {len(intents)} intents")
        print(f"Iterations: {self.warmup} warmup + {self.iterations} measured = {self.warmup + self.iterations} total per intent")
        print(f"Total runs: {total_runs}")
        print(f"{'='*60}\n")
        
        run_idx = 0
        for intent in intents:
            for iteration in range(self.warmup + self.iterations):
                run_idx += 1
                is_warmup = iteration < self.warmup
                iter_label = f"warmup-{iteration+1}" if is_warmup else f"run-{iteration - self.warmup + 1}"
                
                print(f"[{run_idx}/{total_runs}] {intent['intent_id']} ({intent['complexity']}) — {iter_label}")
                
                # Query model
                raw_output, inference_time = self.client.query(intent["input_text"])
                
                # Validate format
                is_valid, parsed, fmt_error = FormatValidator.validate(raw_output)
                
                # Compare parameters if format is valid
                if is_valid and parsed:
                    comparison = ParameterMatcher.compare(parsed, intent["ground_truth"])
                    param_match = comparison["match_ratio"]
                    matched = comparison["matched_keys"]
                    mismatched = comparison["mismatched_keys"]
                    missing = comparison["missing_keys"]
                    extra = comparison["extra_keys"]
                else:
                    param_match = 0.0
                    matched = []
                    mismatched = []
                    missing = []
                    extra = []
                
                # Skip warmup results
                if is_warmup:
                    print(f"  ⏭️  Warmup (discarded)")
                    continue
                
                result = EvaluationResult(
                    intent_id=intent["intent_id"],
                    domain=intent["domain"],
                    complexity=intent["complexity"],
                    model_name=self.model_name,
                    iteration=iteration - self.warmup + 1,
                    input_text=intent["input_text"],
                    ground_truth=intent["ground_truth"],
                    model_output_raw=raw_output,
                    model_output_parsed=parsed,
                    format_valid=is_valid,
                    format_error=fmt_error,
                    parameter_match=param_match,
                    matched_keys=matched,
                    mismatched_keys=mismatched,
                    missing_keys=missing,
                    extra_keys=extra,
                    inference_time_ms=inference_time,
                )
                
                self.results.append(result)
                
                status = "✅" if is_valid and param_match == 1.0 else ("⚠️" if is_valid else "❌")
                print(f"  {status} Format: {'VALID' if is_valid else 'INVALID'}, "
                      f"Match: {param_match:.1%}, Time: {inference_time:.0f}ms")
        
        return self.results
    
    def summary(self) -> dict:
        """Generate summary statistics."""
        if not self.results:
            return {"error": "No results to summarize"}
        
        total = len(self.results)
        format_valid_count = sum(1 for r in self.results if r.format_valid)
        perfect_match = sum(1 for r in self.results if r.parameter_match == 1.0)
        
        avg_param_match = sum(r.parameter_match for r in self.results) / total if total > 0 else 0
        avg_inference_ms = sum(r.inference_time_ms for r in self.results) / total if total > 0 else 0
        
        # By domain
        by_domain = defaultdict(lambda: {"total": 0, "format_valid": 0, "perfect_match": 0, "sum_param_match": 0.0})
        for r in self.results:
            d = by_domain[r.domain]
            d["total"] += 1
            d["format_valid"] += 1 if r.format_valid else 0
            d["perfect_match"] += 1 if r.parameter_match == 1.0 else 0
            d["sum_param_match"] += r.parameter_match
        
        # By complexity
        by_complexity = defaultdict(lambda: {"total": 0, "format_valid": 0, "perfect_match": 0, "sum_param_match": 0.0})
        for r in self.results:
            c = by_complexity[r.complexity]
            c["total"] += 1
            c["format_valid"] += 1 if r.format_valid else 0
            c["perfect_match"] += 1 if r.parameter_match == 1.0 else 0
            c["sum_param_match"] += r.parameter_match
        
        summary = {
            "model": self.model_name,
            "total_evaluations": total,
            "overall": {
                "format_accuracy": format_valid_count / total if total > 0 else 0,
                "parameter_exact_match_rate": perfect_match / total if total > 0 else 0,
                "avg_parameter_match": avg_param_match,
                "avg_inference_time_ms": avg_inference_ms,
            },
            "by_domain": {},
            "by_complexity": {},
        }
        
        for domain, stats in sorted(by_domain.items()):
            summary["by_domain"][domain] = {
                "total": stats["total"],
                "format_accuracy": stats["format_valid"] / stats["total"],
                "avg_parameter_match": stats["sum_param_match"] / stats["total"],
            }
        
        for complexity, stats in sorted(by_complexity.items()):
            summary["by_complexity"][complexity] = {
                "total": stats["total"],
                "format_accuracy": stats["format_valid"] / stats["total"],
                "avg_parameter_match": stats["sum_param_match"] / stats["total"],
            }
        
        return summary
    
    def save_results(self, output_path: str) -> None:
        """Save detailed results to JSON."""
        output = {
            "summary": self.summary(),
            "results": [asdict(r) for r in self.results],
        }
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {path}")
    
    def print_summary(self) -> None:
        """Print summary to console."""
        s = self.summary()
        if "error" in s:
            print(s["error"])
            return
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.model_name}")
        print(f"{'='*60}")
        print(f"\n📊 Overall (n={s['total_evaluations']}):")
        print(f"   Format Accuracy:        {s['overall']['format_accuracy']:.1%}")
        print(f"   Parameter Exact Match:  {s['overall']['parameter_exact_match_rate']:.1%}")
        print(f"   Avg Parameter Match:    {s['overall']['avg_parameter_match']:.1%}")
        print(f"   Avg Inference Time:     {s['overall']['avg_inference_time_ms']:.0f} ms")
        
        print(f"\n📂 By Domain:")
        for domain, stats in s["by_domain"].items():
            print(f"   {domain}:")
            print(f"      Format: {stats['format_accuracy']:.1%}  |  Param Match: {stats['avg_parameter_match']:.1%}")
        
        print(f"\n🎯 By Complexity:")
        for complexity, stats in s["by_complexity"].items():
            print(f"   {complexity}:")
            print(f"      Format: {stats['format_accuracy']:.1%}  |  Param Match: {stats['avg_parameter_match']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="SLM Agent Evaluation Engine")
    parser.add_argument("--dataset", default="dataset/intents.jsonl", help="Path to intent dataset (.jsonl)")
    parser.add_argument("--model", default="gpt-4o", help="Model name/ID")
    parser.add_argument("--provider", default="openai", choices=["rule_based", "ollama", "openai", "anthropic"], help="Model provider")
    parser.add_argument("--endpoint", default=None, help="API endpoint (for Ollama)")
    parser.add_argument("--output", default=None, help="Output path for results JSON")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per intent")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (discarded)")
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    
    dataset_path = args.dataset
    if not Path(dataset_path).is_absolute():
        dataset_path = str(project_root / dataset_path)
    
    output_path = args.output
    if output_path:
        if not Path(output_path).is_absolute():
            output_path = str(project_root / output_path)
    else:
        output_path = str(project_root / "results" / f"{args.model.replace('/', '_')}_results.json")
    
    # Setup
    client = ModelClient(
        provider=args.provider,
        model=args.model,
        endpoint=args.endpoint,
    )
    
    # Run evaluation
    runner = EvaluationRunner(
        dataset_path=dataset_path,
        model_client=client,
        model_name=args.model,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    
    runner.run()
    runner.print_summary()
    runner.save_results(output_path)


if __name__ == "__main__":
    main()
