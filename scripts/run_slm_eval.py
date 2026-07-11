#!/usr/bin/env python3
"""
Real Ollama Client for SLM Evaluation
======================================
Replaces the stub ModelClient in evaluate.py with real Ollama API calls.

Usage:
  python scripts/run_slm_eval.py --model qwen2.5:1.5b --limit 10 --iterations 2
"""

import json
import time
import requests
import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import EvaluationRunner

OLLAMA_URL = "http://localhost:11434"

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


class RealOllamaClient:
    """Real Ollama client for SLM inference."""
    
    def __init__(self, model: str, endpoint: str = OLLAMA_URL):
        self.model = model
        self.endpoint = endpoint.rstrip('/')
        self.timeout = 600
    
    def query(self, user_text: str) -> tuple[str, float]:
        """
        Send prompt to Ollama and get response.
        Returns (response_text, inference_time_ms).
        """
        t_start = time.time()
        
        payload = {
            "model": self.model,
            "prompt": user_text,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temp for deterministic outputs
                "num_predict": 512,   # Max tokens
            }
        }
        
        try:
            resp = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("response", "")
            elapsed = (time.time() - t_start) * 1000
            return response_text, elapsed
            
        except requests.exceptions.Timeout:
            elapsed = (time.time() - t_start) * 1000
            return json.dumps({"action": "ERROR", "error": "TIMEOUT"}), elapsed
        except Exception as e:
            elapsed = (time.time() - t_start) * 1000
            return json.dumps({"action": "ERROR", "error": str(e)}), elapsed


def main():
    parser = argparse.ArgumentParser(description="Run SLM Evaluation with Ollama")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Ollama model name")
    parser.add_argument("--dataset", default="dataset/intents.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="Number of intents to test")
    parser.add_argument("--iterations", type=int, default=2, help="Iterations per intent")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    parser.add_argument("--output", default=None, help="Output JSON path")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    
    dataset_path = args.dataset
    if not Path(dataset_path).is_absolute():
        dataset_path = str(project_root / dataset_path)
    
    output_path = args.output
    if output_path:
        if not Path(output_path).is_absolute():
            output_path = str(project_root / output_path)
    else:
        safe_name = args.model.replace(":", "_").replace("/", "_")
        output_path = str(project_root / "results" / f"{safe_name}_results.json")
    
    # Create real Ollama client
    client = RealOllamaClient(model=args.model)
    
    # Run evaluation
    print(f"\n🚀 Starting evaluation with {args.model}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Limit: {args.limit} intents, {args.iterations} iterations each")
    print(f"   Warmup: {args.warmup} runs")
    print(f"   Output: {output_path}")
    
    runner = EvaluationRunner(
        dataset_path=dataset_path,
        model_client=client,
        model_name=args.model,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    
    # Override dataset loading to support limit
    all_intents = runner.load_dataset()
    runner.dataset = all_intents[:args.limit]  # Tricky: override the dataset
    
    # Hack: override run method for limited dataset
    original_load = runner.load_dataset
    runner.load_dataset = lambda: all_intents[:args.limit]
    
    runner.run()
    runner.print_summary()
    runner.save_results(output_path)
    
    print(f"\n✅ Evaluation complete for {args.model}!")


if __name__ == "__main__":
    main()
