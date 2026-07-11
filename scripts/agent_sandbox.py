#!/usr/bin/env python3
"""
6G Agent Sandbox — LangGraph Workflow + API Translator + Monitoring Logger
==========================================================================
Orchestrates the full intent → agent → core → metrics pipeline.

Architecture:
  Intent Input → [Agent Node] → [Validate Node] → [Translate Node] → [Execute Node] → [Log Node]
                     ↑                                                                    |
                     └──────────────────── (feedback loop) ←─────────────────────────────┘

Modes:
  --mode simulation  : Uses SimulatedNetworkCore (no real K8s needed)
  --mode production  : Uses real kubectl + Free5GC API (requires cluster)
"""

import json
import time
import sys
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.simulated_core import SimulatedNetworkCore, get_core


# ══════════════════════════════════════════════════════════════════════
# Agent Sandbox — Simple State Machine (LangGraph-inspired)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AgentState:
    """State that flows through the agent pipeline."""
    intent_id: str
    input_text: str
    ground_truth: dict = field(default_factory=dict)
    
    # Agent output
    model_output_raw: str = ""
    model_output_parsed: Optional[dict] = None
    model_name: str = "unknown"
    
    # Validation
    format_valid: bool = False
    validation_error: Optional[str] = None
    
    # Translation
    translated_command: Optional[dict] = None
    
    # Execution
    execution_result: Optional[dict] = None
    execution_success: bool = False
    
    # Metrics
    t_inference_ms: float = 0.0
    t_translation_ms: float = 0.0
    t_execution_ms: float = 0.0
    convergence_time_ms: float = 0.0
    
    # System metrics snapshot
    metrics_before: Optional[dict] = None
    metrics_after: Optional[dict] = None
    
    # Energy (simulated)
    energy_joules: float = 0.0
    
    def to_csv_row(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "intent_id": self.intent_id,
            "model_name": self.model_name,
            "input_text": self.input_text,
            "format_valid": self.format_valid,
            "execution_success": self.execution_success,
            "t_inference_ms": round(self.t_inference_ms, 2),
            "t_translation_ms": round(self.t_translation_ms, 2),
            "t_execution_ms": round(self.t_execution_ms, 2),
            "convergence_time_ms": round(self.convergence_time_ms, 2),
            "energy_joules": round(self.energy_joules, 4),
            "cpu_before_pct": self.metrics_before["system"]["cpu_usage_pct"] if self.metrics_before else 0,
            "cpu_after_pct": self.metrics_after["system"]["cpu_usage_pct"] if self.metrics_after else 0,
            "bw_before_gbps": self.metrics_before["system"]["bandwidth_usage_gbps"] if self.metrics_before else 0,
            "bw_after_gbps": self.metrics_after["system"]["bandwidth_usage_gbps"] if self.metrics_after else 0,
        }


# ══════════════════════════════════════════════════════════════════════
# Node 1: Intent Receiver
# ══════════════════════════════════════════════════════════════════════

def node_receive_intent(state: AgentState, core: SimulatedNetworkCore) -> AgentState:
    """Receive and log the intent."""
    print(f"\n{'─'*50}")
    print(f"📥 INTENT RECEIVED: {state.intent_id}")
    print(f"   \"{state.input_text[:80]}{'...' if len(state.input_text) > 80 else ''}\"")
    return state


# ══════════════════════════════════════════════════════════════════════
# Node 2: LLM Processing
# ══════════════════════════════════════════════════════════════════════

def node_process_with_llm(state: AgentState, model_client=None) -> AgentState:
    """
    Send intent to the LLM/SLM model and capture response.
    In simulation mode, we simulate the model output.
    """
    t_start = time.time()
    
    if model_client and hasattr(model_client, 'query'):
        raw_output, inference_time = model_client.query(state.input_text)
        state.model_output_raw = raw_output
        state.t_inference_ms = inference_time
    else:
        # Simulation: produce a plausible agent response based on ground truth
        state.model_output_raw = json.dumps(state.ground_truth)
        state.t_inference_ms = 0.0  # Simulation
    
    state.t_inference_ms = (time.time() - t_start) * 1000
    
    # Try to parse as JSON
    try:
        state.model_output_parsed = json.loads(state.model_output_raw)
    except json.JSONDecodeError:
        # Try extracting JSON from text
        import re
        matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', state.model_output_raw, re.DOTALL)
        for match in matches:
            try:
                state.model_output_parsed = json.loads(match)
                break
            except json.JSONDecodeError:
                continue
    
    print(f"🤖 LLM RESPONSE: {json.dumps(state.model_output_parsed, ensure_ascii=False) if state.model_output_parsed else 'PARSE FAILED'}")
    print(f"   Inference time: {state.t_inference_ms:.1f} ms")
    
    return state


# ══════════════════════════════════════════════════════════════════════
# Node 3: JSON Validation
# ══════════════════════════════════════════════════════════════════════

def node_validate_json(state: AgentState) -> AgentState:
    """Validate that the model output is valid JSON with required fields."""
    
    if state.model_output_parsed is None:
        state.format_valid = False
        state.validation_error = "Could not parse model output as JSON"
        print(f"❌ VALIDATION: FAILED — {state.validation_error}")
        return state
    
    if "action" not in state.model_output_parsed:
        state.format_valid = False
        state.validation_error = "Missing required 'action' field"
        print(f"❌ VALIDATION: FAILED — {state.validation_error}")
        return state
    
    state.format_valid = True
    print(f"✅ VALIDATION: PASSED — action={state.model_output_parsed['action']}")
    
    return state


# ══════════════════════════════════════════════════════════════════════
# Node 4: API Translator (JSON → Core Command)
# ══════════════════════════════════════════════════════════════════════

def node_translate_to_command(state: AgentState) -> AgentState:
    """
    Translate the agent's JSON output into a core network command.
    This is the bridge between AI output and network operations.
    """
    t_start = time.time()
    
    if not state.format_valid or not state.model_output_parsed:
        state.translated_command = None
        state.t_translation_ms = 0
        return state
    
    parsed = state.model_output_parsed
    action = parsed.get("action", "").upper()
    
    command = {"action": action, "params": parsed}
    
    # Map to specific core operations
    if action in ("CREATE", "CONFIGURE"):
        command["core_operation"] = "create_slice"
        command["params"] = {
            "slice_type": parsed.get("slice_type", "eMBB"),
            "latency_ms": parsed.get("latency_ms", 10),
            "bandwidth_gbps": parsed.get("bandwidth_gbps", 1.0),
            "device_density": parsed.get("device_density", 0),
            "priority": parsed.get("priority", "normal"),
        }
    elif action in ("SCALE", "SCALE_UP", "SCALE_DOWN", "OPTIMIZE"):
        command["core_operation"] = "scale_slice"
        command["params"] = {
            "slice_id": parsed.get("slice_id", parsed.get("target", "")),
            "upf_replicas": parsed.get("replicas", parsed.get("upf_replicas")),
            "bandwidth_gbps": parsed.get("bandwidth_gbps"),
            "target": parsed.get("target", "upf"),
        }
    elif action in ("RESOLVE", "PREEMPT", "TERMINATE", "REALLOCATE"):
        command["core_operation"] = "resolve_conflict"
        command["params"] = dict(parsed)
    
    state.translated_command = command
    state.t_translation_ms = (time.time() - t_start) * 1000
    
    print(f"🔄 TRANSLATED: {command['core_operation']} → params={json.dumps(command['params'], ensure_ascii=False)}")
    print(f"   Translation time: {state.t_translation_ms:.2f} ms")
    
    return state


# ══════════════════════════════════════════════════════════════════════
# Node 5: Execute on Core Network
# ══════════════════════════════════════════════════════════════════════

def node_execute(state: AgentState, core: SimulatedNetworkCore) -> AgentState:
    """Execute the translated command on the core network."""
    
    if not state.translated_command or not state.format_valid:
        state.execution_success = False
        state.execution_result = {"error": "No valid command to execute"}
        return state
    
    # Snapshot metrics BEFORE execution
    state.metrics_before = core.get_metrics()
    
    t_start = time.time()
    cmd = state.translated_command
    operation = cmd["core_operation"]
    params = cmd["params"]
    
    try:
        if operation == "create_slice":
            result = core.create_slice(params)
        elif operation == "scale_slice":
            result = core.scale_slice(params)
        elif operation == "resolve_conflict":
            result = core.resolve_conflict(params)
        else:
            result = {"success": False, "error": f"Unknown operation: {operation}"}
        
        state.execution_result = result
        state.execution_success = result.get("success", False)
        state.t_execution_ms = result.get("convergence_time_ms", (time.time() - t_start) * 1000)
        
    except Exception as e:
        state.execution_result = {"success": False, "error": str(e)}
        state.execution_success = False
        state.t_execution_ms = (time.time() - t_start) * 1000
    
    # Snapshot metrics AFTER execution
    state.metrics_after = core.get_metrics()
    
    # Calculate total convergence time
    state.convergence_time_ms = state.t_inference_ms + state.t_translation_ms + state.t_execution_ms
    
    # Simulate energy consumption
    # Rule of thumb: ~0.1 J per inference for tiny models, ~1 J for mid, ~10 J for cloud
    energy_map = {
        "rule_based": 0.001,
        "qwen2.5-1.5b": 0.05,
        "gemma2-2b": 0.08,
        "llama3.2-3b": 0.12,
        "mistral-7b": 0.5,
        "llama3.1-8b": 0.6,
        "qwen2.5-7b": 0.5,
        "phi3-14b": 2.0,
        "gpt-4o": 5.0,
    }
    state.energy_joules = energy_map.get(state.model_name, 1.0)
    
    status = "✅" if state.execution_success else "❌"
    print(f"{status} EXECUTED: {operation} → success={state.execution_success}")
    print(f"   T_execution: {state.t_execution_ms:.1f} ms")
    print(f"   T_convergence: {state.convergence_time_ms:.1f} ms")
    print(f"   Energy: {state.energy_joules:.3f} J")
    
    return state


# ══════════════════════════════════════════════════════════════════════
# Node 6: Metrics Logger
# ══════════════════════════════════════════════════════════════════════

class MetricsLogger:
    """Logs metrics to CSV for later analysis."""
    
    CSV_HEADERS = [
        "timestamp", "intent_id", "model_name", "input_text",
        "format_valid", "execution_success",
        "t_inference_ms", "t_translation_ms", "t_execution_ms", "convergence_time_ms",
        "energy_joules",
        "cpu_before_pct", "cpu_after_pct",
        "bw_before_gbps", "bw_after_gbps",
    ]
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_csv()
    
    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                writer.writeheader()
    
    def log(self, state: AgentState):
        row = state.to_csv_row()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writerow(row)


def node_log_metrics(state: AgentState, logger: MetricsLogger) -> AgentState:
    """Log metrics to CSV and print summary."""
    logger.log(state)
    
    if state.metrics_before and state.metrics_after:
        cpu_delta = state.metrics_after["system"]["cpu_usage_pct"] - state.metrics_before["system"]["cpu_usage_pct"]
        bw_delta = state.metrics_after["system"]["bandwidth_usage_gbps"] - state.metrics_before["system"]["bandwidth_usage_gbps"]
        print(f"📊 METRICS: CPU Δ={cpu_delta:+.1f}% | BW Δ={bw_delta:+.2f} Gbps | Energy={state.energy_joules:.3f} J")
    
    return state


# ══════════════════════════════════════════════════════════════════════
# Pipeline Orchestrator
# ══════════════════════════════════════════════════════════════════════

class AgentPipeline:
    """
    Full end-to-end pipeline: Intent → Agent → Validate → Translate → Execute → Log
    """
    
    def __init__(self, mode: str = "simulation", csv_path: str = None, model_client=None):
        self.mode = mode
        self.core = get_core(mode)
        self.logger = MetricsLogger(
            csv_path or "results/pipeline_metrics.csv"
        )
        self.model_client = model_client
        self.states: list[AgentState] = []
    
    def run_single(self, intent: dict, model_name: str = "simulation") -> AgentState:
        """Run a single intent through the pipeline."""
        state = AgentState(
            intent_id=intent.get("intent_id", "UNKNOWN"),
            input_text=intent.get("input_text", ""),
            ground_truth=intent.get("ground_truth", {}),
            model_name=model_name,
        )
        
        # Pipeline stages
        state = node_receive_intent(state, self.core)
        state = node_process_with_llm(state, self.model_client)
        state = node_validate_json(state)
        state = node_translate_to_command(state)
        state = node_execute(state, self.core)
        state = node_log_metrics(state, self.logger)
        
        self.states.append(state)
        return state
    
    def run_dataset(self, dataset_path: str, model_name: str = "simulation", limit: int = None) -> list[AgentState]:
        """Run all intents in a dataset through the pipeline."""
        intents = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    intents.append(json.loads(line))
        
        if limit:
            intents = intents[:limit]
        
        print(f"\n{'='*60}")
        print(f"🚀 PIPELINE START: {model_name}")
        print(f"   Intents: {len(intents)} | Mode: {self.mode}")
        print(f"   Core: {self.core.TOTAL_CPU_UNITS} CPU | {self.core.TOTAL_MEMORY_GB} GB RAM | {self.core.TOTAL_BANDWIDTH_GBPS} Gbps BW")
        print(f"{'='*60}")
        
        for i, intent in enumerate(intents):
            print(f"\n[{i+1}/{len(intents)}] ", end="")
            self.run_single(intent, model_name)
        
        self.print_summary()
        return self.states
    
    def print_summary(self):
        """Print pipeline summary statistics."""
        if not self.states:
            print("\n⚠️  No pipeline runs completed.")
            return
        
        total = len(self.states)
        valid = sum(1 for s in self.states if s.format_valid)
        executed = sum(1 for s in self.states if s.execution_success)
        avg_convergence = sum(s.convergence_time_ms for s in self.states) / total
        avg_energy = sum(s.energy_joules for s in self.states) / total
        
        print(f"\n{'='*60}")
        print(f"📊 PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"   Total intents processed: {total}")
        print(f"   Format valid:  {valid}/{total} ({valid/total*100:.1f}%)")
        print(f"   Executed:      {executed}/{total} ({executed/total*100:.1f}%)")
        print(f"   Avg T_conv:    {avg_convergence:.1f} ms")
        print(f"   Avg Energy:    {avg_energy:.3f} J/intent")
        print(f"   Metrics saved: {self.logger.csv_path}")
        
        # Core status
        print(f"\n{self.core.status_report()}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="6G Agent Sandbox Pipeline")
    parser.add_argument("--mode", default="simulation", choices=["simulation", "production"])
    parser.add_argument("--dataset", default="dataset/intents.jsonl")
    parser.add_argument("--model", default="simulation", help="Model name for logging")
    parser.add_argument("--limit", type=int, default=None, help="Limit intents (for quick tests)")
    parser.add_argument("--csv", default=None, help="CSV output path")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    
    dataset_path = args.dataset
    if not Path(dataset_path).is_absolute():
        dataset_path = str(project_root / dataset_path)
    
    csv_path = args.csv
    if csv_path:
        if not Path(csv_path).is_absolute():
            csv_path = str(project_root / csv_path)
    
    pipeline = AgentPipeline(
        mode=args.mode,
        csv_path=csv_path,
    )
    
    pipeline.run_dataset(
        dataset_path=dataset_path,
        model_name=args.model,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
