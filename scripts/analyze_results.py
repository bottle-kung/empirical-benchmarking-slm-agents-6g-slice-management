#!/usr/bin/env python3
"""
Results Analyzer & Visualization Dashboard
===========================================
Analyzes evaluation results and generates comparison reports.
Creates ready-to-publish charts for Q1 papers.

Input: Multiple evaluation result JSON files
Output: 
  - Console summary tables
  - Ready-to-plot data structures
  - Pareto frontier analysis

Usage:
  python scripts/analyze_results.py
  python scripts/analyze_results.py --results-dir results/
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_results(filepath: str) -> dict:
    """Load evaluation results from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_all_results(results_dir: str) -> dict[str, dict]:
    """Load all *_results.json files from a directory."""
    path = Path(results_dir)
    results = {}
    for f in sorted(path.glob("*_results.json")):
        if "_judged" in f.name:
            continue
        model_name = f.stem.replace("_results", "")
        results[model_name] = load_results(f)
    return results


# ══════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════

MODEL_PARAMS = {
    "rule_based": 0,
    "qwen2.5-1.5b": 1.5,
    "gemma2-2b": 2.0,
    "llama3.2-3b": 3.0,
    "mistral-7b": 7.0,
    "llama3.1-8b": 8.0,
    "qwen2.5-7b": 7.0,
    "phi3-14b": 14.0,
    "gpt-4o": None,  # Unknown, cloud
}

MODEL_POWER = {  # Estimated watts
    "rule_based": 0.1,
    "qwen2.5-1.5b": 50,
    "gemma2-2b": 60,
    "llama3.2-3b": 70,
    "mistral-7b": 150,
    "llama3.1-8b": 160,
    "qwen2.5-7b": 150,
    "phi3-14b": 250,
    "gpt-4o": 500,
}

MODEL_GROUPS = {
    "rule_based": "Legacy",
    "qwen2.5-1.5b": "Tiny SLM",
    "gemma2-2b": "Tiny SLM",
    "llama3.2-3b": "Tiny SLM",
    "mistral-7b": "Mid SLM",
    "llama3.1-8b": "Mid SLM",
    "qwen2.5-7b": "Mid SLM",
    "phi3-14b": "Mid SLM",
    "gpt-4o": "Cloud LLM",
    "claude-sonnet-4": "Cloud LLM",
}


class ResultsAnalyzer:
    """Analyze and compare results across models."""
    
    def __init__(self):
        self.models: dict[str, dict] = {}
        self.comparison: list[dict] = []
    
    def add_model(self, name: str, results: dict):
        self.models[name] = results
    
    def build_comparison(self):
        """Build comparison table across all models."""
        self.comparison = []
        for model_name, data in self.models.items():
            summary = data.get("summary", data.get("overall", {}))
            if not summary:
                continue
            
            overall = summary.get("overall", summary)
            
            # Get per-domain metrics
            by_domain = summary.get("by_domain", {})
            by_complexity = summary.get("by_complexity", {})
            
            params_b = MODEL_PARAMS.get(model_name)
            power_w = MODEL_POWER.get(model_name, 0)
            group = MODEL_GROUPS.get(model_name, "Unknown")
            
            entry = {
                "model": model_name,
                "group": group,
                "parameters_billions": params_b,
                "estimated_power_watts": power_w,
                "format_accuracy": overall.get("format_accuracy", 0),
                "parameter_match": overall.get("avg_parameter_match", 0),
                "perfect_match_rate": overall.get("parameter_exact_match_rate", 0),
                "avg_inference_ms": overall.get("avg_inference_ms", 0),
            }
            
            # Domain breakdown
            for domain, stats in by_domain.items():
                entry[f"{domain}_format"] = stats.get("format_accuracy", 0)
                entry[f"{domain}_match"] = stats.get("avg_parameter_match", 0)
            
            # Complexity breakdown
            for complexity, stats in by_complexity.items():
                entry[f"{complexity}_format"] = stats.get("format_accuracy", 0)
                entry[f"{complexity}_match"] = stats.get("avg_parameter_match", 0)
            
            self.comparison.append(entry)
    
    def print_comparison_table(self):
        """Print a formatted comparison table."""
        if not self.comparison:
            print("⚠️  No comparison data. Run build_comparison() first.")
            return
        
        # Sort by group then parameter size
        group_order = {"Legacy": 0, "Tiny SLM": 1, "Mid SLM": 2, "Cloud LLM": 3}
        sorted_models = sorted(
            self.comparison,
            key=lambda x: (group_order.get(x["group"], 99), x.get("parameters_billions") or 999)
        )
        
        print(f"\n{'='*90}")
        print(f"  MODEL COMPARISON — Intent Translation Accuracy")
        print(f"{'='*90}")
        print(f"  {'Model':<18} {'Group':<12} {'Params':>7} {'Format':>8} {'P.Match':>8} {'PerfMatch':>9} {'Inf(ms)':>8}")
        print(f"  {'-'*18} {'-'*12} {'-'*7} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")
        
        for m in sorted_models:
            params_str = f"{m['parameters_billions']}B" if m['parameters_billions'] else "N/A"
            print(f"  {m['model']:<18} {m['group']:<12} {params_str:>7} "
                  f"{m['format_accuracy']:>7.1%} {m['parameter_match']:>7.1%} "
                  f"{m['perfect_match_rate']:>8.1%} {m['avg_inference_ms']:>7.0f}")
        
        print(f"{'='*90}")
    
    def print_domain_breakdown(self):
        """Print per-domain comparison."""
        if not self.comparison:
            return
        
        domains = ["Slicing Provisioning", "Scaling Request", "Conflict Resolution"]
        
        print(f"\n{'='*90}")
        print(f"  DOMAIN BREAKDOWN — Avg Parameter Match (%)")
        print(f"{'='*90}")
        header = f"  {'Model':<18}"
        for d in domains:
            header += f" {d[:12]:>12}"
        print(header)
        print(f"  {'-'*18}" + f" {'-'*12}" * 3)
        
        for m in sorted(self.comparison, key=lambda x: -x["parameter_match"]):
            row = f"  {m['model']:<18}"
            for d in domains:
                val = m.get(f"{d}_match", 0)
                row += f" {val:>11.1%}"
            print(row)
        
        print(f"{'='*90}")
    
    def print_pareto_analysis(self):
        """
        Pareto Frontier: Parameter Count vs Accuracy.
        Finds the "Sweet Spot" — smallest model with acceptable accuracy.
        """
        if not self.comparison:
            return
        
        # Filter to models with known parameter counts
        with_params = [m for m in self.comparison if m["parameters_billions"] is not None]
        with_params.sort(key=lambda x: x["parameters_billions"])
        
        print(f"\n{'='*70}")
        print(f"  PARETO FRONTIER ANALYSIS")
        print(f"  Parameters (B) vs Parameter Match Accuracy")
        print(f"{'='*70}")
        print(f"  {'Model':<18} {'Params':>7} {'Match':>8} {'Efficiency':>10}")
        print(f"  {'-'*18} {'-'*7} {'-'*8} {'-'*10}")
        
        for m in with_params:
            # Efficiency = accuracy per billion parameters
            efficiency = m["parameter_match"] / m["parameters_billions"] * 100 if m["parameters_billions"] > 0 else 0
            bar = "█" * int(m["parameter_match"] * 50)
            print(f"  {m['model']:<18} {m['parameters_billions']:>6.1f}B {m['parameter_match']:>7.1%} "
                  f"{efficiency:>9.1f}%/B  {bar}")
        
        # Find sweet spot
        acceptable = [m for m in with_params if m["parameter_match"] >= 0.80]
        if acceptable:
            sweet_spot = min(acceptable, key=lambda x: x["parameters_billions"])
            print(f"\n  🎯 SWEET SPOT: {sweet_spot['model']} ({sweet_spot['parameters_billions']}B)")
            print(f"     Accuracy: {sweet_spot['parameter_match']:.1%} | "
                  f"Smallest model above 80% threshold")
        
        print(f"{'='*70}")
    
    def print_energy_analysis(self):
        """Energy efficiency comparison."""
        with_power = [m for m in self.comparison if m.get("estimated_power_watts", 0) > 0]
        
        print(f"\n{'='*80}")
        print(f"  ENERGY EFFICIENCY ANALYSIS")
        print(f"{'='*80}")
        print(f"  {'Model':<18} {'Power(W)':>8} {'Match':>8} {'Accuracy/W':>10} {'Notes'}")
        print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*10} {'-'*30}")
        
        for m in sorted(with_power, key=lambda x: -(x["parameter_match"] / max(x["estimated_power_watts"], 0.1))):
            acc_per_watt = m["parameter_match"] / max(m["estimated_power_watts"], 0.1) * 1000
            notes = ""
            if "rule" in m["model"]:
                notes = "Cheapest but least accurate"
            elif m["parameters_billions"] and m["parameters_billions"] <= 3:
                notes = "Best energy/accuracy ratio candidate"
            print(f"  {m['model']:<18} {m['estimated_power_watts']:>7.0f}W {m['parameter_match']:>7.1%} "
                  f"{acc_per_watt:>9.1f}‰/W  {notes}")
        
        print(f"{'='*80}")
    
    def export_for_plotting(self, output_path: str):
        """Export data in a format ready for matplotlib/seaborn."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        plot_data = {
            "models": [],
            "params_vs_accuracy": [],
            "domain_radar": [],
            "complexity_breakdown": [],
        }
        
        for m in self.comparison:
            plot_data["models"].append(m["model"])
            plot_data["params_vs_accuracy"].append({
                "x": m.get("parameters_billions", 0),
                "y": m["parameter_match"],
                "label": m["model"],
                "group": m["group"],
                "format_accuracy": m["format_accuracy"],
            })
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(plot_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Plot data exported to: {path}")


def generate_synthetic_comparison(analyzer: ResultsAnalyzer):
    """
    Generate synthetic comparison data to demonstrate the analysis
    when real model results aren't available yet.
    """
    import random
    synthetic = {
        "rule_based": {
            "format_accuracy": 1.00,
            "avg_parameter_match": 0.152,
            "parameter_exact_match_rate": 0.035,
            "avg_inference_ms": 0.1,
        },
        "qwen2.5-1.5b": {
            "format_accuracy": 0.85,
            "avg_parameter_match": 0.55,
            "parameter_exact_match_rate": 0.30,
            "avg_inference_ms": 150,
        },
        "llama3.2-3b": {
            "format_accuracy": 0.92,
            "avg_parameter_match": 0.68,
            "parameter_exact_match_rate": 0.45,
            "avg_inference_ms": 250,
        },
        "qwen2.5-7b": {
            "format_accuracy": 0.96,
            "avg_parameter_match": 0.82,
            "parameter_exact_match_rate": 0.65,
            "avg_inference_ms": 500,
        },
        "llama3.1-8b": {
            "format_accuracy": 0.97,
            "avg_parameter_match": 0.85,
            "parameter_exact_match_rate": 0.70,
            "avg_inference_ms": 600,
        },
        "gpt-4o": {
            "format_accuracy": 0.99,
            "avg_parameter_match": 0.95,
            "parameter_exact_match_rate": 0.88,
            "avg_inference_ms": 2000,
        },
    }
    
    domains = ["Slicing Provisioning", "Scaling Request", "Conflict Resolution"]
    complexities = ["Simple", "Complex", "Ambiguous"]
    
    for model_name, metrics in synthetic.items():
        by_domain = {}
        by_complexity = {}
        
        # Generate plausible per-domain metrics
        base = metrics["avg_parameter_match"]
        for d in domains:
            factor = {"Slicing Provisioning": 1.15, "Scaling Request": 0.85, "Conflict Resolution": 0.7}
            by_domain[d] = {
                "format_accuracy": metrics["format_accuracy"] * random.uniform(0.95, 1.05),
                "avg_parameter_match": min(1.0, base * factor.get(d, 1.0)),
            }
        
        for c in complexities:
            factor = {"Simple": 1.2, "Complex": 0.85, "Ambiguous": 0.65}
            by_complexity[c] = {
                "format_accuracy": metrics["format_accuracy"] * random.uniform(0.95, 1.05),
                "avg_parameter_match": min(1.0, base * factor.get(c, 1.0)),
            }
        
        data = {
            "summary": {
                "overall": metrics,
                "by_domain": by_domain,
                "by_complexity": by_complexity,
            }
        }
        
        analyzer.add_model(model_name, data)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Results Analyzer & Dashboard")
    parser.add_argument("--results-dir", default="results", help="Directory with result JSONs")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for demonstration")
    parser.add_argument("--output", default="results/comparison_data.json", help="Output for plot data")
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer()
    
    project_root = Path(__file__).resolve().parent.parent
    
    results_dir = args.results_dir
    if not Path(results_dir).is_absolute():
        results_dir = str(project_root / results_dir)
    
    if args.synthetic or not list(Path(results_dir).glob("*_results.json*")):
        print("⚡ Using synthetic comparison data (real model results not yet available)")
        generate_synthetic_comparison(analyzer)
    else:
        all_results = load_all_results(results_dir)
        for model_name, data in all_results.items():
            analyzer.add_model(model_name, data)
    
    analyzer.build_comparison()
    
    # Print all reports
    analyzer.print_comparison_table()
    analyzer.print_domain_breakdown()
    analyzer.print_pareto_analysis()
    analyzer.print_energy_analysis()
    
    # Export for plotting
    output_path = args.output
    if not Path(output_path).is_absolute():
        output_path = str(project_root / output_path)
    analyzer.export_for_plotting(output_path)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
