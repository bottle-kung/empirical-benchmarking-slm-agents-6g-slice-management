#!/usr/bin/env python3
"""
Visualization Generator — SLM Agent Benchmark v2
=================================================
Generates all figures for Phase 1 and Phase 2 results.

Phase 1 (Cross-family comparison):
  1. Radar chart — 5 models across all dimensions
  2. Format + Exact + Semantic bar comparison
  3. Inference time comparison
  4. Resource usage (CPU, Memory, Energy) bars
  5. Efficiency scatter (Tokens/Watt vs Accuracy)
  6. Domain breakdown heatmap

Phase 2 (Size scaling):
  7. Performance vs Model Size line chart
  8. Inference Time vs Size
  9. Resource usage vs Size
  10. Energy vs Size
  11. Efficiency (Tokens/Watt) vs Size
  12. Pareto frontier

Usage:
  python3 scripts/visualize.py results/phase1_benchmark_*.json
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Set Thai-friendly font
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["figure.dpi"] = 150

PROJECT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Color palette for families
FAMILY_COLORS = {
    "Llama": "#FF6B35",
    "DeepSeek": "#4ECDC4",
    "GLM": "#7B68EE",
    "Qwen": "#45B7D1",
    "Gemma": "#96CEB4",
}


def load_results(filepath: str) -> list:
    with open(filepath) as f:
        return json.load(f)


def save_fig(fig, name: str):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ {path.name}")
    return str(path)


# ═══════════════════════════════════════════════════════════════
# PHASE 1: CROSS-FAMILY COMPARISON
# ═══════════════════════════════════════════════════════════════
def phase1_radar(results: list):
    """Spider/Radar chart comparing 5 models across all metrics."""
    categories = ["Format\nAccuracy", "Exact\nMatch", "Inference\nSpeed", "Energy\nEfficiency", "Memory\nEfficiency"]
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for r in results:
        s = r["summary"]
        res = s.get("resources", {})
        family = s["family"]

        # Normalize each metric to 0-100
        format_pct = s["format_valid_pct"]
        exact_pct = s["avg_match_pct"]
        speed_norm = min(100, (1 / max(s["avg_time_s"], 0.1)) * 10)  # inverse time
        energy_per = res.get("energy_per_intent_j", 10)
        energy_norm = max(0, 100 - energy_per * 10)  # lower energy = better
        mem_peak = res.get("peak_mem_gb", 8)
        mem_norm = max(0, 100 - mem_peak * 10)  # lower mem = better

        values = [format_pct, exact_pct, speed_norm, energy_norm, mem_norm]
        values += values[:1]

        ax.fill(angles, values, alpha=0.15, color=FAMILY_COLORS.get(family, "#888"), label=family)
        ax.plot(angles, values, color=FAMILY_COLORS.get(family, "#888"), linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("Phase 1: Cross-Family Model Comparison @ ~8B", fontsize=14, fontweight="bold", pad=20)

    return save_fig(fig, "phase1_radar")


def phase1_bars(results: list):
    """Grouped bar chart: Format Accuracy + Exact Match per model."""
    models = []
    format_vals = []
    exact_vals = []
    families = []

    for r in results:
        s = r["summary"]
        models.append(s["family"])
        format_vals.append(s["format_valid_pct"])
        exact_vals.append(s["avg_match_pct"])
        families.append(s["family"])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, format_vals, width, label="Format Accuracy", color="#4ECDC4", edgecolor="white")
    bars2 = ax.bar(x + width/2, exact_vals, width, label="Exact Match", color="#FF6B35", edgecolor="white")

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Phase 1: Format vs Exact Match by Model Family", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    return save_fig(fig, "phase1_format_exact_bars")


def phase1_inference_time(results: list):
    """Bar chart: average inference time per intent."""
    models = [r["summary"]["family"] for r in results]
    times = [r["summary"]["avg_time_s"] for r in results]
    colors = [FAMILY_COLORS.get(m, "#888") for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, times, color=colors, edgecolor="white")

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f"{t:.1f}s", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Time per Intent (seconds)")
    ax.set_title("Phase 1: Inference Time Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    return save_fig(fig, "phase1_inference_time")


def phase1_resources(results: list):
    """Grouped bar chart: CPU%, Peak Memory, Energy per intent."""
    models = [r["summary"]["family"] for r in results]

    cpu_vals = [r["summary"].get("resources", {}).get("avg_cpu_pct", 0) for r in results]
    mem_vals = [r["summary"].get("resources", {}).get("peak_mem_gb", 0) for r in results]
    energy_vals = [r["summary"].get("resources", {}).get("energy_per_intent_j", 0) for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CPU
    axes[0].bar(models, cpu_vals, color=[FAMILY_COLORS.get(m, "#888") for m in models])
    axes[0].set_title("Avg CPU Usage (%)")
    axes[0].set_ylabel("%")
    for i, v in enumerate(cpu_vals):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)

    # Memory
    axes[1].bar(models, mem_vals, color=[FAMILY_COLORS.get(m, "#888") for m in models])
    axes[1].set_title("Peak Memory (GB)")
    axes[1].set_ylabel("GB")
    for i, v in enumerate(mem_vals):
        axes[1].text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)

    # Energy
    axes[2].bar(models, energy_vals, color=[FAMILY_COLORS.get(m, "#888") for m in models])
    axes[2].set_title("Energy per Intent (J)")
    axes[2].set_ylabel("Joules")
    for i, v in enumerate(energy_vals):
        axes[2].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)

    fig.suptitle("Phase 1: Resource Utilization by Model", fontsize=14, fontweight="bold")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)

    return save_fig(fig, "phase1_resources")


def phase1_efficiency(results: list):
    """Scatter plot: Tokens/Watt vs Accuracy (bubble size = format accuracy)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for r in results:
        s = r["summary"]
        family = s["family"]
        color = FAMILY_COLORS.get(family, "#888")

        tokens_per_sec = s.get("avg_tokens_per_sec", 0)
        energy_j = s.get("resources", {}).get("energy_joules", 1) or 1
        tokens_per_joule = tokens_per_sec * s.get("total_time_s", 1) / energy_j if energy_j > 0 else 0
        accuracy = s["avg_match_pct"]
        format_pct = s["format_valid_pct"]

        ax.scatter(accuracy, tokens_per_joule, s=format_pct * 5, c=color,
                   alpha=0.7, edgecolors="black", linewidth=0.5, label=family)
        ax.annotate(family, (accuracy, tokens_per_joule),
                    textcoords="offset points", xytext=(0, 10), fontsize=11, fontweight="bold", ha="center")

    ax.set_xlabel("Exact Match Accuracy (%)")
    ax.set_ylabel("Tokens per Joule (efficiency)")
    ax.set_title("Phase 1: Efficiency Frontier — Tokens/Joule vs Accuracy", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="lower right")

    return save_fig(fig, "phase1_efficiency")


def phase1_domain_heatmap(results: list):
    """Heatmap: Domain vs Model performance."""
    domains = ["Slicing Provisioning", "Scaling Request", "Conflict Resolution"]
    families = [r["summary"]["family"] for r in results]

    data = np.zeros((len(families), len(domains)))
    for i, r in enumerate(results):
        for j, d in enumerate(domains):
            data[i, j] = r["summary"]["by_domain"].get(d, {}).get("avg_match_pct", 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    for i in range(len(families)):
        for j in range(len(domains)):
            ax.text(j, i, f"{data[i,j]:.1f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if data[i,j] < 50 else "black")

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, fontsize=10)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=11)
    ax.set_title("Phase 1: Domain-Level Exact Match by Model", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Match %")

    return save_fig(fig, "phase1_domain_heatmap")


# ═══════════════════════════════════════════════════════════════
# PHASE 2: SIZE SCALING
# ═══════════════════════════════════════════════════════════════
def phase2_performance_vs_size(results: list, family: str = ""):
    """Line chart: Performance metrics vs model size."""
    sizes = [r["summary"].get("params_b", 0) or 
             float(r["summary"]["model"].split(":")[1].replace("b","")) 
             for r in results]
    format_vals = [r["summary"]["format_valid_pct"] for r in results]
    exact_vals = [r["summary"]["avg_match_pct"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sizes, format_vals, "o-", linewidth=2, markersize=8, label="Format Accuracy", color="#4ECDC4")
    ax.plot(sizes, exact_vals, "s-", linewidth=2, markersize=8, label="Exact Match", color="#FF6B35")

    for x, y in zip(sizes, format_vals):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=9)
    for x, y in zip(sizes, exact_vals):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, -16), fontsize=9)

    ax.set_xlabel("Model Size (Billions of Parameters)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Phase 2: Performance vs Model Size{(' — '+family) if family else ''}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    return save_fig(fig, "phase2_performance_vs_size")


def phase2_resources_vs_size(results: list, family: str = ""):
    """Multi-line: CPU, Memory, Energy vs model size."""
    sizes = [r["summary"].get("params_b", 0) or 
             float(r["summary"]["model"].split(":")[1].replace("b",""))
             for r in results]
    time_vals = [r["summary"]["avg_time_s"] for r in results]
    energy_vals = [r["summary"].get("resources", {}).get("energy_per_intent_j", 0) for r in results]
    cpu_vals = [r["summary"].get("resources", {}).get("avg_cpu_pct", 0) for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Time
    axes[0].plot(sizes, time_vals, "o-", linewidth=2, markersize=8, color="#FF6B35")
    for x, y in zip(sizes, time_vals):
        axes[0].annotate(f"{y:.1f}s", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=9)
    axes[0].set_xlabel("Model Size (B params)")
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Inference Time")
    axes[0].grid(alpha=0.3)

    # CPU
    axes[1].plot(sizes, cpu_vals, "s-", linewidth=2, markersize=8, color="#4ECDC4")
    for x, y in zip(sizes, cpu_vals):
        axes[1].annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=9)
    axes[1].set_xlabel("Model Size (B params)")
    axes[1].set_ylabel("CPU %")
    axes[1].set_title("Avg CPU Usage")
    axes[1].grid(alpha=0.3)

    # Energy
    axes[2].plot(sizes, energy_vals, "D-", linewidth=2, markersize=8, color="#7B68EE")
    for x, y in zip(sizes, energy_vals):
        axes[2].annotate(f"{y:.2f}J", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=9)
    axes[2].set_xlabel("Model Size (B params)")
    axes[2].set_ylabel("Joules")
    axes[2].set_title("Energy per Intent")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Phase 2: Resource Scaling{(' — '+family) if family else ''}", fontsize=14, fontweight="bold")
    return save_fig(fig, "phase2_resources_vs_size")


def phase2_efficiency_vs_size(results: list, family: str = ""):
    """Efficiency (Tokens/Watt) vs Model Size."""
    sizes = []
    efficiency = []
    for r in results:
        s = r["summary"]
        size = s.get("params_b", 0)
        if not size:
            size = float(s["model"].split(":")[1].replace("b", ""))
        sizes.append(size)

        energy = s.get("resources", {}).get("energy_joules", 1) or 1
        total_tokens = s.get("total_tokens", 1)
        tok_per_joule = total_tokens / energy if energy > 0 else 0
        efficiency.append(tok_per_joule)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar([f"{s}B" for s in sizes], efficiency, color="#45B7D1", edgecolor="white")

    for i, (s, e) in enumerate(zip(sizes, efficiency)):
        ax.text(i, e + max(efficiency)*0.02, f"{e:.1f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Model Size")
    ax.set_ylabel("Tokens per Joule")
    ax.set_title(f"Phase 2: Energy Efficiency vs Model Size{(' — '+family) if family else ''}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    return save_fig(fig, "phase2_efficiency_vs_size")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Generate visualization for SLM benchmark v2")
    parser.add_argument("results_file", type=str, help="Path to benchmark results JSON")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1, help="Phase to visualize")
    parser.add_argument("--family", type=str, default="", help="Model family name (for titles)")
    args = parser.parse_args()

    results = load_results(args.results_file)
    print(f"📊 Generating Phase {args.phase} figures from {len(results)} model results...")
    print(f"   Output: {FIGURES_DIR}/\n")

    if args.phase == 1:
        phase1_radar(results)
        phase1_bars(results)
        phase1_inference_time(results)
        phase1_resources(results)
        phase1_efficiency(results)
        phase1_domain_heatmap(results)
        print(f"\n✅ 6 figures saved to {FIGURES_DIR}/")
    else:
        phase2_performance_vs_size(results, args.family)
        phase2_resources_vs_size(results, args.family)
        phase2_efficiency_vs_size(results, args.family)
        print(f"\n✅ 3 figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
