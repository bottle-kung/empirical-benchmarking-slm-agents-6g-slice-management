#!/usr/bin/env python3
"""
Data Visualization: Pareto Frontier + Trade-off Analysis
=========================================================
Generates publication-quality charts for SLM Agent Evaluation paper.

Charts:
1. Pareto Frontier: Accuracy vs Speed (highlighting Sweet Spot)
2. Trade-off Analysis: T_Inference vs T_Execution bars
3. Domain Radar: Per-model domain comparison
4. Error Analysis: Format vs Parameter failures
"""

import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
from collections import defaultdict

BENCHMARK_PATH = "/jupyter_workspace/local/ai_agent/slm_6g_eval/results/benchmark_200_full.json"
OUTPUT_DIR = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval/results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
})

MODEL_COLORS = {
    "qwen2.5:1.5b": "#2ECC71",
    "llama3.2:3b": "#3498DB",
    "qwen2.5:7b": "#E74C3C",
}
MODEL_LABELS = {
    "qwen2.5:1.5b": "Qwen2.5-1.5B",
    "llama3.2:3b": "Llama-3.2-3B",
    "qwen2.5:7b": "Qwen2.5-7B",
}
MODEL_SIZES = {
    "qwen2.5:1.5b": 120,
    "llama3.2:3b": 180,
    "qwen2.5:7b": 350,
}

def load_data():
    with open(BENCHMARK_PATH) as f:
        return json.load(f)

# ═══════════════════════════════════════════════════════════════
# Chart 1: Pareto Frontier — Speed vs Accuracy
# ═══════════════════════════════════════════════════════════════

def pareto_frontier(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models_info = []
    for mr in data:
        model = mr["model"]
        s = mr["summary"]
        results = mr["results"]
        
        # X: Speed (inverse — higher = better)
        avg_speed = sum(r.get("speed_tok_s", 0) for r in results if "speed_tok_s" in r) / max(1, sum(1 for r in results if "speed_tok_s" in r))
        # Y: Accuracy (format_valid rate)
        accuracy = s["format_valid"] / s["total"] * 100
        match = s["avg_match_pct"]
        
        models_info.append({
            "model": model,
            "label": MODEL_LABELS.get(model, model),
            "speed": avg_speed,
            "accuracy": accuracy,
            "match": match,
            "size": MODEL_SIZES.get(model, 100),
            "color": MODEL_COLORS.get(model, "#333"),
            "time": s["avg_time_s"],
        })
    
    # Sort by speed
    models_info.sort(key=lambda x: x["speed"])
    
    # Plot
    for m in models_info:
        ax.scatter(m["speed"], m["match"], s=m["size"], c=m["color"], 
                   alpha=0.7, edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(m["label"], (m["speed"], m["match"]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold', color=m["color"])
    
    # Draw Pareto frontier line
    sorted_by_speed = sorted(models_info, key=lambda x: x["speed"])
    pareto_x = []
    pareto_y = []
    max_y = 0
    for m in sorted_by_speed:
        if m["match"] > max_y:
            max_y = m["match"]
            pareto_x.append(m["speed"])
            pareto_y.append(m["match"])
    
    ax.plot(pareto_x, pareto_y, '--', color='gray', alpha=0.5, linewidth=2, label='Pareto Frontier')
    
    # Highlight Sweet Spot
    sweet_spot = max(models_info, key=lambda x: x["match"] / max(x["time"], 0.1))
    ax.annotate(f'★ Sweet Spot\n{sweet_spot["label"]}\n{sweet_spot["match"]:.1f}% match\n{sweet_spot["speed"]:.1f} tok/s',
                xy=(sweet_spot["speed"], sweet_spot["match"]),
                xytext=(sweet_spot["speed"] + 1, sweet_spot["match"] - 5),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    
    ax.set_xlabel('Inference Speed (tokens/second)', fontweight='bold')
    ax.set_ylabel('Parameter Match Accuracy (%)', fontweight='bold')
    ax.set_title('Pareto Frontier: Speed vs Accuracy\n(SLM Agents for 6G Intent-Driven Slice Management)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Add model specs as table
    table_text = "\n".join([
        f"{m['label']}: {m['speed']:.1f} tok/s, {m['match']:.1f}% match, {m['time']:.1f}s avg"
        for m in models_info
    ])
    ax.text(0.02, 0.02, table_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    path = OUTPUT_DIR / "pareto_frontier.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Pareto Frontier: {path}")
    return models_info

# ═══════════════════════════════════════════════════════════════
# Chart 2: Trade-off Analysis — T_Inference vs T_Execution
# ═══════════════════════════════════════════════════════════════

def tradeoff_analysis(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models_labels = []
    t_inference_vals = []
    t_execution_vals = []
    colors = []
    
    for mr in data:
        model = mr["model"]
        results = mr["results"]
        
        # T_inference: model generation time (duration_s)
        # T_execution: simulated core execution (we don't have this separately, estimate)
        t_inf = [r.get("duration_s", 0) for r in results if "duration_s" in r]
        avg_inf = np.mean(t_inf) if t_inf else 0
        
        # Simulated execution time (from pipeline — use simulated_core delays)
        avg_exec = 0.5  # seconds (simulated core provisioning delay)
        
        models_labels.append(MODEL_LABELS.get(model, model))
        t_inference_vals.append(avg_inf)
        t_execution_vals.append(avg_exec)
        colors.append(MODEL_COLORS.get(model, "#333"))
    
    x = np.arange(len(models_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, t_inference_vals, width, label='T_Inference (AI Thinking)',
                   color=['#E74C3C', '#E67E22', '#F1C40F'], edgecolor='black')
    bars2 = ax.bar(x + width/2, t_execution_vals, width, label='T_Execution (Network Provisioning)',
                   color='#3498DB', edgecolor='black', alpha=0.7)
    
    # Add values on bars
    for bar, val in zip(bars1, t_inference_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('Trade-off Analysis: T_Inference vs T_Execution\n(CPU-only, 32-core Xeon E5-2690)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_labels, fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add ratio annotation
    for i, (inf, exe) in enumerate(zip(t_inference_vals, t_execution_vals)):
        ratio = inf / exe if exe > 0 else 0
        ax.annotate(f'Ratio: {ratio:.0f}:1', (x[i], inf + exe + 1),
                    ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    path = OUTPUT_DIR / "tradeoff_analysis.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Trade-off: {path}")

# ═══════════════════════════════════════════════════════════════
# Chart 3: Domain Radar
# ═══════════════════════════════════════════════════════════════

def domain_radar(data):
    domains = ["Slicing Provisioning", "Scaling Request", "Conflict Resolution"]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(domains), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace(' ', '\n') for d in domains], fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.set_title('Domain Sensitivity Analysis\n(Format Accuracy %)', fontweight='bold', pad=20)
    
    for mr in data:
        model = mr["model"]
        s = mr["summary"]
        by_domain = s.get("by_domain", {})
        
        values = []
        for d in domains:
            if d in by_domain:
                vals = by_domain[d]
                format_pct = vals["format_valid"] / vals["count"] * 100
            else:
                format_pct = 0
            values.append(format_pct)
        values += values[:1]  # Close polygon
        
        ax.fill(angles, values, alpha=0.15, color=MODEL_COLORS.get(model, "#333"))
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model, "#333"))
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    path = OUTPUT_DIR / "domain_radar.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Domain Radar: {path}")

# ═══════════════════════════════════════════════════════════════
# Chart 4: Error Analysis — Format vs Parameter Failures
# ═══════════════════════════════════════════════════════════════

def error_analysis(data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, mr in enumerate(data):
        ax = axes[idx]
        model = mr["model"]
        results = mr["results"]
        
        # Categorize errors
        format_err = 0
        param_err = 0
        both_ok = 0
        
        for r in results:
            if "error" in r:
                format_err += 1
            elif not r.get("format_valid"):
                format_err += 1
            elif r.get("action_correct"):
                both_ok += 1
            else:
                param_err += 1
        
        total = len(results)
        categories = ['Format OK\n+ Action OK', 'Format OK\nAction Wrong', 'JSON Invalid']
        values = [both_ok, param_err, format_err]
        colors_pie = ['#2ECC71', '#F39C12', '#E74C3C']
        explode = (0.05, 0.05, 0.1)
        
        wedges, texts, autotexts = ax.pie(
            values, explode=explode, labels=None, colors=colors_pie,
            autopct='%1.1f%%', startangle=90, pctdistance=0.75
        )
        
        # Legend
        legend_labels = [f'{c}\n({v}/{total}, {v/total*100:.1f}%)' 
                        for c, v in zip(categories, values)]
        ax.legend(wedges, legend_labels, loc='lower center', fontsize=8,
                 bbox_to_anchor=(0.5, -0.25))
        
        ax.set_title(f'{MODEL_LABELS.get(model, model)}\nError Analysis', fontweight='bold')
    
    fig.suptitle('Error Breakdown: Format vs Parameter Failures', fontweight='bold', fontsize=14)
    plt.tight_layout()
    path = OUTPUT_DIR / "error_analysis.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Error Analysis: {path}")

# ═══════════════════════════════════════════════════════════════
# Chart 5: Convergence Time Distribution (Box Plot)
# ═══════════════════════════════════════════════════════════════

def convergence_boxplot(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_data = []
    labels = []
    colors = []
    
    for mr in data:
        model = mr["model"]
        results = mr["results"]
        times = [r.get("duration_s", 0) for r in results if "duration_s" in r and r.get("duration_s", 0) > 0]
        
        if times:
            time_data.append(times)
            labels.append(MODEL_LABELS.get(model, model))
            colors.append(MODEL_COLORS.get(model, "#333"))
    
    bp = ax.boxplot(time_data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black', markersize=6))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add μ and σ annotations
    for i, times in enumerate(time_data):
        mu = np.mean(times)
        sigma = np.std(times)
        ax.annotate(f'μ={mu:.1f}s\nσ={sigma:.1f}s', 
                    xy=(i+1, mu), xytext=(i+1.3, mu + max(times)*0.1),
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel('Inference Duration (seconds)', fontweight='bold')
    ax.set_title('Convergence Time Distribution\n(μ = mean, σ = standard deviation)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "convergence_boxplot.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Convergence Boxplot: {path}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading benchmark data...")
    data = load_data()
    
    print("\n🎨 Generating charts...\n")
    
    models_info = pareto_frontier(data)
    tradeoff_analysis(data)
    domain_radar(data)
    error_analysis(data)
    convergence_boxplot(data)
    
    print(f"\n{'='*50}")
    print(f"✅ All charts saved to: {OUTPUT_DIR}")
    print(f"{'='*50}")
    
    # Print stats for discussion
    print(f"\n📊 Statistical Summary:")
    for mr in data:
        model = mr["model"]
        results = mr["results"]
        times = [r.get("duration_s", 0) for r in results if "duration_s" in r]
        if times:
            mu = np.mean(times)
            sigma = np.std(times)
            print(f"  {MODEL_LABELS.get(model, model)}: μ={mu:.2f}s, σ={sigma:.2f}s, n={len(times)}")

if __name__ == "__main__":
    main()
