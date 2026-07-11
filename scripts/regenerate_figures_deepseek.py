#!/usr/bin/env python3
"""
Regenerate Figures 2, 4, 5 with DeepSeek-V4 Pro data
=====================================================
- Fig 2: Error Analysis (pie charts)
- Fig 4: Convergence Boxplot
- Fig 5: Pareto Frontier
"""

import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

BENCHMARK_PATH = "/jupyter_workspace/local/ai_agent/slm_6g_eval/results/benchmark_200_full.json"
DEEPSEEK_PATH = "/jupyter_workspace/local/ai_agent/slm_6g_eval/results/deepseek_enriched_results.json"
OUTPUT_DIR = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval/results/figures")

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12, 'legend.fontsize': 9,
})

# Color scheme — SLMs + DeepSeek
ALL_COLORS = {
    "qwen2.5:1.5b": "#2ECC71",
    "llama3.2:3b": "#3498DB",
    "qwen2.5:7b": "#E74C3C",
    "deepseek-chat": "#9B59B6",  # Purple for DeepSeek
    "DeepSeek-V4 Pro": "#9B59B6",
}
ALL_LABELS = {
    "qwen2.5:1.5b": "Qwen2.5-1.5B",
    "llama3.2:3b": "Llama-3.2-3B",
    "qwen2.5:7b": "Qwen2.5-7B",
    "deepseek-chat": "DeepSeek-V4 Pro",
    "DeepSeek-V4 Pro": "DeepSeek-V4 Pro",
}


def load_all_data():
    with open(BENCHMARK_PATH) as f:
        slm_data = json.load(f)
    with open(DEEPSEEK_PATH) as f:
        ds_raw = json.load(f)
    # Wrap DeepSeek in same format as SLM data
    ds_results = ds_raw["results"]
    ds_wrapped = {
        "model": "deepseek-chat",
        "summary": {
            "total": 200,
            "format_valid": 200,
            "format_accuracy": 100.0,
            "action_accuracy": 83,
            "avg_match_pct": 31.0,
            "avg_time_s": 1.185,
        },
        "results": ds_results,
    }
    return slm_data, ds_wrapped


def make_error_analysis(slm_data, ds_wrapped):
    """Figure 2: Error Analysis — 4 pie charts including DeepSeek"""
    all_models = slm_data + [ds_wrapped]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for idx, mr in enumerate(all_models):
        ax = axes[idx]
        model = mr["model"]
        results = mr["results"]
        
        format_err = 0
        param_err = 0
        both_ok = 0
        
        for r in results:
            has_error = "error" in r
            is_valid = r.get("format_valid", False)
            has_correct_action = r.get("action_correct", False)
            
            if has_error:
                format_err += 1
            elif not is_valid:
                format_err += 1
            elif has_correct_action:
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
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight('bold')
        
        legend_labels = [f'{c}\n({v}/{total}, {v/total*100:.1f}%)' 
                        for c, v in zip(categories, values)]
        ax.legend(wedges, legend_labels, loc='lower center', fontsize=8,
                 bbox_to_anchor=(0.5, -0.3))
        
        label = ALL_LABELS.get(model, model)
        is_deepseek = "deepseek" in model.lower()
        title = f'{label}\nError Analysis'
        ax.set_title(title, fontweight='bold')
    
    fig.suptitle('Error Breakdown: Format vs Action Failures (All Models)', 
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    path = OUTPUT_DIR / "error_analysis.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Error Analysis (4 models): {path}")


def make_convergence_boxplot(slm_data, ds_wrapped):
    """Figure 4: Convergence Time Boxplot with DeepSeek"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_data = []
    labels = []
    colors = []
    
    # SLM data
    for mr in slm_data:
        results = mr["results"]
        times = [r.get("duration_s", 0) for r in results if r.get("duration_s", 0) and r.get("duration_s", 0) > 0]
        if times:
            time_data.append(times)
            model = mr["model"]
            labels.append(ALL_LABELS.get(model, model))
            colors.append(ALL_COLORS.get(model, "#333"))
    
    # DeepSeek data
    ds_times = [r.get("t_inference_ms", 0) / 1000.0 for r in ds_wrapped["results"] 
                if r.get("t_inference_ms", 0) > 0]
    time_data.append(ds_times)
    labels.append("DeepSeek-V4 Pro")
    colors.append("#9B59B6")
    
    bp = ax.boxplot(time_data, tick_labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black', markersize=8))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add μ and σ annotations
    for i, times in enumerate(time_data):
        mu = np.mean(times)
        sigma = np.std(times)
        cv = (sigma / mu * 100) if mu > 0 else 0
        ax.annotate(f'μ={mu:.2f}s\nσ={sigma:.2f}s\nCV={cv:.1f}%', 
                    xy=(i+1, mu), xytext=(i+1.35, mu + max(times)*0.15),
                    fontsize=7.5, ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    ax.set_ylabel('Inference Duration (seconds)', fontweight='bold')
    ax.set_title('Convergence Time Distribution — All Models\n(μ = mean, σ = std, CV = coefficient of variation)', 
                 fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "convergence_boxplot.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Convergence Boxplot (4 models): {path}")


def make_pareto_frontier(slm_data, ds_wrapped):
    """Figure 5: Pareto Frontier with DeepSeek"""
    fig, ax = plt.subplots(figsize=(11, 7))
    
    all_models_data = []
    
    # SLM data
    for mr in slm_data:
        model = mr["model"]
        results = mr["results"]
        s = mr["summary"]
        
        avg_speed = sum(r.get("speed_tok_s", 0) for r in results if "speed_tok_s" in r) 
        n_speed = sum(1 for r in results if "speed_tok_s" in r)
        if n_speed > 0:
            avg_speed /= n_speed
        else:
            avg_speed = 0
        
        all_models_data.append({
            "model": model,
            "label": ALL_LABELS.get(model, model),
            "speed": avg_speed,
            "match": s["avg_match_pct"],
            "time": s["avg_time_s"],
            "color": ALL_COLORS.get(model, "#333"),
            "size": 180,
        })
    
    # DeepSeek data
    ds_results = ds_wrapped["results"]
    ds_tokens = [r.get("completion_tokens", 0) for r in ds_results]
    ds_time_ms = [r.get("t_inference_ms", 0) for r in ds_results]
    ds_speed = sum(ds_tokens) / sum(ds_time_ms) * 1000 if sum(ds_time_ms) > 0 else 0  # tok/s
    
    all_models_data.append({
        "model": "deepseek-chat",
        "label": "DeepSeek-V4 Pro",
        "speed": ds_speed,
        "match": ds_wrapped["summary"]["avg_match_pct"],
        "time": ds_wrapped["summary"]["avg_time_s"],
        "color": "#9B59B6",
        "size": 350,  # Large marker for cloud LLM
    })
    
    # Plot all models
    for m in all_models_data:
        ax.scatter(m["speed"], m["match"], s=m["size"], c=m["color"],
                   alpha=0.7, edgecolors='black', linewidth=1.5, zorder=5)
        # Offset annotation to avoid overlap
        offset_y = 15 if "DeepSeek" in m["label"] else (-8 if m["speed"] < 5 else 8)
        ax.annotate(m["label"] + f'\n({m["match"]:.1f}%)', 
                    (m["speed"], m["match"]),
                    xytext=(8, offset_y), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=m["color"])
    
    # Draw Pareto frontier (upper envelope)
    sorted_by_speed = sorted(all_models_data, key=lambda x: x["speed"])
    pareto_x, pareto_y = [], []
    max_y = 0
    for m in sorted_by_speed:
        if m["match"] > max_y:
            max_y = m["match"]
            pareto_x.append(m["speed"])
            pareto_y.append(m["match"])
    
    ax.plot(pareto_x, pareto_y, '--', color='gray', alpha=0.4, linewidth=2, label='Pareto Frontier')
    
    # Highlight Sweet Spot
    # Sweet spot = best match/time ratio among SLMs only
    slm_only = [m for m in all_models_data if "DeepSeek" not in m["label"]]
    sweet_spot = max(slm_only, key=lambda x: x["match"] / max(x["time"], 0.1))
    ax.annotate(f'★ Sweet Spot\n{sweet_spot["label"]}\n{sweet_spot["match"]:.1f}% match\n{sweet_spot["speed"]:.1f} tok/s',
                xy=(sweet_spot["speed"], sweet_spot["match"]),
                xytext=(sweet_spot["speed"] + 2, sweet_spot["match"] - 10),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    
    # Annotate DeepSeek (cloud vs edge)
    ds_m = [m for m in all_models_data if "DeepSeek" in m["label"]][0]
    ax.annotate('[CLOUD GPU]\n(not edge-\ndeployable)', 
                xy=(ds_m["speed"], ds_m["match"]),
                xytext=(ds_m["speed"] - 25, ds_m["match"] + 12),
                fontsize=8, ha='center', style='italic', color='#7D3C98',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8DAEF', alpha=0.8))
    
    ax.set_xlabel('Inference Speed (tokens/second)', fontweight='bold')
    ax.set_ylabel('Exact Parameter Match (%)', fontweight='bold')
    ax.set_title('Pareto Frontier: Speed vs Accuracy — SLMs + Cloud LLM\n(6G Intent-Driven Slice Management)', 
                 fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Stats table
    table_lines = []
    for m in sorted(all_models_data, key=lambda x: x["speed"], reverse=True):
        edge_note = "[CLOUD]" if "DeepSeek" in m["label"] else "[EDGE]"
        table_lines.append(
            f"{m['label']:<20} {m['speed']:>7.1f} tok/s  {m['match']:>5.1f}% match  {m['time']:>5.1f}s avg  {edge_note}"
        )
    ax.text(0.02, 0.02, "\n".join(table_lines), transform=ax.transAxes, fontsize=7.5,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    plt.tight_layout()
    path = OUTPUT_DIR / "pareto_frontier.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Pareto Frontier (4 models): {path}")


def main():
    print("Loading all data...")
    slm_data, ds_wrapped = load_all_data()
    
    print(f"SLMs: {len(slm_data)} models, DeepSeek: {len(ds_wrapped['results'])} intents\n")
    print("🎨 Regenerating figures with DeepSeek-V4 Pro...\n")
    
    make_error_analysis(slm_data, ds_wrapped)
    make_convergence_boxplot(slm_data, ds_wrapped)
    make_pareto_frontier(slm_data, ds_wrapped)
    
    print(f"\n{'='*50}")
    print(f"✅ All figures saved to: {OUTPUT_DIR}")
    print(f"   - error_analysis.png (4 pie charts)")
    print(f"   - convergence_boxplot.png (4 boxplots)")
    print(f"   - pareto_frontier.png (4 scatter + frontier)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
