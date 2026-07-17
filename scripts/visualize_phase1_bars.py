#!/usr/bin/env python3
"""
Phase 1 individual bar charts (Avg Match, Action Accuracy, Inference Time)
Uses: Llama/Gemma/GLM from results/phase1/ (v1 benchmark)
      Qwen 7B Q4_K_M from results/resource_v3/ (v2 re-run)
Output: results/figures/phase1_bar_{match,action,time}.png
"""
import json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

PROJECT = Path("/jupyter_workspace/rbru/slm_6g_eval_v2")
FIGURES = PROJECT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

p1_dir = PROJECT / "results" / "phase1"
data_p1 = {}

# Old models (v1 benchmark) — skip DeepSeek + old Qwen
for f in sorted(glob.glob(str(p1_dir / "*.json"))):
    name = f.split('/')[-1]
    if 'deepseek' in name.lower() or 'qwen' in name.lower():
        continue
    with open(f) as fh:
        d = json.load(fh)
    s = d['summary']
    data_p1[s['family']] = {
        'match': s['avg_match_pct'], 'action': s['action_correct_pct'],
        'format': s['format_valid_pct'], 'time': s['avg_time_s'],
    }

# Re-run Qwen (v2 benchmark with resource monitoring)
with open(PROJECT / "results" / "resource_v3" / "7B_Q4_K_M.json") as fh:
    d = json.load(fh)
results = d['results']
n = len(results)
data_p1['Qwen'] = {
    'match': np.mean([r['match_pct'] for r in results]),
    'action': sum(1 for r in results if r.get('action_correct')) / n * 100,
    'format': sum(1 for r in results if r.get('format_valid')) / n * 100,
    'time': np.mean([r['duration_s'] for r in results]),
}

fams = ['Llama', 'Qwen', 'Gemma', 'GLM']
colors = {'Llama': '#E74C3C', 'Qwen': '#3498DB', 'Gemma': '#2ECC71', 'GLM': '#9B59B6'}

for metric, title, ylabel, fname in [
    ("match", "Phase 1: Avg Match % (~8B, Q4_K_M)", "Avg Match %", "phase1_bar_match.png"),
    ("action", "Phase 1: Action Accuracy % (~8B, Q4_K_M)", "Action Accuracy %", "phase1_bar_action.png"),
    ("time", "Phase 1: Inference Time (~8B, Q4_K_M)", "Inference Time (s)", "phase1_bar_time.png"),
]:
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = [data_p1[f][metric] for f in fams]
    bar_colors = [colors[f] for f in fams]
    bars = ax.bar(fams, vals, color=bar_colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, vals):
        offset = max(vals) * 0.02
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
               f"{val:.1f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.grid(alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = FIGURES / fname
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {path}")

print("\n✅ Phase 1 individual figures done!")
