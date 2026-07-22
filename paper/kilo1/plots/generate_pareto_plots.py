"""
Pareto Frontier Plot Generator — SLM 6G Benchmark
Generates dark-theme academic Pareto plots for each agent's paper section.
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent.parent  # back to project root

# ── Dark theme styling ──
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e0e0e0', 'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0', 'xtick.color': '#e0e0e0', 'ytick.color': '#e0e0e0',
    'grid.color': '#2a2a4a', 'grid.alpha': 0.6,
    'legend.facecolor': '#16213e', 'legend.edgecolor': '#3a3a5a',
})
COLORS = ['#00d4aa', '#ff6b6b', '#ffd93d', '#6c5ce7', '#48dbfb', '#ff9ff3', '#feca57', '#54a0ff']

# Load data
with open(PROJECT / "results" / "ci_summary_v3.json") as f:
    ci = json.load(f)

# ── Plot function ──
def make_pareto(agent, x_metric, y_metric, xlabel, ylabel, data_filter, 
                highlight_key=None, highlight_label=None):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    points, labels_list, sizes_list = [], [], []
    for key, label in data_filter.items():
        if key in ci:
            xv, yv = ci[key][x_metric], ci[key][y_metric]
            points.append((xv, yv))
            labels_list.append(label)
            s = 120 if '0.5B' in label else 200 if '1.5B' in label else 280
            sizes_list.append(s)
    for i, (x, y) in enumerate(points):
        c = COLORS[i % len(COLORS)]
        ax.scatter(x, y, s=sizes_list[i], c=c, edgecolors='white', linewidth=1.5, zorder=5)
        ax.annotate(labels_list[i], (x, y), textcoords="offset points",
                    xytext=(8, 8 if i % 2 == 0 else -12), fontsize=11, color=c, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#16213e', edgecolor=c, alpha=0.8))
    if highlight_key and highlight_key in ci:
        hx, hy = ci[highlight_key][x_metric], ci[highlight_key][y_metric]
        ax.scatter(hx, hy, s=400, c='none', edgecolors='#ffd93d', linewidth=3, zorder=6, marker='D')
        ax.annotate(f'★ {highlight_label}', (hx, hy), textcoords="offset points",
                    xytext=(15, 15), fontsize=12, color='#ffd93d', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', edgecolor='#ffd93d', alpha=0.9))
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(f'{agent.upper()} — {ylabel} vs {xlabel}', fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.text(0.98, 0.02, f'SLM 6G Benchmark · {agent}', transform=ax.transAxes,
            fontsize=8, color='#555577', ha='right', va='bottom', style='italic')
    plt.tight_layout()
    out = Path(__file__).parent / f"pareto_{y_metric}_vs_{x_metric}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()

# ── Generate plots ──
AGENT = "kilo1"  # replace with your agent name
if __name__ == "__main__":
    # Configure which data_filter to use based on agent
    pass  # See individual agent script for configuration
