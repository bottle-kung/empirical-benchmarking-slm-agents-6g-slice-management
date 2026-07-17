#!/usr/bin/env python3
"""
Paper Pareto Figures — Shape=Size, same color per size, Q = | lines only
Same style as visualize_phase2_pareto_v2.py but:
  - One color per model size (not per Q)
  - Q differentiated ONLY by vertical black lines through markers
"""
import json, math, os, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

PROJECT = Path("/jupyter_workspace/rbru/slm_6g_eval_v2")
RES_DIR = PROJECT / "results" / "resource_v3"
FIGURES = PROJECT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

SIZES = ["0.5B", "1.5B", "3B", "7B"]
SIZE_PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0}
Q_LEVELS = ["Q2_K", "Q4_K_M", "Q8_0"]

# Visual encoding: Shape=Size, Color=Size, Q = vertical lines only
SIZE_SHAPES = {"0.5B": "s", "1.5B": "D", "3B": "^", "7B": "o"}
SIZE_SIZES  = {"0.5B": 100, "1.5B": 150, "3B": 200, "7B": 260}
SIZE_COLORS = {"0.5B": "#E67E22", "1.5B": "#2ECC71", "3B": "#3498DB", "7B": "#9B59B6"}
Q_NLINES = {"Q8_0": 0, "Q4_K_M": 1, "Q2_K": 2}

ALL_POINTS = [(s, q) for s in SIZES for q in Q_LEVELS]

# ─── Load data ───
DATA = {}
for f in sorted(RES_DIR.glob("*.json")):
    name = f.stem
    parts = name.split("_")
    size = parts[0]
    q = "_".join(parts[1:]) if len(parts) > 1 else "Q4_K_M"
    with open(f) as fh:
        d = json.load(fh)
    results = d["results"]
    n = len(results)
    match_pcts = [r["match_pct"] for r in results]
    durations = [r["duration_s"] for r in results]
    energies = [r.get("energy_j", 0) for r in results]
    cpu_pcts = [r.get("cpu_pct", 0) for r in results if r.get("cpu_pct", 0) > 0]
    mem_peaks = [r["peak_memory_mb"] for r in results if r.get("peak_memory_mb", 0) > 0]
    total_tok = sum(r.get("tokens", 0) for r in results)
    action = sum(1 for r in results if r.get("action_correct")) / n * 100
    fmt = sum(1 for r in results if r.get("format_valid")) / n * 100
    
    DATA[(size, q)] = {
        "match": np.mean(match_pcts), "action": action, "format": fmt,
        "time": np.mean(durations), "energy": np.mean(energies),
        "cpu": np.mean(cpu_pcts) if cpu_pcts else 0,
        "peak_mem": max(mem_peaks) if mem_peaks else 0,
        "tps": total_tok / sum(durations) if sum(durations) > 0 else 0,
    }


def draw_vertical_lines(ax, x, y, n_lines, x_range, y_range):
    """Draw n solid black vertical lines through (x,y) center of marker."""
    if n_lines == 0:
        return
    spacing = x_range * 0.015
    half_h = y_range * 0.028
    
    if n_lines == 1:
        ax.plot([x, x], [y - half_h, y + half_h], '-', color='black', linewidth=2.5, zorder=6)
    elif n_lines == 2:
        ax.plot([x - spacing, x - spacing], [y - half_h, y + half_h], '-', color='black', linewidth=2.5, zorder=6)
        ax.plot([x + spacing, x + spacing], [y - half_h, y + half_h], '-', color='black', linewidth=2.5, zorder=6)


def get_value(size, q, key):
    return DATA[(size, q)][key]


def plot_all(ax, x_key, y_key):
    """Plot all 12 points: scatter (same color per size) + vertical Q-lines + labels."""
    all_x = [get_value(s, q, x_key) for s in SIZES for q in Q_LEVELS]
    all_y = [get_value(s, q, y_key) for s in SIZES for q in Q_LEVELS]
    x_range = max(all_x) - min(all_x) + 0.01
    y_range = max(all_y) - min(all_y) + 0.01
    
    # Draw markers — same color per size
    for size in SIZES:
        for q in Q_LEVELS:
            x_val = get_value(size, q, x_key)
            y_val = get_value(size, q, y_key)
            ax.scatter(x_val, y_val, s=SIZE_SIZES[size], c=SIZE_COLORS[size],
                      marker=SIZE_SHAPES[size], edgecolors="white",
                      linewidth=1.5, zorder=5, alpha=0.9)
    
    # Draw vertical lines AFTER all points (render on top)
    for size in SIZES:
        for q in Q_LEVELS:
            x_val = get_value(size, q, x_key)
            y_val = get_value(size, q, y_key)
            draw_vertical_lines(ax, x_val, y_val, Q_NLINES[q], x_range, y_range)
    
    # Labels
    offsets = {
        ("0.5B","Q8_0"):(0.2,0.8), ("0.5B","Q4_K_M"):(0.3,-1.2), ("0.5B","Q2_K"):(0.2,0.8),
        ("1.5B","Q8_0"):(0.3,0.6), ("1.5B","Q4_K_M"):(0.4,-0.8), ("1.5B","Q2_K"):(-0.8,-1.0),
        ("3B","Q8_0"):(0.3,0.6), ("3B","Q4_K_M"):(0.4,-0.8), ("3B","Q2_K"):(0.4,-0.8),
        ("7B","Q8_0"):(-0.8,0.8), ("7B","Q4_K_M"):(0.5,-1.5), ("7B","Q2_K"):(0.5,0.8),
    }
    for size in SIZES:
        for q in Q_LEVELS:
            x_val = get_value(size, q, x_key)
            y_val = get_value(size, q, y_key)
            ox, oy = offsets.get((size, q), (0.2, 0.5))
            label = f"{size}\n{q.replace('_',' ')}"
            ax.annotate(label, (x_val, y_val), textcoords="offset points",
                       xytext=(ox*12, oy*12), ha="center", fontsize=5.5,
                       alpha=0.85,
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


def pareto_frontier(ax, x_key, y_key):
    """Draw dashed Pareto frontier line."""
    pts = sorted([(get_value(s,q,x_key), get_value(s,q,y_key)) for s,q in ALL_POINTS],
                 key=lambda p: p[0])
    fx, fy = [], []
    best_y = -1
    for tx, ty in pts:
        if ty > best_y:
            best_y = ty
            fx.append(tx); fy.append(ty)
    frontier = sorted(zip(fx, fy))
    fx, fy = zip(*frontier)
    ax.plot(fx, fy, "--", color="gray", alpha=0.4, linewidth=2, label="Pareto frontier", zorder=2)


def build_legend(ax, loc1="lower right", loc2="upper left"):
    """Shape=Size, | lines=Q"""
    leg_size = [Line2D([0],[0], marker=SIZE_SHAPES[s], color=c,
                       markerfacecolor=c, markeredgecolor='white',
                       markersize=[6,8,10,12][i], label=s, linestyle='none',
                       markeredgewidth=1.5)
                for i, (s, c) in enumerate(zip(SIZES, SIZE_COLORS.values()))]
    
    leg_q = [
        Patch(facecolor='gray', edgecolor='white', linewidth=1.5, label='Q8_0 (8-bit, no line)'),
        Patch(facecolor='gray', edgecolor='white', linewidth=1.5, label='Q4_K_M (4-bit, | line)'),
        Patch(facecolor='gray', edgecolor='white', linewidth=1.5, label='Q2_K (2-bit, || lines)'),
    ]
    
    leg1 = ax.legend(handles=leg_size, title="Model Size (shape+color)", loc=loc1,
                    fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=leg_q, title="Quantization (| lines)", loc=loc2,
             fontsize=8, title_fontsize=9)


def make_pareto(x_key, y_key, x_label, y_label, title, fname, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_all(ax, x_key, y_key)
    pareto_frontier(ax, x_key, y_key)
    
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=14)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    build_legend(ax)
    fig.tight_layout()
    path = FIGURES / fname
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {path}")


# ═══════════════════════════════════════════════════════════════
print("=== Paper Pareto: Shape=Size, Color=Size, | lines=Q ===\n")

make_pareto("time", "match",
           "Inference Time (s/intent)", "Avg Match %",
           "Accuracy vs Speed  |  ■◆▲● = 0.5/1.5/3/7B  |  | = Q Level",
           "paper_pareto_match_vs_time.png")

make_pareto("time", "action",
           "Inference Time (s/intent)", "Action Accuracy %",
           "Action Accuracy vs Speed  |  ■◆▲● = 0.5/1.5/3/7B  |  | = Q Level",
           "paper_pareto_action_vs_time.png")

make_pareto("time", "format",
           "Inference Time (s/intent)", "Format Validity %",
           "Format Validity vs Speed  |  ■◆▲● = 0.5/1.5/3/7B  |  | = Q Level",
           "paper_pareto_format_vs_time.png", ylim=(93, 102))

make_pareto("energy", "match",
           "Energy per Intent (J)", "Avg Match %",
           "Accuracy vs Energy  |  ■◆▲● = 0.5/1.5/3/7B  |  | = Q Level",
           "paper_pareto_match_vs_energy.png")

make_pareto("energy", "action",
           "Energy per Intent (J)", "Action Accuracy %",
           "Action Accuracy vs Energy  |  ■◆▲● = 0.5/1.5/3/7B  |  | = Q Level",
           "paper_pareto_action_vs_energy.png")

make_pareto("tps", "match",
           "Throughput (tokens/s)", "Avg Match %",
           "Accuracy vs Throughput  |  ■◆▲● = 0.5/1.5/3/7B  |  | = Q Level",
           "paper_pareto_match_vs_throughput.png")

print("\n✅ All 6 Pareto figures generated!")
