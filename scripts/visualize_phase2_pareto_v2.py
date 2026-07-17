#!/usr/bin/env python3
"""
Phase 2 Pareto — Shape=Size, Vertical-line=Q
  Q8_0  : no line (solid marker)
  Q4_K_M: 1 vertical black line through center
  Q2_K  : 2 vertical black lines through center
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

PROJECT = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval_v2")
FIGURES = PROJECT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────
SIZES = ["0.5B", "1.5B", "3B", "7B"]
Q_LEVELS = ["Q2_K", "Q4_K_M", "Q8_0"]
SIZE_PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0}

SIZE_SHAPES = {"0.5B": "s", "1.5B": "D", "3B": "^", "7B": "o"}
SIZE_SIZES  = {"0.5B": 90, "1.5B": 130, "3B": 180, "7B": 240}
Q_COLORS = {"Q8_0": "#2ECC71", "Q4_K_M": "#3498DB", "Q2_K": "#E74C3C"}
Q_NLINES = {"Q8_0": 0, "Q4_K_M": 1, "Q2_K": 2}  # vertical lines through marker

sources = {
    ("0.5B", "Q2_K"):    PROJECT / "results/phase3/0.5B_q2k.json",
    ("0.5B", "Q8_0"):    PROJECT / "results/phase3/0.5B_q8_0.json",
    ("0.5B", "Q4_K_M"):  PROJECT / "results/phase2/benchmark_qwen2_5_0_5b.json",
    ("1.5B", "Q2_K"):    PROJECT / "results/phase3/1.5B_q2k.json",
    ("1.5B", "Q8_0"):    PROJECT / "results/phase3/1.5B_q8_0.json",
    ("1.5B", "Q4_K_M"):  PROJECT / "results/phase2/benchmark_qwen2_5_1_5b.json",
    ("3B", "Q2_K"):      PROJECT / "results/phase3/3B_q2k.json",
    ("3B", "Q8_0"):      PROJECT / "results/phase3/3B_q8_0.json",
    ("3B", "Q4_K_M"):    PROJECT / "results/phase2/benchmark_qwen2_5_3b.json",
    ("7B", "Q2_K"):      PROJECT / "results/phase3/7B_q2k.json",
    ("7B", "Q8_0"):      PROJECT / "results/phase3/7B_q8_0.json",
    ("7B", "Q4_K_M"):    PROJECT / "results/phase1/benchmark_qwen2_5_7b.json",
}

data = {}
for (size, q), path in sources.items():
    with open(path) as f:
        d = json.load(f)
    s = d["summary"]
    data[(size, q)] = {
        "format": s["format_valid_pct"],
        "action": s["action_correct_pct"],
        "exact": s["avg_match_pct"],
        "time": s["avg_time_s"],
        "tokens_per_sec": s["avg_tokens_per_sec"],
        "total_time": s["total_time_s"],
    }

ALL_POINTS = [(s, q) for s in SIZES for q in Q_LEVELS]

# ── Label offsets (computed to avoid overlaps) ─────────────────
OFF = {
    ("0.5B", "Q8_0"):    (0.15, 0.8),
    ("0.5B", "Q4_K_M"):  (0.25, -1.2),
    ("0.5B", "Q2_K"):    (0.15, 0.8),
    ("1.5B", "Q8_0"):    (0.3, 0.6),
    ("1.5B", "Q4_K_M"):  (0.3, -0.8),
    ("1.5B", "Q2_K"):    (-1.0, -1.5),
    ("3B", "Q8_0"):      (0.3, 0.6),
    ("3B", "Q4_K_M"):    (0.3, -0.8),
    ("3B", "Q2_K"):      (0.3, -1.0),
    ("7B", "Q8_0"):      (-1.5, 0.8),
    ("7B", "Q4_K_M"):    (0.5, -1.5),
    ("7B", "Q2_K"):      (0.6, 0.8),
}

OFF_TOK = {
    ("0.5B", "Q8_0"):    (-0.8, 0.8),
    ("0.5B", "Q4_K_M"):  (-0.5, -1.2),
    ("0.5B", "Q2_K"):    (-0.3, 1.0),
    ("1.5B", "Q8_0"):    (-0.5, -1.0),
    ("1.5B", "Q4_K_M"):  (0.3, 0.8),
    ("1.5B", "Q2_K"):    (-0.5, 1.0),
    ("3B", "Q8_0"):      (0.3, -1.0),
    ("3B", "Q4_K_M"):    (0.3, 0.8),
    ("3B", "Q2_K"):      (0.3, -1.0),
    ("7B", "Q8_0"):      (-0.8, 0.8),
    ("7B", "Q4_K_M"):    (-0.5, -1.5),
    ("7B", "Q2_K"):      (-0.3, -1.0),
}

OFF_PARAM = {
    ("0.5B", "Q8_0"):    (0.05, 0.8),
    ("0.5B", "Q4_K_M"):  (0.08, -1.2),
    ("0.5B", "Q2_K"):    (0.12, 0.6),
    ("1.5B", "Q8_0"):    (0.1, 0.6),
    ("1.5B", "Q4_K_M"):  (0.15, -0.8),
    ("1.5B", "Q2_K"):    (0.2, -1.0),
    ("3B", "Q8_0"):      (0.15, -1.0),
    ("3B", "Q4_K_M"):    (0.2, 0.6),
    ("3B", "Q2_K"):      (0.15, -0.8),
    ("7B", "Q8_0"):      (0.15, 0.8),
    ("7B", "Q4_K_M"):    (0.2, -1.5),
    ("7B", "Q2_K"):      (0.15, 0.6),
}

# ═══════════════════════════════════════════════════════════════
def get_value(size, q, key):
    if key == "size_params":
        return SIZE_PARAMS[size]
    return data[(size, q)][key]

def draw_vertical_lines(ax, x, y, n_lines, x_range, y_range):
    """Draw n vertical black lines through (x,y) center of marker."""
    if n_lines == 0:
        return
    spacing = x_range * 0.015  # 1.5% of x data range
    half_h = y_range * 0.025   # 2.5% of y data range

    if n_lines == 1:
        ax.plot([x, x], [y - half_h, y + half_h], '-', color='black', linewidth=2, zorder=6)
    elif n_lines == 2:
        ax.plot([x - spacing, x - spacing], [y - half_h, y + half_h], '-', color='black', linewidth=2, zorder=6)
        ax.plot([x + spacing, x + spacing], [y - half_h, y + half_h], '-', color='black', linewidth=2, zorder=6)

def plot_all(ax, x_key, y_key, offsets):
    """Plot all 12 points: scatter + vertical Q-lines + labels."""
    # Compute data ranges first
    all_x = [get_value(s, q, x_key) for s in SIZES for q in Q_LEVELS]
    all_y = [get_value(s, q, y_key) for s in SIZES for q in Q_LEVELS]
    x_range = max(all_x) - min(all_x) + 0.01
    y_range = max(all_y) - min(all_y) + 0.01
    
    for size in SIZES:
        for q in Q_LEVELS:
            x_val = get_value(size, q, x_key)
            y_val = get_value(size, q, y_key)
            
            # Main marker
            ax.scatter(x_val, y_val, s=SIZE_SIZES[size], c=Q_COLORS[q],
                      marker=SIZE_SHAPES[size], edgecolors="white",
                      linewidth=1.5, zorder=5, alpha=0.9)
        
    # Draw vertical lines AFTER all points (so they render on top)
    for size in SIZES:
        for q in Q_LEVELS:
            x_val = get_value(size, q, x_key)
            y_val = get_value(size, q, y_key)
            draw_vertical_lines(ax, x_val, y_val, Q_NLINES[q], x_range, y_range)
    
    # Labels
    for size in SIZES:
        for q in Q_LEVELS:
            x_val = get_value(size, q, x_key)
            y_val = get_value(size, q, y_key)
            ox, oy = offsets.get((size, q), (0.2, 0.5))
            label = f"{size}\n{q}"
            ax.annotate(label, (x_val, y_val), textcoords="offset points",
                       xytext=(ox*12, oy*12), ha="center", fontsize=5.5,
                       alpha=0.85,
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


def pareto_frontier(ax, x_key, y_key):
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
    ax.plot(fx, fy, "--", color="gray", alpha=0.35, linewidth=2, label="Pareto frontier")


def build_legend(ax, loc1="lower right", loc2="upper left"):
    """Legend: shapes=Size, vertical-lines=Q"""
    # Size legend (shapes)
    leg_size = [Line2D([0],[0], marker=SIZE_SHAPES[s], color='gray',
                       markerfacecolor='gray', markeredgecolor='gray',
                       markersize=[5,7,9,11][i], label=s, linestyle='none')
                for i, s in enumerate(SIZES)]
    
    # Q legend (vertical line count)
    leg_q = [
        Line2D([0],[0], marker='s', color='#3498DB', markerfacecolor='#2ECC71',
               markeredgecolor='white', markersize=10, label='Q8_0 (0 line)', linestyle='none'),
        Line2D([0],[0], marker='s', color='#3498DB', markerfacecolor='#3498DB',
               markeredgecolor='white', markersize=10, label='Q4_K_M', linestyle='none'),
        Line2D([0],[0], marker='s', color='#3498DB', markerfacecolor='#E74C3C',
               markeredgecolor='white', markersize=10, label='Q2_K', linestyle='none'),
    ]
    
    leg1 = ax.legend(handles=leg_size, title="Model Size (shape)", loc=loc1,
                    fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)
    
    # Add vertical line indicators to Q legend manually via text annotation
    # We'll draw them as a separate legend box
    leg_q_custom = [
        Patch(facecolor='#2ECC71', edgecolor='white', linewidth=1.5, label='Q8_0 (8-bit, no line)'),
        Patch(facecolor='#3498DB', edgecolor='white', linewidth=1.5, label='Q4_K_M (4-bit, 1 line)'),
        Patch(facecolor='#E74C3C', edgecolor='white', linewidth=1.5, label='Q2_K (2-bit, 2 lines)'),
    ]
    ax.legend(handles=leg_q_custom, title="Quantization (| lines)", loc=loc2,
             fontsize=8, title_fontsize=9)


# ═══════════════════════════════════════════════════════════════
def make_pareto(x_key, y_key, x_label, y_label, title, fname, offsets,
                xlim=None, ylim=None, annotate_fn=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_all(ax, x_key, y_key, offsets)
    if x_key != "size_params":
        pareto_frontier(ax, x_key, y_key)
    
    if annotate_fn:
        annotate_fn(ax)
    
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
    fig.savefig(FIGURES / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {fname}")


# ── Annotation helpers ─────────────────────────────────────────
def annotate_exact_time(ax):
    bd = data[("7B", "Q8_0")]
    ax.annotate(f"Best: 7B Q8_0\n{bd['exact']:.1f}% @ {bd['time']:.1f}s",
               xy=(bd["time"], bd["exact"]), xytext=(20, 31.5),
               arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=2),
               fontsize=9, color="#E74C3C", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E74C3C", alpha=0.9))
    sd = data[("1.5B", "Q2_K")]
    ax.annotate(f"Fastest: 1.5B Q2_K\n{sd['exact']:.1f}% @ {sd['time']:.1f}s",
               xy=(sd["time"], sd["exact"]), xytext=(6, 23),
               arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=2),
               fontsize=9, color="#2ECC71", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ECC71", alpha=0.9))

def annotate_action_time(ax):
    bd = data[("3B", "Q4_K_M")]
    ax.annotate(f"Best Action: 3B Q4_K_M\n{bd['action']:.1f}% @ {bd['time']:.1f}s",
               xy=(bd["time"], bd["action"]), xytext=(15, 48),
               arrowprops=dict(arrowstyle="->", color="#3498DB", lw=2),
               fontsize=9, color="#3498DB", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#3498DB", alpha=0.9))

def annotate_format_time(ax):
    bd = data[("7B", "Q4_K_M")]
    ax.annotate(f"100%: 7B Q4_K_M, Q8_0, Q2_K\n& 0.5B Q8_0",
               xy=(bd["time"], bd["format"]), xytext=(22, 101),
               arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=2),
               fontsize=8, color="#2ECC71", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ECC71", alpha=0.9))

def annotate_exact_tok(ax):
    bd = data[("7B", "Q8_0")]
    ax.annotate(f"Best: 7B Q8_0\n30.4% @ {bd['tokens_per_sec']:.1f} tok/s",
               xy=(bd["tokens_per_sec"], bd["exact"]), xytext=(4.5, 32),
               arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=2),
               fontsize=9, color="#E74C3C", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E74C3C", alpha=0.9))
    sd = data[("0.5B", "Q4_K_M")]
    ax.annotate(f"Fastest: 0.5B Q4_K_M\n{sd['exact']:.1f}% @ {sd['tokens_per_sec']:.1f} tok/s",
               xy=(sd["tokens_per_sec"], sd["exact"]), xytext=(8, 20.5),
               arrowprops=dict(arrowstyle="->", color="#3498DB", lw=2),
               fontsize=8, color="#3498DB", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#3498DB", alpha=0.9))


# ═══════════════════════════════════════════════════════════════
print("=== Phase 2 Pareto: Shape=Size, | lines=Q ===\n")

make_pareto("time", "exact",
           "Inference Time (s/intent)", "Exact Match (%)",
           "Phase 2: Accuracy vs Speed  |  ■◆▲● = 0.5/1.5/3/7B  |  | = Q Level",
           "phase2_pareto_exact_vs_time.png", OFF,
           annotate_fn=annotate_exact_time)

make_pareto("time", "action",
           "Inference Time (s/intent)", "Action Correct (%)",
           "Phase 2: Action Correct vs Speed  |  Shape=Size, | lines=Q",
           "phase2_pareto_action_vs_time.png", OFF,
           annotate_fn=annotate_action_time)

make_pareto("time", "format",
           "Inference Time (s/intent)", "Format Accuracy (%)",
           "Phase 2: Format Accuracy vs Speed  |  Shape=Size, | lines=Q",
           "phase2_pareto_format_vs_time.png", OFF,
           ylim=(93, 102), annotate_fn=annotate_format_time)

make_pareto("tokens_per_sec", "exact",
           "Throughput (tokens/s)", "Exact Match (%)",
           "Phase 2: Accuracy vs Throughput  |  Shape=Size, | lines=Q",
           "phase2_pareto_exact_vs_throughput.png", OFF_TOK,
           annotate_fn=annotate_exact_tok)

make_pareto("size_params", "exact",
           "Model Size (Billion Parameters)", "Exact Match (%)",
           "Phase 2: Accuracy vs Model Size  |  Shape=Size, | lines=Q",
           "phase2_pareto_exact_vs_size.png", OFF_PARAM,
           xlim=(0, 8))

make_pareto("size_params", "time",
           "Model Size (Billion Parameters)", "Inference Time (s/intent)",
           "Phase 2: Inference Time vs Model Size  |  Shape=Size, | lines=Q",
           "phase2_pareto_time_vs_size.png", OFF_PARAM,
           xlim=(0, 8))

print("\n✅ All 6 Pareto figures regenerated!")
