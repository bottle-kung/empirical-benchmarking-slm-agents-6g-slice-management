#!/usr/bin/env python3
"""
Phase 2 Combined — Pareto Plots with distinct markers + hatch patterns
Size → shape, Q → hatch (none=Q8_0, /=Q4_K_M, x=Q2_K)
Labels repositioned to avoid overlap
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

# Size → shape (distinct)
SIZE_SHAPES = {"0.5B": "s", "1.5B": "D", "3B": "^", "7B": "o"}
SIZE_SIZES  = {"0.5B": 80, "1.5B": 120, "3B": 160, "7B": 220}

# Q → color + hatch
Q_COLORS = {"Q8_0": "#2ECC71", "Q4_K_M": "#3498DB", "Q2_K": "#E74C3C"}
Q_HATCHES = {"Q8_0": "", "Q4_K_M": "//", "Q2_K": "xx"}  # none, single, double
Q_LABELS_FULL = {"Q8_0": "Q8_0 (8-bit)", "Q4_K_M": "Q4_K_M (4-bit)", "Q2_K": "Q2_K (2-bit)"}

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

# ── Smart label offsets (pre-computed to avoid overlaps) ──────
# Each point gets a manual (dx, dy) offset in data coordinates
LABEL_OFFSETS = {
    # 0.5B
    ("0.5B", "Q8_0"):    (0.15, 0.8),
    ("0.5B", "Q4_K_M"):  (0.25, -1.2),
    ("0.5B", "Q2_K"):    (0.15, 0.8),
    # 1.5B
    ("1.5B", "Q8_0"):    (0.3, 0.6),
    ("1.5B", "Q4_K_M"):  (0.3, -0.8),
    ("1.5B", "Q2_K"):    (-1.0, -1.5),
    # 3B
    ("3B", "Q8_0"):      (0.3, 0.6),
    ("3B", "Q4_K_M"):    (0.3, -0.8),
    ("3B", "Q2_K"):      (0.3, -1.0),
    # 7B
    ("7B", "Q8_0"):      (-1.5, 0.8),
    ("7B", "Q4_K_M"):    (0.5, -1.5),
    ("7B", "Q2_K"):      (0.6, 0.8),
}

# Manual label offsets for other x-axis types
LABEL_OFFSETS_TOK = {
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

LABEL_OFFSETS_PARAMS = {
    ("0.5B", "Q8_0"):    (0.05, 0.8),
    ("0.5B", "Q4_K_M"):  (0.05, -1.2),
    ("0.5B", "Q2_K"):    (0.05, 0.8),
    ("1.5B", "Q8_0"):    (0.1, 0.6),
    ("1.5B", "Q4_K_M"):  (0.1, -0.8),
    ("1.5B", "Q2_K"):    (0.1, -1.0),
    ("3B", "Q8_0"):      (0.1, -0.8),
    ("3B", "Q4_K_M"):    (0.1, 0.6),
    ("3B", "Q2_K"):      (0.1, -1.0),
    ("7B", "Q8_0"):      (0.15, 0.8),
    ("7B", "Q4_K_M"):    (0.15, -1.5),
    ("7B", "Q2_K"):      (0.15, 0.6),
}

# ═══════════════════════════════════════════════════════════════
def build_legend(ax, loc="lower right"):
    """Create unified legend: shapes=size, hatch=Q"""
    leg_size = [Line2D([0],[0], marker=SIZE_SHAPES[s], color='w', 
                       markerfacecolor='gray', markeredgecolor='gray',
                       markersize=[4,6,8,10][i], label=s)
                for i, s in enumerate(SIZES)]
    leg_q = [Patch(facecolor=Q_COLORS[q], edgecolor='white', linewidth=1.5,
                   hatch=Q_HATCHES[q], label=Q_LABELS_FULL[q])
             for q in Q_LEVELS]
    
    leg1 = ax.legend(handles=leg_size, title="Model Size", loc=loc, fontsize=8, 
                    title_fontsize=9, bbox_to_anchor=(0.02, 0.02) if "right" in loc else None)
    ax.add_artist(leg1)
    # Position Q legend above/beside
    ax.legend(handles=leg_q, title="Quantization (hatch)", loc="upper left" if "right" in loc else "upper right",
             fontsize=8, title_fontsize=9)


def scatter_all(ax, x_key, y_key, offsets):
    """Plot all 12 points with shapes + hatches."""
    for size in SIZES:
        for q in Q_LEVELS:
            d = data[(size, q)]
            # Handle special keys
            if x_key == "size_params":
                x_val = SIZE_PARAMS[size]
            else:
                x_val = d[x_key]
            if y_key == "size_params":
                y_val = SIZE_PARAMS[size]
            else:
                y_val = d[y_key]
            shape = SIZE_SHAPES[size]
            sz = SIZE_SIZES[size]
            color = Q_COLORS[q]
            hatch = Q_HATCHES[q]
            
            ax.scatter(x_val, y_val, s=sz, c=color, marker=shape,
                      edgecolors="white", linewidth=1.5, zorder=5, alpha=0.9,
                      hatch=hatch)
            
            # Smart label with offset
            ox, oy = offsets.get((size, q), (0.2, 0.5))
            label = f"{size} {q}"
            ax.annotate(label, (x_val, y_val), textcoords="offset points",
                       xytext=(ox*12, oy*12), ha="center", fontsize=5.5,
                       alpha=0.85,
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


def pareto_frontier(ax, x_key, y_key, color="gray"):
    """Draw Pareto frontier line."""
    pts = sorted([(data[(s,q)][x_key], data[(s,q)][y_key]) for s,q in ALL_POINTS],
                 key=lambda p: p[0])
    frontier_x, frontier_y = [], []
    best_y = -1
    for tx, ty in pts:
        if ty > best_y:
            best_y = ty
            frontier_x.append(tx)
            frontier_y.append(ty)
    frontier_pts = sorted(zip(frontier_x, frontier_y))
    fx, fy = zip(*frontier_pts)
    ax.plot(fx, fy, "--", color=color, alpha=0.35, linewidth=2, label="Pareto frontier")


# ═══════════════════════════════════════════════════════════════
# FIG 1: Exact Match vs Time
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

scatter_all(ax, "time", "exact", LABEL_OFFSETS)
pareto_frontier(ax, "time", "exact")

# Annotations
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

ax.set_xlabel("Inference Time (s/intent)", fontsize=11)
ax.set_ylabel("Exact Match (%)", fontsize=11)
ax.set_title("Phase 2: Accuracy vs Inference Speed\n(Shape=Size, Hatch=Quantization)", fontweight="bold", fontsize=14)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

build_legend(ax, "lower right")
fig.tight_layout()
fig.savefig(FIGURES / "phase2_pareto_exact_vs_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_pareto_exact_vs_time.png")

# ═══════════════════════════════════════════════════════════════
# FIG 2: Action Correct vs Time
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

scatter_all(ax, "time", "action", LABEL_OFFSETS)
pareto_frontier(ax, "time", "action")

bd = data[("3B", "Q4_K_M")]
ax.annotate(f"Best Action: 3B Q4_K_M\n{bd['action']:.1f}% @ {bd['time']:.1f}s",
           xy=(bd["time"], bd["action"]), xytext=(15, 48),
           arrowprops=dict(arrowstyle="->", color="#3498DB", lw=2),
           fontsize=9, color="#3498DB", fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#3498DB", alpha=0.9))

ax.set_xlabel("Inference Time (s/intent)", fontsize=11)
ax.set_ylabel("Action Correct (%)", fontsize=11)
ax.set_title("Phase 2: Action Correct vs Inference Speed\n(Shape=Size, Hatch=Quantization)", fontweight="bold", fontsize=14)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

build_legend(ax, "lower right")
fig.tight_layout()
fig.savefig(FIGURES / "phase2_pareto_action_vs_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_pareto_action_vs_time.png")

# ═══════════════════════════════════════════════════════════════
# FIG 3: Format Accuracy vs Time
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

scatter_all(ax, "time", "format", LABEL_OFFSETS)

# Highlight the 100% cluster
bd = data[("7B", "Q4_K_M")]
ax.annotate(f"100% format: 7B Q4_K_M\n{bd['time']:.1f}s | Also: 7B Q8_0, Q2_K, 0.5B Q8_0",
           xy=(bd["time"], bd["format"]), xytext=(22, 101),
           arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=2),
           fontsize=8, color="#2ECC71", fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ECC71", alpha=0.9))

ax.set_xlabel("Inference Time (s/intent)", fontsize=11)
ax.set_ylabel("Format Accuracy (%)", fontsize=11)
ax.set_title("Phase 2: Format Accuracy vs Inference Speed\n(Shape=Size, Hatch=Quantization)", fontweight="bold", fontsize=14)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_ylim(93, 102)

build_legend(ax, "lower right")
fig.tight_layout()
fig.savefig(FIGURES / "phase2_pareto_format_vs_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_pareto_format_vs_time.png")

# ═══════════════════════════════════════════════════════════════
# FIG 4: Exact vs Throughput (tokens/s)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

scatter_all(ax, "tokens_per_sec", "exact", LABEL_OFFSETS_TOK)

bd = data[("7B", "Q8_0")]
ax.annotate(f"Best Acc: 7B Q8_0\n30.4% @ {bd['tokens_per_sec']:.1f} tok/s",
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

ax.set_xlabel("Throughput (tokens/s)", fontsize=11)
ax.set_ylabel("Exact Match (%)", fontsize=11)
ax.set_title("Phase 2: Accuracy vs Throughput\n(Shape=Size, Hatch=Quantization)", fontweight="bold", fontsize=14)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

build_legend(ax, "lower left")
fig.tight_layout()
fig.savefig(FIGURES / "phase2_pareto_exact_vs_throughput.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_pareto_exact_vs_throughput.png")

# ═══════════════════════════════════════════════════════════════
# FIG 5: Exact vs Model Size (params)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

scatter_all(ax, "size_params", "exact", LABEL_OFFSETS_PARAMS)

ax.set_xlabel("Model Size (Billion Parameters)", fontsize=11)
ax.set_ylabel("Exact Match (%)", fontsize=11)
ax.set_title("Phase 2: Accuracy vs Model Size\n(Shape=Size, Hatch=Quantization)", fontweight="bold", fontsize=14)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_xlim(0, 8)

build_legend(ax, "lower right")
fig.tight_layout()
fig.savefig(FIGURES / "phase2_pareto_exact_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_pareto_exact_vs_size.png")

# ═══════════════════════════════════════════════════════════════
# FIG 6: Time vs Model Size
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

scatter_all(ax, "size_params", "time", LABEL_OFFSETS_PARAMS)

ax.set_xlabel("Model Size (Billion Parameters)", fontsize=11)
ax.set_ylabel("Inference Time (s/intent)", fontsize=11)
ax.set_title("Phase 2: Inference Time vs Model Size\n(Shape=Size, Hatch=Quantization)", fontweight="bold", fontsize=14)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_xlim(0, 8)

build_legend(ax, "upper left")
fig.tight_layout()
fig.savefig(FIGURES / "phase2_pareto_time_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_pareto_time_vs_size.png")

print("\n✅ All Phase 2 Pareto figures regenerated with shapes + hatches!")
