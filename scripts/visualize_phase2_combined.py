#!/usr/bin/env python3
"""
Phase 2 Combined — Size Scaling + Quantization Impact
12 data points: 4 sizes (0.5B, 1.5B, 3B, 7B) × 3 Q levels (Q2_K, Q4_K_M, Q8_0)
Outputs: Pareto plots per matrix + Table plots per metric
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

PROJECT = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval_v2")
FIGURES = PROJECT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────
SIZES = ["0.5B", "1.5B", "3B", "7B"]
Q_LEVELS = ["Q2_K", "Q4_K_M", "Q8_0"]
SIZE_PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0}
SIZE_GB = {"0.5B": 0.38, "1.5B": 1.0, "3B": 1.9, "7B": 4.7}

Q_COLORS = {"Q2_K": "#E74C3C", "Q4_K_M": "#3498DB", "Q8_0": "#2ECC71"}
Q_MARKERS = {"Q2_K": "v", "Q4_K_M": "o", "Q8_0": "^"}
SIZE_COLORS = {"0.5B": "#A8E6CF", "1.5B": "#45B7D1", "3B": "#1B7FC7", "7B": "#0B3D91"}

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
        "total_tokens": s["total_tokens"],
    }

ALL_POINTS = [(s, q) for s in SIZES for q in Q_LEVELS]

# ── Helper: make table plot ────────────────────────────────────
def table_plot(metric_key, title, unit="%", fmt=".1f", cmap="YlOrRd", vmin=None, vmax=None):
    """Create a heatmap table for size × Q level."""
    matrix = np.zeros((len(SIZES), len(Q_LEVELS)))
    annot = np.empty((len(SIZES), len(Q_LEVELS)), dtype=object)
    
    all_vals = [data[(s,q)][metric_key] for s in SIZES for q in Q_LEVELS]
    if vmin is None: vmin = min(all_vals) * 0.9
    if vmax is None: vmax = max(all_vals) * 1.05
    
    for i, size in enumerate(SIZES):
        for j, q in enumerate(Q_LEVELS):
            val = data[(size, q)][metric_key]
            matrix[i, j] = val
            annot[i, j] = f"{val:{fmt}}{unit}" if unit else f"{val:{fmt}}"
    
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    
    for i in range(len(SIZES)):
        for j in range(len(Q_LEVELS)):
            mid = (vmin + vmax) / 2
            color = "white" if matrix[i, j] < mid else "black"
            ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=12, fontweight="bold",
                   color=color)
    
    ax.set_xticks(range(len(Q_LEVELS)))
    ax.set_xticklabels(["Q2_K\n2-bit", "Q4_K_M\n4-bit", "Q8_0\n8-bit"], fontsize=10)
    ax.set_yticks(range(len(SIZES)))
    ax.set_yticklabels(SIZES, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=14, pad=15)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(unit if unit else "value", fontsize=10)
    
    fig.tight_layout()
    fname = f"phase2_table_{metric_key}.png"
    fig.savefig(FIGURES / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {fname}")

# ── Helper: pareto plot ────────────────────────────────────────
def pareto_plot(x_key, y_key, x_label, y_label, title, fname, 
                y_unit="%", x_unit="", annotate_best=True, invert_x=False):
    """Create a Pareto frontier scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    all_x = [data[(s,q)][x_key] for s in SIZES for q in Q_LEVELS]
    x_pad = (max(all_x) - min(all_x)) * 0.15
    
    for size in SIZES:
        for q in Q_LEVELS:
            d = data[(size, q)]
            x_val, y_val = d[x_key], d[y_key]
            
            # Color by Q, shape by size
            marker = Q_MARKERS[q]
            size_marker = {"0.5B": 80, "1.5B": 120, "3B": 160, "7B": 220}[size]
            
            ax.scatter(x_val, y_val, s=size_marker, c=Q_COLORS[q], marker=marker,
                      edgecolors="white", linewidth=1.5, zorder=5, alpha=0.9)
            
            # Label
            label = f"{size}\n{q}"
            offset_y = 10 if y_val < max(all_x)/2 else -12
            ax.annotate(label, (x_val, y_val), textcoords="offset points",
                       xytext=(0, offset_y), ha="center", fontsize=5.5, alpha=0.8)
    
    # Pareto frontier
    sort_reverse = not invert_x
    pts = sorted([(data[(s,q)][x_key], data[(s,q)][y_key]) for s,q in ALL_POINTS],
                 key=lambda p: p[0], reverse=sort_reverse)
    frontier_x, frontier_y = [], []
    best_y = -1 if not invert_x else 999
    for tx, ty in pts:
        if (not invert_x and ty > best_y) or (invert_x and ty < best_y):
            best_y = ty
            frontier_x.append(tx)
            frontier_y.append(ty)
    
    # Sort frontier by x
    frontier_pts = sorted(zip(frontier_x, frontier_y))
    fx, fy = zip(*frontier_pts)
    ax.plot(fx, fy, "--", color="gray", alpha=0.4, linewidth=2, label="Pareto frontier")
    
    # Highlight special points
    if annotate_best:
        # Best accuracy point
        best_acc = max(ALL_POINTS, key=lambda sp: data[sp][y_key])
        bd = data[best_acc]
        ax.annotate(f"Best: {best_acc[0]} {best_acc[1]}\n{y_label}={bd[y_key]:.1f}{y_unit}",
                   xy=(bd[x_key], bd[y_key]), xytext=(bd[x_key] + x_pad, bd[y_key]),
                   arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=2),
                   fontsize=8, color="#E74C3C", fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E74C3C", alpha=0.9))
        
        # Best speed point
        best_spd = min(ALL_POINTS, key=lambda sp: data[sp][x_key])
        sd = data[best_spd]
        ax.annotate(f"Fastest: {best_spd[0]} {best_spd[1]}\n{x_label}={sd[x_key]:.1f}{x_unit}",
                   xy=(sd[x_key], sd[y_key]), xytext=(sd[x_key] + x_pad, sd[y_key] - 3),
                   arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=2),
                   fontsize=8, color="#2ECC71", fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ECC71", alpha=0.9))
    
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legend: Q levels (color)
    from matplotlib.lines import Line2D
    legend_q = [Line2D([0],[0], marker=Q_MARKERS[q], color='w', markerfacecolor=Q_COLORS[q],
                        markersize=10, label=q) for q in Q_LEVELS]
    legend_sz = [Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                        markersize=[4,6,8,10][i], label=s) for i,s in enumerate(SIZES)]
    leg1 = ax.legend(handles=legend_q, title="Q Level", loc="lower left", fontsize=7, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=legend_sz, title="Size", loc="lower right", fontsize=7, title_fontsize=8)
    
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {fname}")

# ═══════════════════════════════════════════════════════════════
# TABLE PLOTS — one per metric
# ═══════════════════════════════════════════════════════════════
print("\n=== Table Plots ===")
table_plot("exact",  "Exact Match (%) — Size × Quantization",         fmt=".1f", vmin=15, vmax=32)
table_plot("action", "Action Correct (%) — Size × Quantization",       fmt=".1f", vmin=20, vmax=48)
table_plot("format", "Format Accuracy (%) — Size × Quantization",      fmt=".1f", vmin=94, vmax=101)
table_plot("time",   "Inference Time (s/intent) — Size × Quantization", unit="s", fmt=".1f", cmap="YlGnBu_r", vmin=2, vmax=20)
table_plot("tokens_per_sec", "Throughput (tokens/s) — Size × Quantization", unit="", fmt=".1f", cmap="YlGn", vmin=1, vmax=11)

# ═══════════════════════════════════════════════════════════════
# PARETO PLOTS — multiple matrices
# ═══════════════════════════════════════════════════════════════
print("\n=== Pareto Plots ===")

pareto_plot("time", "exact",
           "Inference Time (s/intent)", "Exact Match (%)",
           "Pareto: Accuracy vs Inference Speed",
           "phase2_pareto_exact_vs_time.png")

pareto_plot("time", "action",
           "Inference Time (s/intent)", "Action Correct (%)",
           "Pareto: Action Correct vs Inference Speed",
           "phase2_pareto_action_vs_time.png")

pareto_plot("time", "format",
           "Inference Time (s/intent)", "Format Accuracy (%)",
           "Pareto: Format Accuracy vs Inference Speed",
           "phase2_pareto_format_vs_time.png")

pareto_plot("tokens_per_sec", "exact",
           "Throughput (tokens/s)", "Exact Match (%)",
           "Pareto: Accuracy vs Throughput",
           "phase2_pareto_exact_vs_throughput.png", invert_x=True)

# Size (params) vs Exact — color by Q
fig, ax = plt.subplots(figsize=(9, 6))
for size in SIZES:
    for q in Q_LEVELS:
        d = data[(size, q)]
        ax.scatter(SIZE_PARAMS[size], d["exact"], s=180, c=Q_COLORS[q], marker=Q_MARKERS[q],
                  edgecolors="white", linewidth=1.5, zorder=5)
        ax.annotate(q, (SIZE_PARAMS[size], d["exact"]), textcoords="offset points",
                   xytext=(0, 8), ha="center", fontsize=6, alpha=0.7)
ax.set_xlabel("Model Size (Billion Parameters)")
ax.set_ylabel("Exact Match (%)")
ax.set_title("Pareto: Accuracy vs Model Size", fontweight="bold", fontsize=14)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker=Q_MARKERS[q], color='w', markerfacecolor=Q_COLORS[q], markersize=10, label=q) for q in Q_LEVELS]
ax.legend(handles=leg, title="Q Level", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "phase2_pareto_exact_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_pareto_exact_vs_size.png")

# Size vs Time
fig, ax = plt.subplots(figsize=(9, 6))
for size in SIZES:
    for q in Q_LEVELS:
        d = data[(size, q)]
        ax.scatter(SIZE_PARAMS[size], d["time"], s=180, c=Q_COLORS[q], marker=Q_MARKERS[q],
                  edgecolors="white", linewidth=1.5, zorder=5)
        ax.annotate(q, (SIZE_PARAMS[size], d["time"]), textcoords="offset points",
                   xytext=(0, 8), ha="center", fontsize=6, alpha=0.7)
ax.set_xlabel("Model Size (Billion Parameters)")
ax.set_ylabel("Inference Time (s/intent)")
ax.set_title("Pareto: Speed vs Model Size", fontweight="bold", fontsize=14)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
leg2 = [Line2D([0],[0], marker=Q_MARKERS[q], color='w', markerfacecolor=Q_COLORS[q], markersize=10, label=q) for q in Q_LEVELS]
ax.legend(handles=leg2, title="Q Level", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "phase2_pareto_time_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_pareto_time_vs_size.png")

# ═══════════════════════════════════════════════════════════════
# COMPREHENSIVE SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 5.5))
ax.axis("off")

headers = ["Size", "Q", "Format %", "Action %", "Exact %", "Time (s)", "Tok/s", "Total (m)", "Acc/Time"]
rows = []
for size in SIZES:
    for q in Q_LEVELS:
        d = data[(size, q)]
        acc_time = d["exact"] / max(d["time"], 0.1)
        rows.append([size, q, f"{d['format']:.1f}", f"{d['action']:.1f}", f"{d['exact']:.1f}",
                    f"{d['time']:.1f}", f"{d['tokens_per_sec']:.1f}", f"{d['total_time']/60:.1f}", 
                    f"{acc_time:.2f}"])

table_data = [headers] + rows
table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(7.5)

for j in range(len(headers)):
    c = table[0, j]
    c.set_facecolor("#2C3E50")
    c.set_text_props(color="white", fontweight="bold", fontsize=8)
    c.set_height(0.035)

for i, (size, q) in enumerate([(s, q) for s in SIZES for q in Q_LEVELS]):
    ri = i + 1
    table[ri, 1].set_facecolor(Q_COLORS[q])
    table[ri, 1].set_text_props(color="white", fontweight="bold")
    table[ri, 0].set_facecolor("#ECEFF1")
    table[ri, 0].set_text_props(fontweight="bold")
    # Highlight best per size
    best_exact = max(data[(size, qq)]["exact"] for qq in Q_LEVELS)
    if data[(size, q)]["exact"] == best_exact:
        table[ri, 4].set_text_props(color="#1565C0", fontweight="bold")
    # Highlight fastest per size
    best_time = min(data[(size, qq)]["time"] for qq in Q_LEVELS)
    if data[(size, q)]["time"] == best_time:
        table[ri, 5].set_text_props(color="#2E7D32", fontweight="bold")
    # Highlight best acc/time
    best_ratio = max(data[(size, qq)]["exact"] / max(data[(size, qq)]["time"], 0.1) for qq in Q_LEVELS)
    ratio = data[(size, q)]["exact"] / max(data[(size, q)]["time"], 0.1)
    if abs(ratio - best_ratio) < 0.01:
        table[ri, 8].set_text_props(color="#E65100", fontweight="bold")

ax.set_title("Phase 2 — Qwen2.5 Size Scaling + Quantization: Complete Results", 
            fontweight="bold", fontsize=12, pad=20)

fig.tight_layout()
fig.savefig(FIGURES / "phase2_complete_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_complete_summary.png")

print("\n✅ All Phase 2 Combined figures generated!")
print(f"   {FIGURES}")
