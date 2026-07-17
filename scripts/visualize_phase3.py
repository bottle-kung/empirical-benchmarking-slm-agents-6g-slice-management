#!/usr/bin/env python3
"""
Phase 3 Visualization — Qwen2.5 Quantization Impact (Q2_K vs Q4_K_M vs Q8_0)
All 4 sizes × 3 Q levels = 12 data points
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

PROJECT = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval_v2")
FIGURES = PROJECT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────
SIZES = ["0.5B", "1.5B", "3B", "7B"]
Q_LEVELS = ["Q2_K", "Q4_K_M", "Q8_0"]
Q_LABELS = ["Q2_K\n(2-bit)", "Q4_K_M\n(4-bit)", "Q8_0\n(8-bit)"]
SIZE_PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0}

# Colors
Q_COLORS = {"Q2_K": "#E74C3C", "Q4_K_M": "#3498DB", "Q8_0": "#2ECC71"}
SIZE_MARKERS = {"0.5B": "o", "1.5B": "s", "3B": "D", "7B": "P"}

sources = {
    ("0.5B", "Q2_K"):  PROJECT / "results/phase3/0.5B_q2k.json",
    ("0.5B", "Q8_0"):  PROJECT / "results/phase3/0.5B_q8_0.json",
    ("0.5B", "Q4_K_M"): PROJECT / "results/phase2/benchmark_qwen2_5_0_5b.json",
    ("1.5B", "Q2_K"):  PROJECT / "results/phase3/1.5B_q2k.json",
    ("1.5B", "Q8_0"):  PROJECT / "results/phase3/1.5B_q8_0.json",
    ("1.5B", "Q4_K_M"): PROJECT / "results/phase2/benchmark_qwen2_5_1_5b.json",
    ("3B", "Q2_K"):    PROJECT / "results/phase3/3B_q2k.json",
    ("3B", "Q8_0"):    PROJECT / "results/phase3/3B_q8_0.json",
    ("3B", "Q4_K_M"):  PROJECT / "results/phase2/benchmark_qwen2_5_3b.json",
    ("7B", "Q2_K"):    PROJECT / "results/phase3/7B_q2k.json",
    ("7B", "Q8_0"):    PROJECT / "results/phase3/7B_q8_0.json",
    ("7B", "Q4_K_M"):  PROJECT / "results/phase1/benchmark_qwen2_5_7b.json",
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

# ═══════════════════════════════════════════════════════════════
# FIG 1: Exact Match Heatmap — Size × Q Level
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

matrix = np.zeros((len(SIZES), len(Q_LEVELS)))
annot = np.empty((len(SIZES), len(Q_LEVELS)), dtype=object)
time_matrix = np.zeros((len(SIZES), len(Q_LEVELS)))

for i, size in enumerate(SIZES):
    for j, q in enumerate(Q_LEVELS):
        d = data[(size, q)]
        matrix[i, j] = d["exact"]
        time_matrix[i, j] = d["time"]
        annot[i, j] = f"{d['exact']:.1f}%\n({d['time']:.0f}s)"

im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=15, vmax=32)

for i in range(len(SIZES)):
    for j in range(len(Q_LEVELS)):
        color = "white" if matrix[i, j] > 26 else "black"
        ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=11, fontweight="bold", color=color)

ax.set_xticks(range(len(Q_LEVELS)))
ax.set_xticklabels(Q_LABELS)
ax.set_yticks(range(len(SIZES)))
ax.set_yticklabels(SIZES)
ax.set_title("Qwen2.5 — Exact Match by Size & Quantization", fontweight="bold", fontsize=14, pad=15)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Exact Match (%)", fontsize=11)

fig.tight_layout()
fig.savefig(FIGURES / "phase3_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase3_heatmap.png")

# ═══════════════════════════════════════════════════════════════
# FIG 2: Grouped Bar — Exact Match by Size, grouped by Q
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(SIZES))
width = 0.22

for j, q in enumerate(Q_LEVELS):
    vals = [data[(s, q)]["exact"] for s in SIZES]
    offset = (j - 1) * width
    bars = ax.bar(x + offset, vals, width, label=q, color=Q_COLORS[q], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(SIZES)
ax.set_ylabel("Exact Match (%)")
ax.set_title("Qwen2.5 — Exact Match: Quantization Level Comparison", fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_ylim(0, 35)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(FIGURES / "phase3_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase3_bars.png")

# ═══════════════════════════════════════════════════════════════
# FIG 3: Pareto — Accuracy vs Speed (all 12 points)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

for size in SIZES:
    for q in Q_LEVELS:
        d = data[(size, q)]
        ax.scatter(d["time"], d["exact"], s=200, c=Q_COLORS[q], marker=SIZE_MARKERS[size],
                  edgecolors="white", linewidth=1.5, zorder=5,
                  label=f"{size} {q}" if size == SIZES[0] else "")
        ax.annotate(f"{size}\n{q}", (d["time"], d["exact"]), textcoords="offset points",
                   xytext=(0, 10), ha="center", fontsize=6)

# Pareto frontier
pts = sorted([(data[(s, q)]["time"], data[(s, q)]["exact"], s, q) for s in SIZES for q in Q_LEVELS], key=lambda x: x[0])
frontier_x, frontier_y = [], []
max_y = -1
for tx, ty, _, _ in pts:
    if ty > max_y:
        max_y = ty
        frontier_x.append(tx)
        frontier_y.append(ty)
ax.plot(frontier_x, frontier_y, "--", color="gray", alpha=0.5, linewidth=1.5, label="Pareto frontier")

# Highlight key tradeoffs
# Q2_K at 7B vs Q8_0 at 3B
ax.annotate("7B Q2_K ≈ 3B Q8_0\nin accuracy,\nfaster!", 
           xy=(data[("7B","Q2_K")]["time"], data[("7B","Q2_K")]["exact"]),
           xytext=(16, 26), arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=1.5),
           fontsize=8, color="#9B59B6", fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9B59B6", alpha=0.9))

ax.set_xlabel("Inference Time (s/intent)")
ax.set_ylabel("Exact Match (%)")
ax.set_title("Pareto Frontier: Accuracy vs Speed (All Q Levels)", fontweight="bold")

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=Q_COLORS[q], markersize=10, label=q) for q in Q_LEVELS
] + [
    Line2D([0], [0], marker=SIZE_MARKERS[s], color='w', markerfacecolor='gray', markersize=10, label=s) for s in SIZES
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, ncol=2)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(FIGURES / "phase3_pareto.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase3_pareto.png")

# ═══════════════════════════════════════════════════════════════
# FIG 4: Line Chart — Exact Match vs Q Level, one line per size
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

q_x = [2, 4, 8]  # bit-width proxy
for size in SIZES:
    vals = [data[(size, q)]["exact"] for q in Q_LEVELS]
    ax1.plot(q_x, vals, f"{SIZE_MARKERS[size]}-", color=plt.cm.viridis(SIZES.index(size)/len(SIZES)),
            linewidth=2, markersize=10, label=size)

ax1.set_xlabel("Quantization (bits)")
ax1.set_ylabel("Exact Match (%)")
ax1.set_title("Accuracy vs Quantization Level", fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_xticks([2, 4, 8])

# Right: Δ from Q4_K_M baseline
for size in SIZES:
    baseline = data[(size, "Q4_K_M")]["exact"]
    deltas = [data[(size, q)]["exact"] - baseline for q in Q_LEVELS]
    ax2.bar(np.arange(3) + SIZES.index(size)*0.15 - 0.2, deltas, 0.15,
           color=plt.cm.viridis(SIZES.index(size)/len(SIZES)), label=size, edgecolor="white")

ax2.axhline(y=0, color="black", linewidth=0.5)
ax2.set_xticks(range(3))
ax2.set_xticklabels(Q_LEVELS)
ax2.set_ylabel("Δ from Q4_K_M (pp)")
ax2.set_title("Quantization Impact (vs Q4_K_M Baseline)", fontweight="bold")
ax2.legend(fontsize=9, ncol=2)
ax2.grid(alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("Qwen2.5 — Quantization Sensitivity Analysis", fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "phase3_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase3_sensitivity.png")

# ═══════════════════════════════════════════════════════════════
# FIG 5: Speed Efficiency
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: Time per intent
ax = axes[0]
x = np.arange(len(SIZES))
width = 0.22
for j, q in enumerate(Q_LEVELS):
    vals = [data[(s, q)]["time"] for s in SIZES]
    offset = (j - 1) * width
    bars = ax.bar(x + offset, vals, width, label=q, color=Q_COLORS[q], edgecolor="white")
    for bar, val in zip(bars, vals):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                   f"{val:.1f}s", ha="center", fontsize=7, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(SIZES)
ax.set_ylabel("Seconds per Intent")
ax.set_title("Inference Time", fontweight="bold")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Center: Tokens/sec
ax = axes[1]
for j, q in enumerate(Q_LEVELS):
    vals = [data[(s, q)]["tokens_per_sec"] for s in SIZES]
    offset = (j - 1) * width
    bars = ax.bar(x + offset, vals, width, label=q, color=Q_COLORS[q], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
               f"{val:.1f}", ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(SIZES)
ax.set_ylabel("Tokens per Second")
ax.set_title("Throughput", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: Total time
ax = axes[2]
for j, q in enumerate(Q_LEVELS):
    vals = [data[(s, q)]["total_time"] / 60 for s in SIZES]
    offset = (j - 1) * width
    bars = ax.bar(x + offset, vals, width, label=q, color=Q_COLORS[q], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
               f"{val:.0f}m", ha="center", fontsize=6)
ax.set_xticks(x)
ax.set_xticklabels(SIZES)
ax.set_ylabel("Minutes (200 intents)")
ax.set_title("Total Benchmark Time", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("Qwen2.5 — Inference Speed by Quantization Level", fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "phase3_speed.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase3_speed.png")

# ═══════════════════════════════════════════════════════════════
# FIG 6: Summary Table
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

headers = ["Size", "Q Level", "Format", "Action", "Exact Match", "Time/Intent", "Tok/s", "Total Time"]
rows = []
for size in SIZES:
    for q in Q_LEVELS:
        d = data[(size, q)]
        rows.append([size, q, f"{d['format']:.1f}%", f"{d['action']:.1f}%", 
                    f"{d['exact']:.1f}%", f"{d['time']:.1f}s", f"{d['tokens_per_sec']:.1f}",
                    f"{d['total_time']/60:.1f}m"])

table_data = [headers] + rows
table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)

# Style
for j in range(len(headers)):
    cell = table[0, j]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold", fontsize=9)
    cell.set_height(0.04)

for i, row in enumerate(rows):
    size, q = row[0], row[1]
    bg = Q_COLORS[q]
    # Q column
    table[i+1, 1].set_facecolor(bg)
    table[i+1, 1].set_text_props(color="white", fontweight="bold")
    # Size column
    table[i+1, 0].set_facecolor("#f0f0f0")
    # Highlight best exact per size
    size_best = max([data[(size, qq)]["exact"] for qq in Q_LEVELS])
    if data[(size, q)]["exact"] == size_best:
        table[i+1, 4].set_text_props(color="#27AE60", fontweight="bold")

ax.set_title("Phase 3 Complete Results — Qwen2.5 Quantization Comparison", fontweight="bold", fontsize=13, pad=15)

fig.tight_layout()
fig.savefig(FIGURES / "phase3_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase3_summary_table.png")

# ═══════════════════════════════════════════════════════════════
# FIG 7: Recommendation — "Best Q per Size"
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

# Calculate value score: exact / (time * size_factor) 
recommendations = []
for size in SIZES:
    scores = []
    for q in Q_LEVELS:
        d = data[(size, q)]
        # Score: accuracy per unit time per GB-equivalent
        score = d["exact"] / (d["time"] + 0.1)
        scores.append((q, score, d["exact"], d["time"]))
    best = max(scores, key=lambda x: x[1])
    recommendations.append((size, best))

x = np.arange(len(SIZES))
y = [r[1][2] for r in recommendations]  # exact values
labels = [f"{r[1][0]}\n{r[1][2]:.1f}%" for r in recommendations]
colors = [Q_COLORS[r[1][0]] for r in recommendations]

bars = ax.bar(x, y, color=colors, edgecolor="white", linewidth=2, width=0.5)
for bar, label, rec in zip(bars, labels, recommendations):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
           label, ha="center", fontsize=12, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2 - 0.5,
           f"{rec[0]}", ha="center", fontsize=10, fontweight="bold", color="white")

ax.set_xticks(x)
ax.set_xticklabels(SIZES, fontsize=12)
ax.set_ylabel("Exact Match (%)")
ax.set_title("Best Quantization per Model Size\n(Balancing Accuracy & Speed)", fontweight="bold", fontsize=14)
ax.set_ylim(0, 38)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(FIGURES / "phase3_recommendation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase3_recommendation.png")

print("\n✅ All Phase 3 figures generated!")
