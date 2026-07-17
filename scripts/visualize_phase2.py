#!/usr/bin/env python3
"""
Phase 2 Visualization — Qwen Size Scaling (0.5B → 7B)
Shows how model size affects performance, speed, and resources.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

PROJECT = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval_v2")
FIGURES = PROJECT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────
SIZES = ["0.5B", "1.5B", "3B", "7B"]
SIZE_GB = {"0.5B": 0.38, "1.5B": 1.0, "3B": 1.9, "7B": 4.7}
SIZE_PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0}

files = {
    "0.5B": PROJECT / "results/phase2/benchmark_qwen2_5_0_5b.json",
    "1.5B": PROJECT / "results/phase2/benchmark_qwen2_5_1_5b.json",
    "3B":   PROJECT / "results/phase2/benchmark_qwen2_5_3b.json",
    "7B":   PROJECT / "results/phase1/benchmark_qwen2_5_7b.json",
}

data = {}
for size, path in files.items():
    with open(path) as f:
        d = json.load(f)
    s = d["summary"]
    # Count domain-level breakdowns
    by_domain = {}
    for r in d["results"]:
        dom = r["domain"]
        if dom not in by_domain:
            by_domain[dom] = {"total": 0, "action_correct": 0, "match_sum": 0.0}
        by_domain[dom]["total"] += 1
        if r.get("action_correct"):
            by_domain[dom]["action_correct"] += 1
        by_domain[dom]["match_sum"] += r.get("match_pct", 0)
    
    for dom in by_domain:
        by_domain[dom]["action_pct"] = 100 * by_domain[dom]["action_correct"] / by_domain[dom]["total"]
        by_domain[dom]["avg_match"] = by_domain[dom]["match_sum"] / by_domain[dom]["total"]

    data[size] = {
        "format": s["format_valid_pct"],
        "action": s["action_correct_pct"],
        "exact": s["avg_match_pct"],
        "time": s["avg_time_s"],
        "tokens_per_sec": s["avg_tokens_per_sec"],
        "total_time": s["total_time_s"],
        "total_tokens": s["total_tokens"],
        "size_gb": SIZE_GB[size],
        "size_params": SIZE_PARAMS[size],
        "by_domain": by_domain,
    }

# ── Color Palette ──────────────────────────────────────────────
QCOLORS = ["#A8E6CF", "#45B7D1", "#1B7FC7", "#0B3D91"]  # Light → Dark blue gradient
SIZE_LABELS_EN = ["0.5B", "1.5B", "3B", "7B"]
SIZE_LABELS = ["0.5B\n(0.4 GB)", "1.5B\n(1.0 GB)", "3B\n(1.9 GB)", "7B\n(4.7 GB)"]

# ═══════════════════════════════════════════════════════════════
# FIG 1: Performance Metrics vs Model Size (Line chart)
# ═══════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(SIZES))
width = 0.22

format_vals = [data[s]["format"] for s in SIZES]
action_vals = [data[s]["action"] for s in SIZES]
exact_vals  = [data[s]["exact"] for s in SIZES]

bars1 = ax1.bar(x - width, format_vals, width, label="Format Accuracy", color="#2ECC71", edgecolor="white", linewidth=0.5)
bars2 = ax1.bar(x,        action_vals, width, label="Action Correct", color="#3498DB", edgecolor="white", linewidth=0.5)
bars3 = ax1.bar(x + width, exact_vals,  width, label="Exact Match",    color="#9B59B6", edgecolor="white", linewidth=0.5)

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(0, 105)
ax1.set_xticks(x)
ax1.set_xticklabels(SIZE_LABELS)
ax1.set_title("Qwen2.5 — Accuracy vs Model Size\n", fontweight="bold")
ax1.legend(loc="lower right", framealpha=0.9)
ax1.grid(axis="y", alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(FIGURES / "phase2_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_accuracy.png")

# ═══════════════════════════════════════════════════════════════
# FIG 2: Performance Line + Pareto Front
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Line chart
param_vals = [data[s]["size_params"] for s in SIZES]
ax1.plot(param_vals, format_vals, "o-", color="#2ECC71", linewidth=2.5, markersize=10, label="Format Accuracy")
ax1.plot(param_vals, action_vals, "s-", color="#3498DB", linewidth=2.5, markersize=10, label="Action Correct")
ax1.plot(param_vals, exact_vals,  "D-", color="#9B59B6", linewidth=2.5, markersize=10, label="Exact Match")

for i, s in enumerate(SIZES):
    ax1.annotate(s, (param_vals[i], exact_vals[i]), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

ax1.set_xlabel("Model Size (Billion Parameters)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Accuracy vs Parameters", fontweight="bold")
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylim(0, 105)

# Right: Pareto — Accuracy vs Speed
time_vals = [data[s]["time"] for s in SIZES]
for i, s in enumerate(SIZES):
    ax2.scatter(time_vals[i], exact_vals[i], s=300, c=QCOLORS[i], edgecolors="white",
               linewidth=2, zorder=5)
    ax2.annotate(SIZE_LABELS_EN[i], (time_vals[i], exact_vals[i]),
                textcoords="offset points", xytext=(0, 14), ha="center",
                fontsize=10, fontweight="bold")

# Pareto frontier
pts = sorted(zip(time_vals, exact_vals), key=lambda x: x[0])
frontier_x, frontier_y = [], []
max_y = -1
for tx, ty in pts:
    if ty > max_y:
        max_y = ty
        frontier_x.append(tx)
        frontier_y.append(ty)
if len(frontier_x) >= 2:
    ax2.plot(frontier_x, frontier_y, "--", color="gray", alpha=0.5, linewidth=1.5, label="Pareto frontier")
    # Shade dominated region (skip fill_between to avoid shape mismatch)

# Sweet spot annotation
ax2.annotate("Sweet Spot\n92% of 7B accuracy\n1.6× faster",
            xy=(time_vals[2], exact_vals[2]), xytext=(time_vals[2] + 4, exact_vals[2] - 2),
            arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5),
            fontsize=9, color="#E74C3C", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E74C3C", alpha=0.9))

ax2.set_xlabel("Inference Time (s/intent)")
ax2.set_ylabel("Exact Match (%)")
ax2.set_title("Pareto: Accuracy vs Speed", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(FIGURES / "phase2_performance_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_performance_vs_size.png")

# ═══════════════════════════════════════════════════════════════
# FIG 3: Resources — Time, Tokens/s, Energy
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: Inference Time (bar)
ax = axes[0]
bars = ax.bar(SIZE_LABELS_EN, time_vals, color=QCOLORS, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, time_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f"{val:.2f}s", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Seconds per Intent")
ax.set_title("Avg Inference Time", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Center: Throughput (tokens/s)
ax = axes[1]
tps_vals = [data[s]["tokens_per_sec"] for s in SIZES]
bars = ax.bar(SIZE_LABELS_EN, tps_vals, color=QCOLORS, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, tps_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
            f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Tokens per Second")
ax.set_title("Inference Throughput", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: Total Time (minutes)
ax = axes[2]
total_min = [data[s]["total_time"] / 60 for s in SIZES]
bars = ax.bar(SIZE_LABELS_EN, total_min, color=QCOLORS, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, total_min):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f"{val:.1f}m", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Minutes (200 intents)")
ax.set_title("Total Benchmark Time", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("Qwen2.5 — Resource Utilization by Model Size", fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "phase2_resources_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_resources_vs_size.png")

# ═══════════════════════════════════════════════════════════════
# FIG 4: Efficiency — Tokens/Watt, J/intent, Match%/GB
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Energy efficiency (J/intent) + Match%/GB
ax = axes[0]
# Estimate joules based on runtime × estimated power (CPU ~135W)
power_w = 135  # Watts (TDP estimate for 16-core CPU at 95% utilization)
joules = [data[s]["time"] * power_w for s in SIZES]  # J/intent
match_per_gb = [data[s]["exact"] / data[s]["size_gb"] for s in SIZES]

ax2_ = ax.twinx()
bars1 = ax.bar(np.arange(len(SIZES)) - 0.15, joules, 0.3, color="#E74C3C", alpha=0.7,
               edgecolor="white", label="Energy (J/intent)")
bars2 = ax2_.bar(np.arange(len(SIZES)) + 0.15, match_per_gb, 0.3, color="#2ECC71", alpha=0.7,
                 edgecolor="white", label="Match% / GB")

ax.set_xticks(np.arange(len(SIZES)))
ax.set_xticklabels(SIZE_LABELS_EN)
ax.set_ylabel("Joules per Intent", color="#E74C3C")
ax2_.set_ylabel("Exact Match % per GB", color="#2ECC71")
ax.set_title("Energy & Memory Efficiency", fontweight="bold")
ax.spines["top"].set_visible(False)

# Right: Speed efficiency
ax = axes[1]
tokens_per_joule = [data[s]["total_tokens"] / (data[s]["total_time"] * power_w) for s in SIZES]
intents_per_min = [60 / data[s]["time"] for s in SIZES]

ax3_ = ax.twinx()
bars3 = ax.bar(np.arange(len(SIZES)) - 0.15, tokens_per_joule, 0.3, color="#F39C12", alpha=0.7,
               edgecolor="white", label="Tokens/Joule")
bars4 = ax3_.bar(np.arange(len(SIZES)) + 0.15, intents_per_min, 0.3, color="#3498DB", alpha=0.7,
                  edgecolor="white", label="Intents/min")

ax.set_xticks(np.arange(len(SIZES)))
ax.set_xticklabels(SIZE_LABELS_EN)
ax.set_ylabel("Tokens per Joule", color="#F39C12")
ax3_.set_ylabel("Intents per Minute", color="#3498DB")
ax.set_title("Throughput Efficiency", fontweight="bold")
ax.spines["top"].set_visible(False)

fig.suptitle("Qwen2.5 — Efficiency Metrics vs Model Size", fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "phase2_efficiency_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_efficiency_vs_size.png")

# ═══════════════════════════════════════════════════════════════
# FIG 5: Domain Breakdown
# ═══════════════════════════════════════════════════════════════
domains = ["Slicing Provisioning", "Scaling Request", "Conflict Resolution"]
domain_colors = {"Slicing Provisioning": "#3498DB", "Scaling Request": "#E67E22", "Conflict Resolution": "#E74C3C"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: Action by domain (grouped bars)
ax = axes[0]
x = np.arange(len(SIZES))
width = 0.2
for j, dom in enumerate(domains):
    vals = [data[s]["by_domain"][dom]["action_pct"] for s in SIZES]
    offset = (j - 1) * width
    bars = ax.bar(x + offset, vals, width, label=dom.replace("Slicing ", ""), 
                  color=domain_colors[dom], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{val:.0f}%", ha="center", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(SIZE_LABELS_EN)
ax.set_ylabel("Action Correct (%)")
ax.set_title("Action Accuracy by Domain", fontweight="bold")
ax.legend(fontsize=8, ncol=3, loc="lower right")
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: Improvement Curve (Δ from 0.5B)
ax = axes[1]
base_exact = data["0.5B"]["exact"]
markers = ["o", "s", "D", "P"]
for j, dom in enumerate(domains):
    vals = [data[s]["by_domain"][dom]["avg_match"] for s in SIZES]
    ax.plot(param_vals, vals, f"{markers[j]}-", color=domain_colors[dom],
            linewidth=2, markersize=8, label=dom.replace("Slicing ", ""))

ax.set_xlabel("Model Size (Billion Parameters)")
ax.set_ylabel("Avg Match (%)")
ax.set_title("Match Quality Scaling by Domain", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("Qwen2.5 — Domain-Level Performance Scaling", fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "phase2_domain_scaling.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_domain_scaling.png")

# ═══════════════════════════════════════════════════════════════
# FIG 6: Summary Table
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.axis("off")

headers = ["Model", "Format", "Action", "Exact\nMatch", "Time\n(s/intent)", "Tokens\n/s", "Total\n(min)", "Size\n(GB)"]
rows = []
for s in SIZES:
    d = data[s]
    rows.append([
        f"Qwen2.5-{s}",
        f"{d['format']:.1f}%",
        f"{d['action']:.1f}%",
        f"{d['exact']:.1f}%",
        f"{d['time']:.1f}s",
        f"{d['tokens_per_sec']:.1f}",
        f"{d['total_time']/60:.1f}",
        f"{d['size_gb']:.2f}"
    ])

table_data = [headers] + rows
table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)

# Style header
for j in range(len(headers)):
    cell = table[0, j]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold")
    cell.set_height(0.06)

# Style rows
for i in range(len(rows)):
    bg = QCOLORS[i]
    for j in range(len(headers)):
        cell = table[i+1, j]
        if j == 0:
            cell.set_facecolor(bg)
            cell.set_text_props(fontweight="bold", color="white" if i >= 2 else "#333")
        else:
            cell.set_facecolor("white")
            cell.set_edgecolor("#E0E0E0")
    # Highlight best values
    table[i+1, 1].set_text_props(color="#27AE60" if i == 3 else "#555", fontweight="bold")
    table[i+1, 3].set_text_props(color="#27AE60" if i == 3 else "#555")

# Highlight best row (7B)
for j in range(len(headers)):
    table[4, j].set_edgecolor("#27AE60")
    table[4, j].set_linewidth(1.5)

ax.set_title("Qwen2.5 Size Scaling — Summary", fontweight="bold", fontsize=13, pad=15)

fig.tight_layout()
fig.savefig(FIGURES / "phase2_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_summary_table.png")

# ═══════════════════════════════════════════════════════════════
# FIG 7: Diminishing Returns Analysis
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Top: Marginal gain per parameter doubling
doublings = ["0.5B→1.5B\n(3×)", "1.5B→3B\n(2×)", "3B→7B\n(2.3×)"]
marginal_exact = [
    data["1.5B"]["exact"] - data["0.5B"]["exact"],
    data["3B"]["exact"]   - data["1.5B"]["exact"],
    data["7B"]["exact"]   - data["3B"]["exact"],
]
marginal_time = [
    data["1.5B"]["time"] - data["0.5B"]["time"],
    data["3B"]["time"]   - data["1.5B"]["time"],
    data["7B"]["time"]   - data["3B"]["time"],
]

colors_marginal = ["#A8E6CF", "#45B7D1", "#0B3D91"]
x = np.arange(3)
width = 0.3

bars1 = ax1.bar(x - width/2, marginal_exact, width, color=colors_marginal, edgecolor="white",
                label="Δ Exact Match (%)")
for bar, val in zip(bars1, marginal_exact):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
            f"+{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(doublings)
ax1.set_ylabel("Percentage Points Gain")
ax1.set_title("Marginal Accuracy Gain per Size Increase", fontweight="bold")
ax1.legend(fontsize=10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="y", alpha=0.3)

# Add annotation arrows
ax1.annotate("Biggest jump\n+7.6pp", xy=(0, marginal_exact[0]), xytext=(0, marginal_exact[0] + 2),
            arrowprops=dict(arrowstyle="->", color="#E74C3C"), fontsize=9, color="#E74C3C",
            ha="center", fontweight="bold")
ax1.annotate("Diminishing\nreturns", xy=(2, marginal_exact[2]), xytext=(2, marginal_exact[2] + 1.5),
            arrowprops=dict(arrowstyle="->", color="#E74C3C"), fontsize=9, color="#E74C3C",
            ha="center", fontweight="bold")

# Bottom: ROI — Accuracy per GB of model size
ax2_obj = ax2.twinx()
match_per_gb_vals = [data[s]["exact"] / data[s]["size_gb"] for s in SIZES]
time_per_match = [data[s]["time"] / max(data[s]["exact"], 0.1) for s in SIZES]

ax2.plot(param_vals, match_per_gb_vals, "D-", color="#2ECC71", linewidth=2.5, markersize=12,
         label="Match% per GB")
ax2_obj.plot(param_vals, time_per_match, "s-", color="#E74C3C", linewidth=2.5, markersize=12,
             label="Seconds per Match% point")

ax2.set_xlabel("Model Size (Billion Parameters)")
ax2.set_ylabel("Match% / GB", color="#2ECC71")
ax2_obj.set_ylabel("Seconds / Match% point", color="#E74C3C")
ax2.set_title("ROI: Accuracy Efficiency vs Model Size", fontweight="bold")
ax2.grid(alpha=0.3)
ax2.spines["top"].set_visible(False)

# Combine legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_obj.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

fig.tight_layout()
fig.savefig(FIGURES / "phase2_diminishing_returns.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ phase2_diminishing_returns.png")

print("\n✅ All Phase 2 figures generated!")
