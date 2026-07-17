#!/usr/bin/env python3
"""
SLM 6G Eval v2 — Full Visualization with Resource Metrics
Uses new re-run results from /results/resource/
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

PROJECT = Path("/jupyter_workspace/rbru/slm_6g_eval_v2")
RES_DIR = PROJECT / "results" / "resource"
FIGURES = PROJECT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ─── LOAD DATA ─────────────────────────────────────────────────
SIZES = ["0.5B", "1.5B", "3B", "7B"]
Q_LEVELS = ["Q2_K", "Q4_K_M", "Q8_0"]
SIZE_VALS = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0}

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
    
    # Accuracy
    match_pcts = [r.get("match_pct", 0) for r in results]
    format_valids = sum(1 for r in results if r.get("format_valid"))
    action_corrects = sum(1 for r in results if r.get("action_correct"))
    
    # Time & Tokens
    durations = [r.get("duration_s", 0) for r in results]
    tokens_all = [r.get("tokens", 0) for r in results]
    total_tok = sum(tokens_all)
    total_time = sum(durations)
    
    # Resource
    energies = [r.get("energy_j", 0) for r in results]
    mem_avgs = [r.get("avg_memory_mb", 0) for r in results]
    mem_peaks = [r.get("peak_memory_mb", 0) for r in results]
    cpu_pcts = [r.get("cpu_pct", 0) for r in results]
    
    # Domain breakdown
    domains = defaultdict(lambda: {"count": 0, "action": 0, "match_sum": 0.0, "time_sum": 0.0, "energy_sum": 0.0})
    for r in results:
        dom = r.get("domain", "Unknown")
        domains[dom]["count"] += 1
        if r.get("action_correct"):
            domains[dom]["action"] += 1
        domains[dom]["match_sum"] += r.get("match_pct", 0)
        domains[dom]["time_sum"] += r.get("duration_s", 0)
        domains[dom]["energy_sum"] += r.get("energy_j", 0)
    
    DATA[(size, q)] = {
        "n": n,
        "avg_match_pct": np.mean(match_pcts),
        "format_pct": format_valids / n * 100,
        "action_pct": action_corrects / n * 100,
        "avg_time_s": np.mean(durations),
        "total_time_s": total_time,
        "avg_tokens": np.mean(tokens_all),
        "tokens_per_sec": total_tok / total_time if total_time > 0 else 0,
        "avg_energy_j": np.mean(energies),
        "total_energy_j": sum(energies),
        "energy_per_token_j": sum(energies) / total_tok if total_tok > 0 else 0,
        "avg_memory_mb": np.mean(mem_avgs),
        "peak_memory_mb": max(mem_peaks) if mem_peaks else 0,
        "avg_cpu_pct": np.mean(cpu_pcts),
        "domains": {d: {
            "action_pct": v["action"] / v["count"] * 100,
            "avg_match_pct": v["match_sum"] / v["count"],
            "avg_time": v["time_sum"] / v["count"],
            "avg_energy": v["energy_sum"] / v["count"],
        } for d, v in domains.items()},
    }

# ─── STYLE ─────────────────────────────────────────────────────
SIZE_SHAPES = {"0.5B": "s", "1.5B": "D", "3B": "^", "7B": "o"}
SIZE_SIZES = {"0.5B": 80, "1.5B": 120, "3B": 160, "7B": 200}
SIZE_COLORS = {"0.5B": "#E67E22", "1.5B": "#2ECC71", "3B": "#3498DB", "7B": "#9B59B6"}
Q_COLORS = {"Q8_0": "#2ECC71", "Q4_K_M": "#3498DB", "Q2_K": "#E74C3C"}
Q_STYLES = {"Q8_0": "-", "Q4_K_M": "--", "Q2_K": ":"}

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Size Scaling — Avg Match Pct vs Model Size (by Q)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
metrics = [
    ("avg_match_pct", "Avg Match %", "%", axes[0, 0]),
    ("action_pct", "Action Accuracy %", "%", axes[0, 1]),
    ("format_pct", "Format Valid %", "%", axes[0, 2]),
    ("avg_time_s", "Avg Inference Time (s)", "s", axes[1, 0]),
    ("tokens_per_sec", "Tokens/sec", "tok/s", axes[1, 1]),
    ("avg_energy_j", "Avg Energy per Intent (J)", "J", axes[1, 2]),
]

for metric, title, unit, ax in metrics:
    for q in Q_LEVELS:
        xs, ys = [], []
        for s in SIZES:
            if (s, q) in DATA:
                xs.append(SIZE_VALS[s])
                ys.append(DATA[(s, q)][metric])
        ax.plot(xs, ys, Q_STYLES[q] + "o", color=Q_COLORS[q], 
                label=q.replace("_", " "), markersize=8, linewidth=2)
    ax.set_xlabel("Model Size (B params)")
    ax.set_ylabel(unit)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Phase 2: Size Scaling & Quantization Impact — All Metrics", 
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
path = FIGURES / "v2_size_scaling_all_metrics.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ {path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Pareto Frontier — Accuracy vs Time (all 12 points)
# ═══════════════════════════════════════════════════════════════
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(10, 8))

for (size, q), d in DATA.items():
    marker = SIZE_SHAPES[size]
    ms = SIZE_SIZES[size] * 0.7
    color = Q_COLORS[q]
    ax.scatter(d["avg_time_s"], d["avg_match_pct"], 
               marker=marker, s=ms, c=color, edgecolors="black", linewidth=1.5,
               zorder=3)
    ax.annotate(f"{size}\n{q}", (d["avg_time_s"], d["avg_match_pct"]),
                fontsize=7, ha="center", va="bottom",
                xytext=(0, 8), textcoords="offset points")

size_legend = [Line2D([0], [0], marker=m, color="gray", markersize=np.sqrt(ss)/2,
                       label=s, linestyle="None", markerfacecolor="gray")
               for s, m, ss in zip(SIZES, SIZE_SHAPES.values(), SIZE_SIZES.values())]
q_legend = [Patch(facecolor=c, label=q) for q, c in Q_COLORS.items()]

leg1 = ax.legend(handles=size_legend, title="Model Size", loc="upper left", fontsize=8)
ax.add_artist(leg1)
ax.legend(handles=q_legend, title="Quantization", loc="lower right", fontsize=8)

ax.set_xlabel("Avg Inference Time (s)")
ax.set_ylabel("Avg Match %")
ax.set_title("Pareto Frontier: Accuracy vs Inference Time")
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = FIGURES / "v2_pareto_accuracy_vs_time.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ {path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Pareto Frontier — Accuracy vs Energy
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))

for (size, q), d in DATA.items():
    marker = SIZE_SHAPES[size]
    ms = SIZE_SIZES[size] * 0.7
    color = Q_COLORS[q]
    ax.scatter(d["avg_energy_j"], d["avg_match_pct"],
               marker=marker, s=ms, c=color, edgecolors="black", linewidth=1.5,
               zorder=3)
    ax.annotate(f"{size}\n{q}", (d["avg_energy_j"], d["avg_match_pct"]),
                fontsize=7, ha="center", va="bottom",
                xytext=(0, 8), textcoords="offset points")

size_legend2 = [Line2D([0], [0], marker=m, color="gray", markersize=np.sqrt(ss)/2,
                        label=s, linestyle="None", markerfacecolor="gray")
                for s, m, ss in zip(SIZES, SIZE_SHAPES.values(), SIZE_SIZES.values())]
q_legend2 = [Patch(facecolor=c, label=q) for q, c in Q_COLORS.items()]

leg2a = ax.legend(handles=size_legend2, title="Model Size", loc="upper left", fontsize=8)
ax.add_artist(leg2a)
ax.legend(handles=q_legend2, title="Quantization", loc="lower right", fontsize=8)

ax.set_xlabel("Avg Energy per Intent (J)")
ax.set_ylabel("Avg Match %")
ax.set_title("Pareto Frontier: Accuracy vs Energy Consumption")
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = FIGURES / "v2_pareto_accuracy_vs_energy.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ {path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Resource Usage — Memory, Energy, Efficiency
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 4a: Peak Memory vs Model Size
ax = axes[0, 0]
for q in Q_LEVELS:
    xs, ys = [], []
    for s in SIZES:
        if (s, q) in DATA:
            xs.append(SIZE_VALS[s])
            ys.append(DATA[(s, q)]["peak_memory_mb"])
    ax.plot(xs, ys, Q_STYLES[q] + "o", color=Q_COLORS[q],
            label=q.replace("_", " "), markersize=8, linewidth=2)
ax.set_xlabel("Model Size (B params)")
ax.set_ylabel("Peak Memory (MB)")
ax.set_title("Peak Memory Usage")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 4b: Avg Energy vs Model Size
ax = axes[0, 1]
for q in Q_LEVELS:
    xs, ys = [], []
    for s in SIZES:
        if (s, q) in DATA:
            xs.append(SIZE_VALS[s])
            ys.append(DATA[(s, q)]["avg_energy_j"])
    ax.plot(xs, ys, Q_STYLES[q] + "o", color=Q_COLORS[q],
            label=q.replace("_", " "), markersize=8, linewidth=2)
ax.set_xlabel("Model Size (B params)")
ax.set_ylabel("Energy per Intent (J)")
ax.set_title("Energy Consumption per Intent")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 4c: Energy Efficiency — Avg Match / Energy (higher = more efficient)
ax = axes[1, 0]
for q in Q_LEVELS:
    xs, ys = [], []
    for s in SIZES:
        if (s, q) in DATA:
            d = DATA[(s, q)]
            eff = d["avg_match_pct"] / d["avg_energy_j"] * 100 if d["avg_energy_j"] > 0 else 0
            xs.append(SIZE_VALS[s])
            ys.append(eff)
    ax.plot(xs, ys, Q_STYLES[q] + "o", color=Q_COLORS[q],
            label=q.replace("_", " "), markersize=8, linewidth=2)
ax.set_xlabel("Model Size (B params)")
ax.set_ylabel("Match% / Joule × 100")
ax.set_title("Accuracy-per-Energy Efficiency")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 4d: Tokens/sec vs Model Size
ax = axes[1, 1]
for q in Q_LEVELS:
    xs, ys = [], []
    for s in SIZES:
        if (s, q) in DATA:
            xs.append(SIZE_VALS[s])
            ys.append(DATA[(s, q)]["tokens_per_sec"])
    ax.plot(xs, ys, Q_STYLES[q] + "o", color=Q_COLORS[q],
            label=q.replace("_", " "), markersize=8, linewidth=2)
ax.set_xlabel("Model Size (B params)")
ax.set_ylabel("Tokens/sec")
ax.set_title("Inference Throughput")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

fig.suptitle("Resource Metrics — Memory, Energy, Throughput", fontsize=13, fontweight="bold")
fig.tight_layout()
path = FIGURES / "v2_resource_metrics.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ {path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Combined Heatmap — Size × Q for all metrics
# ═══════════════════════════════════════════════════════════════
heat_metrics = [
    ("avg_match_pct", "Avg Match %", "RdYlGn"),
    ("action_pct", "Action Accuracy %", "RdYlGn"),
    ("avg_time_s", "Avg Time (s)", "YlOrRd_r"),
    ("avg_energy_j", "Energy/Intent (J)", "YlOrRd_r"),
    ("peak_memory_mb", "Peak Memory (MB)", "YlOrRd_r"),
    ("tokens_per_sec", "Tokens/sec", "RdYlGn"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
for idx, (metric, title, cmap) in enumerate(heat_metrics):
    ax = axes[idx // 3, idx % 3]
    mat = np.zeros((len(SIZES), len(Q_LEVELS)))
    for i, s in enumerate(SIZES):
        for j, q in enumerate(Q_LEVELS):
            if (s, q) in DATA:
                mat[i, j] = DATA[(s, q)][metric]
            else:
                mat[i, j] = np.nan
    
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(Q_LEVELS)))
    ax.set_xticklabels([q.replace("_", " ") for q in Q_LEVELS])
    ax.set_yticks(range(len(SIZES)))
    ax.set_yticklabels(SIZES)
    ax.set_title(title)
    
    # Annotate values
    for i in range(len(SIZES)):
        for j in range(len(Q_LEVELS)):
            if not np.isnan(mat[i, j]):
                val = mat[i, j]
                text = f"{val:.1f}" if val < 100 else f"{val:.0f}"
                ax.text(j, i, text, ha="center", va="center", 
                       fontsize=8, fontweight="bold",
                       color="white" if val > np.nanmean(mat) else "black")
    
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("Phase 2: Full Results Heatmap — Size × Quantization", 
             fontsize=14, fontweight="bold")
fig.tight_layout()
path = FIGURES / "v2_heatmap_all_metrics.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ {path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Domain-Level Analysis by Size
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

q_fixed = "Q4_K_M"  # Use Q4_K_M for fair domain comparison

# 6a: Action Accuracy by Domain
ax = axes[0, 0]
domains_list = ["Slicing Provisioning", "Scaling Request", "Conflict Resolution"]
x = np.arange(len(domains_list))
width = 0.2
for i, s in enumerate(SIZES):
    if (s, q_fixed) in DATA:
        vals = [DATA[(s, q_fixed)]["domains"].get(d, {}).get("action_pct", 0) for d in domains_list]
        ax.bar(x + i * width, vals, width, label=s, color=SIZE_COLORS[s], edgecolor="black")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([d.replace(" ", "\n") for d in domains_list], fontsize=8)
ax.set_ylabel("Action Accuracy %")
ax.set_title(f"Action Accuracy by Domain ({q_fixed})")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

# 6b: Avg Match Pct by Domain
ax = axes[0, 1]
for i, s in enumerate(SIZES):
    if (s, q_fixed) in DATA:
        vals = [DATA[(s, q_fixed)]["domains"].get(d, {}).get("avg_match_pct", 0) for d in domains_list]
        ax.bar(x + i * width, vals, width, label=s, color=SIZE_COLORS[s], edgecolor="black")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([d.replace(" ", "\n") for d in domains_list], fontsize=8)
ax.set_ylabel("Avg Match %")
ax.set_title(f"Avg Match by Domain ({q_fixed})")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

# 6c: Avg Time by Domain
ax = axes[1, 0]
for i, s in enumerate(SIZES):
    if (s, q_fixed) in DATA:
        vals = [DATA[(s, q_fixed)]["domains"].get(d, {}).get("avg_time", 0) for d in domains_list]
        ax.bar(x + i * width, vals, width, label=s, color=SIZE_COLORS[s], edgecolor="black")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([d.replace(" ", "\n") for d in domains_list], fontsize=8)
ax.set_ylabel("Avg Time (s)")
ax.set_title(f"Avg Time by Domain ({q_fixed})")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

# 6d: Avg Energy by Domain
ax = axes[1, 1]
for i, s in enumerate(SIZES):
    if (s, q_fixed) in DATA:
        vals = [DATA[(s, q_fixed)]["domains"].get(d, {}).get("avg_energy", 0) for d in domains_list]
        ax.bar(x + i * width, vals, width, label=s, color=SIZE_COLORS[s], edgecolor="black")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([d.replace(" ", "\n") for d in domains_list], fontsize=8)
ax.set_ylabel("Avg Energy (J)")
ax.set_title(f"Avg Energy by Domain ({q_fixed})")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Domain-Level Analysis — Provisioning vs Scaling vs Conflict", 
             fontsize=13, fontweight="bold")
fig.tight_layout()
path = FIGURES / "v2_domain_analysis.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ {path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 7: Quantization Impact (fixed size, vary Q)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

q_labels = ["Q2_K", "Q4_K_M", "Q8_0"]
q_x = np.arange(len(q_labels))
q_width = 0.2

# 7a: Avg Match by Q level
ax = axes[0, 0]
for i, s in enumerate(SIZES):
    vals = [DATA[(s, q)]["avg_match_pct"] for q in q_labels if (s, q) in DATA]
    ax.bar(q_x + i * q_width, vals, q_width, label=s, color=SIZE_COLORS[s], edgecolor="black")
ax.set_xticks(q_x + q_width * 1.5)
ax.set_xticklabels([q.replace("_", " ") for q in q_labels], fontsize=9)
ax.set_ylabel("Avg Match %")
ax.set_title("Avg Match vs Quantization")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

# 7b: Energy vs Q level
ax = axes[0, 1]
for i, s in enumerate(SIZES):
    vals = [DATA[(s, q)]["avg_energy_j"] for q in q_labels if (s, q) in DATA]
    ax.bar(q_x + i * q_width, vals, q_width, label=s, color=SIZE_COLORS[s], edgecolor="black")
ax.set_xticks(q_x + q_width * 1.5)
ax.set_xticklabels([q.replace("_", " ") for q in q_labels], fontsize=9)
ax.set_ylabel("Energy per Intent (J)")
ax.set_title("Energy vs Quantization")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

# 7c: Time vs Q level
ax = axes[1, 0]
for i, s in enumerate(SIZES):
    vals = [DATA[(s, q)]["avg_time_s"] for q in q_labels if (s, q) in DATA]
    ax.bar(q_x + i * q_width, vals, q_width, label=s, color=SIZE_COLORS[s], edgecolor="black")
ax.set_xticks(q_x + q_width * 1.5)
ax.set_xticklabels([q.replace("_", " ") for q in q_labels], fontsize=9)
ax.set_ylabel("Avg Time (s)")
ax.set_title("Inference Time vs Quantization")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

# 7d: Peak Memory vs Q level
ax = axes[1, 1]
for i, s in enumerate(SIZES):
    vals = [DATA[(s, q)]["peak_memory_mb"] for q in q_labels if (s, q) in DATA]
    ax.bar(q_x + i * q_width, vals, q_width, label=s, color=SIZE_COLORS[s], edgecolor="black")
ax.set_xticks(q_x + q_width * 1.5)
ax.set_xticklabels([q.replace("_", " ") for q in q_labels], fontsize=9)
ax.set_ylabel("Peak Memory (MB)")
ax.set_title("Memory vs Quantization")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Quantization Impact Analysis — Q2_K vs Q4_K_M vs Q8_0", 
             fontsize=13, fontweight="bold")
fig.tight_layout()
path = FIGURES / "v2_quantization_impact.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ {path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 8: Summary Rankings Table
# ═══════════════════════════════════════════════════════════════
all_configs = sorted(DATA.keys(), key=lambda x: DATA[x]["avg_match_pct"], reverse=True)

fig, ax = plt.subplots(figsize=(16, 0.5 * len(all_configs) + 2))
ax.axis("off")

headers = ["Config", "Match%", "Action%", "Format%", "Time(s)", "tok/s", "Energy(J)", "Mem(MB)"]
col_widths = [0.18, 0.10, 0.10, 0.10, 0.10, 0.10, 0.12, 0.12]

table_data = []
for (size, q) in all_configs:
    d = DATA[(size, q)]
    table_data.append([
        f"{size} {q.replace('_', ' ')}",
        f"{d['avg_match_pct']:.1f}",
        f"{d['action_pct']:.1f}",
        f"{d['format_pct']:.1f}",
        f"{d['avg_time_s']:.1f}",
        f"{d['tokens_per_sec']:.1f}",
        f"{d['avg_energy_j']:.0f}",
        f"{d['peak_memory_mb']:.0f}",
    ])

table = ax.table(cellText=table_data, colLabels=headers,
                 colWidths=col_widths, loc="center",
                 cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Color code best/worst per column
for col_idx in [1, 2, 3, 5]:  # Higher is better
    vals = [float(row[col_idx]) for row in table_data]
    best_val = max(vals)
    worst_val = min(vals)
    for row_idx in range(len(table_data)):
        cell = table[row_idx + 1, col_idx]
        v = float(table_data[row_idx][col_idx])
        if v == best_val:
            cell.set_facecolor("#90EE90")
        elif v == worst_val:
            cell.set_facecolor("#FFB6C1")

for col_idx in [4, 6, 7]:  # Lower is better
    vals = [float(row[col_idx]) for row in table_data]
    best_val = min(vals)
    worst_val = max(vals)
    for row_idx in range(len(table_data)):
        cell = table[row_idx + 1, col_idx]
        v = float(table_data[row_idx][col_idx])
        if v == best_val:
            cell.set_facecolor("#90EE90")
        elif v == worst_val:
            cell.set_facecolor("#FFB6C1")

ax.set_title("Phase 2: Complete Results Summary (sorted by Avg Match %)", 
             fontsize=13, fontweight="bold", pad=20)

fig.tight_layout()
path = FIGURES / "v2_summary_table.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ {path}")

print("\n🎉 All 8 figures generated!")
print(f"   Output: {FIGURES}/")
