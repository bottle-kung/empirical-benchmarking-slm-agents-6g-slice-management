#!/usr/bin/env python3
"""
claude1 — Phase 1 Cross-Family SLM Benchmark figure generation.

Reads:
  results/complete_benchmark.json   (raw per-intent results, ground truth for Phase 1)
  results/ci_summary_v3.json        (Qwen size x quant CIs; used for the scaling bridge)

Writes publication-quality PNGs to paper/claude1/figures/
Also emits phase1_stats.json with all derived numbers used in the paper section.

Run:  python3 paper/claude1/plots/gen_figures.py
"""
import json, os, math, statistics
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "paper", "claude1", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Global style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
})

# Consistent colour per family (colour-blind friendly)
FAM_COLOR = {
    "Qwen":     "#0072B2",
    "Gemma":    "#D55E00",
    "Llama":    "#009E73",
    "GLM":      "#CC79A7",
}

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
with open(os.path.join(RESULTS, "complete_benchmark.json")) as f:
    complete = json.load(f)
with open(os.path.join(RESULTS, "ci_summary_v3.json")) as f:
    ci_v3 = json.load(f)

# Phase 1 blocks (DeepSeek excluded from Phase 1 entirely; dedupe qwen)
p1 = OrderedDict()
for blk in complete:
    s = blk["summary"]
    if s.get("phase") != 1:
        continue
    fam = s["family"]
    if fam == "DeepSeek":
        continue  # excluded from Phase 1
    if fam in p1:
        continue  # first occurrence (qwen appears twice)
    p1[fam] = blk

# Order families by exact-match descending for consistent presentation
FAMILY_ORDER = ["Qwen", "Gemma", "Llama", "GLM"]

DOMAINS = ["Slicing Provisioning", "Scaling Request", "Conflict Resolution"]
COMPLEXITY = ["Simple", "Ambiguous", "Complex"]


def mean_ci95(values):
    """Return (mean, half-width of 95% CI) using normal approximation."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = statistics.mean(values)
    if n == 1:
        return m, 0.0
    sd = statistics.stdev(values)
    hw = 1.96 * sd / math.sqrt(n)
    return m, hw


# ----------------------------------------------------------------------------
# Derive per-family statistics
# ----------------------------------------------------------------------------
stats = OrderedDict()
for fam in FAMILY_ORDER:
    s = p1[fam]["summary"]
    rs = p1[fam]["results"]
    matches = [r["match_pct"] for r in rs]
    m_mean, m_hw = mean_ci95(matches)

    by_dom = {}
    for dom in DOMAINS:
        g = [r for r in rs if r["domain"] == dom]
        ac = 100.0 * sum(1 for r in g if r["action_correct"]) / len(g) if g else 0.0
        em = statistics.mean(r["match_pct"] for r in g) if g else 0.0
        by_dom[dom] = {"n": len(g), "action": ac, "exact": em}

    by_cx = {}
    for cx in COMPLEXITY:
        g = [r for r in rs if r["complexity"] == cx]
        em = statistics.mean(r["match_pct"] for r in g) if g else 0.0
        ac = 100.0 * sum(1 for r in g if r["action_correct"]) / len(g) if g else 0.0
        by_cx[cx] = {"n": len(g), "exact": em, "action": ac}

    stats[fam] = {
        "model": s["model"],
        "format": s["format_valid_pct"],
        "action": s["action_correct_pct"],
        "exact": s["avg_match_pct"],
        "exact_ci95": m_hw,
        "time_s": s["avg_time_s"],
        "tps": s["avg_tokens_per_sec"],
        "avg_tokens": s["avg_tokens"],
        "total_time_min": s["avg_time_s"] * s["total"] / 60.0,
        "by_domain": by_dom,
        "by_complexity": by_cx,
    }

# ----------------------------------------------------------------------------
# FIGURE 1 — Aggregate accuracy: grouped bars (Format / Action / Exact)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 4.6))
metrics = ["format", "action", "exact"]
labels = ["Format Valid", "Action Correct", "Exact Match"]
x = np.arange(len(FAMILY_ORDER))
w = 0.26
hatches = ["", "//", ".."]
greys = ["#b8b8b8", "#7a7a7a", "#3a3a3a"]
for i, (mk, lab) in enumerate(zip(metrics, labels)):
    vals = [stats[f][mk] for f in FAMILY_ORDER]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=greys[i],
                  edgecolor="black", linewidth=0.6, hatch=hatches[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"{f}\n{stats[f]['model'].split(':')[0]}" for f in FAMILY_ORDER])
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 112)
ax.set_title("Phase 1: Aggregate Accuracy by Model Family (~8B, Q4_K_M)")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig1_aggregate_accuracy.png"), bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# FIGURE 2 — Accuracy vs Latency Pareto scatter
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 5.2))
for fam in FAMILY_ORDER:
    st = stats[fam]
    ax.errorbar(st["time_s"], st["exact"], yerr=st["exact_ci95"],
                fmt="o", ms=13, color=FAM_COLOR[fam], capsize=4,
                markeredgecolor="black", markeredgewidth=0.8, zorder=3,
                label=f"{fam} ({st['model'].split(':')[1] if ':' in st['model'] else ''})")
    ax.annotate(f"{fam}\n{st['tps']:.1f} tok/s", (st["time_s"], st["exact"]),
                textcoords="offset points", xytext=(12, 6), fontsize=9)
# Pareto frontier annotation
ax.axvspan(0, 20, color="#0072B2", alpha=0.04)
ax.set_xlabel("Latency per Intent (s)  → slower")
ax.set_ylabel("Exact Match (%)  → more accurate")
ax.set_title("Phase 1: Accuracy–Latency Trade-off (error bars = 95% CI)")
ax.set_xlim(0, 48)
ax.set_ylim(20, 34)
ax.legend(loc="lower right", title="Family")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig2_accuracy_latency.png"), bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# FIGURE 3 — Domain-level Action Correct (grouped bars)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.4, 4.8))
x = np.arange(len(DOMAINS))
w = 0.2
for i, fam in enumerate(FAMILY_ORDER):
    vals = [stats[fam]["by_domain"][d]["action"] for d in DOMAINS]
    ax.bar(x + (i - 1.5) * w, vals, w, label=fam, color=FAM_COLOR[fam],
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(["Slicing\nProvisioning", "Scaling\nRequest", "Conflict\nResolution"])
ax.set_ylabel("Action Correct (%)")
ax.set_ylim(0, 105)
ax.set_title("Phase 1: Domain Difficulty — Action Correct by Domain")
# difficulty tier annotations
ax.text(0, 100, "SOLVED", ha="center", fontsize=9, color="#009E73", fontweight="bold")
ax.text(1, 100, "HARD", ha="center", fontsize=9, color="#D55E00", fontweight="bold")
ax.text(2, 100, "NEAR-RANDOM", ha="center", fontsize=9, color="#CC0000", fontweight="bold")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig3_domain_difficulty.png"), bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# FIGURE 4 — Complexity robustness (Exact Match by complexity)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 4.8))
x = np.arange(len(COMPLEXITY))
w = 0.2
for i, fam in enumerate(FAMILY_ORDER):
    vals = [stats[fam]["by_complexity"][c]["exact"] for c in COMPLEXITY]
    ax.plot(x, vals, "-o", color=FAM_COLOR[fam], label=fam, lw=2, ms=8,
            markeredgecolor="black", markeredgewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(["Simple", "Ambiguous", "Complex"])
ax.set_ylabel("Exact Match (%)")
ax.set_title("Phase 1: Robustness to Intent Complexity")
ax.set_ylim(0, 55)
ax.legend(title="Family", ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig4_complexity_robustness.png"), bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# FIGURE 5 — Efficiency: latency + throughput dual view
# ----------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.4))
fams = FAMILY_ORDER
times = [stats[f]["time_s"] for f in FAMILY_ORDER]
tps = [stats[f]["tps"] for f in FAMILY_ORDER]
cols = [FAM_COLOR[f] for f in fams]
a1.barh(fams[::-1], times[::-1], color=cols[::-1], edgecolor="black", linewidth=0.5)
for i, v in enumerate(times[::-1]):
    a1.text(v + 0.6, i, f"{v:.1f}s", va="center", fontsize=9)
a1.set_xlabel("Latency per Intent (s)")
a1.set_title("Inference Latency")
a1.set_xlim(0, 48)
a2.barh(fams[::-1], tps[::-1], color=cols[::-1], edgecolor="black", linewidth=0.5)
for i, v in enumerate(tps[::-1]):
    a2.text(v + 0.05, i, f"{v:.1f}", va="center", fontsize=9)
a2.set_xlabel("Throughput (tokens/s)")
a2.set_title("Decode Throughput")
a2.set_xlim(0, 4)
fig.suptitle("Phase 1: CPU Efficiency Profile", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig5_efficiency.png"), bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# FIGURE 6 — Bridge to Phase 2: Qwen size scaling from ci_summary_v3 (Q4_K_M)
# Places the Phase-1 winner (Qwen 7B) in its scaling context.
# ci_match array = [mean, std, ci_low, ci_high]
# ----------------------------------------------------------------------------
sizes = ["0.5B", "1.5B", "3B", "7B"]
q = "Q4_K_M"
xm, lo, hi, tps_s = [], [], [], []
for sz in sizes:
    rec = ci_v3[f"{sz}|{q}"]
    m, sd, l, h = rec["ci_match"]
    xm.append(m); lo.append(m - l); hi.append(h - m); tps_s.append(rec["tps"])
fig, ax = plt.subplots(figsize=(7.4, 4.6))
xi = np.arange(len(sizes))
ax.errorbar(xi, xm, yerr=[lo, hi], fmt="-o", color=FAM_COLOR["Qwen"], lw=2,
            ms=9, capsize=5, markeredgecolor="black", markeredgewidth=0.6,
            label="Qwen 2.5 Exact Match (95% CI)")
for i, (m, t) in enumerate(zip(xm, tps_s)):
    ax.annotate(f"{m:.1f}%", (xi[i], m), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9)
ax.axhline(stats["Qwen"]["exact"], color="#888", ls=":", lw=1)
ax.set_xticks(xi)
ax.set_xticklabels(sizes)
ax.set_xlabel("Qwen 2.5 Model Size (Q4_K_M)")
ax.set_ylabel("Exact Match (%)")
ax.set_title("Bridge: Phase-1 Winner (Qwen 7B) in Its Size-Scaling Context")
ax.set_ylim(10, 35)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig6_qwen_scaling_bridge.png"), bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# Dump derived stats for the paper section
# ----------------------------------------------------------------------------
out = {"phase1": stats,
       "qwen_scaling_q4": {sz: ci_v3[f"{sz}|{q}"]["ci_match"] for sz in sizes}}
with open(os.path.join(os.path.dirname(__file__), "phase1_stats.json"), "w") as f:
    json.dump(out, f, indent=2)

print("Figures written to", FIGDIR)
for fn in sorted(os.listdir(FIGDIR)):
    print("  ", fn)
print("\nDerived numbers:")
for fam in FAMILY_ORDER:
    st = stats[fam]
    print(f"  {fam:6s} exact={st['exact']:.1f}±{st['exact_ci95']:.1f} "
          f"action={st['action']:.1f} fmt={st['format']:.1f} "
          f"t={st['time_s']:.1f}s tps={st['tps']:.1f}")
