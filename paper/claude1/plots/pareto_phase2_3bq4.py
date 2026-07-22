#!/usr/bin/env python3
"""
Phase 2 Pareto — headline figure focused on the 3B Q4_K_M sweet spot.

Story: across all 12 (size × quantization) configs, 3B|Q4_K_M is the accuracy
champion (45.5% action-correct, the global max) while sitting ON the Pareto
frontier for BOTH memory and latency — it beats every 7B config on speed and
uses only ~3.3 GB peak memory (a third of 7B|Q8_0).

Source of truth: results/ci_summary_v3.json (12 configs, n=200 each).
Output: pareto_phase2_action_vs_memory.png, pareto_phase2_action_vs_time.png,
        pareto_phase2_headline.png (2-panel).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke

PROJECT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
CI = json.load(open(PROJECT / "results" / "ci_summary_v3.json"))

# ── Encoding: shape = model size, colour = quantization ────────────
SIZES = ["0.5B", "1.5B", "3B", "7B"]
QLEVELS = ["Q2_K", "Q4_K_M", "Q8_0"]
SHAPE = {"0.5B": "o", "1.5B": "s", "3B": "^", "7B": "D"}
MSIZE = {"0.5B": 130, "1.5B": 170, "3B": 230, "7B": 210}
QCOLOR = {"Q2_K": "#E76F51", "Q4_K_M": "#2A9D8F", "Q8_0": "#264653"}
QNICE = {"Q2_K": "Q2_K · 2-bit", "Q4_K_M": "Q4_K_M · 4-bit", "Q8_0": "Q8_0 · 8-bit"}

HERO = "3B|Q4_K_M"          # the point we tell the story about
GOLD = "#E9B44C"

# ── Typography / canvas ────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.facecolor": "#FBFBFA", "figure.facecolor": "white",
    "axes.edgecolor": "#8A8A8A", "axes.linewidth": 1.0,
})

def rows():
    for s in SIZES:
        for q in QLEVELS:
            k = f"{s}|{q}"
            if k in CI:
                yield s, q, k, CI[k]

def pareto_min_x(pts):
    """Frontier maximising y while minimising x. pts=[(x,y,key)]."""
    front, best = [], -1e9
    for x, y, k in sorted(pts, key=lambda p: p[0]):
        if y > best:
            best = y
            front.append((x, y, k))
    return front

def draw(ax, x_metric, xlabel, invert_hint):
    pts = [(d[x_metric], d["action"], k) for _, _, k, d in rows()]

    # Pareto frontier (shade the dominated region lightly)
    front = pareto_min_x(pts)
    fx = [p[0] for p in front]; fy = [p[1] for p in front]
    ax.plot(fx, fy, "--", color="#B0B0B0", lw=1.6, zorder=1)
    ax.fill_between(fx, fy, min(fy) - 4, color="#2A9D8F", alpha=0.04, zorder=0)

    # all points
    for s, q, k, d in rows():
        is_hero = (k == HERO)
        ax.scatter(d[x_metric], d["action"],
                   s=MSIZE[s] * (1.5 if is_hero else 1.0),
                   marker=SHAPE[s], c=QCOLOR[q],
                   edgecolors=GOLD if is_hero else "white",
                   linewidths=2.6 if is_hero else 1.3,
                   zorder=6 if is_hero else 4, alpha=0.96)

    # hero halo + callout
    hx, hy = CI[HERO][x_metric], CI[HERO]["action"]
    ax.scatter(hx, hy, s=620, facecolors="none", edgecolors=GOLD,
               linewidths=2.2, zorder=5)
    txt = (f"★ 3B · Q4_K_M\n"
           f"{CI[HERO]['action']:.1f}% action-correct  (highest)\n"
           f"{CI[HERO]['peak_mem']:.0f} MB peak · {CI[HERO]['avg_time']:.1f}s/intent")
    dx, dy = (26, -60) if invert_hint == "mem" else (34, -74)
    ha = "left"
    ax.annotate(txt, (hx, hy), textcoords="offset points", xytext=(dx, dy),
                ha=ha, fontsize=9.5, color="#3a3a3a",
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=GOLD, lw=1.6),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.8,
                                connectionstyle="arc3,rad=0.15"))

    ax.set_xlabel(xlabel, fontweight="bold", fontsize=11.5)
    ax.set_ylabel("Action-correct (%)", fontweight="bold", fontsize=11.5)
    ax.grid(True, ls=":", color="#D5D5D5", alpha=0.9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.margins(0.12)
    # "better" arrow hint
    ax.annotate("better", xy=(0.06, 0.94), xycoords="axes fraction",
                fontsize=8.5, style="italic", color="#2A9D8F", ha="left")


def legend(fig):
    size_h = [Line2D([0], [0], marker=SHAPE[s], color="#555", lw=0,
                     markerfacecolor="#555", markersize=9 + i, label=f"{s}")
              for i, s in enumerate(SIZES)]
    q_h = [Line2D([0], [0], marker="o", color=QCOLOR[q], lw=0,
                  markerfacecolor=QCOLOR[q], markersize=10, label=QNICE[q])
           for q in QLEVELS]
    l1 = fig.legend(handles=size_h, title="Model size (shape)",
                    loc="lower left", bbox_to_anchor=(0.09, -0.02),
                    ncol=4, frameon=False, fontsize=9, title_fontsize=9.5)
    fig.add_artist(l1)
    fig.legend(handles=q_h, title="Quantization (colour)",
               loc="lower right", bbox_to_anchor=(0.93, -0.02),
               ncol=3, frameon=False, fontsize=9, title_fontsize=9.5)


# ── Headline: two panels side by side ──────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2))
draw(a1, "peak_mem", "Peak memory (MB)  →  lower is better", "mem")
draw(a2, "avg_time", "Latency (s / intent)  →  lower is better", "time")
a1.set_title("Accuracy vs. Memory", fontweight="bold", fontsize=13, pad=10)
a2.set_title("Accuracy vs. Latency", fontweight="bold", fontsize=13, pad=10)
fig.suptitle("Phase 2 — the 3B Q4_K_M sweet spot on the accuracy/efficiency Pareto front",
             fontweight="bold", fontsize=15, y=1.0)
legend(fig)
fig.text(0.5, -0.05,
         "Qwen2.5 · 6G slice-management intents · n=200/config · source: ci_summary_v3.json",
         ha="center", fontsize=8.5, style="italic", color="#777")
fig.tight_layout(rect=(0, 0.03, 1, 0.98))
fig.savefig(OUT / "pareto_phase2_headline.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote pareto_phase2_headline.png")

# ── Also emit the two panels standalone (for slides / single-column) ─
for metric, xlabel, hint, fname in [
    ("peak_mem", "Peak memory (MB)  →  lower is better", "mem",
     "pareto_phase2_action_vs_memory.png"),
    ("avg_time", "Latency (s / intent)  →  lower is better", "time",
     "pareto_phase2_action_vs_time.png"),
]:
    f, ax = plt.subplots(figsize=(8.2, 6))
    draw(ax, metric, xlabel, hint)
    ax.set_title("Phase 2 — Action accuracy Pareto front", fontweight="bold",
                 fontsize=13, pad=10)
    legend(f)
    f.tight_layout(rect=(0, 0.04, 1, 1))
    f.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(f)
    print("wrote", fname)
