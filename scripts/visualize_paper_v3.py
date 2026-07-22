#!/usr/bin/env python3
"""
Regenerate the 13 paper figures (v3) — self-contained, reads real data.

Figures written to paper/images/:
  Phase 1 (3 bars):     phase1_bar_{match,action,time}.png
  Phase 2 Resource (3): paper_bar_{peak_mem,cpu,energy}.png
  Phase 2 Perf (4):     paper_bar_{tps,match,action,time}.png
  Phase 2 Pareto (3):   paper_pareto_{match_vs_time,action_vs_time,action_vs_energy}.png

Data sources (per DATA_TIMELINE.md):
  Phase 1  : results/phase1/*.json (Llama/Gemma/GLM, v1) + resource_v3/7B_Q4_K_M.json (Qwen re-run)
  Phase 2  : results/resource_v3/*.json (12 configs, v2 instrumented)
Encoding matches paper_comprehensive_v3.

Run:  python3 scripts/visualize_paper_v3.py
"""
import json, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

BASE   = Path(__file__).resolve().parent.parent
RES    = BASE / "results" / "resource_v3"
P1     = BASE / "results" / "phase1"
OUT    = BASE / "paper" / "images"
OUT.mkdir(parents=True, exist_ok=True)

SIZES    = ["0.5B", "1.5B", "3B", "7B"]
Q_LEVELS = ["Q2_K", "Q4_K_M", "Q8_0"]           # display order (2/4/8-bit)
Q_COLORS = {"Q2_K": "#E74C3C", "Q4_K_M": "#3498DB", "Q8_0": "#2ECC71"}

# Pareto: shape + color per size, Q = black vertical lines through marker
SIZE_SHAPES = {"0.5B": "s", "1.5B": "D", "3B": "^", "7B": "o"}
SIZE_MSIZE  = {"0.5B": 110, "1.5B": 150, "3B": 200, "7B": 260}
SIZE_COLORS = {"0.5B": "#E67E22", "1.5B": "#2ECC71", "3B": "#3498DB", "7B": "#9B59B6"}
Q_NLINES    = {"Q8_0": 0, "Q4_K_M": 1, "Q2_K": 2}


# ─────────────────────────── load Phase 2 (resource_v3) ───────────────────────────
def load_config(path):
    d = json.load(open(path))
    r = d["results"]; n = len(r)
    match = [x["match_pct"] for x in r]
    dur   = [x["duration_s"] for x in r]
    energy = [x.get("energy_j", 0) for x in r]
    cpu   = [x["cpu_pct"] for x in r if x.get("cpu_pct", 0) > 0]
    mem   = [x.get("peak_memory_mb", 0) for x in r if x.get("peak_memory_mb", 0) > 0]
    tok   = sum(x.get("tokens", 0) for x in r)
    return {
        "match":   float(np.mean(match)),
        "action":  sum(1 for x in r if x.get("action_correct")) / n * 100,
        "time":    float(np.mean(dur)),
        "tps":     tok / sum(dur) if sum(dur) else 0,
        "cpu":     float(np.mean(cpu)) if cpu else 0,
        "energy":  float(np.mean(energy)),
        "peak_mem": max(mem) if mem else 0,
    }

DATA = {}
for f in RES.glob("*.json"):
    parts = f.stem.split("_")
    size = parts[0]; q = "_".join(parts[1:]) if len(parts) > 1 else "Q4_K_M"
    DATA[(size, q)] = load_config(f)


# ─────────────────────────── load Phase 1 ───────────────────────────
P1_DATA = {}
for f in sorted(glob.glob(str(P1 / "*.json"))):
    name = f.split("/")[-1].lower()
    if "deepseek" in name or "qwen" in name:          # skip diagnostic + old Qwen
        continue
    s = json.load(open(f))["summary"]
    P1_DATA[s["family"]] = {"match": s["avg_match_pct"], "action": s["action_correct_pct"],
                            "time": s["avg_time_s"]}
# Qwen from v2 re-run
_q = load_config(RES / "7B_Q4_K_M.json")
P1_DATA["Qwen"] = {"match": _q["match"], "action": _q["action"], "time": _q["time"]}

P1_FAMS   = ["Llama", "Qwen", "Gemma", "GLM"]
P1_COLORS = {"Llama": "#E74C3C", "Qwen": "#3498DB", "Gemma": "#2ECC71", "GLM": "#9B59B6"}


# ─────────────────────────── Phase 1 bars ───────────────────────────
def phase1_bar(metric, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = [P1_DATA[f][metric] for f in P1_FAMS]
    bars = ax.bar(P1_FAMS, vals, color=[P1_COLORS[f] for f in P1_FAMS],
                  edgecolor="white", linewidth=1.5)
    off = max(vals) * 0.02
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + off,
                f"{v:.1f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.grid(alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig); print("✅", fname)

phase1_bar("match",  "Phase 1: Avg Match % (~8B, Q4_K_M)",       "Avg Match %",        "phase1_bar_match.png")
phase1_bar("action", "Phase 1: Action Accuracy % (~8B, Q4_K_M)", "Action Accuracy %",  "phase1_bar_action.png")
phase1_bar("time",   "Phase 1: Inference Time (~8B, Q4_K_M)",    "Inference Time (s)", "phase1_bar_time.png")


# ─────────────────────────── Phase 2 grouped bars ───────────────────────────
def phase2_bar(metric, title, tag, ylabel, fname, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    x = np.arange(len(SIZES)); width = 0.26
    all_vals = []
    for i, q in enumerate(Q_LEVELS):
        vals = [DATA[(s, q)][metric] for s in SIZES]
        all_vals += vals
        bars = ax.bar(x + (i - 1) * width, vals, width, label=q.replace("_", " "),
                      color=Q_COLORS[q], edgecolor="black", linewidth=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(all_vals) * 0.012,
                    fmt.format(v), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(SIZES)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(all_vals) * 1.15)
    ax.set_title(f"{title}\n[{tag}]", fontweight="bold", fontsize=14)
    ax.legend(title="Quantization", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig); print("✅", fname)

# Resource (3)
phase2_bar("peak_mem", "Peak Memory (MB)",         "Resource", "Peak Memory (MB)",        "paper_bar_peak_mem.png", "{:.0f}")
phase2_bar("cpu",      "CPU Utilization (%)",       "Resource", "CPU Utilization (%)",     "paper_bar_cpu.png")
phase2_bar("energy",   "Energy per Intent (J)",     "Resource", "Energy per Intent (J)",   "paper_bar_energy.png", "{:.0f}")
# Performance (4)
phase2_bar("tps",      "Throughput (tokens/s)",     "Performance", "Throughput (tokens/s)","paper_bar_tps.png")
phase2_bar("match",    "Avg Match %",               "Performance", "Avg Match %",          "paper_bar_match.png")
phase2_bar("action",   "Action Accuracy %",         "Performance", "Action Accuracy %",    "paper_bar_action.png")
phase2_bar("time",     "Inference Time (s/intent)", "Performance", "Inference Time (s)",   "paper_bar_time.png")


# ─────────────────────────── Pareto ───────────────────────────
def draw_q_lines(ax, x, y, n, xr, yr):
    if n == 0:
        return
    sp = xr * 0.012; hh = yr * 0.026
    xs = [x] if n == 1 else [x - sp, x + sp]
    for xi in xs:
        ax.plot([xi, xi], [y - hh, y + hh], "-", color="black", linewidth=2.4, zorder=6)

def pareto(x_key, y_key, x_label, y_label, title, fname):
    pts = {(s, q): (DATA[(s, q)][x_key], DATA[(s, q)][y_key])
           for s in SIZES for q in Q_LEVELS}
    xs = [p[0] for p in pts.values()]; ys = [p[1] for p in pts.values()]
    xr = max(xs) - min(xs) + 1e-6; yr = max(ys) - min(ys) + 1e-6
    cx, cy = np.mean(xs), np.mean(ys)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Pareto frontier (maximize y, minimize x)
    fr = sorted(pts.values(), key=lambda p: p[0])
    fx, fy, best = [], [], -1e9
    for tx, ty in fr:
        if ty > best:
            best = ty; fx.append(tx); fy.append(ty)
    ax.plot(fx, fy, "--", color="gray", alpha=0.45, linewidth=2, zorder=2, label="Pareto frontier")

    # markers
    for (s, q), (xv, yv) in pts.items():
        ax.scatter(xv, yv, s=SIZE_MSIZE[s], c=SIZE_COLORS[s], marker=SIZE_SHAPES[s],
                   edgecolors="white", linewidth=1.5, zorder=5, alpha=0.92)
    for (s, q), (xv, yv) in pts.items():
        draw_q_lines(ax, xv, yv, Q_NLINES[q], xr, yr)

    # labels — pushed radially outward from centroid to avoid central overlap
    for (s, q), (xv, yv) in pts.items():
        dx = (xv - cx) / xr; dy = (yv - cy) / yr
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        ox, oy = dx / norm * 26, dy / norm * 26
        ax.annotate(f"{s}\n{q.replace('_', ' ')}", (xv, yv), textcoords="offset points",
                    xytext=(ox, oy), ha="center", va="center", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    ax.set_xlabel(x_label, fontsize=11); ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # margins so labels fit
    ax.set_xlim(min(xs) - xr * 0.10, max(xs) + xr * 0.12)
    ax.set_ylim(min(ys) - yr * 0.12, max(ys) + yr * 0.12)

    leg_size = [Line2D([0], [0], marker=SIZE_SHAPES[s], color=SIZE_COLORS[s],
                       markerfacecolor=SIZE_COLORS[s], markeredgecolor="white",
                       markersize=ms, linestyle="none", markeredgewidth=1.5, label=s)
                for s, ms in zip(SIZES, [7, 9, 11, 13])]
    leg_q = [Patch(facecolor="gray", edgecolor="white", label=lbl) for lbl in
             ["Q8_0 (8-bit, no line)", "Q4_K_M (4-bit, | line)", "Q2_K (2-bit, || lines)"]]
    l1 = ax.legend(handles=leg_size, title="Model Size (shape+color)", loc="lower right",
                   fontsize=8, title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=leg_q, title="Quantization (| lines)", loc="upper left",
              fontsize=8, title_fontsize=9)

    fig.tight_layout(); fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig); print("✅", fname)

pareto("time", "match",   "Inference Time (s/intent)", "Avg Match %",
       "Accuracy vs Speed  |  ■◆▲● = 0.5/1.5/3/7B  |  | = Q Level",
       "paper_pareto_match_vs_time.png")
pareto("time", "action",  "Inference Time (s/intent)", "Action Accuracy %",
       "Action Accuracy vs Speed  |  ■◆▲● = 0.5/1.5/3/7B  |  | = Q Level",
       "paper_pareto_action_vs_time.png")
pareto("energy", "action", "Energy per Intent (J)",    "Action Accuracy %",
       "Action Accuracy vs Energy  |  ■◆▲● = 0.5/1.5/3/7B  |  | = Q Level",
       "paper_pareto_action_vs_energy.png")

print("\n✅ 13 figures regenerated →", OUT)
