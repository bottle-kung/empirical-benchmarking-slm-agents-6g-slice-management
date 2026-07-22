# claude1 — Data & Experiment Timeline (Phase 1: Cross-Family)

> **Author**: claude1 (Claude Code · Opus 4.8) — huddle Round 1/3
> **Scope**: Phase 1 — Cross-family model comparison (Qwen / Gemma / Llama / GLM @ ~8B, Q4_K_M; DeepSeek excluded)
> **Paper section**: §4 → `huddle/claude1.md` → `paper/claude1/claude1_paper.docx`
> **Repo**: `github.com/bottle-kung/empirical-benchmarking-slm-agents-6g-slice-management` (branch `v2`)
> **Updated**: 2026-07-18

This is claude1's **own** data-provenance sheet for the huddle task, kept next to the
claude1 paper artifacts. For the shared/original paper (`paper_comprehensive_v3`) see
`../DATA_TIMELINE.md`. This file records only what the **Phase 1 / claude1** deliverable
actually consumes and produces — verified against the files on disk.

---

## 📊 Data Sources Used (claude1 · Phase 1)

| Purpose | Data Source | Files | Status |
|---------|-------------|-------|:------:|
| **Phase 1 raw per-intent** (ranking, domain, complexity, efficiency) | Per-model Phase-1 benchmark, 200 intents each | `results/phase1/benchmark_*.json` (×4 ranked families) | ✅ USE |
| **Cross-check / raw records** | Consolidated per-intent list (9 entries) | `results/complete_benchmark.json` | ✅ USE (verify) |
| **§4.5 scaling bridge only** | Qwen size×quant 95% CIs | `results/ci_summary_v3.json` (12 keys, Qwen-only) | ✅ USE (bridge) |
| **Derived stats (machine-generated)** | Recomputed by claude1's own script | `paper/claude1/plots/phase1_stats.json` | ✅ generated |

### `results/phase1/` — the four models loaded

| Model | File | Role |
|-------|------|------|
| Qwen 2.5 7B | `benchmark_qwen2_5_7b.json` | 🥇 winner |
| Gemma 2 9B (Q4_K_M) | `benchmark_gemma2_9b-q4_K_M.json` | 🥈 accuracy tie, slow |
| Llama 3.1 8B | `benchmark_llama3_1_8b.json` | 🥉 balanced runner-up |
| GLM-4 9B (Q4_K_M) | `benchmark_glm4_9b-q4_K_M.json` | 4th |

> `benchmark_deepseek-r1_8b.json` exists on disk but is **excluded from Phase 1** (see below) — the loader in `gen_figures.py` skips `family == "DeepSeek"`.

### ⚠️ Why `ci_summary_v3.json` is used **only** for §4.5

`ci_summary_v3.json` keys are `<size>|<quant>` (`0.5B|Q2_K` … `7B|Q8_0`) — it is a
**Qwen size×quantization** matrix and contains **no cross-family rows**. The cross-family
ranking (§4.1–4.4) therefore takes **all** its numbers from `results/phase1/*.json`
(cross-checked against `complete_benchmark.json`); `ci_summary_v3.json` is touched only
for the Qwen diminishing-returns bridge in §4.5.

### 📌 DeepSeek-R1 — EXCLUDED from Phase 1

DeepSeek-R1 is **not part of Phase 1** (per project decision, 2026-07-18). It appears in
no table, figure, or narrative — the earlier "diagnostic case study" (§4.5) was removed
and sections renumbered (old §4.6 Bridge → §4.5, old §4.7 Summary → §4.6). The raw file
`results/phase1/benchmark_deepseek-r1_8b.json` is left on disk but ignored by the loader.

---

## 🔬 claude1 Work Timeline (huddle Round 1)

| Date/Time | Event | Artifact |
|-----------|-------|----------|
| 2026-07-18 10:00 | Joined huddle; assigned Phase 1 (`huddle/task.md`) | `huddle/participants.md` |
| 2026-07-18 11:02 | Wrote `gen_figures.py` — recompute all Phase-1 stats from raw records | `paper/claude1/plots/gen_figures.py` |
| 2026-07-18 11:03 | Generated derived stats + 6 figures (fig1–fig6) | `plots/phase1_stats.json`, `figures/fig1..6.png` |
| 2026-07-18 11:10 | Built styled DOCX (matches `paper_comprehensive_v3` style) | `plots/build_docx.py` → `claude1_paper.docx` |
| 2026-07-18 11:11 | Section markdown finalized (Round 1/3) | `huddle/claude1.md` |
| 2026-07-18 | **This provenance sheet** — claude1's own DATA_TIMELINE | `paper/claude1/DATA_TIMELINE.md` |
| 2026-07-18 | **DeepSeek-R1 removed from Phase 1** — dropped from loader, Table 1, §4.4, old §4.5 case study deleted, sections renumbered; figures + docx rebuilt | `gen_figures.py`, `build_docx.py`, `huddle/claude1.md` |

> Upstream experiment history (Phase 1 runs on lab-slm-01/02, 2025-07-11..12, and the
> 2026-07-14 v2 instrumented re-run) is recorded in `../DATA_TIMELINE.md`; not duplicated
> here to avoid drift.

---

## 📁 claude1 Artifacts (`paper/claude1/`)

```
paper/claude1/
├── DATA_TIMELINE.md          ← this file (Phase 1 provenance)
├── README.md                 ← reproduce instructions
├── claude1_paper.docx        ← final Section 4 (styled DOCX)
├── figures/
│   ├── fig1_aggregate_accuracy.png      Format/Action/Exact by family
│   ├── fig2_accuracy_latency.png        accuracy–latency Pareto (95% CI)
│   ├── fig3_domain_difficulty.png       action correct per domain (3 tiers)
│   ├── fig4_complexity_robustness.png   Simple→Ambiguous→Complex trajectory
│   ├── fig5_efficiency.png              latency + throughput bars
│   └── fig6_qwen_scaling_bridge.png     Qwen exact-match vs size (Q4_K_M), 95% CI
└── plots/
    ├── gen_figures.py        ← derives stats from raw results, renders fig1–6
    ├── build_docx.py         ← assembles claude1_paper.docx
    └── phase1_stats.json     ← machine-generated derived numbers
```

### Reproduce
```bash
cd <repo root>
python3 paper/claude1/plots/gen_figures.py   # -> figures/ + phase1_stats.json
python3 paper/claude1/plots/build_docx.py     # -> claude1_paper.docx
```

---

## 🔑 Scoring Metrics (as reported in §4)

| Metric | Formula | Source |
|--------|---------|--------|
| **Exact Match** | `mean(match_pct)`, `match_pct = exact_keys/total_keys × 100` | per-intent records |
| **Action Correct** | `parsed["action"].lower() == truth["action"].lower()` | per-intent records |
| **Format Valid** | response parses as a JSON dict | per-intent records |
| **Latency / Tok-s** | wall-clock per intent; decode tokens/sec | `results/phase1/*.json` summaries |
| **95% CI (exact)** | normal-approx over the n=200 score distribution per model | recomputed in `gen_figures.py` |

**Verified headline numbers** (from `plots/phase1_stats.json`):
Qwen — format 100.0 · action 43.5 · exact 29.8 (±3.98) · 18.68 s · 2.2 tok/s · 62.3 min.

---

## 🚫 Data NOT Used by claude1 (and why)

| Data | Reason |
|------|--------|
| `results/phase1/benchmark_deepseek-r1_8b.json` | **DeepSeek-R1 excluded from Phase 1** (project decision, 2026-07-18). Not in any table/figure/narrative. |
| `results/resource_v3/*.json`, `results/resource/*.json` | Resource telemetry belongs to Phase 2/3 (kilo1/agy1); Phase 1 ranking does not use per-intent CPU/Mem/Energy. |
| `results/phase2_final.json`, `results/phase1_final.json`, `phase1_prelim*.json` | Superseded / not the per-model Phase-1 files; Phase 1 uses `results/phase1/benchmark_*.json`. |
| `results/ci_summary.json` (old) | Replaced by `ci_summary_v3.json` for the §4.5 bridge. |
| cross-family rows of `ci_summary_v3.json` | None exist — it is Qwen size×quant only. |

---

## 🔗 Data-Integrity Contract

Every number in Tables 1–3 of `huddle/claude1.md` is **recomputed directly from the raw
per-intent records** and independently cross-checked against `results/phase1/*.json`
summaries. Any figure not traceable to a source file is intentionally excluded.

*Generated by claude1 · 2026-07-18 · sibling reference: `../DATA_TIMELINE.md` (agy1/shared).*
