# Paper v3 — Data & Experiment Timeline

> **Paper**: `paper_comprehensive_v3.docx/pdf`  
> **Date**: July 2026  
> **Repo**: `github.com/bottle-kung/empirical-benchmarking-slm-agents-6g-slice-management` (branch `v2`)

---

## 📊 Data Sources Used in the Paper

| Section | Data Source | Files |
|---------|------------|-------|
| **Phase 1 (Section 4)** | Original Phase 1 benchmark (Llama/Gemma/GLM) + Qwen re-run | `results/phase1/benchmark_*.json` (×4), `results/resource_v3/7B_Q4_K_M.json` |
| **Phase 2 (Section 5)** | Instrumented re-run with per-intent CPU/Mem/Energy telemetry | `results/resource_v3/*.json` (×12) |
| **Phase 2 domain/complexity** | Same re-run, computed from raw per-intent records | Computed in Section 5.4/5.5 from `resource_v3` |

### ⚠️ Why two data sources for Phase 1?

- The **original Phase 1 run** (v1 benchmark, no resource telemetry) was used for Llama 3.1, Gemma 2, and GLM-4.
- **Qwen 2.5 7B** was re-run under the **v2 instrumented benchmark** (`benchmark_runner_v2.py`) to get resource data for the Phase 1 winner and to ensure consistent scoring with the Phase 2 re-run.
- The scoring formula (`match_pct = exact_keys / total_keys × 100`) is identical between v1 and v2; the difference is v2 **also** samples CPU/Mem/Energy via psutil.

### 📌 DeepSeek-R1 — Included as Diagnostic Case Study ONLY

DeepSeek-R1 is discussed in **Section 4.4** as a failure analysis — it is **NOT** included in accuracy rankings or conclusions. The paper explicitly states:
- 0% format accuracy on CPU
- Infinite chain-of-thought loops
- "Reasoning-oriented architectures should be avoided for CPU-only edge deployment"

---

## 🔬 Experiment Timeline

### Phase 1 — Cross-Family Benchmark (~8B, Q4_K_M)

| Date | Event | VM | Script |
|------|-------|----|--------|
| 2025-07-11 | Pull models + create custom Q4_K_M for Gemma2/GLM4 | lab-slm-01/02 | Ollama CLI |
| 2025-07-11 | Phase 1 run: Llama 3.1 8B, Qwen 2.5 7B (v1) | lab-slm-01 | `benchmark_runner.py` (v1) |
| 2025-07-11 | Phase 1 run: Gemma 2 9B, GLM-4 9B (custom Q4_K_M) | lab-slm-02 | `benchmark_runner.py` (v1) |
| 2025-07-11 | Phase 1 run: DeepSeek-R1 8B → **FAILED** (0% format) | lab-slm-02 | `benchmark_runner.py` (v1) |
| 2025-07-12 | Re-attempt DeepSeek-R1 (thinking+content parser fix) → **FAILED** (infinite loop) | lab-slm-01 | `benchmark_runner.py` (v1) |
| **2026-07-14** | **Qwen 2.5 7B re-run (v2 instrumented)** — CPU/Mem/Energy telemetry | lab-slm-02 | `benchmark_runner_v2.py` |

### Phase 2 — Size Scaling + Quantization (Qwen 2.5)

| Date | Event | VM | Script |
|------|-------|----|--------|
| 2025-07-12 | Phase 2 run: 0.5B/1.5B/3B Q4_K_M (v1, no resource data) | lab-slm-01 | `benchmark_runner.py` (v1) |
| 2025-07-12-13 | Phase 3 (now merged into Phase 2): Q2_K + Q8_0 variants (v1) | lab-slm-01/02 | `phase3_runner.py` |
| **2026-07-13** | **First re-run attempt (v2)** — CPU bug (psutil first-call-returns-0) → cpu_pct=0 for all | lab-slm-01/02 | `benchmark_runner_v2.py` (early) |
| **2026-07-14** | **Fix psutil priming + CI computation + re-deploy** | — | `benchmark_runner_v2.py` (fixed) |
| **2026-07-14** | **Full re-run all 12 configs** — 9 on lab-slm-01, 3 on lab-slm-02 (~4 hrs) | lab-slm-01/02 | `benchmark_runner_v2.py` + `rerun_v2.sh` |
| **2026-07-14** | Copy results → `results/resource_v3/` (12 JSON files) | — | scp |
| **2026-07-14** | Compute CI summary → `results/ci_summary_v3.json` | — | `compute_ci_summary.py` |

### Paper Generation

| Date | Event | Script |
|------|-------|--------|
| 2025-07-13 | Paper v1 (`paper_comprehensive.docx`) — deepseek session | — |
| 2026-07-14 | Paper v2 (`paper_comprehensive_v2.docx`) — with CI, no real resource data | `generate_paper.py` |
| 2026-07-14 | Paper v3 (`paper_final_v3.docx`) — real CPU%, memory bandwidth finding | `generate_paper_final.py` |
| 2026-07-14 | Figures: Phase 2 bars + Pareto (shape=size, \|=Q) | `visualize_paper_figures.py` |
| 2026-07-14 | Figures: Phase 1 individual bars | `visualize_phase1_bars.py` |
| 2026-07-14 | Paper with figures (`paper_with_figures.docx`) | `generate_paper_with_figures.py` |
| **2026-07-17** | **Paper comprehensive v3** — updated with real CPU% telemetry, energy, CI, 12 findings, cleaner analysis (this version) | `paper_comprehensive_v3.md` → pandoc → docx/pdf |

---

## 📁 Key Files

### Paper (this directory)
```
paper/
├── paper_comprehensive_v3.docx    ← CURRENT PAPER (use this)
├── paper_comprehensive_v3.pdf
└── images/                                        (13 figures — final set)
    ├── phase1_bar_{match,action,time}.png          (Phase 1 bars — 3)
    ├── paper_bar_{peak_mem,cpu,energy}.png         (Phase 2 Resource bars — 3)
    ├── paper_bar_{tps,match,action,time}.png       (Phase 2 Performance bars — 4)
    └── paper_pareto_{match_vs_time,                (Pareto — 3)
                 action_vs_time, action_vs_energy}.png
```

> **2026-07-18 update:** figure set trimmed to the 13 used in the paper (removed
> `paper_bar_format`, `paper_pareto_format_vs_time`, `paper_pareto_match_vs_energy`,
> `paper_pareto_match_vs_throughput`). All 13 are now regenerated by a single
> self-contained script `scripts/visualize_paper_v3.py` (reads `results/phase1/` +
> `results/resource_v3/`, writes straight to `paper/images/`).

### Data (project root `results/`)
```
results/
├── phase1/          ← Phase 1 v1 runs (Llama/Gemma/GLM/DeepSeek)
├── phase2/          ← Phase 2 v1 runs (0.5B/1.5B/3B Q4_K_M)
├── phase3/          ← Q2_K + Q8_0 v1 runs
├── resource/        ← First re-run attempt (cpu_pct=0 — DO NOT USE)
├── resource_v3/     ← ✅ VALID RE-RUN (12 files, real CPU/Mem/Energy/CI) ← USE THIS
├── ci_summary_v3.json  ← Computed CI summary from resource_v3
└── figures/         ← All generated figures (~70 files)
```

### Scripts (project root `scripts/`)
```
scripts/
├── benchmark_runner.py         ← Phase 1 v1 (no resource monitoring)
├── benchmark_runner_v2.py      ← ✅ Phase 2 v2 (psutil CPU/Mem/Energy + CI)
├── rerun_v2.sh                 ← Orchestration for 12-config re-run
├── compute_ci_summary.py       ← CI computation → ci_summary_v3.json
├── visualize_paper_v3.py       ← ✅ Regenerates all 13 paper figures → paper/images/
├── visualize_phase1_bars.py    ← Phase 1 individual bar charts (superseded)
├── visualize_paper_figures.py  ← Phase 2 bars + Pareto (shape/line encoding)
└── visualize_phase2_pareto_v2.py ← Earlier Pareto script (superseded)
```

---

## 🔑 Scoring Metrics (as used in the paper)

| Metric | Formula | Source |
|--------|---------|--------|
| **Exact Match / Avg Match%** | `mean(match_pct)` where `match_pct = exact_keys/total_keys × 100` | `benchmark_runner_v2.py` line 72 |
| **Action Correct** | `parsed["action"].lower() == truth["action"].lower()` | `benchmark_runner_v2.py` line 69 |
| **Format Valid** | Response parses as valid JSON dict | `benchmark_runner_v2.py` line 68 |
| **CPU%** | `psutil.cpu_percent(interval=0)`, system-wide, sampled at 0.5s during inference | `benchmark_runner_v2.py` line 140 |
| **Memory (MB)** | `psutil.Process(pid).memory_info().rss`, peak over sampling window | `benchmark_runner_v2.py` line 147 |
| **Energy (J)** | `(mean CPU% / 100) × 135W × inference_time_s` | `benchmark_runner_v2.py` line 195 |
| **95% CI** | `mean ± t₀.₀₂₅ × SD/√n`, n=200, df=199, t≈1.97 | `compute_ci_summary.py` |

---

## 🚫 Data NOT Used (and why)

| Data | Reason |
|------|--------|
| `results/resource/*.json` | First re-run — CPU bug (psutil returned 0 for all intents). Energy is correct (95% fallback) but CPU% is invalid. |
| `results/phase1/benchmark_qwen2_5_7b.json` | Old v1 Qwen run — replaced by `resource_v3/7B_Q4_K_M.json` re-run for Phase 1 Qwen row |
| `results/phase1/benchmark_deepseek-r1_8b.json` | 0% format accuracy — used only for Section 4.4 diagnostic case study |
| `results/phase2/*.json` | Old v1 Phase 2 runs — superseded by `resource_v3` re-runs |

---

*Generated: 2026-07-17 — For questions: refer to `timeline.md` in project root for full session history.*
