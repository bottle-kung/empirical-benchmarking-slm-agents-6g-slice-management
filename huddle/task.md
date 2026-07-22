# Huddle Task: SLM Benchmark Paper

Each agent creates their own DOCX paper version analyzing the benchmark results.

## Assignment
- **claude1**: Phase 1 — Cross-family comparison (Llama vs Qwen vs Gemma vs GLM @ ~8B; DeepSeek excluded)
- **kilo1**: Phase 2 — Size scaling (Qwen 0.5B → 7B)
- **agy1**: Phase 3 — Quantization impact (Q2_K vs Q8_0)

## Data
- `results/ci_summary_v3.json` — main results with 95% CI
- `results/complete_benchmark.json` — raw per-intent results
- `results/figures/` — pre-generated figures

## Output
1. `huddle/<name>.md` — your paper section (markdown, Round 1/3)
2. `paper/<name>/` — plots directory with Python scripts
3. `paper/<name>/<name>_paper.docx` — final DOCX

## Style Reference
`paper/paper_comprehensive_v3.docx`

## Protocol
Follow `role/skill/huddle.md` — 3 rounds of improve → review → revise
