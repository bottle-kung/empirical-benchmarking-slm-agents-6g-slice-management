#!/usr/bin/env python3
"""
Generate final paper WITH figures + tables + analysis
(paper_with_figures.docx / .pdf)

Structure:
  - Phase 1: Table 1 + Fig 1-3 (match, action, time bars) + analysis
  - Phase 2: Table 2 (12 configs with 95% CI)
  - Resource: Fig 4-6 (memory, cpu, energy bars) + analysis
  - Performance: Fig 7-10 (tps, match, action, time bars) + analysis
  - Pareto: Fig 11-13 (match/time, action/time, action/energy) + analysis
  - Discussion + Conclusion + References

Requires:
  - results/ci_summary_v3.json (from analysis of resource_v3/)
  - results/figures/phase1_bar_*.png (from visualize_phase1_bars.py)
  - results/figures/paper_bar_*.png (from visualize_paper_figures.py — bar section)
  - results/figures/paper_pareto_*.png (from visualize_paper_figures.py)
"""
import json, os, subprocess
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

PROJECT = Path("/jupyter_workspace/rbru/slm_6g_eval_v2")
FIGURES = PROJECT / "results" / "figures"

with open(PROJECT / "results" / "ci_summary_v3.json") as f:
    ci_data = json.load(f)

doc = Document()
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(11)

def add_fig(path, caption, width=4.5):
    if os.path.exists(path):
        doc.add_picture(str(path), width=Inches(width))
        p = doc.add_paragraph()
        p.add_run(caption).bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()

# ═══ TITLE ═══
title = doc.add_heading('Empirical Evaluation of Small Language Models for\nIntent-Driven Slice Lifecycle Management in 6G Core Networks', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in title.runs:
    run.font.size = Pt(16)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run('Teerapong Somsri, [Co-authors TBD]').italic = True
p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p2.add_run('Rajabhat Ratchanakarin University (RBRU)').font.size = Pt(10)
doc.add_paragraph()

# ═══ ABSTRACT ═══
doc.add_heading('Abstract', level=1)
doc.add_paragraph(
    'The deployment of intent-based networking (IBN) in 6G core networks requires lightweight, '
    'efficient language models capable of translating natural language intents into network '
    'orchestration commands. This paper presents a comprehensive empirical evaluation of small '
    'language models (SLMs) for intent-driven slice lifecycle management, examining both '
    'cross-family performance (Llama, Qwen, Gemma, GLM at ~8B parameters) and the impact of '
    'model size scaling (0.5B–7B) with quantization (Q2_K, Q4_K_M, Q8_0). Using a dataset of '
    '200 intents spanning provisioning, scaling, and conflict resolution domains, we evaluate '
    'models across seven dimensions with per-intent CPU, memory, and energy monitoring. '
    'Our results reveal that Qwen2.5-7B achieves the highest overall accuracy (29.8% exact match, '
    '42.5% action accuracy), while Qwen2.5-3B-Q4_K_M offers the best cost-effectiveness. '
    'A key finding is that CPU utilization scales with model size (69% at 0.5B to 94% at 7B), '
    'indicating that memory bandwidth — not compute — is the primary bottleneck for larger models. '
    'We also find that Q4_K_M quantization reduces memory by ~50% with less than 1pp accuracy loss, '
    'and identify a Pareto "knee" at 3B beyond which accuracy gains diminish while costs grow linearly.'
).paragraph_format.first_line_indent = Cm(1.27)

p = doc.add_paragraph()
p.add_run('Keywords: ').bold = True
p.add_run('Small Language Models, 6G Networks, Intent-Based Networking, Network Slicing, Model Quantization, Edge AI, Resource Efficiency')
doc.add_paragraph()

# ═══ 1. INTRODUCTION ═══
doc.add_heading('1. Introduction', level=1)
for text in [
    'Sixth-generation (6G) networks must support diverse use cases including eMBB, URLLC, and mMTC. '
    'Network slicing is a cornerstone technology for meeting these heterogeneous requirements [1,2]. '
    'Intent-based networking (IBN) allows operators to express high-level goals in natural language, '
    'automatically translated into low-level configurations [3]. Large language models (LLMs) have '
    'shown remarkable capabilities for this translation task [4,5], but their computational requirements '
    'are often incompatible with edge infrastructure where power budgets may be as low as 10–50W [6].',
    'Small language models (SLMs) with 0.5–8B parameters, combined with quantization (2–8 bits), '
    'offer a potential solution — but their performance on domain-specific networking tasks remains '
    'underexplored. This paper addresses: RQ1) How do different SLM architectures compare in intent '
    'translation accuracy? RQ2) How does model size affect performance and resource consumption? '
    'RQ3) How does quantization impact the accuracy-efficiency trade-off? RQ4) What are the real '
    'resource characteristics of SLM inference on CPU-only hardware?',
]:
    p = doc.add_paragraph(text)
    p.paragraph_format.first_line_indent = Cm(1.27)
doc.add_paragraph()

# ═══ 2. METHODOLOGY ═══
doc.add_heading('2. Methodology', level=1)
doc.add_paragraph(
    'Dataset: 200 intents across three domains — Slicing Provisioning (70), Scaling Request (70), '
    'Conflict Resolution (60). Each intent pairs natural language input with ground-truth JSON. '
    'Metrics: Format Validity, Action Accuracy, Avg Match% (mean key-value accuracy), Inference Time, '
    'Throughput (tokens/s), CPU Utilization (system-wide psutil), Peak Memory (RSS MB), Energy (J) = '
    'measured_CPU_util × TDP_135W × time. All continuous metrics report 95% CI (t-distribution, n=200).'
)
doc.add_paragraph(
    'Phase 1: Four open-weight families at ~8B Q4_K_M — Llama 3.1-8B [10], Qwen2.5-7B [11] '
    '(re-run with v2 benchmark for resource data), Gemma 2-9B [12] (custom Q4_K_M), GLM-4-9B [13] '
    '(custom Q4_K_M). DeepSeek-R1 excluded (infinite chain-of-thought on CPU). '
    'Phase 2: Qwen2.5 across 4 sizes × 3 quantization levels = 12 configurations on identical hardware.'
)
doc.add_paragraph()

# ═══ 3. PHASE 1 ═══
doc.add_heading('3. Phase 1: Cross-Family Model Comparison', level=1)
doc.add_paragraph(
    'Table 1 presents the cross-family benchmark results at ~8B parameters with Q4_K_M quantization. '
    'Figures 1–3 visualize the key accuracy and performance metrics.'
)

table = doc.add_table(rows=5, cols=5, style='Light Grid Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['Model', 'Format %', 'Action %', 'Avg Match %', 'Time (s)']):
    table.rows[0].cells[i].text = h
    for p in table.rows[0].cells[i].paragraphs:
        for run in p.runs: run.bold = True
for i, row in enumerate([
    ['Llama 3.1 8B', '100.0', '36.5', '26.0', '19.11'],
    ['Qwen2.5 7B (re-run)', '100.0', '42.5', '29.8', '15.00'],
    ['Gemma 2 9B', '98.5', '45.5', '29.3', '39.79'],
    ['GLM-4 9B', '100.0', '29.0', '24.6', '22.88'],
]):
    for j, val in enumerate(row):
        table.rows[i+1].cells[j].text = val
p = doc.add_paragraph()
p.add_run('Table 1: ').bold = True
p.add_run('Phase 1 cross-family benchmark results (~8B, Q4_K_M, n=200). Qwen data from re-run with v2 benchmark.')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

add_fig(FIGURES / "phase1_bar_match.png", "Figure 1: Avg Match% — Phase 1 cross-family (~8B, Q4_K_M).")
add_fig(FIGURES / "phase1_bar_action.png", "Figure 2: Action Accuracy% — Phase 1 cross-family (~8B, Q4_K_M).")
add_fig(FIGURES / "phase1_bar_time.png", "Figure 3: Inference Time — Phase 1 cross-family (~8B, Q4_K_M).")

doc.add_heading('3.1 Analysis', level=2)
for f in [
    'Avg Match (Fig. 1): Qwen2.5-7B achieves the highest average key-value match (29.8%), followed by '
    'Gemma 2-9B (29.3%), Llama 3.1-8B (26.0%), and GLM-4-9B (24.6%). Qwen outperforms Llama by 14.6% '
    'relative. The 9B models (Gemma, GLM) show no proportional gains over 7–8B, indicating architecture '
    'and training quality outweigh raw parameter count for parameter-level matching.',
    'Action Accuracy (Fig. 2): Gemma 2-9B leads at 45.5%, demonstrating strong intent understanding. '
    'Qwen achieves 42.5% — a competitive second place. Llama (36.5%) and GLM (29.0%) trail significantly. '
    'GLM\'s poor action accuracy suggests its instruction-tuning is less aligned with structured JSON '
    'output for networking tasks.',
    'Inference Time (Fig. 3): Qwen2.5-7B is the fastest (15.0s), followed by Llama (19.1s) and GLM '
    '(22.9s). Gemma 2-9B is 2.7× slower (39.8s) than Qwen despite only 21% more parameters, making it '
    'impractical for latency-sensitive edge applications. The speed gap likely reflects architectural '
    'differences in vocabulary size and attention mechanisms.',
    'Overall: Qwen2.5-7B wins across all three metrics — highest match, competitive action accuracy, '
    'and fastest inference. Selected as the reference architecture for Phase 2.',
]:
    doc.add_paragraph(f, style='List Bullet')
doc.add_paragraph()

# ═══ 4. PHASE 2 TABLE ═══
doc.add_heading('4. Phase 2: Size Scaling & Quantization Results', level=1)
doc.add_paragraph('Table 2 presents complete Phase 2 results with 95% confidence intervals.')

configs_order = [
    ('0.5B', 'Q2_K'), ('0.5B', 'Q4_K_M'), ('0.5B', 'Q8_0'),
    ('1.5B', 'Q2_K'), ('1.5B', 'Q4_K_M'), ('1.5B', 'Q8_0'),
    ('3B', 'Q2_K'), ('3B', 'Q4_K_M'), ('3B', 'Q8_0'),
    ('7B', 'Q2_K'), ('7B', 'Q4_K_M'), ('7B', 'Q8_0'),
]

table = doc.add_table(rows=13, cols=11, style='Light Grid Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ['Config', 'Match%', '95% CI', 'CPU%', '95% CI', 'Time(s)', '95% CI', 'Energy(J)', '95% CI', 'Action%', 'Mem(MB)']
for i, h in enumerate(headers):
    table.rows[0].cells[i].text = h
    for p in table.rows[0].cells[i].paragraphs:
        for run in p.runs: run.bold = True
        run.font.size = Pt(6)
for idx, (s, q) in enumerate(configs_order):
    key = f"{s}|{q}"
    d = ci_data[key]
    row = table.rows[idx+1]
    vals = [
        f"{s} {q.replace('_',' ')}",
        f"{d['avg_match']:.1f}", f"[{d['ci_match'][2]:.1f},{d['ci_match'][3]:.1f}]",
        f"{d['avg_cpu']:.1f}", f"[{d['ci_cpu'][2]:.1f},{d['ci_cpu'][3]:.1f}]",
        f"{d['avg_time']:.1f}", f"[{d['ci_time'][2]:.1f},{d['ci_time'][3]:.1f}]",
        f"{d['avg_energy']:.0f}", f"[{d['ci_energy'][2]:.0f},{d['ci_energy'][3]:.0f}]",
        f"{d['action']:.1f}", f"{d['peak_mem']:.0f}",
    ]
    for j, val in enumerate(vals):
        row.cells[j].text = val
        for p in row.cells[j].paragraphs:
            for run in p.runs: run.font.size = Pt(6)
p = doc.add_paragraph()
p.add_run('Table 2: ').bold = True
p.add_run('Phase 2 complete results — Qwen2.5 across 4 sizes × 3 Q levels (n=200, 95% CI). '
          'CPU% measured via psutil; Energy from actual CPU × TDP(135W) × time.')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

# ═══ 5. RESOURCE ═══
doc.add_heading('5. Resource Consumption Analysis', level=1)
doc.add_paragraph('Figures 4–6 present the resource consumption characteristics across all 12 configurations.')

add_fig(FIGURES / "paper_bar_peak_mem.png", "Figure 4: Peak Memory (RSS MB) by model size and quantization.")
add_fig(FIGURES / "paper_bar_cpu.png", "Figure 5: CPU Utilization (%) by model size and quantization.")
add_fig(FIGURES / "paper_bar_energy.png", "Figure 6: Energy per Intent (J) by model size and quantization.")

doc.add_heading('5.1 Resource Analysis', level=2)
for f in [
    'Memory (Fig. 4): Peak RSS ranges from 575 MB (0.5B-Q4_K_M) to 11,429 MB (7B-Q8_0). '
    'Q4_K_M consistently reduces memory by ~50% vs Q8_0. 7B-Q8_0 (11.4 GB) exceeds typical 8 GB '
    'edge budgets, making 7B-Q4_K_M (4.9 GB) the practical maximum for edge deployment.',
    'CPU Utilization (Fig. 5): CPU% scales with model size — 68.8% (0.5B-Q2_K) to 94.1% (7B-Q8_0). '
    'This reveals that inference is memory-bandwidth-limited: smaller models cannot saturate the '
    'memory bus. Within the same size class, quantization has minimal impact on CPU% (≤3pp).',
    'Energy (Fig. 6): 0.5B-Q2_K is cheapest (436 J/intent); 7B-Q8_0 most expensive (2,613 J). '
    'Q4_K_M saves 20–30% energy vs Q8_0. The 1.5B–3B Q4_K_M range (850–1,122 J) balances '
    'energy efficiency with competitive accuracy.',
]:
    doc.add_paragraph(f, style='List Bullet')
doc.add_paragraph()

# ═══ 6. PERFORMANCE ═══
doc.add_heading('6. Performance and Accuracy Analysis', level=1)
doc.add_paragraph('Figures 7–10 present throughput, accuracy, and speed across all configurations.')

add_fig(FIGURES / "paper_bar_tps.png", "Figure 7: Throughput (tokens/s) by model size and quantization.")
add_fig(FIGURES / "paper_bar_match.png", "Figure 8: Avg Match% by model size and quantization.")
add_fig(FIGURES / "paper_bar_action.png", "Figure 9: Action Accuracy% by model size and quantization.")
add_fig(FIGURES / "paper_bar_time.png", "Figure 10: Inference Time (s) by model size and quantization.")

doc.add_heading('6.1 Performance Analysis', level=2)
for f in [
    'Throughput (Fig. 7): Inversely proportional to model size. 0.5B-Q2_K: 8.9 tok/s; 7B-Q8_0: 1.4 tok/s. '
    'Larger models produce more output tokens per intent, partially offsetting the lower token rate.',
    'Accuracy — Size Scaling (Figs. 8–9): 0.5B→1.5B yields the largest jump (+5.8pp match, +9.5pp action). '
    '1.5B→3B shows marginal gain (+1.6pp match). 3B→7B adds +3.7pp match but at 67% more time and 80% '
    'more energy. The Pareto "knee" is at 3B — beyond this, costs grow faster than accuracy.',
    'Accuracy — Quantization: Q8_0→Q4_K_M loses only 0.2pp match; Q4_K_M→Q2_K loses 1.6pp. 0.5B least '
    'sensitive (±1.1pp); 1.5B most sensitive (3.7pp gap). 7B-Q2_K (29.0%) outperforms 3B-Q8_0 (26.8%).',
    'Inference Time (Fig. 10): Scales near-linearly with model size. 0.5B-Q4_K_M: 4.7s; 7B-Q8_0: 20.4s. '
    'Q2_K accelerates 0.5B and 1.5B but slows 7B (likely due to dequantization overhead).',
]:
    doc.add_paragraph(f, style='List Bullet')
doc.add_paragraph()

# ═══ 7. PARETO ═══
doc.add_heading('7. Pareto Frontier Analysis', level=1)
doc.add_paragraph(
    'Figures 11–13 present Pareto frontiers. Visual encoding: shape = model size '
    '(■ 0.5B, ◆ 1.5B, ▲ 3B, ● 7B), vertical lines = quantization (no line = Q8_0, | = Q4_K_M, '
    '|| = Q2_K). Dashed gray line = Pareto frontier.'
)

add_fig(FIGURES / "paper_pareto_match_vs_time.png", "Figure 11: Pareto Frontier — Avg Match% vs Inference Time.")
add_fig(FIGURES / "paper_pareto_action_vs_time.png", "Figure 12: Pareto Frontier — Action Accuracy% vs Inference Time.")
add_fig(FIGURES / "paper_pareto_action_vs_energy.png", "Figure 13: Pareto Frontier — Action Accuracy% vs Energy.")

doc.add_heading('7.1 Pareto Analysis', level=2)
for f in [
    'Match vs Time (Fig. 11): 7B-Q8_0 dominates upper-right (30.5%, 20.4s). 3B-Q4_K_M at knee '
    '(27.3%, 9.4s) — 89% of max accuracy at 46% of time. Q2_K configurations shift leftward, '
    'demonstrating speed gains at accuracy cost.',
    'Action vs Time (Fig. 12): 3B models achieve highest action accuracy (44.5%), matching 7B models '
    '(42.5–43.5%). Action classification saturates at moderate sizes — fine-grained parameter matching '
    'is the harder task. 0.5B models lag (25–35% action) due to insufficient capacity.',
    'Action vs Energy (Fig. 13): 0.5B-Q4_K_M is most energy-efficient (35% action, 477J). '
    '3B-Q4_K_M offers 44.5% action at 1,122J — 2.3× energy for 9.5pp gain. 7B-Q8_0 (43.5%, 2,613J) '
    'costs 5.5× energy of 0.5B for only 8.5pp improvement — quantifying the steep energy premium for '
    'marginal accuracy gains at larger scales.',
]:
    doc.add_paragraph(f, style='List Bullet')
doc.add_paragraph()

# ═══ 8. DISCUSSION ═══
doc.add_heading('8. Discussion', level=1)
doc.add_heading('8.1 Deployment Recommendations', level=2)
for r in [
    'Latency-critical edge: 0.5B-Q8_0 (4.8s, 19.4% match) or 1.5B-Q4_K_M (7.5s, 25.8% match).',
    'Balanced edge: 3B-Q4_K_M (9.4s, 27.3% match, 1,122J, 3.3GB) — 91% of 7B accuracy at 60% cost.',
    'Maximum accuracy: 7B-Q4_K_M (15.0s, 29.8% match, 1,884J, 4.9GB). Q8_0 not worth +40% energy for +0.7pp.',
    'Quantization: Q4_K_M recommended for all deployments. Q2_K acceptable when memory-constrained.',
    'Energy estimation: Use measured CPU% values — small models consume 15–25% less than flat 95% TDP-based estimates.',
]:
    doc.add_paragraph(r, style='List Bullet')

doc.add_heading('8.2 Limitations', level=2)
for l in [
    'CPU-only inference; TDP-based energy (±15%); 200-intent dataset; Qwen-only Phase 2.',
    'DeepSeek-R1 excluded — reasoning models require GPU. Future GPU evaluation warranted.',
]:
    doc.add_paragraph(l, style='List Bullet')
doc.add_paragraph()

# ═══ 9. CONCLUSION ═══
doc.add_heading('9. Conclusion', level=1)
doc.add_paragraph(
    'This comprehensive evaluation of 16 SLM configurations for 6G intent-driven slice management '
    'yields four key findings: (1) Qwen2.5-7B achieves best accuracy (29.8% match, 42.5% action); '
    '(2) CPU utilization scales with model size (69→94%), revealing memory bandwidth as bottleneck; '
    '(3) 3B-Q4_K_M is the deployment sweet spot — 91% accuracy at 60% cost; (4) Q4_K_M quantization '
    'provides optimal trade-off with ≤1pp loss. These results provide actionable guidance for 6G edge '
    'infrastructure deployment of intent-based network automation.'
)
doc.add_paragraph()

# ═══ REFERENCES ═══
doc.add_heading('References', level=1)
for ref in [
    '[1] I. Afolabi et al., "Network Slicing and Softwarization: A Survey," IEEE Commun. Surveys Tuts., 2018.',
    '[2] K. Abbas et al., "Intent-Based Networking for 6G," IEEE Commun. Mag., 2024.',
    '[3] ETSI GS ZSM 002, "Zero-touch Network and Service Management," 2019.',
    '[4] J. Wei et al., "Chain-of-Thought Prompting," NeurIPS, 2022.',
    '[5] A. Maatouk et al., "LLMs for Telecom," IEEE Commun. Mag., 2024.',
    '[6] M. Murshed et al., "ML at the Network Edge," ACM Comput. Surv., 2022.',
    '[10] Meta AI, "Llama 3.1: Open Foundation Models," 2024.',
    '[11] Alibaba Cloud, "Qwen2.5 Technical Report," 2024.',
    '[12] Google DeepMind, "Gemma 2 Technical Report," 2024.',
    '[13] Zhipu AI, "GLM-4 Technical Report," 2024.',
    '[14] G. Gerganov, "GGUF Format," github.com/ggerganov/ggml, 2024.',
    '[15] T. Dettmers et al., "QLoRA," NeurIPS, 2023.',
    '[16] D. Patterson et al., "Carbon Emissions and NN Training," arXiv:2104.10350, 2021.',
    '[17] R. Schwartz et al., "Green AI," Commun. ACM, 2020.',
    '[18] T. Somsri et al., "SLM Agent Eval for 6G," GitHub, 2025.',
]:
    p = doc.add_paragraph(ref)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Cm(1)
    for run in p.runs: run.font.size = Pt(9)

docx_path = PROJECT / "paper_with_figures.docx"
doc.save(str(docx_path))
print(f"✅ DOCX: {docx_path} ({os.path.getsize(docx_path)/1024:.1f} KB)")

pdf_path = PROJECT / "paper_with_figures.pdf"
result = subprocess.run(
    ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', str(PROJECT), str(docx_path)],
    capture_output=True, text=True, timeout=120
)
if result.returncode == 0:
    print(f"✅ PDF: {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)")
else:
    print(f"❌ PDF: {result.stderr[:200]}")
