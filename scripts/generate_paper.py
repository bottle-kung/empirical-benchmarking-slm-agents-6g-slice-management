#!/usr/bin/env python3
"""Generate comprehensive paper for SLM 6G Eval v2"""
import json, os, subprocess
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

PROJECT = "/jupyter_workspace/rbru/slm_6g_eval_v2"
with open(f"{PROJECT}/results/ci_summary.json") as f:
    ci_data = json.load(f)

doc = Document()
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(11)

# ── TITLE ──
title = doc.add_heading('Empirical Evaluation of Small Language Models for\nIntent-Driven Slice Lifecycle Management in 6G Core Networks', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in title.runs:
    run.font.size = Pt(16)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('Teerapong Somsri, [Co-authors TBD]')
run.font.italic = True
p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p2.add_run('Rajabhat Ratchanakarin University (RBRU)').font.size = Pt(10)
doc.add_paragraph()

# ── ABSTRACT ──
doc.add_heading('Abstract', level=1)
abs_text = (
    'The deployment of intent-based networking (IBN) in 6G core networks requires lightweight, '
    'efficient language models capable of translating natural language intents into network '
    'orchestration commands. This paper presents a comprehensive empirical evaluation of small '
    'language models (SLMs) for intent-driven slice lifecycle management, examining both '
    'cross-family performance (Llama, Qwen, Gemma, GLM at ~8B parameters) and the impact of '
    'model size scaling (0.5B-7B) with quantization (Q2_K, Q4_K_M, Q8_0). Using a dataset of '
    '200 intents spanning provisioning, scaling, and conflict resolution domains, we evaluate '
    'models across seven dimensions: format validity, action accuracy, exact parameter match, '
    'inference latency, throughput, memory consumption, and energy efficiency. '
    'Our results reveal that Qwen2.5-7B achieves the highest overall accuracy (29.8% exact match, '
    '43.5% action accuracy) while Qwen2.5-3B-Q4_K_M offers the best cost-effectiveness ratio. '
    'We also identify diminishing returns beyond 3B parameters and find that quantization '
    'degradation (Q8_0 to Q2_K) impacts accuracy by only 1-3 percentage points while '
    'reducing energy consumption by up to 40%. These findings provide practical guidelines for '
    'deploying SLMs in resource-constrained 6G edge environments.'
)
p = doc.add_paragraph(abs_text)
p.paragraph_format.first_line_indent = Cm(1.27)

p = doc.add_paragraph()
run = p.add_run('Keywords: ')
run.bold = True
p.add_run('Small Language Models, 6G Networks, Intent-Based Networking, Network Slicing, Model Quantization, Edge AI, Resource Efficiency')
doc.add_paragraph()

# ═══════════════════════ 1. INTRODUCTION ═══════════════════════
doc.add_heading('1. Introduction', level=1)
intro_paras = [
    'Sixth-generation (6G) networks are expected to support diverse and demanding use cases '
    'including enhanced mobile broadband (eMBB), ultra-reliable low-latency communications (URLLC), '
    'and massive machine-type communications (mMTC). Network slicing — the ability to create '
    'multiple virtualized and independent logical networks on a shared physical infrastructure — '
    'is a cornerstone technology for meeting these heterogeneous requirements [1,2].',
    'Intent-based networking (IBN) has emerged as a promising paradigm that allows network operators '
    'to express high-level business goals in natural language, which are then automatically translated '
    'into low-level configurations [3]. Large language models (LLMs) have demonstrated remarkable '
    'capabilities in understanding and generating structured outputs from natural language, making '
    'them natural candidates for the intent translation task in 6G IBN systems [4,5].',
    'However, deploying LLMs in 6G edge environments presents significant challenges. The computational '
    'and memory requirements of models with 7-70 billion parameters are often incompatible with the '
    'resource constraints of edge nodes, where power budgets may be as low as 10-50W and memory limited '
    'to a few gigabytes [6]. Small language models (SLMs) with 0.5-8B parameters, combined with '
    'quantization techniques (2-8 bits), offer a potential solution — but their performance on '
    'domain-specific networking tasks remains underexplored.',
    'This paper addresses four research questions:',
]

for text in intro_paras:
    p = doc.add_paragraph(text)
    p.paragraph_format.first_line_indent = Cm(1.27)

rqs = [
    'RQ1: How do different SLM architectures (Llama, Qwen, Gemma, GLM) compare in intent translation accuracy?',
    'RQ2: What is the relationship between model size (0.5B-7B) and intent translation performance?',
    'RQ3: How does quantization (Q2_K, Q4_K_M, Q8_0) affect the accuracy-efficiency trade-off?',
    'RQ4: What are the resource consumption characteristics of SLM inference on CPU-only hardware?',
]
for rq in rqs:
    p = doc.add_paragraph(rq, style='List Bullet')

doc.add_paragraph(
    'We answer these questions through a systematic, reproducible benchmark of 16 model configurations '
    'on identical CPU-only hardware. Our contributions include: (1) the first comprehensive comparison '
    'of open-weight SLMs for 6G intent translation, (2) a detailed Pareto frontier analysis across '
    'model sizes and quantization levels, (3) quantitative evidence of diminishing returns beyond 3B, and '
    '(4) practical deployment recommendations for resource-constrained 6G edge infrastructure.'
).paragraph_format.first_line_indent = Cm(1.27)
doc.add_paragraph()

# ═══════════════════════ 2. RELATED WORK ═══════════════════════
doc.add_heading('2. Related Work', level=1)
rw = [
    ('2.1 Intent-Based Networking in 6G',
     'IBN has been extensively studied in the context of 5G and beyond. ETSI ZSM [7] and 3GPP [8] '
     'have standardized frameworks for intent-driven closed-loop automation. Recent works by Mechtri '
     'et al. [9] and Abbas et al. [2] have proposed natural language interfaces for network slice '
     'management. However, these approaches typically assume cloud-based LLMs, which are impractical '
     'for edge deployment.'),
    ('2.2 Small Language Models for Domain Tasks',
     'The emergence of efficient open-weight models such as Llama 3.1 [10], Qwen2.5 [11], Gemma 2 [12], '
     'and GLM-4 [13] has made SLM deployment feasible on consumer hardware. Recent benchmarks have '
     'evaluated these models on general NLP tasks, but domain-specific evaluation for networking '
     'applications remains scarce.'),
    ('2.3 Model Quantization for Edge Deployment',
     'The GGUF format [14] with K-quant methods has become the standard for CPU inference. '
     'Dettmers et al. [15] demonstrated that 4-bit quantization preserves model quality for most '
     'NLP tasks. Our work extends this analysis to the networking domain.'),
    ('2.4 Resource-Aware Model Selection',
     'Prior work on energy-aware ML has focused on training efficiency [16] or cloud inference cost [17]. '
     'For edge deployment, memory bandwidth and CPU utilization are primary bottlenecks [18]. Our work '
     'contributes per-intent CPU, memory, and energy measurements across 12 configurations on identical hardware.'),
]
for title, text in rw:
    doc.add_heading(title, level=2)
    p = doc.add_paragraph(text)
    p.paragraph_format.first_line_indent = Cm(1.27)
doc.add_paragraph()

# ═══════════════════════ 3. METHODOLOGY ═══════════════════════
doc.add_heading('3. Methodology', level=1)

doc.add_heading('3.1 Dataset', level=2)
doc.add_paragraph(
    'We constructed a dataset of 200 natural language intents spanning three network slice '
    'lifecycle domains: Slicing Provisioning (70 intents), Scaling Request (70 intents), and '
    'Conflict Resolution (60 intents). Each intent consists of an input_text and a ground_truth JSON '
    'object containing expected action, slice type, and domain-specific parameters.'
)
p = doc.add_paragraph()
p.add_run('Example Intent: ').bold = True
p.add_run('Input: "Create a new eMBB slice for a 4K video streaming event in Bangkok." '
          'Ground Truth: {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 1}')

doc.add_heading('3.2 Models', level=2)
doc.add_paragraph(
    'We evaluate models across two experimental phases. Phase 1 compares four open-weight SLM families '
    'at ~8B parameters under Q4_K_M quantization: Llama 3.1-8B, Qwen2.5-7B, Gemma 2-9B (custom Q4_K_M), '
    'and GLM-4-9B (custom Q4_K_M). DeepSeek-R1-8B was excluded due to CPU-only incompatibility — its '
    'reasoning architecture caused infinite chain-of-thought loops on CPU inference [20].'
)
doc.add_paragraph(
    'Phase 2 systematically evaluates Qwen2.5 (selected as the top performer from Phase 1) across '
    'four sizes (0.5B, 1.5B, 3B, 7B) and three quantization levels (Q2_K, Q4_K_M, Q8_0), '
    'resulting in 12 configurations.'
)

doc.add_heading('3.3 Metrics & Statistical Framework', level=2)
doc.add_paragraph(
    'We define seven evaluation metrics: Format Validity (valid JSON with required fields), '
    'Action Accuracy (correct CREATE/MODIFY/DELETE), Avg Match% (mean percentage of ground-truth '
    'key-value pairs correctly predicted), Inference Time (wall-clock seconds), Throughput (tokens/s), '
    'Memory Footprint (peak RSS in MB via psutil), and Energy Consumption (estimated as '
    'CPU_utilization × TDP_135W × inference_time). For every continuous metric, we report '
    'the mean, standard deviation, and 95% confidence interval [μ ± t₀.₀₂₅ · σ/√n] across '
    'n = 200 intents using the t-distribution (df = 199, t ≈ 1.972).'
)

doc.add_heading('3.4 Hardware & Software', level=2)
doc.add_paragraph(
    'All experiments run on identical Proxmox VE 8.2 KVM virtual machines with 16 vCPUs '
    '(Intel Xeon E5-2690 v4 @ 2.60GHz), 32GB RAM, and 80GB SSD. Software: Ubuntu 24.04, '
    'Ollama 0.5.x (llama.cpp backend), Python 3.12, psutil 6.x. Each configuration is evaluated '
    'independently with a 2-intent warmup to ensure consistent model loading and KV-cache state.'
)
doc.add_paragraph()

# ═══════════════════════ 4. PHASE 1 ═══════════════════════
doc.add_heading('4. Phase 1: Cross-Family Model Comparison', level=1)
doc.add_paragraph(
    'Table 1 presents the results of the cross-family comparison at ~8B parameters with Q4_K_M quantization.'
)

# Phase 1 table
table = doc.add_table(rows=5, cols=5, style='Light Grid Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['Model', 'Format %', 'Action %', 'Avg Match %', 'Time (s)']):
    table.rows[0].cells[i].text = h
    for p in table.rows[0].cells[i].paragraphs:
        for run in p.runs: run.bold = True
p1 = [
    ['Llama 3.1 8B', '100.0', '36.5', '26.0', '19.11'],
    ['Qwen2.5 7B', '100.0', '43.5', '29.8', '18.68'],
    ['Gemma 2 9B', '98.5', '45.5', '29.3', '39.79'],
    ['GLM-4 9B', '100.0', '29.0', '24.6', '22.88'],
]
for i, row in enumerate(p1):
    for j, val in enumerate(row):
        table.rows[i+1].cells[j].text = val

p = doc.add_paragraph()
p.add_run('Table 1: ').bold = True
p.add_run('Phase 1 cross-family benchmark results (~8B, Q4_K_M, n=200 intents).')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

doc.add_heading('4.1 Key Findings', level=2)
findings = [
    'Qwen2.5-7B achieves the best overall balance: highest exact match (29.8%) with lowest inference time (18.68s), representing a 14.6% relative improvement over Llama 3.1.',
    'Gemma 2-9B has the highest action accuracy (45.5%) but at 2.1× the latency of Qwen (39.79s vs 18.68s), making it unsuitable for latency-sensitive edge applications.',
    'GLM-4-9B significantly underperforms in action accuracy (29.0%), suggesting its instruction-tuning may be less aligned with structured JSON output tasks.',
    'The 9B models (Gemma, GLM) do not show proportional gains over 7-8B models, indicating that architecture and training quality outweigh raw parameter count for this task.',
]
for f in findings:
    doc.add_paragraph(f, style='List Bullet')
doc.add_paragraph(
    'Based on these results, Qwen2.5 is selected as the reference architecture for Phase 2.'
)
doc.add_paragraph()

# ═══════════════════════ 5. PHASE 2 ═══════════════════════
doc.add_heading('5. Phase 2: Size Scaling & Quantization', level=1)
doc.add_heading('5.1 Complete Results with 95% Confidence Intervals', level=2)
doc.add_paragraph('Table 2 presents the complete Phase 2 results with 95% confidence intervals for all metrics.')

configs_order = [
    ('0.5B', 'Q2_K'), ('0.5B', 'Q4_K_M'), ('0.5B', 'Q8_0'),
    ('1.5B', 'Q2_K'), ('1.5B', 'Q4_K_M'), ('1.5B', 'Q8_0'),
    ('3B', 'Q2_K'), ('3B', 'Q4_K_M'), ('3B', 'Q8_0'),
    ('7B', 'Q2_K'), ('7B', 'Q4_K_M'), ('7B', 'Q8_0'),
]

table = doc.add_table(rows=13, cols=9, style='Light Grid Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ['Config', 'Match%', '95% CI', 'Action%', 'Format%', 'Time(s)', '95% CI', 'Energy(J)', 'Mem(MB)']
for i, h in enumerate(headers):
    table.rows[0].cells[i].text = h
    for p in table.rows[0].cells[i].paragraphs:
        for run in p.runs: run.bold = True
        run.font.size = Pt(7)

for idx, (s, q) in enumerate(configs_order):
    key = f"{s}|{q}"
    d = ci_data[key]
    row = table.rows[idx+1]
    vals = [
        f"{s} {q.replace('_',' ')}",
        f"{d['avg_match']:.1f}",
        f"[{d['ci_match'][2]:.1f},{d['ci_match'][3]:.1f}]",
        f"{d['action']:.1f}", f"{d['format']:.1f}",
        f"{d['avg_time']:.1f}",
        f"[{d['ci_time'][2]:.1f},{d['ci_time'][3]:.1f}]",
        f"{d['avg_energy']:.0f}", f"{d['peak_mem']:.0f}",
    ]
    for j, val in enumerate(vals):
        table.rows[idx+1].cells[j].text = val
        for p in table.rows[idx+1].cells[j].paragraphs:
            for run in p.runs:
                run.font.size = Pt(7)

p = doc.add_paragraph()
p.add_run('Table 2: ').bold = True
p.add_run('Phase 2 complete results — Qwen2.5 across 4 sizes × 3 quantization levels (n=200, 95% CI).')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

doc.add_heading('5.2 Size Scaling Analysis', level=2)
size_findings = [
    '0.5B → 1.5B: Avg match improves from 19.0% to 24.7% (+5.7pp, +30% relative). This is the largest single-step gain, indicating 0.5B models lack capacity for structured JSON generation.',
    '1.5B → 3B: Avg match improves from 24.7% to 26.1% (+1.4pp, +5.7% relative). Marginal gain per billion parameters drops significantly.',
    '3B → 7B: Avg match improves from 26.1% to 29.6% (+3.5pp, +13.4% relative), but at 67% more time and 80% more energy. The Pareto frontier reveals a "knee" at 3B.',
]
for f in size_findings:
    doc.add_paragraph(f, style='List Bullet')

doc.add_heading('5.3 Quantization Impact', level=2)
quant_findings = [
    'Q8_0 → Q4_K_M: Avg match loss of only 0.2pp across all sizes, while reducing model size by ~50% and memory by ~40%. Q4_K_M is the optimal accuracy-efficiency trade-off.',
    'Q4_K_M → Q2_K: Avg match loss of 1.6pp, with energy savings of 20-30%. The 7B-Q2_K configuration (28.8% match, 12.2s, 1,566J) outperforms 3B-Q8_0 (27.0%) while consuming only 72% more energy.',
    'Quantization sensitivity varies by size: 0.5B shows minimal impact (±1.5pp), while 1.5B is most sensitive (up to 5.3pp loss from Q4_K_M to Q2_K).',
]
for f in quant_findings:
    doc.add_paragraph(f, style='List Bullet')
doc.add_paragraph()

# ═══════════════════════ 6. RESOURCE ANALYSIS ═══════════════════════
doc.add_heading('6. Resource Consumption Analysis', level=1)

doc.add_heading('6.1 Memory Footprint', level=2)
doc.add_paragraph(
    'Memory consumption scales primarily with model size and quantization level. '
    'Peak RSS ranges from 575 MB (0.5B-Q4_K_M) to 11,408 MB (7B-Q8_0). '
    'Table 3 shows the memory footprint across all configurations.'
)
mem_table = doc.add_table(rows=5, cols=4, style='Light Grid Accent 1')
mem_table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['Size', 'Q2_K (MB)', 'Q4_K_M (MB)', 'Q8_0 (MB)']):
    mem_table.rows[0].cells[i].text = h
    for p in mem_table.rows[0].cells[i].paragraphs:
        for run in p.runs: run.bold = True
for i, s in enumerate(['0.5B', '1.5B', '3B', '7B']):
    mem_table.rows[i+1].cells[0].text = s
    for j, q in enumerate(['Q2_K', 'Q4_K_M', 'Q8_0']):
        key = f"{s}|{q}"
        mem_table.rows[i+1].cells[j+1].text = f"{ci_data[key]['peak_mem']:.0f}"

p = doc.add_paragraph()
p.add_run('Table 3: ').bold = True
p.add_run('Peak memory footprint (RSS in MB) by model size and quantization.')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

doc.add_heading('6.2 Energy Efficiency Ranking', level=2)
doc.add_paragraph(
    'Energy consumption is dominated by inference time, as CPU utilization remains at ~95% for all '
    'configurations. Table 4 ranks all configurations by energy efficiency (Match% per Joule).'
)

energy_rankings = sorted(
    [(s, q, ci_data[f"{s}|{q}"]) for s, q in configs_order],
    key=lambda x: x[2]['avg_match'] / x[2]['avg_energy'] if x[2]['avg_energy'] > 0 else 0
)
table = doc.add_table(rows=len(energy_rankings)+1, cols=6, style='Light Grid Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['Rank', 'Config', 'Match%', 'Energy(J)', 'Match/J ×100', 'Mem(MB)']):
    table.rows[0].cells[i].text = h
    for p in table.rows[0].cells[i].paragraphs:
        for run in p.runs: run.bold = True
for idx, (s, q, d) in enumerate(energy_rankings):
    eff = d['avg_match'] / d['avg_energy'] * 100 if d['avg_energy'] > 0 else 0
    vals = [str(idx+1), f"{s} {q.replace('_',' ')}", f"{d['avg_match']:.1f}",
            f"{d['avg_energy']:.0f}", f"{eff:.3f}", f"{d['peak_mem']:.0f}"]
    for j, val in enumerate(vals):
        table.rows[idx+1].cells[j].text = val
        for p in table.rows[idx+1].cells[j].paragraphs:
            for run in p.runs: run.font.size = Pt(8)

p = doc.add_paragraph()
p.add_run('Table 4: ').bold = True
p.add_run('Energy efficiency ranking — sorted by Match%/Joule.')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

doc.add_paragraph(
    'The 1.5B-Q8_0 configuration achieves the highest energy efficiency (0.036 Match/J), '
    'followed by 0.5B-Q4_K_M (0.034 Match/J). The 7B models are the least efficient '
    '(0.011-0.013 Match/J), consuming ~3× more energy per unit of accuracy compared to 1.5B models.'
)

doc.add_paragraph()

# ═══════════════════════ 7. DOMAIN ANALYSIS ═══════════════════════
doc.add_heading('7. Domain-Level Analysis', level=1)
doc.add_paragraph(
    'We decompose performance across three network slice lifecycle domains (Q4_K_M configurations).'
)

d_table = doc.add_table(rows=5, cols=5, style='Light Grid Accent 1')
d_table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['Size', 'Provisioning\nMatch%/Act%', 'Scaling\nMatch%/Act%', 'Conflict\nMatch%/Act%', 'Overall']):
    d_table.rows[0].cells[i].text = h
    for p in d_table.rows[0].cells[i].paragraphs:
        for run in p.runs: run.bold = True
for i, s in enumerate(['0.5B', '1.5B', '3B', '7B']):
    key = f"{s}|Q4_K_M"
    d = ci_data.get(key, {})
    dom = d.get('domain', {})
    d_table.rows[i+1].cells[0].text = s
    dp = dom.get('Slicing Provisioning', {})
    d_table.rows[i+1].cells[1].text = f"{dp.get('match',[0])[0]:.1f}%/{dp.get('action_pct',0):.1f}%"
    ds = dom.get('Scaling Request', {})
    d_table.rows[i+1].cells[2].text = f"{ds.get('match',[0])[0]:.1f}%/{ds.get('action_pct',0):.1f}%"
    dc = dom.get('Conflict Resolution', {})
    d_table.rows[i+1].cells[3].text = f"{dc.get('match',[0])[0]:.1f}%/{dc.get('action_pct',0):.1f}%"
    d_table.rows[i+1].cells[4].text = f"{d.get('avg_match',0):.1f}%/{d.get('action',0):.1f}%"

p = doc.add_paragraph()
p.add_run('Table 5: ').bold = True
p.add_run('Domain-level performance breakdown (Q4_K_M).')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

doc.add_paragraph(
    'Provisioning intents consistently achieve the highest accuracy (35-42% match, 70-76% action) '
    'due to well-defined JSON schemas. Scaling intents show moderate performance (15-22% match) '
    'reflecting ambiguity in scaling decisions. Conflict resolution is the most challenging domain '
    '(2-4% match), requiring multi-constraint optimization beyond current SLM capabilities.'
)
doc.add_paragraph()

# ═══════════════════════ 8. DISCUSSION ═══════════════════════
doc.add_heading('8. Discussion', level=1)
doc.add_heading('8.1 Deployment Recommendations', level=2)
recs = [
    'Latency-critical edge (<10s): Qwen2.5-3B-Q4_K_M (27.1% match, 9.9s, 1,274J, 3.3GB). Offers 91% of 7B accuracy at 60% cost.',
    'Accuracy-prioritized: Qwen2.5-7B-Q8_0 (30.3% match, 20.6s, 2,647J). Upper bound for CPU-only inference.',
    'Ultra-light edge (<2GB): Qwen2.5-1.5B-Q4_K_M (26.4% match, 8.05s, 1,033J, 1.7GB). Competitive accuracy at minimal resource.',
    'Quantization: Always prefer Q4_K_M over Q8_0. Q2_K acceptable for memory-constrained scenarios but avoid for 1.5B models.',
]
for r in recs:
    doc.add_paragraph(r, style='List Bullet')

doc.add_heading('8.2 Limitations', level=2)
lims = [
    'CPU-only inference: Results are specific to CPU inference. GPU acceleration would change the latency-energy trade-off.',
    'Energy estimation: Energy is TDP-based approximation (±15%). Direct RAPL measurement was unavailable on VM infrastructure.',
    'Dataset scope: 200-intent dataset may not capture full production diversity. Future work should validate on operational data.',
    'Single architecture (Phase 2): Findings are specific to Qwen2.5. Other architectures may exhibit different scaling behavior.',
    'DeepSeek-R1 exclusion: Reasoning models were incompatible with CPU inference. Future GPU-based evaluation is warranted.',
]
for l in lims:
    doc.add_paragraph(l, style='List Bullet')

doc.add_heading('8.3 Future Work', level=2)
fw = [
    'Multi-agent orchestration with domain-specialized SLMs leveraging observed domain strengths.',
    'Instruction-tuning on 6G network data, particularly for conflict resolution underperformance.',
    'GPU benchmarking to establish performance upper bounds.',
    'Online learning through operator feedback on edge deployments.',
    'Integration with formal verification for production safety guarantees.',
]
for f_text in fw:
    doc.add_paragraph(f_text, style='List Bullet')
doc.add_paragraph()

# ═══════════════════════ 9. CONCLUSION ═══════════════════════
doc.add_heading('9. Conclusion', level=1)
doc.add_paragraph(
    'This paper presented a comprehensive empirical evaluation of small language models for '
    'intent-driven slice lifecycle management in 6G core networks. Through systematic benchmarking '
    'of 16 model configurations across 200 networking intents with per-intent resource monitoring, '
    'we demonstrated that:'
)
concs = [
    'Qwen2.5-7B achieves the highest intent translation accuracy (29.8% match, 43.5% action) among open-weight SLMs.',
    'Model size scaling exhibits diminishing returns beyond 3B, with 3B models achieving 91% of 7B accuracy at 60% cost.',
    'Q4_K_M provides the optimal accuracy-efficiency trade-off (≤1pp loss, ~50% memory reduction).',
    'Energy consumption is dominated by inference time, making throughput the primary lever for efficiency.',
    'Domain difficulty varies significantly: provisioning is well-handled even by 0.5B models; conflict resolution challenges all SLMs.',
    'The 1.5B-3B Q4_K_M configurations represent the sweet spot for edge deployment, balancing accuracy with practical resource budgets.',
]
for c in concs:
    doc.add_paragraph(c, style='List Bullet')
doc.add_paragraph(
    'These findings provide actionable guidance for network operators deploying intent-based '
    'automation in 6G infrastructure, and establish a foundation for future research in '
    'domain-adapted SLMs for network management.'
)
doc.add_paragraph()

# ═══════════════════════ REFERENCES ═══════════════════════
doc.add_heading('References', level=1)
refs = [
    '[1] I. Afolabi et al., "Network Slicing and Softwarization: A Survey," IEEE Commun. Surveys Tuts., vol. 20, no. 3, 2018.',
    '[2] K. Abbas et al., "Intent-Based Networking for 6G: Vision, Challenges, and Enablers," IEEE Commun. Mag., vol. 62, no. 1, 2024.',
    '[3] ETSI GS ZSM 002, "Zero-touch Network and Service Management (ZSM); Reference Architecture," 2019.',
    '[4] J. Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," NeurIPS, 2022.',
    '[5] A. Maatouk et al., "Large Language Models for Telecom: The Next Frontier," IEEE Commun. Mag., vol. 62, no. 3, 2024.',
    '[6] M. G. S. Murshed et al., "Machine Learning at the Network Edge: A Survey," ACM Comput. Surv., vol. 54, no. 8, 2022.',
    '[7] ETSI GS ZSM 001, "Zero-touch Network and Service Management; Requirements," 2019.',
    '[8] 3GPP TS 28.530, "Management and Orchestration; Concepts, Use Cases and Requirements," 2023.',
    '[9] M. Mechtri et al., "Intent-Based Networking: A Comprehensive Survey," IEEE Access, vol. 11, 2023.',
    '[10] Meta AI, "Llama 3.1: Open and Efficient Foundation Language Models," 2024.',
    '[11] Alibaba Cloud, "Qwen2.5 Technical Report," 2024.',
    '[12] Google DeepMind, "Gemma 2: Improving Open Language Models at a Practical Size," 2024.',
    '[13] Zhipu AI, "GLM-4 Technical Report," 2024.',
    '[14] G. Gerganov, "GGUF: GPT-Generated Unified Format," github.com/ggerganov/ggml, 2024.',
    '[15] T. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS, 2023.',
    '[16] D. Patterson et al., "Carbon Emissions and Large Neural Network Training," arXiv:2104.10350, 2021.',
    '[17] R. Schwartz et al., "Green AI," Commun. ACM, vol. 63, no. 12, 2020.',
    '[18] S. Han et al., "EIE: Efficient Inference Engine on Compressed Sparse Neural Network," ISCA, 2016.',
    '[19] T. Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," NeurIPS, 2022.',
    '[20] T. Somsri et al., "SLM Agent Evaluation for Intent-Driven 6G Slice Management," GitHub: bottle-kung/empirical-benchmarking-slm-agents-6g-slice-management, 2025.',
]
for ref in refs:
    p = doc.add_paragraph(ref)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Cm(1)
    for run in p.runs:
        run.font.size = Pt(9)

# ── Save ──
docx_path = os.path.join(PROJECT, "paper_comprehensive_v2.docx")
doc.save(docx_path)
print(f"✅ DOCX: {docx_path} ({os.path.getsize(docx_path)/1024:.1f} KB)")

pdf_path = os.path.join(PROJECT, "paper_comprehensive_v2.pdf")
result = subprocess.run(
    ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', PROJECT, docx_path],
    capture_output=True, text=True, timeout=120
)
if result.returncode == 0:
    print(f"✅ PDF: {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)")
else:
    print(f"❌ PDF failed: {result.stderr[:200]}")
