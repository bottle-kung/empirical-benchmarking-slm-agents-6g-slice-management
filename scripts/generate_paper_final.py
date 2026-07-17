#!/usr/bin/env python3
"""Generate final paper with selected figures and analysis"""
import json, math, os, glob, subprocess
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

PROJECT = Path("/jupyter_workspace/rbru/slm_6g_eval_v2")
FIGURES = PROJECT / "results" / "figures"

# Load Phase 2 data
with open(PROJECT / "results" / "ci_summary_v3.json") as f:
    ci_data = json.load(f)

def add_figure(doc, path, caption, width=5.5):
    """Add a figure with caption."""
    if os.path.exists(path):
        doc.add_picture(str(path), width=Inches(width))
        p = doc.add_paragraph()
        p.add_run(caption).bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()

doc = Document()
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(11)

# ═══════════════ TITLE ═══════════════
title = doc.add_heading('Empirical Evaluation of Small Language Models for\nIntent-Driven Slice Lifecycle Management in 6G Core Networks', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in title.runs:
    run.font.size = Pt(15)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run('Teerapong Somsri').italic = True
p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p2.add_run('Rajabhat Ratchanakarin University (RBRU)').font.size = Pt(10)
doc.add_paragraph()

# ═══════════════ ABSTRACT ═══════════════
doc.add_heading('Abstract', level=1)
doc.add_paragraph(
    'This paper presents a comprehensive empirical evaluation of small language models (SLMs) for '
    'intent-driven slice lifecycle management in 6G core networks. We conduct a two-phase benchmark: '
    'Phase 1 compares four open-weight SLM families (Llama, Qwen, Gemma, GLM) at ~8B parameters '
    'under Q4_K_M quantization, while Phase 2 systematically evaluates Qwen2.5 across four model '
    'sizes (0.5B–7B) and three quantization levels (Q2_K, Q4_K_M, Q8_0) with per-intent CPU, memory, '
    'and energy monitoring. Our results show Qwen2.5-7B achieves the highest accuracy (29.8% match, '
    '42.5% action), but CPU utilization scales with model size (69%→94%), revealing memory bandwidth '
    'as the primary bottleneck. We identify 3B-Q4_K_M as the optimal deployment sweet spot and '
    'demonstrate that Q4_K_M quantization reduces memory by ~50% with under 1pp accuracy loss.'
).paragraph_format.first_line_indent = Cm(1.27)

p = doc.add_paragraph()
p.add_run('Keywords: ').bold = True
p.add_run('Small Language Models, 6G Networks, Intent-Based Networking, Model Quantization, Edge AI')
doc.add_paragraph()

# ═══════════════ 1. INTRODUCTION ═══════════════
doc.add_heading('1. Introduction', level=1)
for text in [
    'Sixth-generation (6G) networks must support diverse use cases via network slicing — the creation '
    'of multiple virtualized logical networks on shared infrastructure [1,2]. Intent-based networking '
    '(IBN) allows operators to express goals in natural language, translated into configurations by AI [3]. '
    'Large language models (LLMs) excel at this translation [4,5], but their computational requirements '
    'conflict with edge infrastructure constraints [6].',
    'Small language models (SLMs) with 0.5–8B parameters, combined with quantization, offer a potential '
    'solution for edge deployment. However, their performance on domain-specific networking tasks remains '
    'underexplored. This paper provides the first systematic evaluation of open-weight SLMs for 6G intent '
    'translation with per-intent resource monitoring, answering: (1) Which SLM architecture performs best? '
    '(2) How does model size affect accuracy-efficiency trade-offs? (3) What is the optimal quantization '
    'level for edge deployment? (4) What are the real resource consumption characteristics?',
]:
    p = doc.add_paragraph(text)
    p.paragraph_format.first_line_indent = Cm(1.27)
doc.add_paragraph()

# ═══════════════ 2. METHODOLOGY ═══════════════
doc.add_heading('2. Methodology', level=1)
doc.add_paragraph(
    'Dataset: 200 intents across three domains — Slicing Provisioning (70), Scaling Request (70), '
    'Conflict Resolution (60). Each intent pairs natural language input with ground-truth JSON. '
    'Metrics: Format Validity, Action Accuracy, Avg Match% (mean key-value accuracy), Inference Time, '
    'Throughput (tokens/s), CPU Utilization (system-wide psutil), Peak Memory (RSS MB), Energy (J) = '
    'CPU_util × TDP_135W × time. All continuous metrics report 95% CI (t-distribution, n=200). '
    'Hardware: Proxmox VMs with 16 vCPUs (Xeon E5-2690 v4), 32GB RAM, 80GB SSD; Ollama 0.5.x, '
    'Python 3.12, psutil 6.x.'
)
doc.add_paragraph(
    'Phase 1: Four open-weight families at ~8B — Llama 3.1-8B [10], Qwen2.5-7B [11] (re-run with v2 '
    'benchmark), Gemma 2-9B [12] (custom Q4_K_M), GLM-4-9B [13] (custom Q4_K_M). DeepSeek-R1 excluded '
    '(infinite chain-of-thought on CPU). Phase 2: Qwen2.5 across 4 sizes × 3 Q levels = 12 configurations.'
)
doc.add_paragraph()

# ═══════════════ 3. PHASE 1 ═══════════════
doc.add_heading('3. Phase 1: Cross-Family Model Comparison', level=1)
doc.add_paragraph(
    'Phase 1 evaluates four SLM families at comparable parameter counts (~8B) under identical Q4_K_M '
    'quantization. Qwen2.5-7B was re-run with the v2 benchmark to ensure resource metric compatibility. '
    'Figure 1 presents the key accuracy and performance metrics.'
)

# Phase 1 Figures
add_figure(doc, FIGURES / "paper_phase1_comparison_v2.png",
           "Figure 1: Phase 1 cross-family comparison — Avg Match%, Action Accuracy%, and Inference Time (~8B, Q4_K_M).")

doc.add_heading('3.1 Analysis', level=2)
for f in [
    'Accuracy: Qwen2.5-7B achieves the highest Avg Match (29.8%), followed by Gemma 2-9B (29.3%), '
    'Llama 3.1-8B (26.0%), and GLM-4-9B (24.6%). Qwen outperforms Llama by 14.6% relative improvement '
    'in key-value matching accuracy.',
    'Action Accuracy: Gemma 2-9B leads at 45.5%, demonstrating strong intent understanding despite '
    'lower parameter-level precision. Qwen (42.5%) and Llama (36.5%) follow. GLM-4 (29.0%) significantly '
    'underperforms, suggesting its instruction-tuning is less aligned with structured JSON output.',
    'Inference Time: Qwen2.5-7B is the fastest (15.0s), followed by Llama 3.1-8B (19.1s) and GLM-4-9B '
    '(22.9s). Gemma 2-9B is 2.7× slower (39.8s) than Qwen despite only 21% more parameters, making it '
    'unsuitable for latency-sensitive edge applications.',
    'Winner: Qwen2.5-7B — best balance of accuracy (29.8% match, 42.5% action) and speed (15.0s). '
    'The ~9B models show no proportional accuracy gains over 7-8B, indicating architecture quality '
    'outweighs raw parameter count. Qwen2.5 is selected as the reference architecture for Phase 2.',
]:
    doc.add_paragraph(f, style='List Bullet')
doc.add_paragraph()

# ═══════════════ 4. PHASE 2: RESOURCE ANALYSIS ═══════════════
doc.add_heading('4. Phase 2: Resource Consumption Analysis', level=1)
doc.add_paragraph(
    'Phase 2 evaluates Qwen2.5 across four sizes (0.5B, 1.5B, 3B, 7B) and three quantization levels '
    '(Q2_K, Q4_K_M, Q8_0) with per-intent CPU, memory, and energy monitoring. We first analyze '
    'resource consumption characteristics, then examine performance-accuracy trade-offs.'
)

doc.add_heading('4.1 Memory, CPU, and Energy', level=2)
add_figure(doc, FIGURES / "paper_bar_peak_mem.png",
           "Figure 2: Peak Memory (RSS MB) by model size and quantization.")
add_figure(doc, FIGURES / "paper_bar_cpu.png",
           "Figure 3: CPU Utilization (%) by model size and quantization.")
add_figure(doc, FIGURES / "paper_bar_energy.png",
           "Figure 4: Energy per Intent (J) by model size and quantization.")

doc.add_heading('4.1.1 Resource Analysis', level=3)
for f in [
    'Memory: Peak RSS scales from 575 MB (0.5B-Q4_K_M) to 11,429 MB (7B-Q8_0). Q4_K_M reduces memory '
    'by ~50% vs Q8_0 with negligible accuracy loss. Q2_K further reduces memory but at a 1.6pp accuracy '
    'cost. 7B-Q8_0 (11.4 GB) exceeds typical 8 GB edge budgets, making 7B-Q4_K_M (4.9 GB) the practical '
    'maximum for edge deployment.',
    'CPU Utilization: A key finding — CPU% scales with model size, from 68.8% (0.5B-Q2_K) to 94.1% '
    '(7B-Q8_0). This reveals that inference on CPU is memory-bandwidth-limited: smaller models cannot '
    'fully saturate the memory bus, leaving CPU cores idle. Quantization has minimal impact on CPU% '
    'within the same size class (e.g., 7B: 91.2–94.1%).',
    'Energy: Directly proportional to CPU% × time. 0.5B-Q2_K is most efficient (436 J/intent), '
    '7B-Q8_0 most expensive (2,613 J/intent). Q4_K_M provides 20–30% energy savings vs Q8_0. '
    'The 1.5B–3B range with Q4_K_M represents the sweet spot: 850–1,122 J with competitive accuracy.',
]:
    doc.add_paragraph(f, style='List Bullet')
doc.add_paragraph()

# ═══════════════ 5. PHASE 2: PERFORMANCE ═══════════════
doc.add_heading('5. Phase 2: Performance and Accuracy', level=1)
doc.add_heading('5.1 Throughput, Accuracy, and Speed', level=2)

add_figure(doc, FIGURES / "paper_bar_tps.png",
           "Figure 5: Throughput (tokens/s) by model size and quantization.")
add_figure(doc, FIGURES / "paper_bar_match.png",
           "Figure 6: Avg Match% by model size and quantization.")
add_figure(doc, FIGURES / "paper_bar_action.png",
           "Figure 7: Action Accuracy% by model size and quantization.")
add_figure(doc, FIGURES / "paper_bar_time.png",
           "Figure 8: Inference Time (s) by model size and quantization.")

doc.add_heading('5.1.1 Performance Analysis', level=3)
for f in [
    'Throughput: Inversely proportional to model size. 0.5B-Q2_K achieves 8.9 tok/s, 7B-Q8_0 only 1.4 tok/s. '
    'Larger models produce more output tokens per intent, partially offsetting the lower token rate.',
    'Size Scaling — Diminishing Returns: 0.5B→1.5B yields the largest accuracy jump (+5.8pp match). '
    '1.5B→3B shows marginal gain (+1.6pp). 3B→7B gains +3.7pp but at 67% more time and 80% more energy. '
    'The Pareto "knee" is at 3B — beyond this, accuracy gains diminish while costs grow linearly.',
    'Quantization Impact: Q8_0→Q4_K_M loses only 0.2pp match on average while halving memory. '
    'Q4_K_M→Q2_K loses 1.6pp with 20–30% energy savings. 0.5B is least sensitive (±1.1pp), '
    '1.5B most sensitive (3.7pp loss Q2_K→Q4_K_M). 7B-Q2_K (29.0% match) outperforms 3B-Q8_0 (26.8%).',
    'Q4_K_M is the recommended quantization for all deployments — optimal accuracy-efficiency trade-off.',
]:
    doc.add_paragraph(f, style='List Bullet')
doc.add_paragraph()

# ═══════════════ 6. PHASE 2: PARETO ═══════════════
doc.add_heading('6. Phase 2: Pareto Frontier Analysis', level=1)
doc.add_paragraph(
    'Figures 9–11 present Pareto frontier analyses using the visual encoding: shape = model size '
    '(■ 0.5B, ◆ 1.5B, ▲ 3B, ● 7B), vertical lines = quantization (no line = Q8_0, | = Q4_K_M, '
    '|| = Q2_K). Dashed gray lines indicate the Pareto frontier — configurations on or near the '
    'frontier represent optimal trade-offs.'
)

add_figure(doc, FIGURES / "paper_pareto_match_vs_time.png",
           "Figure 9: Pareto Frontier — Avg Match% vs Inference Time.")
add_figure(doc, FIGURES / "paper_pareto_action_vs_time.png",
           "Figure 10: Pareto Frontier — Action Accuracy% vs Inference Time.")
add_figure(doc, FIGURES / "paper_pareto_action_vs_energy.png",
           "Figure 11: Pareto Frontier — Action Accuracy% vs Energy per Intent.")

doc.add_heading('6.1 Pareto Analysis', level=3)
for f in [
    'Accuracy vs Speed (Fig. 9): 7B-Q8_0 dominates the upper-right with highest accuracy (30.5%) '
    'but slowest inference (20.4s). 3B-Q4_K_M lies near the Pareto knee — 27.3% match at 9.4s. '
    '0.5B-Q8_0 offers the fastest inference (4.8s) with 19.4% match, suitable for ultra-low-latency '
    'applications.',
    'Action Accuracy vs Speed (Fig. 10): 3B-Q4_K_M and 3B-Q8_0 achieve the highest action accuracy '
    '(44.5%), matching or exceeding 7B models (42.5–43.5%). This confirms that action classification '
    '(CREATE/MODIFY/DELETE) saturates at moderate model sizes, unlike fine-grained parameter matching.',
    'Action Accuracy vs Energy (Fig. 11): 0.5B-Q4_K_M is the most energy-efficient for action accuracy '
    '(35.0% action at 477J). 3B-Q4_K_M provides 44.5% action at 1,122J — 2.3× energy for 9.5pp gain. '
    '7B-Q8_0 achieves 43.5% action at 2,613J — 5.5× energy of 0.5B for only 8.5pp improvement, '
    'quantifying the steep energy cost of marginal accuracy gains.',
]:
    doc.add_paragraph(f, style='List Bullet')
doc.add_paragraph()

# ═══════════════ 7. DISCUSSION ═══════════════
doc.add_heading('7. Discussion', level=1)

doc.add_heading('7.1 Deployment Recommendations', level=2)
for r in [
    'Latency-critical edge (<500ms budget): 0.5B-Q8_0 (4.8s, 19.4% match). Fastest option.',
    'Balanced edge deployment: 1.5B-Q4_K_M (7.5s, 25.8% match, 850J, 1.7GB) or 3B-Q4_K_M (9.4s, 27.3% match, 1,122J, 3.3GB).',
    'Maximum accuracy (cloud/regional): 7B-Q4_K_M (15.0s, 29.8% match, 1,884J, 4.9GB). Q8_0 adds only 0.7pp at 40% more energy.',
    'Quantization guidance: Q4_K_M for all deployments. Q2_K acceptable for memory-constrained scenarios. Q8_0 not recommended for CPU-only — negligible gain, high cost.',
    'Energy estimator: CPU utilization is NOT constant — use our measured values (Table 1) instead of flat 95% assumption. Small models consume 15–25% less than TDP-based estimates.',
]:
    doc.add_paragraph(r, style='List Bullet')

# CPU table
doc.add_heading('7.2 CPU Utilization Reference', level=2)
doc.add_paragraph('Table 1 provides measured CPU utilization for energy estimation:')

cpu_table = doc.add_table(rows=5, cols=4, style='Light Grid Accent 1')
cpu_table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['Size', 'Q2_K CPU%', 'Q4_K_M CPU%', 'Q8_0 CPU%']):
    cpu_table.rows[0].cells[i].text = h
    for p in cpu_table.rows[0].cells[i].paragraphs:
        for run in p.runs: run.bold = True
for i, s in enumerate(['0.5B', '1.5B', '3B', '7B']):
    cpu_table.rows[i+1].cells[0].text = s
    for j, q in enumerate(['Q2_K', 'Q4_K_M', 'Q8_0']):
        key = f"{s}|{q}"
        cpu_table.rows[i+1].cells[j+1].text = f"{ci_data[key]['avg_cpu']:.1f}"

p = doc.add_paragraph()
p.add_run('Table 1: ').bold = True
p.add_run('Measured CPU utilization — system-wide psutil, n=200 intents per configuration.')
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

doc.add_heading('7.3 Limitations', level=2)
for l in [
    'CPU-only: Results specific to CPU inference. GPU acceleration would change trade-offs.',
    'Energy: TDP-based estimation (±15%). Direct RAPL was unavailable on VM infrastructure.',
    'Scope: 200 intents, Qwen-only Phase 2. Validation on larger datasets and architectures needed.',
]:
    doc.add_paragraph(l, style='List Bullet')
doc.add_paragraph()

# ═══════════════ 8. CONCLUSION ═══════════════
doc.add_heading('8. Conclusion', level=1)
doc.add_paragraph(
    'This paper presented a comprehensive empirical evaluation of SLMs for 6G intent-driven slice '
    'management. Key findings: (1) Qwen2.5-7B achieves best overall accuracy (29.8% match, 42.5% action). '
    '(2) CPU utilization scales with model size (69%→94%), revealing memory bandwidth bottleneck. '
    '(3) Diminishing returns beyond 3B — the 3B-Q4_K_M sweet spot offers 91% of 7B accuracy at 60% cost. '
    '(4) Q4_K_M is the optimal quantization with ≤1pp accuracy loss and ~50% memory reduction. '
    'These findings provide actionable deployment guidance for 6G edge infrastructure.'
)
doc.add_paragraph()

# ═══════════════ REFERENCES ═══════════════
doc.add_heading('References', level=1)
for ref in [
    '[1] I. Afolabi et al., "Network Slicing: A Survey," IEEE Commun. Surveys Tuts., 2018.',
    '[2] K. Abbas et al., "Intent-Based Networking for 6G," IEEE Commun. Mag., 2024.',
    '[3] ETSI GS ZSM 002, "ZSM Reference Architecture," 2019.',
    '[4] J. Wei et al., "Chain-of-Thought Prompting," NeurIPS, 2022.',
    '[5] A. Maatouk et al., "LLMs for Telecom," IEEE Commun. Mag., 2024.',
    '[6] M. Murshed et al., "ML at the Network Edge," ACM Comput. Surv., 2022.',
    '[10] Meta AI, "Llama 3.1," 2024.',
    '[11] Alibaba Cloud, "Qwen2.5 Technical Report," 2024.',
    '[12] Google DeepMind, "Gemma 2," 2024.',
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

# ── Save ──
docx_path = PROJECT / "paper_final_complete.docx"
doc.save(str(docx_path))
print(f"✅ DOCX: {docx_path} ({os.path.getsize(docx_path)/1024:.1f} KB)")

pdf_path = PROJECT / "paper_final_complete.pdf"
result = subprocess.run(
    ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', str(PROJECT), str(docx_path)],
    capture_output=True, text=True, timeout=120
)
if result.returncode == 0:
    print(f"✅ PDF: {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)")
else:
    print(f"❌ PDF failed: {result.stderr[:200]}")
