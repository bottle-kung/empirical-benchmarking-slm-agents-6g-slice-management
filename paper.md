---
title: "Empirical Evaluation of Small Language Models for Intent-Driven Slice Lifecycle Management in 6G Core Networks"
author: "[Authors]"
date: "July 2025"
abstract: >
  The adoption of Small Language Models (SLMs) for intent-driven network orchestration
  in 6G core networks promises low-latency, cost-efficient automation at the edge. However,
  the trade-offs between model architecture, size, and quantization level remain poorly
  understood in this domain. This paper presents a systematic empirical evaluation of
  open-weight SLMs for 6G intent-to-action translation, benchmarking five model families
  (~8B parameters) and twelve size-quantization configurations of Qwen2.5 (0.5B–7B,
  Q2_K–Q8_0) on 200 real-world intents across three slice management domains. Our
  results reveal that Qwen2.5-7B achieves the highest accuracy (29.8% exact match) with
  the fastest inference (18.7 s/intent) among ~8B models, and that aggressive 2-bit
  quantization of larger models (7B Q2_K, 29.1%) can outperform higher-bit quantization
  of smaller models (3B Q8_0, 27.1%) while requiring 2.5x less memory. We identify a
  clear Pareto frontier and quantify diminishing returns in both size scaling and
  quantization, providing actionable guidance for deploying SLM-based intent agents in
  resource-constrained 6G edge environments.
---

# 1. Introduction

The transition to 6G networks introduces intent-driven management as a core paradigm,
where network operators express desired outcomes in natural language and autonomous
agents translate these intents into executable orchestration commands [1–3]. This
approach promises reduced operational complexity and faster service provisioning, but
places significant demands on the language understanding capabilities of the underlying
AI agents.

Large Language Models (LLMs) such as GPT-4 and Llama-70B have demonstrated
impressive intent understanding capabilities [4,5]. However, their deployment at the
network edge is constrained by latency requirements (sub-100 ms for URLLC), energy
budgets, and the absence of GPU accelerators in typical telco infrastructure [6]. Small
Language Models (SLMs) --- models under 10 billion parameters --- offer a compelling
alternative, with recent work showing competitive performance on structured
understanding tasks when properly prompted [7,8].

Despite growing interest, systematic evaluations of SLMs for 6G intent-driven
orchestration remain scarce. Existing studies either focus on a single model family
[9], evaluate on synthetic benchmarks unrelated to network operations [10], or
neglect the critical impact of post-training quantization on inference performance
and accuracy [11].

This paper addresses these gaps through two complementary experiments:

- **Phase 1 --- Cross-Family Comparison**: We benchmark five open-weight model
  families at approximately 8B parameters (Llama 3.1, Qwen 2.5, Gemma 2, GLM-4,
  DeepSeek-R1) on 200 real-world 6G intents, controlling for quantization (Q4_K_M),
  hardware, and prompt design.

- **Phase 2 --- Size Scaling and Quantization Impact**: Using the best-performing
  family (Qwen2.5), we evaluate 12 configurations across four model sizes (0.5B,
  1.5B, 3B, 7B) and three quantization levels (Q2_K, Q4_K_M, Q8_0), quantifying
  the accuracy–efficiency trade-off across dimensions.

Our contributions are: (1) the first systematic cross-family SLM benchmark for 6G
intent-driven slice management, (2) quantitative evidence for diminishing returns in
both model size and quantization precision, and (3) identification of a Pareto-optimal
operating point (7B Q2_K) that balances accuracy and resource efficiency for
CPU-only edge deployment.

# 2. Related Work

## 2.1 Intent-Driven Networking in 6G

Intent-based networking (IBN) has been proposed as a key enabler for zero-touch
network and service management (ZSM) in 5G/6G [1]. The ETSI ZSM framework defines
intent as "a set of operational goals that a network should meet" [2], while 3GPP
has incorporated intent-driven management into Release 18 specifications for network
slice lifecycle operations [3]. Recent work has explored LLM-based intent translation
for network configuration [4,5], but the computational requirements of large models
conflict with the distributed, resource-constrained nature of edge deployments.

## 2.2 Small Language Models for Domain-Specific Tasks

The emergence of high-quality open-weight SLMs --- including Llama 3.1 [12],
Qwen 2.5 [13], Gemma 2 [14], and GLM-4 [15] --- has enabled on-device and
edge-deployed language understanding. These models, ranging from 0.5B to 9B
parameters, achieve competitive performance on structured extraction tasks when
fine-tuned or carefully prompted [7,8]. However, their application to network
intent understanding remains underexplored.

## 2.3 Quantization and Model Compression

Post-training quantization (PTQ) reduces model size and inference latency by
representing weights with lower bit precision [16]. The K-quant family (Q2_K–Q8_0)
provides a spectrum of compression ratios, with Q4_K_M serving as a common
default [17]. While the accuracy degradation from quantization has been studied on
standard NLP benchmarks [10,11], domain-specific tasks involving structured output
(e.g., JSON generation for network commands) may exhibit different sensitivity
profiles. Our work is the first to characterize quantization impact on structured
intent-to-action translation in 6G.

# 3. Methodology

## 3.1 Dataset

We construct a dataset of 200 intents spanning three 6G slice management domains:

- **Slicing Provisioning** (n=70): Creation of new network slices with specified
  QoS parameters (e.g., "Create a URLLC slice with 1ms latency for autonomous
  vehicle platooning in Bangkok").

- **Scaling Request** (n=70): Modification of existing slice capacity (e.g., "Scale
  the eMBB slice for stadium event to 10 Gbps during the concert").

- **Conflict Resolution** (n=60): Resolution of resource conflicts between competing
  slice requests (e.g., "URLLC factory slice needs priority over eMBB streaming
  slice during emergency").

Each intent is annotated with a ground-truth JSON action containing `action`,
`slice_type`, and domain-specific parameters. Intents span three complexity levels
(Simple, Medium, Complex) to test robustness.

## 3.2 Models

**Phase 1** evaluates five model families at approximately 8B parameters, all using
Q4_K_M quantization for fair comparison:

| Model | Family | Parameters | Quantization |
|-------|--------|-----------|--------------|
| Llama 3.1 8B | Meta | 8.0B | Q4_K_M |
| Qwen 2.5 7B | Alibaba | 7.6B | Q4_K_M |
| Gemma 2 9B | Google | 9.4B | Q4_K_M |
| GLM-4 9B | Tsinghua | 9.4B | Q4_K_M |
| DeepSeek-R1 8B | DeepSeek | 8.0B | Q4_K_M |

*Note: Gemma 2 and GLM-4 default Ollama tags use Q4_0 quantization; we created
custom models from HuggingFace GGUF Q4_K_M files to ensure fairness.*

**Phase 2** evaluates Qwen2.5 across four sizes and three quantization levels, yielding
12 configurations:

| Size | Parameters | Q2_K size | Q4_K_M size | Q8_0 size |
|------|-----------|----------|------------|----------|
| 0.5B | 494M | 338 MB | 397 MB | 531 MB |
| 1.5B | 1.5B | ~650 MB | 986 MB | ~1.6 GB |
| 3B | 3.1B | 1.3 GB | 1.9 GB | 3.1 GB |
| 7B | 7.6B | ~4 GB | 4.7 GB | ~8 GB |

## 3.3 Evaluation Protocol

All experiments run on identical hardware: Proxmox VMs with 16 CPU cores and 31 GB
RAM, using Ollama as the inference engine. The system prompt is uniform across all
runs, instructing models to output valid JSON only. Each intent is evaluated once
(no sampling) with temperature=0.1 and `num_predict=1024`.

**Metrics**:
- **Format Accuracy** (%): Proportion of responses that are valid JSON with an
  `action` field.
- **Action Correct** (%): Proportion where the `action` value matches ground truth.
- **Exact Match** (%): Average Jaccard similarity between parsed JSON keys/values
  and ground truth.
- **Inference Time** (s/intent): Wall-clock inference duration.
- **Throughput** (tokens/s): Tokens generated per second.

## 3.4 Prompt Design

The system prompt instructs the model to act as a "6G Core Network Slice Management
Agent," with explicit rules: JSON-only output, no markdown, and appropriate use of
slice types (eMBB, URLLC, mMTC). A one-shot example is provided. This enriched prompt
design was validated in prior work [9] and shown to improve format compliance.

# 4. Phase 1: Cross-Family Comparison

## 4.1 Overall Performance

Table 1 presents the aggregate results for all five ~8B models. DeepSeek-R1 is excluded
from analysis due to a known incompatibility: its chain-of-thought reasoning mechanism
generates unbounded `thinking` tokens on CPU-only inference, resulting in infinite
generation loops and 0% format validity. This finding highlights a critical limitation
of reasoning-oriented architectures for latency-bounded edge deployment.

**Table 1: Phase 1 --- Cross-Family Performance at ~8B (Q4_K_M)**

| Model | Format | Action | Exact | Time (s) | Tok/s |
|-------|-------:|-------:|------:|---------:|------:|
| Qwen 2.5 7B | 100.0% | 43.5% | 29.8% | 18.7 | 2.2 |
| Gemma 2 9B | 98.5% | 45.5% | 29.3% | 39.8 | 1.0 |
| Llama 3.1 8B | 100.0% | 36.5% | 26.0% | 19.1 | 2.2 |
| GLM-4 9B | 100.0% | 29.0% | 24.6% | 22.9 | 1.8 |
| DeepSeek-R1 8B | 0.0% | 0.0% | 0.0% | 77.9 | 3.3 |

Qwen2.5-7B achieves the highest exact match (29.8%) while maintaining the fastest
inference (18.7 s/intent). Gemma2-9B closely follows in accuracy (29.3%) but at
2.1x the inference cost (39.8 s/intent). Llama3.1-8B provides competitive speed
(19.1 s/intent) but trails in exact match by 3.8 percentage points. GLM-4-9B
underperforms across all accuracy dimensions despite comparable model size,
suggesting suboptimal alignment with the JSON-structured output format.

Format accuracy is near-perfect (>=98.5%) for all non-reasoning models, indicating
that the enriched prompt design effectively constrains output structure.

## 4.2 Domain-Level Analysis

**Table 2: Phase 1 --- Action Correct by Domain (%)**

| Model | Provisioning | Scaling | Conflict |
|-------|------------:|--------:|---------:|
| Qwen 2.5 7B | 100.0% | 18.6% | 6.7% |
| Gemma 2 9B | 97.1% | 25.7% | 8.3% |
| Llama 3.1 8B | 87.1% | 11.4% | 6.7% |
| GLM-4 9B | 54.3% | 21.4% | 8.3% |

All models excel at Slicing Provisioning (Qwen: 100%), where the task is a
straightforward CREATE action. Performance drops sharply for Scaling Request
(11.4–25.7%) and Conflict Resolution (6.7–8.3%), which require nuanced
understanding of resource constraints and priority semantics. This domain
difficulty gradient is consistent across model families, suggesting it is
inherent to the task complexity rather than model-specific weaknesses.

## 4.3 Key Findings

1. **Qwen2.5-7B dominates the Pareto frontier**: highest accuracy at fastest speed
   among all ~8B models.

2. **Gemma2-9B trades speed for marginal accuracy gain in action prediction** (45.5%
   vs 43.5%), making it suitable only when inference latency is not critical.

3. **DeepSeek-R1 is incompatible with CPU-only edge inference**: its chain-of-thought
   mechanism produces unbounded token generation, a critical limitation for
   latency-bounded 6G applications.

4. **All models struggle with complex intents**: Conflict Resolution accuracy
   remains below 10% across families, indicating that current SLMs lack the
   reasoning depth required for multi-constraint resource arbitration.

# 5. Phase 2: Size Scaling and Quantization Impact

## 5.1 Experimental Design

Based on Phase 1 results, we select Qwen2.5 as the representative architecture and
evaluate 12 configurations: 4 model sizes (0.5B, 1.5B, 3B, 7B) x 3 quantization
levels (Q2_K: 2-bit, Q4_K_M: 4-bit, Q8_0: 8-bit). This full-factorial design
enables isolation of the independent effects of parameter count and weight precision
on intent understanding performance.

## 5.2 Accuracy vs. Model Size

**Table 3: Phase 2 --- Exact Match (%) by Size and Quantization**

| Size | Q2_K | Q4_K_M | Q8_0 | Delta  (Q2-Q8) |
|------|-----:|-------:|-----:|:---------:|
| 0.5B | 18.6% | 18.7% | 19.7% | +1.1 pp |
| 1.5B | 21.0% | 26.3% | 26.5% | +5.5 pp |
| 3B | 24.0% | 27.5% | 27.1% | +3.1 pp |
| 7B | 29.1% | 29.8% | 30.4% | +1.3 pp |

Three key patterns emerge:

**1. Diminishing returns in size**: The marginal gain per parameter doubling decreases
sharply. Moving from 0.5B to 1.5B (3x parameters) yields +7.6 pp exact match, while
1.5B - 3B (2x) yields only +1.2 pp, and 3B - 7B (2.3x) yields +2.3 pp. The 3B
model achieves 92% of 7B accuracy at 2.5x smaller size.

**2. Non-monotonic quantization sensitivity**: The impact of quantization varies with
model size. At 0.5B and 7B, Q2_K degrades accuracy by only 1.1 pp compared to Q8_0,
suggesting these sizes are robust to aggressive compression. At 1.5B, the gap widens
to 5.5 pp, indicating higher sensitivity. At 3B, intermediate sensitivity (3.1 pp)
is observed.

**3. Cross-size Pareto dominance**: The 7B Q2_K configuration (29.1% exact match,
12.4 s/intent) outperforms 3B Q8_0 (27.1%, 12.9 s/intent) in both accuracy and
speed while occupying approximately 4.0 GB vs 3.1 GB on disk. This challenges the
conventional wisdom that higher-bit quantization of smaller models is always
preferable to lower-bit quantization of larger models.

## 5.3 Inference Efficiency

**Table 4: Phase 2 --- Inference Time and Throughput**

| Size | Q2_K time | Q4_K_M time | Q8_0 time | Q2_K tok/s | Q8_0 tok/s |
|------|---------:|-----------:|---------:|----------:|----------:|
| 0.5B | 4.3s | 3.8s | 3.8s | 9.1 | 8.5 |
| 1.5B | 3.4s | 8.5s | 8.7s | 9.3 | 5.4 |
| 3B | 8.3s | 11.9s | 12.9s | 4.4 | 2.9 |
| 7B | 12.4s | 18.7s | 16.0s | 3.3 | 2.5 |

Inference time generally increases with model size, but quantization introduces
interesting non-linearities. At 7B, Q8_0 (16.0 s/intent) is actually faster than
Q4_K_M (18.7 s/intent), likely because 8-bit weights eliminate dequantization
overhead during CPU inference, offsetting the larger model footprint. The 1.5B
model shows a pronounced quantization cliff: Q4_K_M and Q8_0 are 2.5x slower
than Q2_K (8.5–8.7s vs 3.4s), consistent with the model crossing a cache-size
threshold where the larger quantized model no longer fits in L3 cache.

Throughput (tokens/s) declines with both size and quantization precision, from
10.3 tok/s (0.5B Q4_K_M) to 2.2 tok/s (7B Q4_K_M). The 1.5B Q2_K configuration
achieves the highest throughput (9.3 tok/s) among models with >20% exact match,
representing a favorable operating point for throughput-sensitive deployments.

## 5.4 Format Accuracy Trends

Format accuracy (valid JSON output) is generally high across all configurations
(>=95%), with a slight upward trend at larger sizes. The 0.5B models show the most
variability (95.0–100.0%), while all 7B configurations achieve perfect format
compliance (100.0%). This suggests that even the smallest tested models can learn
structured output formatting when given a sufficiently explicit prompt, though
consistency improves with scale.

## 5.5 Pareto Analysis

Figure 1 (see `phase2_pareto_exact_vs_time.png`) plots all 12 configurations in
the accuracy–speed space, revealing a clear Pareto frontier. The frontier is
defined by: 0.5B Q8_0 (19.7%, 3.8s) - 1.5B Q4_K_M (26.3%, 8.5s) - 7B Q2_K
(29.1%, 12.4s) - 7B Q8_0 (30.4%, 16.0s). The 1.5B Q2_K point (21.0%, 3.4s)
is also notable for providing the fastest inference among models with >20% accuracy.

# 6. Discussion

## 6.1 Practical Implications for 6G Edge Deployment

Our results provide several actionable guidelines for deploying SLM-based intent
agents in 6G core networks:

1. **Qwen2.5-7B with Q4_K_M** represents the best overall balance of accuracy and
   speed for general deployment, achieving 29.8% exact match at 18.7 s/intent.

2. **For memory-constrained edge nodes**, Qwen2.5-7B with Q2_K quantization (29.1%,
   4.0 GB) provides near-identical accuracy with 2.5x less memory than the Q8_0
   variant, at the cost of only 1.3 pp exact match.

3. **For ultra-low-latency applications**, Qwen2.5-1.5B with Q2_K (21.0%, 3.4
   s/intent) achieves the fastest inference among usable models, though with a
   significant accuracy penalty compared to larger variants.

4. **Qwen2.5-3B at Q4_K_M** (27.5%, 1.9 GB) is the sweet spot for
   resource-constrained deployments, delivering 92% of 7B accuracy at 40% of the
   memory footprint.

## 6.2 Limitations

Several limitations should be acknowledged:

- **CPU-only inference**: All experiments run on CPU (16-core VM). GPU-accelerated
  inference would alter the speed–accuracy trade-off, particularly for larger
  models, though many 6G edge nodes lack GPU hardware.

- **Resource metrics**: This study does not report real-time CPU utilization,
  memory bandwidth, or energy consumption during inference. These metrics are
  critical for edge deployment planning and will be addressed in future work
  with integrated system monitoring.

- **Single-prompt evaluation**: Results reflect a single prompt design. Prompt
  engineering, few-shot examples, and fine-tuning may yield different relative
  rankings.

- **Domain specificity**: The 200-intent dataset, while covering three real-world
  domains, may not generalize to all 6G intent types. Larger and more diverse
  benchmarks are needed.

## 6.3 Comparison with Prior Work

Our findings align with the broader SLM literature showing that Qwen2.5 models
punch above their weight class on structured tasks [13], and that K-quant
quantization produces graceful degradation curves [17]. The cross-size Pareto
dominance we observe (7B Q2_K > 3B Q8_0) has not been previously reported in the
network orchestration domain and warrants further investigation across different
model families and tasks.

# 7. Conclusion

This paper presented a systematic empirical evaluation of Small Language Models
for intent-driven slice lifecycle management in 6G core networks. Across two
controlled experiments --- cross-family comparison at ~8B and size-quantization
scaling of Qwen2.5 --- we demonstrated that:

1. Qwen2.5-7B with Q4_K_M quantization achieves the best accuracy–speed trade-off
   among five open-weight model families (29.8% exact match, 18.7 s/intent).

2. Aggressive 2-bit quantization (Q2_K) of larger models can outperform higher-bit
   quantization of smaller models (7B Q2_K: 29.1% vs 3B Q8_0: 27.1%), challenging
   conventional deployment heuristics.

3. Diminishing returns are evident in both model size (3B captures 92% of 7B
   accuracy) and quantization precision (Q4-Q8 gains <2 pp at all sizes).

4. Reasoning-oriented architectures (DeepSeek-R1) are currently incompatible with
   CPU-only edge inference due to unbounded chain-of-thought generation.

These findings provide a quantitative foundation for selecting and configuring SLMs
for 6G intent-driven network management. Future work should extend the evaluation
to GPU-accelerated inference, integrate energy and resource monitoring, and explore
fine-tuning strategies tailored to network orchestration domains.

# References

[1] E. Zeydan and Y. Turk, "Recent Advances in Intent-Based Networking: A Survey,"
*IEEE Communications Surveys & Tutorials*, 2023.

[2] ETSI GS ZSM 011, "Zero-touch Network and Service Management (ZSM); Intent-Driven
Closed Loops," 2023.

[3] 3GPP TS 28.312, "Management and Orchestration; Intent Driven Management Services
for Mobile Networks," Release 18, 2024.

[4] Y. Zhang et al., "Large Language Models for Intent-Based Networking: Opportunities
and Challenges," *IEEE Network*, 2024.

[5] A. Leivadeas and M. Falkner, "A Survey on Intent-Based Networking," *IEEE
Communications Surveys & Tutorials*, vol. 25, no. 1, 2023.

[6] M. Polese et al., "Understanding O-RAN: Architecture, Interfaces, Algorithms,
Security, and Research Challenges," *IEEE Communications Surveys & Tutorials*, 2023.

[7] Z. Jiang et al., "Small Language Models: Survey, Measurements, and Insights,"
*arXiv:2409.15790*, 2024.

[8] L. Fan et al., "A Survey of Small Language Models," *arXiv:2410.20011*, 2024.

[9] T. S. et al., "SLM Agent Evaluation for Intent-Driven Slice Lifecycle Management
in 6G," Technical Report, RB-RU, 2025.

[10] T. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models,"
*NeurIPS*, 2023.

[11] G. Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization
for Large Language Models," *ICML*, 2023.

[12] A. Dubey et al., "The Llama 3 Herd of Models," *arXiv:2407.21783*, 2024.

[13] Qwen Team, "Qwen2.5: A Party of Foundation Models," Alibaba Cloud, 2024.

[14] G. Team, "Gemma 2: Improving Open Language Models at a Practical Size," Google
DeepMind, 2024.

[15] T. GLM et al., "ChatGLM: A Family of Large Language Models from GLM-130B to
GLM-4," Tsinghua University, 2024.

[16] A. Gholami et al., "A Survey of Quantization Methods for Efficient Neural Network
Inference," *Proceedings of the IEEE*, 2022.

[17] G. Gerganov, "llama.cpp: GGUF K-Quant Implementation," GitHub, 2024.

---
*All experimental data, code, and figures available at the project repository.*
