# Empirical Benchmarking of Small Language Model Agents for Intent-Driven 6G Network Slice Management

Empirical evaluation of Small Language Models (SLMs) for translating natural-language intents into JSON commands for 6G Core network slice lifecycle management.

## 📊 Key Results

| Model | Size | Format Acc. | Lenient Score | Speed | Sweet Spot? |
|-------|------|:-----------:|:------------:|:-----:|:-----------:|
| **Qwen2.5-1.5B** | 986 MB | 95.5% | 44.8% (2.24/5) | 12.4 tok/s | 🏆 **Yes** |
| Llama-3.2-3B | 2.0 GB | 99.5% | 48.6% (2.43/5) | 7.7 tok/s | |
| Qwen2.5-7B | 4.7 GB | 96.5% | 49.6% (2.48/5) | 3.6 tok/s | |
| Rule-Based (Regex) | <1 MB | 100% | ~10% (est.) | Instant | ❌ No flexibility |

> **Finding**: Qwen2.5-1.5B is the Pareto-optimal choice — **3.7× faster** and **4.7× smaller** than 7B with only **4.8% lower lenient accuracy**. All SLMs achieve >95% JSON format validity.

## 🗂️ Project Structure

```
slm_6g_eval/
├── dataset/
│   └── intents.jsonl          # 200 natural-language intents (3 domains)
├── scripts/
│   ├── evaluate.py             # Core evaluation engine
│   ├── run_slm_eval.py         # Ollama-based evaluation runner
│   ├── deepseek_baseline.py    # DeepSeek API baseline
│   ├── deepseek_enriched.py    # Enriched intent evaluation
│   ├── lenient_scoring.py      # LLM-as-a-Judge scoring
│   ├── visualize.py            # Result visualization
│   ├── analyze_results.py      # Statistical analysis
│   └── ...
├── config/
│   ├── schema.yaml             # Intent schema definition
│   └── baselines.yaml          # Baseline configurations
├── results/
│   ├── RESULTS_AND_DISCUSSION.md  # Full results & analysis
│   ├── benchmark_200_full.json    # Complete benchmark results
│   └── figures/                   # All charts & visualizations
└── paper/
    ├── paper.tex               # Academic paper (LaTeX)
    └── figures/                # Publication-ready figures
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install requests pyyaml
```

### 2. Run Evaluation with Ollama (Local)
```bash
# Pull a model
ollama pull qwen2.5:1.5b

# Run evaluation (200 intents, 5 iterations each)
python scripts/run_slm_eval.py --model qwen2.5:1.5b --limit 200 --iterations 5
```

### 3. Run with DeepSeek API
```bash
export DEEPSEEK_API_KEY="sk-..."
python scripts/deepseek_baseline.py --limit 200
```

## 📈 Domains Evaluated

| Domain | Intents | Difficulty | Description |
|--------|:-------:|:----------:|-------------|
| **Slicing Provisioning** | 70 | ⭐ Easy | Create eMBB/URLLC/mMTC slices |
| **Scaling Request** | 70 | ⭐⭐ Moderate | Scale UPF/AMF, bandwidth |
| **Conflict Resolution** | 60 | ⭐⭐⭐ Hard | Preempt, resolve, reallocate |

## 🖥️ Infrastructure

All experiments run on **Proxmox VM** with:
- 32-core Intel Xeon E5-2690 (CPU-only)
- 32 GB RAM
- Ollama with Q4_K_M quantization
- Ubuntu 24.04

## 📝 Citation

```bibtex
@misc{slm-6g-benchmark-2026,
  title   = {Empirical Benchmarking of Small Language Model Agents for Intent-Driven 6G Network Slice Management},
  author  = {T S},
  year    = {2026},
  howpublished = {GitHub Repository}
}
```

## 📄 License

This project is for academic research purposes. See paper/ directory for full publication.
