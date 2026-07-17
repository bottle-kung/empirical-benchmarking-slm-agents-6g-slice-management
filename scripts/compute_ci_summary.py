#!/usr/bin/env python3
"""
Compute CI summary (mean, SD, 95% CI) for all metrics from resource_v3 results.
Output: results/ci_summary_v3.json — used by generate_paper_with_figures.py
"""
import json, math, os, glob
from collections import defaultdict

def t_value_95(n):
    df = n - 1
    if df < 1: return float('inf')
    if df > 100: return 1.984
    if df > 50: return 2.009
    if df > 30: return 2.042
    return 2.0 + 2.0 / math.sqrt(df)

def ci95(values):
    """Returns (mean, sd, ci_lower, ci_upper)."""
    n = len(values)
    if n < 2: return (values[0] if n else 0, 0, 0, 0)
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    t = t_value_95(n)
    margin = t * sd / math.sqrt(n)
    return (round(mean, 2), round(sd, 2), round(mean - margin, 2), round(mean + margin, 2))

res_dir = "/jupyter_workspace/rbru/slm_6g_eval_v2/results/resource_v3"
out_path = "/jupyter_workspace/rbru/slm_6g_eval_v2/results/ci_summary_v3.json"

all_data = {}
for f in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
    name = os.path.basename(f).replace('.json', '')
    with open(f) as fh:
        data = json.load(fh)
    results = data['results']
    n = len(results)

    match_pcts = [r['match_pct'] for r in results]
    durations = [r['duration_s'] for r in results]
    energies = [r['energy_j'] for r in results]
    cpu_pcts = [r.get('cpu_pct', 0) for r in results if r.get('cpu_pct', 0) > 0]
    mem_avgs = [r['avg_memory_mb'] for r in results if r.get('avg_memory_mb', 0) > 0]
    mem_peaks = [r['peak_memory_mb'] for r in results if r.get('peak_memory_mb', 0) > 0]
    tps_vals = [r['tokens_per_sec'] for r in results]
    total_tok = sum(r.get('tokens', 0) for r in results)

    action = sum(1 for r in results if r.get('action_correct')) / n * 100
    fmt = sum(1 for r in results if r.get('format_valid')) / n * 100

    parts = name.split('_')
    size = parts[0]
    q = '_'.join(parts[1:]) if len(parts) > 1 else 'Q4_K_M'

    all_data[f"{size}|{q}"] = {
        'n': n,
        'avg_match': round(sum(match_pcts)/n, 1),
        'action': round(action, 1),
        'format': round(fmt, 1),
        'avg_time': round(sum(durations)/n, 2),
        'avg_energy': round(sum(energies)/n, 1),
        'avg_cpu': round(sum(cpu_pcts)/len(cpu_pcts), 1) if cpu_pcts else 0,
        'avg_mem': round(sum(mem_avgs)/len(mem_avgs), 0) if mem_avgs else 0,
        'peak_mem': round(max(mem_peaks), 0) if mem_peaks else 0,
        'tps': round(total_tok/sum(durations), 1) if sum(durations) > 0 else 0,
        'ci_match': ci95(match_pcts),
        'ci_time': ci95(durations),
        'ci_energy': ci95(energies),
        'ci_cpu': ci95(cpu_pcts) if cpu_pcts else (0, 0, 0, 0),
        'ci_mem': ci95(mem_avgs) if mem_avgs else (0, 0, 0, 0),
        'ci_peak_mem': ci95(mem_peaks) if mem_peaks else (0, 0, 0, 0),
        'ci_tps': ci95(tps_vals),
    }
    print(f"{name}: match={all_data[f'{size}|{q}']['avg_match']}%, cpu={all_data[f'{size}|{q}']['avg_cpu']}%")

with open(out_path, 'w') as f:
    json.dump(all_data, f, indent=2)
print(f"\n✅ Saved: {out_path}")
