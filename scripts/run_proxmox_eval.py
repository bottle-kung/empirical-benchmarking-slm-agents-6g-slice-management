#!/usr/bin/env python3
"""Run SLM evaluation on Proxmox host via SSH."""
import json, time, subprocess, sys

SSH = ["sshpass", "-p", "cpeadmin", "ssh", "-o", "StrictHostKeyChecking=no", "root@172.16.1.206"]

MODELS = ["qwen2.5:1.5b", "llama3.2:3b", "qwen2.5:7b"]
INTENTS = [
    {"id": "PROV-001", "domain": "Provisioning", "prompt": "Create a new eMBB slice for a 4K video streaming event in Bangkok.", "truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 1}},
    {"id": "SCAL-026", "domain": "Scaling", "prompt": "The stadium is filling up for the concert. Gradually increase the bandwidth of the public Wi-Fi slice by 20% every hour for the next 4 hours.", "truth": {"action": "SCALE", "target": "bandwidth", "strategy": "gradual", "step_percent": 20, "duration_hours": 4}},
    {"id": "CONF-023", "domain": "Conflict", "prompt": "A natural disaster just occurred. Ensure emergency responder communications have absolute priority over regular mobile users, even if the system is overloaded.", "truth": {"action": "RESOLVE", "priority": "emergency_responders", "preempt": "regular_users", "slice_type": "URLLC"}},
]

SYSTEM = "You are a 6G Core Network Slice Manager. Output ONLY valid JSON. No markdown, no explanation."

results = []

for model in MODELS:
    # Pull model first
    print(f"\n{'='*60}")
    print(f"🔄 Pulling {model}...")
    subprocess.run(SSH + [f"ollama pull {model} 2>&1 | tail -2"], timeout=300)
    
    for intent in INTENTS:
        print(f"  📋 {model} → {intent['id']} ({intent['domain']})")
        
        payload = json.dumps({
            "model": model,
            "prompt": intent["prompt"],
            "system": SYSTEM,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 60}
        })
        
        t0 = time.time()
        try:
            resp = subprocess.run(
                SSH + ["curl", "-s", "--max-time", "120", "http://localhost:11434/api/generate", "-d", payload],
                capture_output=True, text=True, timeout=130
            )
            elapsed = time.time() - t0
            
            data = json.loads(resp.stdout)
            response_text = data.get("response", "")
            dur = data.get("total_duration", 0) / 1e9
            tokens = data.get("eval_count", 0)
            
            # Parse JSON
            import re
            parsed = None
            try:
                parsed = json.loads(response_text)
            except:
                matches = re.findall(r'\{[^{}]*\}', response_text)
                for m in matches:
                    try: parsed = json.loads(m); break
                    except: pass
            
            # Score
            truth = intent["truth"]
            if parsed and "action" in parsed:
                gt_keys = set(truth.keys())
                p_keys = set(k for k in parsed.keys() if k != "action")  # exclude action for key count
                all_keys = set(truth.keys())
                matched = sum(1 for k in all_keys if k in parsed and str(parsed[k]).lower() == str(truth[k]).lower())
                action_ok = str(parsed.get("action","")).lower() == str(truth["action"]).lower()
                match_pct = (matched + (1 if action_ok else 0)) / (len(all_keys)) * 100
            else:
                match_pct = 0
                action_ok = False
            
            status = "✅" if parsed and action_ok else ("⚠️" if parsed else "❌")
            print(f"    {status} Match: {match_pct:.0f}% | {dur:.1f}s | {tokens} tok | {tokens/dur:.1f} tok/s" if dur > 0 else f"    {status}")
            
            results.append({
                "model": model,
                "intent_id": intent["id"],
                "domain": intent["domain"],
                "wall_time_s": round(elapsed, 1),
                "duration_s": round(dur, 1),
                "tokens": tokens,
                "speed_tok_s": round(tokens/dur, 2) if dur > 0 else 0,
                "response": response_text,
                "parsed": parsed,
                "ground_truth": truth,
                "match_pct": round(match_pct, 1),
                "action_correct": action_ok,
                "format_valid": parsed is not None,
            })
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({"model": model, "intent_id": intent["id"], "error": str(e)})

# Save
outpath = "/jupyter_workspace/local/ai_agent/slm_6g_eval/results/proxmox_3models_3domains.json"
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)

# Summary
print(f"\n{'='*60}")
print(f"📊 RESULTS SUMMARY — Proxmox Host (32 cores)")
print(f"{'='*60}")
print(f"{'Model':<18} {'Intent':<10} {'Match':>7} {'Time':>7} {'Speed':>8} {'Status'}")
print(f"{'-'*18} {'-'*10} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
for r in results:
    if "error" in r:
        print(f"{r['model']:<18} {r['intent_id']:<10} {'—':>7} {'—':>7} {'—':>8} ❌ ERROR")
    else:
        status = "✅" if r.get("action_correct") else ("⚠️" if r.get("format_valid") else "❌")
        print(f"{r['model']:<18} {r['intent_id']:<10} {r.get('match_pct',0):>6.0f}% {r.get('wall_time_s',0):>6.0f}s {r.get('speed_tok_s',0):>7.1f}/s {status}")

print(f"\n💾 Saved to: {outpath}")
