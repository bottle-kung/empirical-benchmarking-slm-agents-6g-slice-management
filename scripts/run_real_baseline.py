#!/usr/bin/env python3
"""Run real inference on Qwen2.5-1.5B for Domain 2 (Scaling) and Domain 3 (Conflict)."""
import json, re, time, requests

OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5:1.5b"
SYSTEM = "You are a 6G Core Network Slice Manager. Output ONLY valid JSON."

TESTS = [
    {
        "domain": "Scaling Request",
        "complexity": "Complex",
        "id": "SCAL-026",
        "prompt": "The stadium is filling up for the concert. Gradually increase the bandwidth of the public Wi-Fi slice by 20% every hour for the next 4 hours.",
        "hint": "Include action, target, strategy, step_percent, duration_hours.",
        "truth": {"action": "SCALE", "target": "bandwidth", "strategy": "gradual", "step_percent": 20, "duration_hours": 4},
        "num_predict": 50,
    },
    {
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous", 
        "id": "CONF-023",
        "prompt": "A natural disaster just occurred. Ensure emergency responder communications have absolute priority over regular mobile users, even if the system is overloaded.",
        "hint": "Include action, priority, preempt, slice_type.",
        "truth": {"action": "RESOLVE", "priority": "emergency_responders", "preempt": "regular_users", "slice_type": "URLLC"},
        "num_predict": 50,
    },
]

results = []

for t in TESTS:
    print(f"\n{'='*60}")
    print(f"📋 {t['domain']}: {t['id']} ({t['complexity']})")
    print(f"{'='*60}")
    print(f"Prompt: \"{t['prompt'][:80]}...\"")
    
    t0 = time.time()
    try:
        resp = requests.post(f"{OLLAMA}/api/generate", json={
            "model": MODEL,
            "prompt": t["prompt"],
            "system": f"{SYSTEM} {t['hint']}",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": t["num_predict"]}
        }, timeout=480)
        elapsed = time.time() - t0
        
        data = resp.json()
        if "error" in data:
            print(f"❌ ERROR: {data['error'][:100]}")
            continue
            
        response_text = data.get("response", "")
        dur = data.get("total_duration", 0) / 1e9
        tokens = data.get("eval_count", 0)
        speed = tokens / dur if dur > 0 else 0
        
        # Parse JSON
        parsed = None
        try:
            parsed = json.loads(response_text)
        except:
            matches = re.findall(r'\{[^{}]*\}', response_text)
            for m in matches:
                try:
                    parsed = json.loads(m)
                    break
                except: pass
        
        print(f"⏱️  Wall: {elapsed:.0f}s | Gen: {dur:.1f}s | Tokens: {tokens} | Speed: {speed:.3f} tok/s")
        print(f"📝 Raw: {response_text[:200]}")
        print(f"🔍 Parsed: {json.dumps(parsed, ensure_ascii=False) if parsed else 'PARSE FAILED'}")
        print(f"🎯 Truth:  {json.dumps(t['truth'])}")
        
        if parsed:
            checks = []
            for key in t["truth"]:
                if key == "action":
                    continue  # check separately
                gt_val = str(t["truth"][key]).lower()
                p_val = str(parsed.get(key, "")).lower()
                if gt_val in p_val or p_val == gt_val:
                    checks.append(f"✅ {key}")
                else:
                    checks.append(f"❌ {key} (expected={gt_val}, got={p_val}")
            
            gt_action = t["truth"]["action"].lower()
            p_action = str(parsed.get("action", "")).lower()
            action_ok = gt_action in p_action or p_action == gt_action
            checks.insert(0, f"{'✅' if action_ok else '❌'} action")
            
            print(f"🎯 Results: {' | '.join(checks)}")
        
        result = {
            "domain": t["domain"],
            "complexity": t["complexity"],
            "intent_id": t["id"],
            "model": MODEL,
            "raw_response": response_text,
            "parsed": parsed,
            "ground_truth": t["truth"],
            "duration_s": round(dur, 1),
            "wall_time_s": round(elapsed, 1),
            "tokens": tokens,
            "speed_tok_s": round(speed, 4),
        }
        results.append(result)
        
    except requests.exceptions.Timeout:
        elapsed = time.time() - t0
        print(f"⏱️  TIMEOUT after {elapsed:.0f}s")
        results.append({"domain": t["domain"], "intent_id": t["id"], "error": "TIMEOUT", "wall_time_s": elapsed})
    except Exception as e:
        print(f"❌ Exception: {e}")

# Save
outpath = "/jupyter_workspace/local/ai_agent/slm_6g_eval/results/qwen2.5_1.5b_real_baseline.json"
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n💾 Results saved to: {outpath}")
print(f"✅ {len([r for r in results if 'parsed' in r and r['parsed']])}/{len(results)} intents produced valid JSON")
