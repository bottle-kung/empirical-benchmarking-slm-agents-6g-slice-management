#!/usr/bin/env python3
"""Enhanced evaluation with explicit schema in system prompt."""
import json, re, time, sys

# Better system prompt with explicit schema
SYSTEM = """You are a 6G Core Network Slice Management Agent.
Your ONLY task: translate natural language intents into a specific JSON format.

REQUIRED JSON SCHEMA — you MUST use these exact field names:
{
  "action": "<CREATE|SCALE|RESOLVE|TERMINATE>",
  "slice_type": "<eMBB|URLLC|mMTC>",
  "bandwidth_gbps": <number>,
  "latency_ms": <number>,
  "device_density": <number>,
  "priority": "<low|normal|high|critical>",
  "target": "<upf|smf|amf|bandwidth|all>",
  "replicas": <number>,
  "strategy": "<gradual|immediate|burst>",
  "step_percent": <number>,
  "duration_hours": <number>,
  "preempt": "<slice_type or user_group>"
}

RULES:
1. Output ONLY the JSON object — no markdown, no ```json fences, no explanation
2. ALWAYS include "action" as the first field
3. Use the EXACT field names shown above — do not invent new names
4. Only include fields that are relevant to the intent

Example: {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 5}"""

OLLAMA = "http://localhost:11434"

TESTS = [
    ("qwen2.5:1.5b", "PROV-001", "Provisioning", 
     "Create a new eMBB slice for a 4K video streaming event in Bangkok.",
     {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 1}),
    ("qwen2.5:1.5b", "SCAL-026", "Scaling",
     "The stadium is filling up for the concert. Gradually increase the bandwidth of the public Wi-Fi slice by 20% every hour for the next 4 hours.",
     {"action": "SCALE", "target": "bandwidth", "strategy": "gradual", "step_percent": 20, "duration_hours": 4}),
    ("qwen2.5:1.5b", "CONF-023", "Conflict",
     "A natural disaster just occurred. Ensure emergency responder communications have absolute priority over regular mobile users, even if the system is overloaded.",
     {"action": "RESOLVE", "priority": "emergency_responders", "preempt": "regular_users", "slice_type": "URLLC"}),
    ("llama3.2:3b", "PROV-001", "Provisioning",
     "Create a new eMBB slice for a 4K video streaming event in Bangkok.",
     {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 1}),
    ("qwen2.5:7b", "PROV-001", "Provisioning",
     "Create a new eMBB slice for a 4K video streaming event in Bangkok.",
     {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 1}),
]

import requests

for model, iid, domain, prompt, truth in TESTS:
    print(f"\n{'='*60}")
    print(f"📋 {model} → {iid} ({domain})")
    
    t0 = time.time()
    try:
        resp = requests.post(f"{OLLAMA}/api/generate", json={
            "model": model, "prompt": prompt, "system": SYSTEM,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 80}
        }, timeout=120)
        
        data = resp.json()
        response_text = data.get("response", "")
        dur = data.get("total_duration", 0) / 1e9
        tokens = data.get("eval_count", 0)
        
        # Parse JSON - strip markdown fences
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        
        parsed = None
        try:
            parsed = json.loads(cleaned)
        except:
            matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned)
            for m in matches:
                try: parsed = json.loads(m); break
                except: pass
        
        # Soft scoring: check key intent signals
        if parsed:
            action_ok = "action" in parsed
            has_slice = "slice_type" in parsed
            has_params = len(parsed) >= 2
            
            # Fuzzy match: check if model got the right IDEA even if values differ
            truth_action = truth["action"].lower()
            model_action = str(parsed.get("action", "")).lower()
            action_match = truth_action in model_action or model_action == truth_action
            
            if "slice_type" in truth and "slice_type" in parsed:
                type_ok = truth["slice_type"].lower() == str(parsed["slice_type"]).lower()
            else:
                type_ok = None
            
            exact_keys = sum(1 for k in truth if k in parsed and str(parsed[k]).lower() == str(truth[k]).lower())
            total_keys = len(truth)
            
            print(f"  📝 Raw: {response_text[:150]}")
            print(f"  🔍 Parsed: {json.dumps(parsed)}")
            print(f"  🎯 Truth:  {json.dumps(truth)}")
            print(f"  ⏱️  {dur:.1f}s | {tokens} tok | {tokens/dur:.1f} tok/s" if dur > 0 else "")
            print(f"  🎯 Action: {'✅' if action_match else '❌'} (got={model_action}, expected={truth_action})")
            if type_ok is not None:
                print(f"  🎯 Slice Type: {'✅' if type_ok else '❌'}")
            print(f"  📊 Exact key match: {exact_keys}/{total_keys} ({exact_keys/total_keys*100:.0f}%)")
            print(f"  📊 Schema compliance: action={'✅' if action_ok else '❌'} | slice_type={'✅' if has_slice else '❌'} | params={'✅' if has_params else '❌'}")
        else:
            print(f"  ❌ Could not parse JSON from: {response_text[:100]}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

print(f"\n{'='*60}")
print("✅ Done!")
