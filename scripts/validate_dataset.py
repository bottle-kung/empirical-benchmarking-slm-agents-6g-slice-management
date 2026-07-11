#!/usr/bin/env python3
"""
Dataset Validation Script
=========================
Validates the intent dataset for completeness, consistency, and correctness.

Checks:
  1. All required fields present
  2. Valid domain and complexity values
  3. All ground_truth contain "action" key
  4. No duplicate intent_ids
  5. Input texts are non-empty and unique
  6. Distribution matches expected targets
  7. JSONL format is valid

Usage:
  python scripts/validate_dataset.py [--dataset path/to/intents.jsonl]
"""

import json
import sys
from pathlib import Path
from collections import Counter


REQUIRED_FIELDS = ["intent_id", "domain", "complexity", "input_text", "ground_truth"]
VALID_DOMAINS = ["Slicing Provisioning", "Scaling Request", "Conflict Resolution"]
VALID_COMPLEXITIES = ["Simple", "Complex", "Ambiguous"]

EXPECTED_DISTRIBUTION = {
    "Slicing Provisioning": {"Simple": 25, "Complex": 22, "Ambiguous": 23, "total": 70},
    "Scaling Request": {"Simple": 25, "Complex": 23, "Ambiguous": 22, "total": 70},
    "Conflict Resolution": {"Simple": 20, "Complex": 20, "Ambiguous": 20, "total": 60},
}

EXPECTED_TOTAL = 200


class ValidationReport:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {}
    
    def error(self, msg: str):
        self.errors.append(msg)
    
    def warn(self, msg: str):
        self.warnings.append(msg)
    
    def ok(self):
        return len(self.errors) == 0
    
    def print(self):
        print("\n" + "=" * 60)
        print("DATASET VALIDATION REPORT")
        print("=" * 60)
        
        if self.stats:
            print(f"\n📊 Statistics:")
            for key, value in self.stats.items():
                print(f"   {key}: {value}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for e in self.errors:
                print(f"   • {e}")
        else:
            print(f"\n✅ No errors found!")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"   • {w}")
        
        if self.ok():
            print(f"\n🎉 Dataset is valid and ready for benchmarking!")
        else:
            print(f"\n🔴 Dataset has {len(self.errors)} error(s) — fix before proceeding.")


def validate_dataset(dataset_path: str) -> ValidationReport:
    report = ValidationReport()
    path = Path(dataset_path)
    
    if not path.exists():
        report.error(f"Dataset file not found: {dataset_path}")
        return report
    
    report.stats["file_size"] = f"{path.stat().st_size:,} bytes"
    
    # Load all intents
    intents = []
    line_errors = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                intents.append(json.loads(line))
            except json.JSONDecodeError as e:
                report.error(f"Line {line_num}: Invalid JSON — {e}")
                line_errors += 1
    
    report.stats["total_lines"] = len(intents)
    
    if line_errors > 0:
        report.error(f"{line_errors} lines could not be parsed as JSON")
    
    # Check total count
    if len(intents) != EXPECTED_TOTAL:
        report.error(f"Expected {EXPECTED_TOTAL} intents, found {len(intents)}")
    else:
        report.stats["target_count"] = f"{EXPECTED_TOTAL} ✅"
    
    # Check required fields
    missing_fields = Counter()
    for i, item in enumerate(intents):
        for field in REQUIRED_FIELDS:
            if field not in item:
                missing_fields[field] += 1
                if missing_fields[field] <= 5:  # Only report first 5
                    report.error(f"Intent #{i+1} ({item.get('intent_id', 'UNKNOWN')}): Missing field '{field}'")
    
    if missing_fields:
        report.error(f"Field missing summary: {dict(missing_fields)}")
    else:
        report.stats["required_fields"] = "All present ✅"
    
    # Check domain values
    invalid_domains = []
    for item in intents:
        if item.get("domain") not in VALID_DOMAINS:
            invalid_domains.append(item.get("intent_id", "UNKNOWN"))
    if invalid_domains:
        report.error(f"Invalid domains: {invalid_domains}")
    else:
        report.stats["domain_values"] = "All valid ✅"
    
    # Check complexity values
    invalid_complexities = []
    for item in intents:
        if item.get("complexity") not in VALID_COMPLEXITIES:
            invalid_complexities.append(item.get("intent_id", "UNKNOWN"))
    if invalid_complexities:
        report.error(f"Invalid complexities: {invalid_complexities}")
    else:
        report.stats["complexity_values"] = "All valid ✅"
    
    # Check ground_truth has "action"
    missing_actions = []
    for item in intents:
        gt = item.get("ground_truth", {})
        if "action" not in gt:
            missing_actions.append(item.get("intent_id", "UNKNOWN"))
    if missing_actions:
        report.error(f"Missing 'action' in ground_truth: {missing_actions[:10]}...")
    else:
        report.stats["ground_truth_actions"] = "All present ✅"
    
    # Check duplicate intent_ids
    ids = [item["intent_id"] for item in intents if "intent_id" in item]
    dupes = {id: count for id, count in Counter(ids).items() if count > 1}
    if dupes:
        report.error(f"Duplicate intent_ids: {dupes}")
    else:
        report.stats["unique_ids"] = f"{len(ids)} unique ✅"
    
    # Check empty input_text
    empty_inputs = [item.get("intent_id", "UNKNOWN") for item in intents 
                    if not item.get("input_text", "").strip()]
    if empty_inputs:
        report.error(f"Empty input_text: {empty_inputs}")
    else:
        report.stats["empty_inputs"] = "None ✅"
    
    # Check input_text uniqueness (warn only for exact duplicates)
    input_texts = [item.get("input_text", "") for item in intents]
    input_dupes = {text: count for text, count in Counter(input_texts).items() if count > 1}
    if input_dupes:
        report.warn(f"Duplicate input_texts ({len(input_dupes)} texts appear multiple times)")
    else:
        report.stats["unique_inputs"] = "All unique ✅"
    
    # Distribution check
    domain_complexity = Counter((item["domain"], item["complexity"]) 
                                 for item in intents 
                                 if "domain" in item and "complexity" in item)
    
    domain_totals = Counter(item["domain"] for item in intents if "domain" in item)
    
    print("\n📊 Actual Distribution:")
    print(f"   {'Domain':<25} {'Simple':>7} {'Complex':>8} {'Ambiguous':>10} {'Total':>6}")
    print(f"   {'-'*25} {'-'*7} {'-'*8} {'-'*10} {'-'*6}")
    
    all_match = True
    for domain in VALID_DOMAINS:
        expected = EXPECTED_DISTRIBUTION[domain]
        actual_s = domain_complexity.get((domain, "Simple"), 0)
        actual_c = domain_complexity.get((domain, "Complex"), 0)
        actual_a = domain_complexity.get((domain, "Ambiguous"), 0)
        actual_total = domain_totals.get(domain, 0)
        
        s_ok = "✅" if actual_s == expected["Simple"] else f"❌(exp {expected['Simple']})"
        c_ok = "✅" if actual_c == expected["Complex"] else f"❌(exp {expected['Complex']})"
        a_ok = "✅" if actual_a == expected["Ambiguous"] else f"❌(exp {expected['Ambiguous']})"
        t_ok = "✅" if actual_total == expected["total"] else f"❌(exp {expected['total']})"
        
        if actual_s != expected["Simple"] or actual_c != expected["Complex"] or \
           actual_a != expected["Ambiguous"] or actual_total != expected["total"]:
            all_match = False
        
        print(f"   {domain:<25} {actual_s:>3} {s_ok:<12} {actual_c:>3} {c_ok:<12} {actual_a:>3} {a_ok:<14} {actual_total:>3} {t_ok}")
    
    if all_match:
        print(f"\n   ✅ All distribution targets match!")
    else:
        report.warn("Distribution does not match expected targets (check table above)")
    
    # Check intent_id format
    bad_ids = []
    for item in intents:
        iid = item.get("intent_id", "")
        if not iid:
            continue
        # Expected format: PREFIX-NNN
        parts = iid.split("-")
        if len(parts) != 2:
            bad_ids.append(iid)
        elif not parts[1].isdigit():
            bad_ids.append(iid)
        elif len(parts[1]) != 3:
            bad_ids.append(iid)
    if bad_ids:
        report.warn(f"Non-standard intent_id format: {bad_ids}")
    else:
        report.stats["id_format"] = "All standard ✅"
    
    # Check average input_text length
    avg_len = sum(len(item.get("input_text", "")) for item in intents) / len(intents) if intents else 0
    report.stats["avg_input_len"] = f"{avg_len:.0f} chars"
    
    # Action type distribution
    actions = Counter(item["ground_truth"]["action"] for item in intents if "ground_truth" in item and "action" in item["ground_truth"])
    print(f"\n📊 Action Distribution:")
    for action, count in sorted(actions.items(), key=lambda x: -x[1]):
        pct = count / len(intents) * 100
        bar = "█" * int(pct / 2)
        print(f"   {action:<25} {count:>4} ({pct:5.1f}%) {bar}")
    
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate the intent dataset")
    parser.add_argument("--dataset", default="dataset/intents.jsonl", help="Path to intent dataset")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = args.dataset
    if not Path(dataset_path).is_absolute():
        dataset_path = str(project_root / dataset_path)
    
    report = validate_dataset(dataset_path)
    report.print()
    
    sys.exit(0 if report.ok() else 1)


if __name__ == "__main__":
    main()
