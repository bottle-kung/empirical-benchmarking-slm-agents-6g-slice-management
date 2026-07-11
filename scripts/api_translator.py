#!/usr/bin/env python3
"""
6G Core API Translator — Paper 2, Mission 1
=============================================
Translates Qwen2.5-1.5B + Synonym Mapper JSON output into 
real Kubernetes/Free5GC commands.

Architecture:
  Intent → Qwen 1.5B + Synonym Mapper → JSON → API Translator → K8s/Free5GC
    
Actions:
  CREATE  → kubectl create namespace/deployment/configmap for network slice
  SCALE   → kubectl scale deployment upf-slice-XX --replicas=N
  RESOLVE → kubectl delete pod (preempt) / patch priority
  TERMINATE → kubectl delete namespace slice-XX
  
Logging: Timestamps from intent reception → Pod Ready for convergence measurement.
"""

import json
import subprocess
import time
import logging
import sys
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field, asdict

# ═══════════════════════════════════════════════════════════════
# Synonym Mapper (from Paper 1 — 28 entries)
# ═══════════════════════════════════════════════════════════════
SYNONYM_MAP = {
    "action": {
        "create": "CREATE", "deploy": "CREATE", "provision": "CREATE",
        "initialize": "CREATE", "setup": "CREATE", "set up": "CREATE",
        "add": "CREATE", "new": "CREATE",
        "scale": "SCALE", "scale_up": "SCALE_UP", "increase": "SCALE",
        "scale_down": "SCALE_DOWN", "decrease": "SCALE_DOWN", "reduce": "SCALE_DOWN",
        "optimize": "OPTIMIZE",
        "resolve": "RESOLVE", "preempt": "PREEMPT",
        "terminate": "TERMINATE", "reallocate": "REALLOCATE",
        "configure": "CONFIGURE", "config": "CONFIGURE", "modify": "CONFIGURE",
        "update": "CONFIGURE",
    },
    "slice_type": {
        "embb": "eMBB", "enhanced mobile broadband": "eMBB", "broadband": "eMBB",
        "urllc": "URLLC", "ultra reliable low latency": "URLLC", "low latency": "URLLC",
        "mmtc": "mMTC", "massive iot": "mMTC", "iot": "mMTC",
    },
    "target": {"bandwidth": "bandwidth", "throughput": "bandwidth", "latency": "latency",
               "upf": "upf", "user plane": "upf"},
    "strategy": {"gradual": "gradual", "immediate": "immediate", "instant": "immediate"},
    "priority": {"high": "critical", "urgent": "critical", "emergency": "critical"},
}

def apply_synonym_mapper(parsed: dict) -> dict:
    """Map model output to canonical schema terms."""
    if not parsed: return parsed
    result = {}
    for key, value in parsed.items():
        norm_key = key.lower().strip().replace(" ", "_").replace("-", "_")
        mapped_value = value
        if isinstance(value, str) and norm_key in SYNONYM_MAP:
            val_lower = value.lower().strip()
            if val_lower in SYNONYM_MAP[norm_key]:
                mapped_value = SYNONYM_MAP[norm_key][val_lower]
        result[key] = mapped_value
    return result


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════
@dataclass
class ConvergenceMetrics:
    """Tracks end-to-end timing for each intent."""
    intent_id: str
    action: str
    domain: str
    t_reception: str           # ISO timestamp when intent received
    t_inference_start: str = ""
    t_inference_end: str = ""
    t_inference_ms: float = 0  # AI processing time
    t_execution_start: str = ""
    t_execution_end: str = ""
    t_execution_ms: float = 0  # K8s/Free5GC overhead
    t_total_ms: float = 0      # End-to-end convergence time
    status: str = "pending"    # pending, running, ready, failed
    error_message: str = ""
    
    def to_csv_row(self) -> str:
        return (f"{self.intent_id},{self.action},{self.domain},"
                f"{self.t_inference_ms:.1f},{self.t_execution_ms:.1f},"
                f"{self.t_total_ms:.1f},{self.status},{self.error_message}")


@dataclass
class SliceManifest:
    """Represents a Free5GC network slice as K8s resources."""
    slice_id: str
    slice_type: str  # eMBB, URLLC, mMTC
    action: str      # CREATE, SCALE, etc.
    namespace: str
    bandwidth_gbps: float = 0
    latency_ms: float = 0
    device_density: int = 0
    priority: str = "normal"
    replicas: int = 1
    target: str = ""
    step_percent: int = 0
    duration_hours: int = 0
    preempt_slice: str = ""
    

# ═══════════════════════════════════════════════════════════════
# K8s Command Builder
# ═══════════════════════════════════════════════════════════════
class K8sCommandBuilder:
    """Builds kubectl commands from canonical JSON."""
    
    NAMESPACE_PREFIX = "slice"
    DEPLOYMENT_PREFIX = "free5gc"
    
    # Free5GC Network Functions
    NFS = ["amf", "smf", "upf", "nrf", "ausf", "udm", "pcf", "nssf", "udr"]
    
    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Convert to valid K8s resource name."""
        return re.sub(r'[^a-z0-9-]', '-', name.lower())[:63].strip('-')
    
    @classmethod
    def build_create_commands(cls, manifest: SliceManifest) -> list[dict]:
        """Build kubectl commands to CREATE a new network slice.
        
        Returns list of {command: str, description: str} dicts.
        """
        ns = cls._sanitize_name(f"{cls.NAMESPACE_PREFIX}-{manifest.slice_id}")
        commands = []
        
        # 1. Create namespace
        commands.append({
            "command": f"kubectl create namespace {ns}",
            "description": f"Create namespace {ns} for {manifest.slice_type} slice",
        })
        
        # 2. Create ConfigMap with slice parameters
        config_data = {
            "slice_type": manifest.slice_type,
            "bandwidth_gbps": str(manifest.bandwidth_gbps),
            "latency_ms": str(manifest.latency_ms),
            "device_density": str(manifest.device_density),
            "priority": manifest.priority,
        }
        config_json = json.dumps(config_data)
        cm_name = f"{ns}-config"
        commands.append({
            "command": f"kubectl create configmap {cm_name} -n {ns} --from-literal=config='{config_json}'",
            "description": f"Create ConfigMap {cm_name} with slice parameters",
        })
        
        # 3. Create Deployment YAML for key NFs
        for nf in ["upf", "smf", "amf"]:
            dep_name = cls._sanitize_name(f"{cls.DEPLOYMENT_PREFIX}-{nf}-{manifest.slice_id}")
            deploy_yaml = cls._generate_deployment_yaml(
                name=dep_name,
                namespace=ns,
                nf_type=nf,
                replicas=manifest.replicas,
                slice_type=manifest.slice_type,
                latency_ms=manifest.latency_ms,
            )
            commands.append({
                "command": f"cat <<'EOF' | kubectl apply -f -\n{deploy_yaml}\nEOF",
                "description": f"Create {nf.upper()} Deployment {dep_name} ({manifest.replicas} replicas)",
                "yaml": deploy_yaml,
            })
        
        # 4. Wait for pods to be ready
        commands.append({
            "command": f"kubectl wait --for=condition=ready pod -l app=free5gc -n {ns} --timeout=120s",
            "description": f"Wait for all pods in namespace {ns} to be Ready",
        })
        
        return commands
    
    @classmethod
    def build_scale_commands(cls, manifest: SliceManifest) -> list[dict]:
        """Build kubectl commands to SCALE a network slice."""
        commands = []
        target_nf = manifest.target if manifest.target in cls.NFS else "upf"
        dep_name = cls._sanitize_name(f"{cls.DEPLOYMENT_PREFIX}-{target_nf}-{manifest.slice_id}")
        ns = cls._sanitize_name(f"{cls.NAMESPACE_PREFIX}-{manifest.slice_id}")
        
        new_replicas = manifest.replicas
        if manifest.step_percent > 0:
            # Gradual scaling — multiple steps
            current = 1  # default, should be queried
            new_replicas = max(1, int(current * (1 + manifest.step_percent / 100)))
        
        commands.append({
            "command": f"kubectl scale deployment {dep_name} -n {ns} --replicas={new_replicas}",
            "description": f"Scale {target_nf.upper()} Deployment {dep_name} to {new_replicas} replicas",
        })
        
        commands.append({
            "command": f"kubectl rollout status deployment/{dep_name} -n {ns} --timeout=120s",
            "description": f"Wait for {dep_name} rollout to complete",
        })
        
        return commands
    
    @classmethod
    def build_resolve_commands(cls, manifest: SliceManifest) -> list[dict]:
        """Build kubectl commands to RESOLVE a conflict (preempt lower-priority slice)."""
        commands = []
        
        if manifest.preempt_slice:
            preempt_ns = cls._sanitize_name(f"{cls.NAMESPACE_PREFIX}-{manifest.preempt_slice}")
            commands.append({
                "command": f"kubectl scale deployment -n {preempt_ns} --replicas=0 --all",
                "description": f"Preempt slice {manifest.preempt_slice}: scale all deployments to 0",
            })
        
        # Set high priority on the protected slice
        target_ns = cls._sanitize_name(f"{cls.NAMESPACE_PREFIX}-{manifest.slice_id}")
        priority_cmd = f"kubectl annotate namespace {target_ns} priority={manifest.priority} --overwrite"
        commands.append({
            "command": priority_cmd,
            "description": f"Set namespace {target_ns} priority to {manifest.priority}",
        })
        
        return commands
    
    @classmethod
    def build_terminate_commands(cls, manifest: SliceManifest) -> list[dict]:
        """Build kubectl commands to TERMINATE a network slice."""
        ns = cls._sanitize_name(f"{cls.NAMESPACE_PREFIX}-{manifest.slice_id}")
        
        return [{
            "command": f"kubectl delete namespace {ns} --timeout=60s",
            "description": f"Terminate slice {manifest.slice_id}: delete namespace {ns}",
        }]
    
    @classmethod
    def _generate_deployment_yaml(cls, name: str, namespace: str, nf_type: str, 
                                    replicas: int, slice_type: str, latency_ms: float) -> str:
        """Generate a K8s Deployment YAML for a Free5GC NF."""
        resource_requests = {"cpu": "250m", "memory": "256Mi"}
        resource_limits = {"cpu": "500m", "memory": "512Mi"}
        
        # URLLC slices get more resources
        if slice_type == "URLLC":
            resource_requests = {"cpu": "500m", "memory": "512Mi"}
            resource_limits = {"cpu": "1000m", "memory": "1Gi"}
        
        yaml = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: free5gc
    nf-type: {nf_type}
    slice-type: {slice_type}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: free5gc
      nf-type: {nf_type}
  template:
    metadata:
      labels:
        app: free5gc
        nf-type: {nf_type}
        slice-type: {slice_type}
    spec:
      containers:
      - name: {nf_type}
        image: free5gc/{nf_type}:latest
        ports:
        - containerPort: 8000
        env:
        - name: SLICE_TYPE
          value: "{slice_type}"
        - name: LATENCY_TARGET_MS
          value: "{latency_ms}"
        resources:
          requests:
            cpu: {resource_requests['cpu']}
            memory: {resource_requests['memory']}
          limits:
            cpu: {resource_limits['cpu']}
            memory: {resource_limits['memory']}"""
        return yaml


# ═══════════════════════════════════════════════════════════════
# API Translator (Main Engine)
# ═══════════════════════════════════════════════════════════════
class APITranslator:
    """Translates JSON intents → K8s commands → Execution → Metrics."""
    
    def __init__(self, k8s_context: str = None, dry_run: bool = False, 
                 metrics_file: str = "paper2_convergence_metrics.csv"):
        self.k8s_context = k8s_context
        self.dry_run = dry_run
        self.metrics_file = Path(metrics_file)
        self.metrics: list[ConvergenceMetrics] = []
        
        # Setup logging
        self.logger = logging.getLogger("APITranslator")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        ))
        self.logger.addHandler(handler)
        
        # Initialize CSV
        if not self.metrics_file.exists():
            self.metrics_file.write_text(
                "intent_id,action,domain,t_inference_ms,t_execution_ms,"
                "t_total_ms,status,error_message\n"
            )
    
    def _kubectl(self, command: str, timeout: int = 120) -> tuple[int, str, str]:
        """Execute a kubectl command. Returns (exit_code, stdout, stderr)."""
        if self.dry_run:
            self.logger.info(f"[DRY-RUN] {command[:100]}")
            return 0, "dry-run-ok", ""
        
        full_cmd = command
        if self.k8s_context:
            full_cmd = command.replace("kubectl", f"kubectl --context {self.k8s_context}", 1)
        
        try:
            result = subprocess.run(
                full_cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, "", f"Timeout after {timeout}s"
        except Exception as e:
            return -1, "", str(e)
    
    def _build_manifest(self, parsed: dict, intent_id: str) -> SliceManifest:
        """Convert parsed JSON to SliceManifest."""
        action = parsed.get("action", "UNKNOWN").upper()
        slice_type = parsed.get("slice_type", "eMBB")
        
        # Generate slice ID from intent or parameters
        slice_id = parsed.get("slice_id", "")
        if not slice_id:
            slice_id = f"{slice_type.lower()}-{intent_id.lower()}"
        
        return SliceManifest(
            slice_id=slice_id,
            slice_type=slice_type,
            action=action,
            namespace=f"slice-{slice_id}",
            bandwidth_gbps=float(parsed.get("bandwidth_gbps", 0)),
            latency_ms=float(parsed.get("latency_ms", 0)),
            device_density=int(parsed.get("device_density", 0)),
            priority=parsed.get("priority", "normal"),
            replicas=int(parsed.get("replicas", 1)),
            target=parsed.get("target", "upf"),
            step_percent=int(parsed.get("step_percent", 0)),
            duration_hours=int(parsed.get("duration_hours", 0)),
            preempt_slice=parsed.get("preempt_slice", parsed.get("terminate_slice", "")),
        )
    
    def translate_and_execute(self, intent_id: str, domain: str, 
                               raw_json: dict, t_inference_ms: float) -> ConvergenceMetrics:
        """Main entry point: translate JSON → execute K8s commands → measure time.
        
        Args:
            intent_id: Intent identifier (e.g., PROV-001)
            domain: Domain (Provisioning, Scaling, Conflict)
            raw_json: Parsed JSON from the model (may contain non-canonical terms)
            t_inference_ms: Time spent by the AI model (pre-measured)
            
        Returns:
            ConvergenceMetrics with full timing breakdown
        """
        # Apply Synonym Mapper
        canonical_json = apply_synonym_mapper(raw_json)
        manifest = self._build_manifest(canonical_json, intent_id)
        
        # Initialize metrics
        metrics = ConvergenceMetrics(
            intent_id=intent_id,
            action=manifest.action,
            domain=domain,
            t_reception=datetime.now(timezone.utc).isoformat(),
            t_inference_ms=t_inference_ms,
        )
        
        # Select command builder based on action
        action_map = {
            "CREATE": K8sCommandBuilder.build_create_commands,
            "SCALE": K8sCommandBuilder.build_scale_commands,
            "SCALE_UP": K8sCommandBuilder.build_scale_commands,
            "SCALE_DOWN": K8sCommandBuilder.build_scale_commands,
            "RESOLVE": K8sCommandBuilder.build_resolve_commands,
            "PREEMPT": K8sCommandBuilder.build_resolve_commands,
            "TERMINATE": K8sCommandBuilder.build_terminate_commands,
            "REALLOCATE": K8sCommandBuilder.build_resolve_commands,
            "CONFIGURE": K8sCommandBuilder.build_create_commands,
        }
        
        builder = action_map.get(manifest.action)
        if not builder:
            metrics.status = "failed"
            metrics.error_message = f"Unsupported action: {manifest.action}"
            metrics.t_total_ms = t_inference_ms
            self._save_metric(metrics)
            return metrics
        
        # Execute commands
        commands = builder(manifest)
        t_exec_start = time.time()
        metrics.t_execution_start = datetime.now(timezone.utc).isoformat()
        
        all_ok = True
        for i, cmd in enumerate(commands):
            self.logger.info(f"[{intent_id}] {cmd['description']}")
            exit_code, stdout, stderr = self._kubectl(cmd["command"])
            
            if exit_code != 0:
                self.logger.error(f"[{intent_id}] Failed: {stderr[:200]}")
                if not self.dry_run:
                    all_ok = False
                    metrics.error_message = f"Command {i+1}/{len(commands)} failed: {stderr[:200]}"
                    break
        
        t_exec_end = time.time()
        metrics.t_execution_end = datetime.now(timezone.utc).isoformat()
        metrics.t_execution_ms = (t_exec_end - t_exec_start) * 1000
        metrics.t_total_ms = metrics.t_inference_ms + metrics.t_execution_ms
        
        if all_ok:
            metrics.status = "ready"
        elif not metrics.error_message:
            metrics.status = "failed"
            metrics.error_message = "Unknown execution error"
        
        self._save_metric(metrics)
        return metrics
    
    def _save_metric(self, metrics: ConvergenceMetrics):
        """Append metric to CSV file."""
        self.metrics.append(metrics)
        with open(self.metrics_file, "a") as f:
            f.write(metrics.to_csv_row() + "\n")
    
    def get_summary(self) -> dict:
        """Calculate aggregate statistics from all metrics."""
        if not self.metrics:
            return {"error": "No metrics collected"}
        
        ready = [m for m in self.metrics if m.status == "ready"]
        failed = [m for m in self.metrics if m.status == "failed"]
        
        def stats(values):
            if not values: return {"mean": 0, "std": 0, "cv_pct": 0}
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5
            cv = (std / mean * 100) if mean > 0 else 0
            return {"mean": round(mean, 1), "std": round(std, 1), "cv_pct": round(cv, 1)}
        
        return {
            "total_intents": len(self.metrics),
            "success_rate": round(len(ready) / max(len(self.metrics), 1) * 100, 1),
            "t_inference": stats([m.t_inference_ms for m in ready]),
            "t_execution": stats([m.t_execution_ms for m in ready]),
            "t_total": stats([m.t_total_ms for m in ready]),
            "failed_count": len(failed),
        }
    
    def print_report(self):
        """Print a formatted summary report."""
        s = self.get_summary()
        if "error" in s:
            print(f"❌ {s['error']}")
            return
        
        print(f"\n{'='*70}")
        print(f"  API Translator — Convergence Time Report")
        print(f"{'='*70}")
        print(f"  Total Intents:     {s['total_intents']}")
        print(f"  Success Rate:      {s['success_rate']}%")
        print(f"  Failed:            {s['failed_count']}")
        print(f"\n  {'Metric':<20} {'Mean (ms)':>12} {'Std (ms)':>12} {'CV%':>8}")
        print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*8}")
        
        for metric, label in [("t_inference", "T_Inference (AI)"), 
                               ("t_execution", "T_Execution (K8s)"),
                               ("t_total", "T_Total (E2E)")]:
            m = s[metric]
            print(f"  {label:<20} {m['mean']:>10.1f}  {m['std']:>10.1f}  {m['cv_pct']:>6.1f}%")
        
        print(f"\n  📁 Metrics saved to: {self.metrics_file}")


# ═══════════════════════════════════════════════════════════════
# Demo / Test
# ═══════════════════════════════════════════════════════════════
def demo():
    """Demonstrate the API Translator with sample intents (dry-run)."""
    translator = APITranslator(dry_run=True, 
                               metrics_file="/jupyter_workspace/local/ai_agent/slm_6g_eval/results/paper2_convergence_metrics.csv")
    
    # Sample intents with their Qwen 1.5B output (pre-synonym-mapper)
    test_intents = [
        {
            "intent_id": "PROV-001",
            "domain": "Slicing Provisioning",
            "raw_json": {"action": "create", "slice_type": "eMBB", "bandwidth_gbps": 10, "latency_ms": 5},
            "t_inference_ms": 4100,
        },
        {
            "intent_id": "PROV-007",
            "domain": "Slicing Provisioning", 
            "raw_json": {"action": "deploy", "slice_type": "URLLC", "latency_ms": 1, "priority": "high"},
            "t_inference_ms": 3800,
        },
        {
            "intent_id": "SCAL-026",
            "domain": "Scaling Request",
            "raw_json": {"action": "increase", "target": "bandwidth", "strategy": "gradual", 
                         "step_percent": 20, "duration_hours": 4},
            "t_inference_ms": 4200,
        },
        {
            "intent_id": "CONF-023",
            "domain": "Conflict Resolution",
            "raw_json": {"action": "resolve", "priority": "emergency", "preempt": "regular_users", "slice_type": "URLLC"},
            "t_inference_ms": 3900,
        },
    ]
    
    print("=" * 70)
    print("  API Translator — Demo (Dry-Run Mode)")
    print("=" * 70)
    
    for intent in test_intents:
        print(f"\n{'─'*70}")
        print(f"  Intent: {intent['intent_id']} ({intent['domain']})")
        print(f"  Raw JSON: {json.dumps(intent['raw_json'])}")
        
        metrics = translator.translate_and_execute(
            intent_id=intent["intent_id"],
            domain=intent["domain"],
            raw_json=intent["raw_json"],
            t_inference_ms=intent["t_inference_ms"],
        )
        
        print(f"  → Action: {metrics.action}, Status: {metrics.status}")
        print(f"  → T_Inference: {metrics.t_inference_ms:.0f}ms, "
              f"T_Execution: {metrics.t_execution_ms:.0f}ms, "
              f"T_Total: {metrics.t_total_ms:.0f}ms")
    
    translator.print_report()


if __name__ == "__main__":
    demo()
