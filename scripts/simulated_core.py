"""
Simulated 5G/6G Core Network Testbed
=====================================
Mimics Free5GC + Kubernetes behavior for agent evaluation.
Provides realistic API responses for slice lifecycle operations.

Supports two modes:
  - 'simulation': Fake responses with configurable delays/metrics
  - 'production': Real K8s + Free5GC (requires kubectl + cluster)
"""

import time
import json
import random
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SliceType(Enum):
    eMBB = "eMBB"
    URLLC = "URLLC"
    mMTC = "mMTC"
    HYBRID = "hybrid"


@dataclass
class SliceState:
    """Represents a network slice in the simulated core."""
    slice_id: str
    slice_type: SliceType
    status: str = "CREATING"  # CREATING → RUNNING → SCALING → TERMINATED
    upf_replicas: int = 1
    smf_replicas: int = 1
    amf_replicas: int = 1
    bandwidth_gbps: float = 1.0
    latency_ms: float = 10.0
    device_density: int = 0
    priority: str = "normal"
    cpu_usage_pct: float = 0.0
    memory_usage_gb: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "slice_id": self.slice_id,
            "slice_type": self.slice_type.value,
            "status": self.status,
            "upf_replicas": self.upf_replicas,
            "smf_replicas": self.smf_replicas,
            "amf_replicas": self.amf_replicas,
            "bandwidth_gbps": self.bandwidth_gbps,
            "latency_ms": self.latency_ms,
            "device_density": self.device_density,
            "priority": self.priority,
            "cpu_usage_pct": self.cpu_usage_pct,
            "memory_usage_gb": self.memory_usage_gb,
            "uptime_seconds": time.time() - self.created_at,
        }


class SimulatedNetworkCore:
    """
    Simulates a 6G/5G Core Network (Free5GC-like) with:
    - NF management (AMF, SMF, UPF, NRF, UDM, AUSF, PCF, NSSF)
    - Slice lifecycle (create, scale, terminate, resolve conflicts)
    - Realistic delays and resource constraints
    """
    
    TOTAL_CPU_UNITS = 100.0     # Total CPU capacity
    TOTAL_MEMORY_GB = 32.0      # Total memory
    TOTAL_BANDWIDTH_GBPS = 50.0 # Total throughput
    
    def __init__(self, mode: str = "simulation"):
        self.mode = mode
        self.slices: dict[str, SliceState] = {}
        self._slice_counter = 0
        
        # NF health states
        self.nf_health = {
            "amf": {"status": "healthy", "replicas": 2, "cpu_pct": 15.0},
            "smf": {"status": "healthy", "replicas": 2, "cpu_pct": 12.0},
            "upf": {"status": "healthy", "replicas": 3, "cpu_pct": 25.0},
            "nrf": {"status": "healthy", "replicas": 1, "cpu_pct": 5.0},
            "udm": {"status": "healthy", "replicas": 1, "cpu_pct": 8.0},
            "ausf": {"status": "healthy", "replicas": 1, "cpu_pct": 5.0},
            "pcf": {"status": "healthy", "replicas": 1, "cpu_pct": 3.0},
            "nssf": {"status": "healthy", "replicas": 1, "cpu_pct": 4.0},
        }
        
        # Pre-populate some baseline slices
        self._create_baseline_slices()
        
        logger.info(f"🌐 Simulated 6G Core Network initialized (mode={mode})")
        logger.info(f"   CPU: {self.TOTAL_CPU_UNITS} units | RAM: {self.TOTAL_MEMORY_GB} GB | BW: {self.TOTAL_BANDWIDTH_GBPS} Gbps")
    
    def _create_baseline_slices(self):
        """Create existing baseline slices for conflict resolution scenarios."""
        baseline = [
            {"slice_type": SliceType.eMBB, "bandwidth_gbps": 5.0, "priority": "normal"},
            {"slice_type": SliceType.eMBB, "bandwidth_gbps": 3.0, "priority": "low"},
            {"slice_type": SliceType.URLLC, "latency_ms": 5.0, "priority": "high"},
            {"slice_type": SliceType.mMTC, "device_density": 10000, "priority": "low"},
            {"slice_type": SliceType.eMBB, "bandwidth_gbps": 8.0, "priority": "normal"},
        ]
        for cfg in baseline:
            self._slice_counter += 1
            sid = str(self._slice_counter)
            stype = cfg["slice_type"]
            s = SliceState(
                slice_id=sid,
                slice_type=stype,
                status="RUNNING",
                bandwidth_gbps=cfg.get("bandwidth_gbps", 1.0),
                latency_ms=cfg.get("latency_ms", 10.0),
                device_density=cfg.get("device_density", 0),
                priority=cfg.get("priority", "normal"),
                cpu_usage_pct=random.uniform(5, 20),
                memory_usage_gb=random.uniform(0.5, 3.0),
            )
            self.slices[sid] = s
    
    def _next_id(self) -> str:
        self._slice_counter += 1
        return str(self._slice_counter)
    
    # ─── Core Operations ─────────────────────────────────────────────
    
    def create_slice(self, params: dict) -> dict:
        """
        Create a new network slice.
        Simulates: NSSF selection → NRF registration → SMF/UPF instantiation
        
        params example:
        {
            "action": "CREATE",
            "slice_type": "URLLC",
            "latency_ms": 5,
            "bandwidth_gbps": 1.0,
            "device_density": 0,
            "priority": "high"
        }
        """
        sid = self._next_id()
        stype_str = params.get("slice_type", "eMBB")
        try:
            stype = SliceType(stype_str)
        except ValueError:
            stype = SliceType.eMBB
        
        # Check resource availability
        total_bw = sum(s.bandwidth_gbps for s in self.slices.values())
        requested_bw = params.get("bandwidth_gbps", 1.0)
        if total_bw + requested_bw > self.TOTAL_BANDWIDTH_GBPS:
            # Simulate resource conflict
            logger.warning(f"⚠️  Bandwidth exhausted: {total_bw:.1f}/{self.TOTAL_BANDWIDTH_GBPS} Gbps")
            return {
                "success": False,
                "slice_id": sid,
                "error": "INSUFFICIENT_BANDWIDTH",
                "available_gbps": self.TOTAL_BANDWIDTH_GBPS - total_bw,
                "requested_gbps": requested_bw,
            }
        
        # Simulate NF provisioning delay (URLLC is faster, eMBB slower)
        provision_time = {
            SliceType.URLLC: random.uniform(0.3, 0.8),
            SliceType.eMBB: random.uniform(0.5, 1.5),
            SliceType.mMTC: random.uniform(0.8, 2.0),
        }.get(stype, 1.0)
        time.sleep(min(provision_time, 0.1))  # Cap simulation delay
        
        slice_state = SliceState(
            slice_id=sid,
            slice_type=stype,
            status="RUNNING",
            upf_replicas=params.get("upf_replicas", 1),
            bandwidth_gbps=requested_bw,
            latency_ms=params.get("latency_ms", 10.0),
            device_density=params.get("device_density", 0),
            priority=params.get("priority", "normal"),
            cpu_usage_pct=random.uniform(5, 15),
            memory_usage_gb=random.uniform(0.5, 2.0),
        )
        
        self.slices[sid] = slice_state
        
        # Update NF health
        self._update_nf_health_after_operation("create", stype)
        
        logger.info(f"✅ Slice {sid} created: {stype.value} | BW={requested_bw}Gbps | Lat={params.get('latency_ms', 'N/A')}ms")
        
        return {
            "success": True,
            "slice_id": sid,
            "slice_state": slice_state.to_dict(),
            "convergence_time_ms": provision_time * 1000,
            "t_inference_ms": 0,  # Will be filled by agent layer
            "t_execution_ms": provision_time * 1000,
        }
    
    def scale_slice(self, params: dict) -> dict:
        """
        Scale a slice's resources (UPF replicas, bandwidth, etc.)
        """
        target_sid = params.get("slice_id", "")
        if target_sid not in self.slices:
            return {"success": False, "error": f"Slice {target_sid} not found"}
        
        s = self.slices[target_sid]
        s.status = "SCALING"
        
        # Update replicas
        if "upf_replicas" in params or "replicas" in params:
            replicas = params.get("upf_replicas", params.get("replicas", s.upf_replicas))
            s.upf_replicas = replicas
        
        if "smf_replicas" in params:
            s.smf_replicas = params["smf_replicas"]
        
        # Update bandwidth
        if "bandwidth_gbps" in params:
            total_bw = sum(sl.bandwidth_gbps for sl in self.slices.values())
            delta = params["bandwidth_gbps"] - s.bandwidth_gbps
            if total_bw + delta > self.TOTAL_BANDWIDTH_GBPS:
                return {"success": False, "error": "INSUFFICIENT_BANDWIDTH"}
            s.bandwidth_gbps = params["bandwidth_gbps"]
        
        # Simulate scaling delay
        scale_time = random.uniform(0.2, 0.6)
        time.sleep(min(scale_time, 0.05))
        
        s.status = "RUNNING"
        s.updated_at = time.time()
        s.cpu_usage_pct = random.uniform(10, 30)
        
        logger.info(f"📈 Slice {target_sid} scaled: UPF={s.upf_replicas}, BW={s.bandwidth_gbps}Gbps")
        
        return {
            "success": True,
            "slice_state": s.to_dict(),
            "convergence_time_ms": scale_time * 1000,
        }
    
    def resolve_conflict(self, params: dict) -> dict:
        """
        Resolve a resource conflict (preempt, terminate, reallocate).
        """
        action = params.get("action", "RESOLVE")
        
        if "terminate_slice" in params:
            target = params["terminate_slice"]
            if target in self.slices:
                terminated = self.slices.pop(target)
                logger.info(f"🛑 Slice {target} terminated to resolve conflict")
                return {
                    "success": True,
                    "action": "TERMINATED",
                    "terminated_slice": terminated.to_dict(),
                    "freed_bandwidth_gbps": terminated.bandwidth_gbps,
                    "freed_memory_gb": terminated.memory_usage_gb,
                }
        
        if "preempt" in params:
            preempt_target = params["preempt"]
            affected_slices = []
            freed_bw = 0.0
            
            if preempt_target == "eMBB":
                for sid, s in list(self.slices.items()):
                    if s.slice_type == SliceType.eMBB:
                        old_bw = s.bandwidth_gbps
                        s.bandwidth_gbps *= 0.3  # Reduce to 30%
                        freed_bw += old_bw - s.bandwidth_gbps
                        affected_slices.append(sid)
            
            logger.info(f"⚡ Preempted {len(affected_slices)} slices, freed {freed_bw:.1f} Gbps")
            return {
                "success": True,
                "action": "PREEMPTED",
                "affected_slices": affected_slices,
                "freed_bandwidth_gbps": freed_bw,
            }
        
        return {"success": False, "error": "No valid conflict resolution strategy"}
    
    # ─── Monitoring / Metrics ────────────────────────────────────────
    
    def get_metrics(self) -> dict:
        """Get current system-wide metrics (Prometheus-style)."""
        total_cpu = sum(s.cpu_usage_pct for s in self.slices.values())
        total_mem = sum(s.memory_usage_gb for s in self.slices.values())
        total_bw = sum(s.bandwidth_gbps for s in self.slices.values())
        
        return {
            "timestamp": time.time(),
            "system": {
                "cpu_usage_pct": total_cpu + sum(n["cpu_pct"] for n in self.nf_health.values()),
                "cpu_total": self.TOTAL_CPU_UNITS,
                "memory_usage_gb": total_mem,
                "memory_total_gb": self.TOTAL_MEMORY_GB,
                "bandwidth_usage_gbps": total_bw,
                "bandwidth_total_gbps": self.TOTAL_BANDWIDTH_GBPS,
            },
            "slices": {
                sid: s.to_dict() for sid, s in self.slices.items()
            },
            "nfs": self.nf_health,
            "slice_count": len(self.slices),
            "active_slices": sum(1 for s in self.slices.values() if s.status == "RUNNING"),
        }
    
    def get_slice_metrics(self, slice_id: str) -> Optional[dict]:
        """Get metrics for a specific slice."""
        if slice_id in self.slices:
            return self.slices[slice_id].to_dict()
        return None
    
    def _update_nf_health_after_operation(self, op: str, stype: SliceType):
        """Update NF health metrics after operations."""
        cpu_delta = {
            SliceType.eMBB: random.uniform(2, 5),
            SliceType.URLLC: random.uniform(1, 3),
            SliceType.mMTC: random.uniform(0.5, 2),
        }.get(stype, 1)
        
        for nf in ["upf", "smf"]:
            if nf in self.nf_health:
                self.nf_health[nf]["cpu_pct"] = min(100, self.nf_health[nf]["cpu_pct"] + cpu_delta)
    
    def status_report(self) -> str:
        """Generate a human-readable status report."""
        metrics = self.get_metrics()
        lines = [
            "=" * 50,
            "  6G CORE NETWORK — STATUS REPORT",
            "=" * 50,
            f"  Slices: {metrics['active_slices']} active / {metrics['slice_count']} total",
            f"  CPU:    {metrics['system']['cpu_usage_pct']:.1f}% / {metrics['system']['cpu_total']:.0f} units",
            f"  Memory: {metrics['system']['memory_usage_gb']:.1f} GB / {metrics['system']['memory_total_gb']:.0f} GB",
            f"  BW:     {metrics['system']['bandwidth_usage_gbps']:.1f} Gbps / {metrics['system']['bandwidth_total_gbps']:.0f} Gbps",
            "",
            "  Active Slices:",
        ]
        for sid, s in self.slices.items():
            if s.status == "RUNNING":
                lines.append(
                    f"    [{sid}] {s.slice_type.value:<6} | "
                    f"BW={s.bandwidth_gbps:.1f}Gbps | "
                    f"Lat={s.latency_ms:.1f}ms | "
                    f"UPF={s.upf_replicas} | "
                    f"Pri={s.priority}"
                )
        
        lines.append("=" * 50)
        return "\n".join(lines)


# ─── Singleton for easy access ───────────────────────────────────────

_core_instance: Optional[SimulatedNetworkCore] = None


def get_core(mode: str = "simulation") -> SimulatedNetworkCore:
    global _core_instance
    if _core_instance is None:
        _core_instance = SimulatedNetworkCore(mode=mode)
    return _core_instance


# ─── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test
    core = SimulatedNetworkCore()
    print(core.status_report())
    
    # Test create
    print("\n--- Creating URLLC slice ---")
    result = core.create_slice({
        "action": "CREATE",
        "slice_type": "URLLC",
        "latency_ms": 5,
        "priority": "high",
    })
    print(json.dumps(result, indent=2))
    
    # Test metrics
    print("\n--- System Metrics ---")
    print(json.dumps(core.get_metrics(), indent=2))
