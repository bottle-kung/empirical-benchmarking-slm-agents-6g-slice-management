#!/usr/bin/env python3
"""
Generate 200 Intents for SLM Agent Evaluation Benchmark
Intent-Driven Slice Lifecycle Management in 6G Core

Domains:
  - Slicing Provisioning (70 intents)
  - Scaling Request (70 intents)
  - Conflict Resolution (60 intents)
"""

import json
import random
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path("/jupyter_workspace/local/ai_agent/slm_6g_eval/dataset/intents.jsonl")

# =============================================================================
# DOMAIN 1: Slicing Provisioning (70 intents: 25 Simple + 22 Complex + 23 Ambiguous)
# =============================================================================

PROV_SIMPLE = [
    # --- eMBB slices ---
    {
        "intent_id": "PROV-001",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Create a new eMBB slice for a 4K video streaming event in Bangkok.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 1}
    },
    {
        "intent_id": "PROV-002",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Provision an eMBB slice for the upcoming sports tournament live broadcast.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 2}
    },
    {
        "intent_id": "PROV-003",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Set up an eMBB network slice for a virtual reality concert streaming event.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 1.5}
    },
    {
        "intent_id": "PROV-004",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Create a high-bandwidth slice for a cloud gaming tournament with 500 players.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 3}
    },
    {
        "intent_id": "PROV-005",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Deploy an eMBB slice for 8K video surveillance in a smart city district.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 1.5}
    },
    {
        "intent_id": "PROV-006",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Initialize an eMBB slice for large file transfer between two data centers.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 5}
    },
    # --- URLLC slices ---
    {
        "intent_id": "PROV-007",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Create a URLLC slice with 1ms latency for industrial robot control.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 1}
    },
    {
        "intent_id": "PROV-008",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Provision a low-latency URLLC slice for autonomous vehicle platooning.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 5}
    },
    {
        "intent_id": "PROV-009",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Set up a URLLC slice for remote surgical equipment with 3ms maximum latency.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 3}
    },
    {
        "intent_id": "PROV-010",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Create an ultra-reliable slice for smart grid power distribution control.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 10, "priority": "critical"}
    },
    {
        "intent_id": "PROV-011",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Initialize a URLLC network slice for real-time drone swarm coordination.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 5}
    },
    # --- mMTC slices ---
    {
        "intent_id": "PROV-012",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Initialize a massive IoT network slice for 50,000 smart water meters across the city.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "device_density": 50000}
    },
    {
        "intent_id": "PROV-013",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Create an mMTC slice for 10,000 environmental sensors in a national park.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "device_density": 10000}
    },
    {
        "intent_id": "PROV-014",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Provision an mMTC slice for 200,000 smart parking sensors citywide.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "device_density": 200000}
    },
    {
        "intent_id": "PROV-015",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Set up an mMTC slice for a fleet of 5,000 agricultural soil moisture sensors.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "device_density": 5000}
    },
    {
        "intent_id": "PROV-016",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Initialize a massive IoT slice for 30,000 smart streetlights in a metropolitan area.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "device_density": 30000}
    },
    {
        "intent_id": "PROV-017",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Create an mMTC network slice for 100,000 temperature loggers in a cold chain logistics network.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "device_density": 100000}
    },
    # --- Additional Simple Provisioning ---
    {
        "intent_id": "PROV-018",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Deploy an eMBB slice for a 360-degree VR tourism platform serving 1,000 concurrent users.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 2}
    },
    {
        "intent_id": "PROV-019",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Create a URLLC slice for haptic feedback systems in teleoperation with 2ms latency.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 2}
    },
    {
        "intent_id": "PROV-020",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Provision an mMTC slice for 15,000 wearable health monitors in a hospital network.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "device_density": 15000}
    },
    {
        "intent_id": "PROV-021",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Set up an eMBB slice for holographic video conferencing with 2 Gbps requirement.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 2}
    },
    {
        "intent_id": "PROV-022",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Create a URLLC slice for automated train control system with 5ms latency guarantee.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 5}
    },
    {
        "intent_id": "PROV-023",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Initialize an mMTC slice for 75,000 energy meter readers in a smart grid deployment.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "device_density": 75000}
    },
    {
        "intent_id": "PROV-024",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Provision an eMBB slice for a massive online multiplayer game server with 10 Gbps.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 10}
    },
    {
        "intent_id": "PROV-025",
        "domain": "Slicing Provisioning",
        "complexity": "Simple",
        "input_text": "Create a URLLC slice for automated port crane operations with sub-5ms latency.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 5}
    },
]

PROV_COMPLEX = [
    {
        "intent_id": "PROV-026",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Create a hybrid slice that supports both eMBB for video streaming and URLLC for emergency stop signals in a mining automation system.",
        "ground_truth": {"action": "CREATE", "slice_type": "hybrid", "sub_slices": [
            {"type": "eMBB", "bandwidth_gbps": 1},
            {"type": "URLLC", "latency_ms": 2, "priority": "high"}
        ]}
    },
    {
        "intent_id": "PROV-027",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Provision a network slice with mandatory QoS flow for real-time ECG monitoring and a secondary flow for bulk medical imaging uploads.",
        "ground_truth": {"action": "CREATE", "slice_type": "multi_qos", "flows": [
            {"type": "URLLC", "priority": "critical", "latency_ms": 10, "purpose": "ECG"},
            {"type": "eMBB", "priority": "low", "bandwidth_mbps": 500, "purpose": "imaging"}
        ]}
    },
    {
        "intent_id": "PROV-028",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Set up a network slice for a connected vehicle ecosystem that requires 99.9999% reliability for V2X safety messages and background telemetry for 20,000 vehicles.",
        "ground_truth": {"action": "CREATE", "slice_type": "v2x", "reliability": "99.9999%", "v2x_type": "safety", "telemetry_devices": 20000}
    },
    {
        "intent_id": "PROV-029",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Create a slice for a smart factory with three isolated network segments: one for robot control (URLLC), one for video quality inspection (eMBB), and one for environmental sensors (mMTC).",
        "ground_truth": {"action": "CREATE", "slice_type": "multi_segment", "segments": [
            {"segment": "robot_control", "type": "URLLC", "latency_ms": 1},
            {"segment": "quality_inspection", "type": "eMBB", "bandwidth_gbps": 1},
            {"segment": "environmental", "type": "mMTC", "device_density": 5000}
        ]}
    },
    {
        "intent_id": "PROV-030",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Deploy a slice with dynamic QoS that scales from baseline mMTC to burst eMBB mode during peak hours for a smart retail chain with 200 stores.",
        "ground_truth": {"action": "CREATE", "slice_type": "dynamic_qos", "baseline": "mMTC", "burst": "eMBB", "device_density": 200, "peak_hours": True}
    },
    {
        "intent_id": "PROV-031",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Provision a multi-tenant slice with guaranteed isolation between three enterprise customers sharing the same physical infrastructure.",
        "ground_truth": {"action": "CREATE", "slice_type": "multi_tenant", "tenant_count": 3, "isolation": "strict", "per_tenant_bandwidth_gbps": 2}
    },
    {
        "intent_id": "PROV-032",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Set up a time-sensitive networking slice for synchronized robotic assembly lines with end-to-end latency under 500 microseconds and jitter under 1 microsecond.",
        "ground_truth": {"action": "CREATE", "slice_type": "TSN", "latency_us": 500, "jitter_us": 1}
    },
    {
        "intent_id": "PROV-033",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Create a slice with AI-driven predictive resource allocation for a mobile broadband network serving a sports stadium that fills and empties predictably.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 5, "resource_policy": "predictive_ai", "scenario": "sports_stadium"}
    },
    {
        "intent_id": "PROV-034",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Deploy a slice with geographic fencing — full performance within a 5km industrial zone, degraded but available outside for logistics tracking.",
        "ground_truth": {"action": "CREATE", "slice_type": "geo_fenced", "zone_radius_km": 5, "inside_qos": {"type": "URLLC", "latency_ms": 5}, "outside_qos": {"type": "mMTC"}}
    },
    {
        "intent_id": "PROV-035",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Provision an energy-aware slice that prioritizes green energy nodes for routing in an mMTC smart agriculture deployment.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "energy_policy": "green_preferred", "device_density": 25000, "sector": "agriculture"}
    },
    {
        "intent_id": "PROV-036",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Create a network slice for disaster recovery with automatic failover to satellite backhaul if fiber connection drops.",
        "ground_truth": {"action": "CREATE", "slice_type": "resilient", "backup": "satellite", "failover_policy": "automatic", "primary": "fiber"}
    },
    {
        "intent_id": "PROV-037",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Set up a secure slice with end-to-end quantum-resistant encryption for a government classified communications network.",
        "ground_truth": {"action": "CREATE", "slice_type": "secure", "encryption": "quantum_resistant", "classification": "government"}
    },
    {
        "intent_id": "PROV-038",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Deploy a slice with redundant UPF instances across two geographically separated data centers for a banking transaction system requiring zero downtime.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "redundancy": "geo_redundant", "upf_replicas": 2, "target_sector": "banking"}
    },
    {
        "intent_id": "PROV-039",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Create a slice with network slicing as a service (NSaaS) capability — allow the tenant to dynamically create sub-slices via API.",
        "ground_truth": {"action": "CREATE", "slice_type": "nsaas", "api_enabled": True, "max_sub_slices": 50}
    },
    {
        "intent_id": "PROV-040",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Provision a slice for a telemedicine hub serving 50 rural clinics — each clinic needs video consultation (eMBB) and vitals monitoring (mMTC) on the same slice.",
        "ground_truth": {"action": "CREATE", "slice_type": "composite", "clinics": 50, "services": [
            {"type": "eMBB", "purpose": "video_consult", "bandwidth_mbps": 50},
            {"type": "mMTC", "purpose": "vitals", "device_density": 500}
        ]}
    },
    {
        "intent_id": "PROV-041",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Set up a slice for a large-scale digital twin system that mirrors a factory floor in real-time with bidirectional data streams.",
        "ground_truth": {"action": "CREATE", "slice_type": "digital_twin", "data_direction": "bidirectional", "update_frequency_hz": 60, "bandwidth_gbps": 2}
    },
    {
        "intent_id": "PROV-042",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Create a slice with network coding enabled for an ultra-reliable multicast video distribution to 100 emergency response vehicles.",
        "ground_truth": {"action": "CREATE", "slice_type": "multicast", "network_coding": True, "clients": 100, "slice_type_qos": "URLLC"}
    },
    {
        "intent_id": "PROV-043",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Deploy a slice for an air traffic control drone management system that must support 500 concurrent drone flights with real-time telemetry.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "concurrent_drones": 500, "telemetry": "real_time", "latency_ms": 10}
    },
    {
        "intent_id": "PROV-044",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Provision an edge-computing integrated slice where video analytics runs on MEC nodes within the slice boundary for a smart surveillance system.",
        "ground_truth": {"action": "CREATE", "slice_type": "edge_integrated", "mec_enabled": True, "application": "video_analytics", "bandwidth_gbps": 1.5}
    },
    {
        "intent_id": "PROV-045",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Create a slice with end-to-end network slicing across multi-domain (RAN + Transport + Core) for an intercontinental enterprise VPN.",
        "ground_truth": {"action": "CREATE", "slice_type": "e2e", "domains": ["RAN", "Transport", "Core"], "service": "enterprise_vpn", "bandwidth_gbps": 1}
    },
    {
        "intent_id": "PROV-046",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Set up a slice that integrates with a blockchain-based SLA monitoring system for a smart contract-governed logistics network.",
        "ground_truth": {"action": "CREATE", "slice_type": "blockchain_sla", "sla_monitoring": "blockchain", "sector": "logistics"}
    },
    {
        "intent_id": "PROV-047",
        "domain": "Slicing Provisioning",
        "complexity": "Complex",
        "input_text": "Deploy a network slice with intent-based closed-loop automation — the slice self-optimizes based on declared intent KPIs.",
        "ground_truth": {"action": "CREATE", "slice_type": "closed_loop", "automation": "intent_based", "kpi_targets": {"latency_ms": 10, "availability": "99.999%"}}
    },
]

PROV_AMBIGUOUS = [
    {
        "intent_id": "PROV-048",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Set up a reliable connection for the hospital's remote surgery robots.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 5}
    },
    {
        "intent_id": "PROV-049",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "I need a slice for a fleet of autonomous delivery drones.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "priority": "high"}
    },
    {
        "intent_id": "PROV-050",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Give me a network for my factory — it needs to handle all the machines and cameras.",
        "ground_truth": {"action": "CREATE", "slice_type": "hybrid", "sub_slices": [
            {"type": "URLLC", "purpose": "machines"},
            {"type": "eMBB", "purpose": "cameras"}
        ]}
    },
    {
        "intent_id": "PROV-051",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "We need connectivity for a smart city project — sensors everywhere, lots of them.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "device_density": 100000}
    },
    {
        "intent_id": "PROV-052",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "The power company wants a network for their grid monitoring — it has to be bulletproof.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "priority": "critical", "reliability": "99.9999%"}
    },
    {
        "intent_id": "PROV-053",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Can you spin up something fast for the stadium? There's a big match tonight.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 5, "scenario": "stadium_event"}
    },
    {
        "intent_id": "PROV-054",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "I need a slice that will never go down, even during an earthquake — it's for emergency services.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "priority": "critical", "resiliency": "disaster_proof"}
    },
    {
        "intent_id": "PROV-055",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "My logistics company tracks thousands of packages — I just need basic connectivity, nothing fancy.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "device_density": 5000, "sector": "logistics"}
    },
    {
        "intent_id": "PROV-056",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Set up network for a university campus — students stream videos, researchers transfer huge datasets, and the admin runs VoIP phones.",
        "ground_truth": {"action": "CREATE", "slice_type": "composite", "services": ["eMBB", "eMBB", "URLLC"]}
    },
    {
        "intent_id": "PROV-057",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "We're deploying self-driving tractors across a 50-square-kilometer farm. Make it work.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "area_sqkm": 50, "latency_ms": 10}
    },
    {
        "intent_id": "PROV-058",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Need a secure channel for police body cameras streaming back to headquarters in real-time.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "bandwidth_mbps": 50, "encryption": "required", "sector": "law_enforcement"}
    },
    {
        "intent_id": "PROV-059",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Can you set up connectivity for a cruise ship? Hundreds of passengers streaming and posting on social media.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 3, "scenario": "maritime"}
    },
    {
        "intent_id": "PROV-060",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "We're running augmented reality guided tours in a museum — needs to be super responsive.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 5, "application": "AR"}
    },
    {
        "intent_id": "PROV-061",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Oil rig in the middle of the ocean — need connectivity for equipment monitoring and crew communication.",
        "ground_truth": {"action": "CREATE", "slice_type": "composite", "services": [
            {"type": "mMTC", "purpose": "equipment_monitoring"},
            {"type": "URLLC", "purpose": "crew_comm"}
        ], "scenario": "offshore"}
    },
    {
        "intent_id": "PROV-062",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Build a network for an esports arena — latency is everything.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "latency_ms": 2, "scenario": "esports"}
    },
    {
        "intent_id": "PROV-063",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "My delivery company wants to track every package in real-time. There could be millions per day.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "device_density": 1000000, "update_frequency": "real_time"}
    },
    {
        "intent_id": "PROV-064",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Need a network for a temporary field hospital after a natural disaster. Quick setup, must work perfectly.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "priority": "emergency", "deployment": "rapid", "latency_ms": 5}
    },
    {
        "intent_id": "PROV-065",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Set up a slice for remote mining operations — autonomous trucks, conveyor belts, and worker safety sensors.",
        "ground_truth": {"action": "CREATE", "slice_type": "composite", "services": [
            {"type": "URLLC", "purpose": "autonomous_trucks"},
            {"type": "mMTC", "purpose": "safety_sensors"}
        ], "sector": "mining"}
    },
    {
        "intent_id": "PROV-066",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "We need connectivity for a high-speed rail line — trains go 300 km/h, passengers expect perfect 5G.",
        "ground_truth": {"action": "CREATE", "slice_type": "eMBB", "bandwidth_gbps": 1, "mobility_kmh": 300, "scenario": "high_speed_rail"}
    },
    {
        "intent_id": "PROV-067",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Connect a wind farm — hundreds of turbines sending telemetry and maintenance data.",
        "ground_truth": {"action": "CREATE", "slice_type": "mMTC", "device_density": 500, "sector": "renewable_energy"}
    },
    {
        "intent_id": "PROV-068",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Need a slice for financial trading — every microsecond counts, downtime is not an option.",
        "ground_truth": {"action": "CREATE", "slice_type": "URLLC", "latency_us": 500, "reliability": "99.99999%", "sector": "finance"}
    },
    {
        "intent_id": "PROV-069",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "Set up a network for a large mall — thousands of shoppers, digital signage everywhere, and the security cameras.",
        "ground_truth": {"action": "CREATE", "slice_type": "composite", "services": [
            {"type": "eMBB", "purpose": "shoppers"},
            {"type": "mMTC", "purpose": "digital_signage"},
            {"type": "eMBB", "purpose": "security"}
        ]}
    },
    {
        "intent_id": "PROV-070",
        "domain": "Slicing Provisioning",
        "complexity": "Ambiguous",
        "input_text": "I'm opening a chain of automated convenience stores — need network for payment terminals, inventory sensors, and facial recognition cameras.",
        "ground_truth": {"action": "CREATE", "slice_type": "composite", "services": [
            {"type": "mMTC", "purpose": "inventory"},
            {"type": "URLLC", "purpose": "payments"},
            {"type": "eMBB", "purpose": "facial_recognition"}
        ]}
    },
]

# =============================================================================
# DOMAIN 2: Scaling Request (70 intents: 25 Simple + 23 Complex + 22 Ambiguous)
# =============================================================================

SCAL_SIMPLE = [
    {
        "intent_id": "SCAL-001",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Scale up the UPF instances for Slice ID 55 to 10 nodes immediately.",
        "ground_truth": {"action": "SCALE", "target": "upf", "slice_id": "55", "replicas": 10}
    },
    {
        "intent_id": "SCAL-002",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Scale down the AMF instances for Slice ID 12 to 2 replicas.",
        "ground_truth": {"action": "SCALE", "target": "amf", "slice_id": "12", "replicas": 2}
    },
    {
        "intent_id": "SCAL-003",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Increase bandwidth of Slice ID 88 from 1 Gbps to 5 Gbps.",
        "ground_truth": {"action": "SCALE", "target": "bandwidth", "slice_id": "88", "bandwidth_gbps": 5}
    },
    {
        "intent_id": "SCAL-004",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Add 3 more SMF instances to Slice ID 42.",
        "ground_truth": {"action": "SCALE_UP", "target": "smf", "slice_id": "42", "replicas_increment": 3}
    },
    {
        "intent_id": "SCAL-005",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Reduce the number of UDM instances for Slice ID 99 to 1.",
        "ground_truth": {"action": "SCALE_DOWN", "target": "udm", "slice_id": "99", "replicas": 1}
    },
    {
        "intent_id": "SCAL-006",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Scale the NRF service for Slice ID 7 to 4 replicas.",
        "ground_truth": {"action": "SCALE", "target": "nrf", "slice_id": "7", "replicas": 4}
    },
    {
        "intent_id": "SCAL-007",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Increase the UPF buffer size for Slice ID 34 to 2 GB.",
        "ground_truth": {"action": "SCALE", "target": "upf_buffer", "slice_id": "34", "buffer_gb": 2}
    },
    {
        "intent_id": "SCAL-008",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Scale up ingress bandwidth for Slice ID 21 to 10 Gbps.",
        "ground_truth": {"action": "SCALE", "target": "ingress_bandwidth", "slice_id": "21", "bandwidth_gbps": 10}
    },
    {
        "intent_id": "SCAL-009",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Scale down all idle PDU sessions in Slice ID 67 to free resources.",
        "ground_truth": {"action": "SCALE_DOWN", "target": "pdu_sessions", "slice_id": "67", "filter": "idle"}
    },
    {
        "intent_id": "SCAL-010",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Double the CPU allocation for the UPF pod in Slice ID 45.",
        "ground_truth": {"action": "SCALE", "target": "cpu", "slice_id": "45", "component": "upf", "scale_factor": 2}
    },
    {
        "intent_id": "SCAL-011",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Increase memory limit for all NFs in Slice ID 19 to 8 GB each.",
        "ground_truth": {"action": "SCALE", "target": "memory", "slice_id": "19", "memory_gb": 8, "scope": "all_nfs"}
    },
    {
        "intent_id": "SCAL-012",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Scale up the AUSF instances for Slice ID 83 from 3 to 6.",
        "ground_truth": {"action": "SCALE", "target": "ausf", "slice_id": "83", "replicas": 6}
    },
    {
        "intent_id": "SCAL-013",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Scale down Slice ID 44 bandwidth from 8 Gbps to 3 Gbps during off-peak.",
        "ground_truth": {"action": "SCALE_DOWN", "target": "bandwidth", "slice_id": "44", "bandwidth_gbps": 3, "condition": "off_peak"}
    },
    {
        "intent_id": "SCAL-014",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Add 5 more gNB connections to Slice ID 31.",
        "ground_truth": {"action": "SCALE_UP", "target": "gnb_connections", "slice_id": "31", "count": 5}
    },
    {
        "intent_id": "SCAL-015",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Increase the number of QoS flows for Slice ID 90 to 1000.",
        "ground_truth": {"action": "SCALE", "target": "qos_flows", "slice_id": "90", "max_flows": 1000}
    },
    {
        "intent_id": "SCAL-016",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Scale down Slice ID 11 to bare minimum — 1 UPF and 1 AMF only.",
        "ground_truth": {"action": "SCALE_DOWN", "target": "all", "slice_id": "11", "minimal": True}
    },
    {
        "intent_id": "SCAL-017",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Increase the packet processing rate for Slice ID 73 UPF to 100 Gbps throughput.",
        "ground_truth": {"action": "SCALE", "target": "upf_throughput", "slice_id": "73", "throughput_gbps": 100}
    },
    {
        "intent_id": "SCAL-018",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Add 3 edge UPFs to Slice ID 56 for lower latency at branch offices.",
        "ground_truth": {"action": "SCALE_UP", "target": "edge_upf", "slice_id": "56", "replicas": 3}
    },
    {
        "intent_id": "SCAL-019",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Scale the PCF instances of Slice ID 28 to 5 replicas.",
        "ground_truth": {"action": "SCALE", "target": "pcf", "slice_id": "28", "replicas": 5}
    },
    {
        "intent_id": "SCAL-020",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Boost the NSSF capacity of Slice ID 63 to handle 1 million subscribers.",
        "ground_truth": {"action": "SCALE", "target": "nssf", "slice_id": "63", "subscriber_capacity": 1000000}
    },
    {
        "intent_id": "SCAL-021",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Reduce Slice ID 77's NRF cache TTL from 60 seconds to 30 seconds.",
        "ground_truth": {"action": "SCALE", "target": "nrf_cache_ttl", "slice_id": "77", "ttl_seconds": 30}
    },
    {
        "intent_id": "SCAL-022",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Scale the UDR database for Slice ID 5 to handle 10 million subscriber records.",
        "ground_truth": {"action": "SCALE", "target": "udr", "slice_id": "5", "records_million": 10}
    },
    {
        "intent_id": "SCAL-023",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Increase the number of parallel N4 sessions for Slice ID 39 UPF to 5000.",
        "ground_truth": {"action": "SCALE", "target": "n4_sessions", "slice_id": "39", "max_parallel": 5000}
    },
    {
        "intent_id": "SCAL-024",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Double the egress bandwidth for Slice ID 86 to 20 Gbps.",
        "ground_truth": {"action": "SCALE", "target": "egress_bandwidth", "slice_id": "86", "bandwidth_gbps": 20}
    },
    {
        "intent_id": "SCAL-025",
        "domain": "Scaling Request",
        "complexity": "Simple",
        "input_text": "Scale up Slice ID 33 to support 500,000 concurrent PDU sessions.",
        "ground_truth": {"action": "SCALE", "target": "pdu_sessions", "slice_id": "33", "concurrent_max": 500000}
    },
]

SCAL_COMPLEX = [
    {
        "intent_id": "SCAL-026",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "The stadium is filling up for the concert. Gradually increase the bandwidth of the public Wi-Fi slice by 20% every hour for the next 4 hours.",
        "ground_truth": {"action": "SCALE", "target": "bandwidth", "strategy": "gradual", "step_percent": 20, "duration_hours": 4}
    },
    {
        "intent_id": "SCAL-027",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "CPU utilization on UPF pods of Slice 80 exceeded 85% for 5 consecutive minutes — trigger auto-scaling to add 2 replicas and increase the CPU limit by 50%.",
        "ground_truth": {"action": "SCALE", "trigger": "cpu_threshold", "target": "upf", "slice_id": "80", "replicas_increment": 2, "cpu_limit_multiplier": 1.5}
    },
    {
        "intent_id": "SCAL-028",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "We expect a 300% traffic surge during the New Year countdown for Slice 25. Pre-scale now and then auto-scale down after 6 hours.",
        "ground_truth": {"action": "SCALE", "target": "all", "slice_id": "25", "scale_factor": 4, "strategy": "pre_scale", "auto_scale_down_after_hours": 6}
    },
    {
        "intent_id": "SCAL-029",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Slice ID 41 is experiencing packet loss above 1% at the UPF. Add an intermediate UPF with load balancing and increase buffer size to 4 GB.",
        "ground_truth": {"action": "SCALE", "trigger": "packet_loss", "target": "upf", "slice_id": "41", "load_balancing": True, "buffer_gb": 4}
    },
    {
        "intent_id": "SCAL-030",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Implement elastic scaling for Slice ID 60's SMF — scale the number of SMF instances between 2 and 10 based on the number of active PDU sessions with a target of 5000 sessions per SMF.",
        "ground_truth": {"action": "SCALE", "target": "smf", "slice_id": "60", "strategy": "elastic", "min_replicas": 2, "max_replicas": 10, "sessions_per_smf": 5000}
    },
    {
        "intent_id": "SCAL-031",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Migrate Slice ID 72 from a centralized UPF to a distributed edge UPF architecture — deploy 6 edge UPFs across 3 regions while maintaining the central UPF as fallback.",
        "ground_truth": {"action": "SCALE", "target": "upf", "slice_id": "72", "strategy": "distribute", "edge_upfs": 6, "regions": 3, "central_fallback": True}
    },
    {
        "intent_id": "SCAL-032",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Slice ID 15's control plane is overloaded — offload 30% of signaling traffic to a secondary NRF cluster while scaling the primary NRF to handle remaining load.",
        "ground_truth": {"action": "SCALE", "trigger": "overload", "target": "control_plane", "slice_id": "15", "offload_percent": 30, "primary_scale": True}
    },
    {
        "intent_id": "SCAL-033",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Apply a time-based scaling schedule for Slice ID 92: Monday-Friday 8AM-8PM scale to 100% capacity, 8PM-8AM scale to 25%, weekends scale to 50%.",
        "ground_truth": {"action": "SCALE", "target": "all", "slice_id": "92", "strategy": "time_based", "schedule": {"weekday_peak": 100, "weekday_offpeak": 25, "weekend": 50}}
    },
    {
        "intent_id": "SCAL-034",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "For Slice ID 3, scale the UPF externally while performing a rolling upgrade — maintain 80% capacity during the upgrade window.",
        "ground_truth": {"action": "SCALE", "target": "upf", "slice_id": "3", "strategy": "rolling_upgrade", "maintain_capacity_pct": 80}
    },
    {
        "intent_id": "SCAL-035",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Slice ID 48 is approaching its guaranteed bit rate limit — burst scale to 150% of GBR for the next 15 minutes with excess usage billed at premium rate.",
        "ground_truth": {"action": "SCALE", "trigger": "gbr_limit", "target": "bandwidth", "slice_id": "48", "burst_percent": 150, "burst_duration_min": 15, "billing": "premium"}
    },
    {
        "intent_id": "SCAL-036",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Implement geo-aware scaling for Slice ID 17 — deploy NF instances closest to user clusters in Asia-Pacific while scaling down unused instances in Europe.",
        "ground_truth": {"action": "SCALE", "target": "all", "slice_id": "17", "strategy": "geo_aware", "scale_up_region": "asia_pacific", "scale_down_region": "europe"}
    },
    {
        "intent_id": "SCAL-037",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "During the DDOS mitigation for Slice ID 69, scale up the NRF and security gateway instances by 300% while throttling non-essential traffic to 20%.",
        "ground_truth": {"action": "SCALE", "trigger": "ddos", "targets": [{"name": "security_gateway", "scale_factor": 3}, {"name": "nrf", "scale_factor": 3}], "throttle_non_essential_pct": 20}
    },
    {
        "intent_id": "SCAL-038",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Slice ID 82 is serving a multi-tenant environment. Scale Tenant A to 200% bandwidth, Tenant B to 50%, and keep Tenant C unchanged — within the slice's total capacity.",
        "ground_truth": {"action": "SCALE", "target": "tenant_bandwidth", "slice_id": "82", "tenants": {"A": 200, "B": 50, "C": 100}}
    },
    {
        "intent_id": "SCAL-039",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Energy prices are spiking — shift Slice ID 6's UPF processing to edge nodes powered by solar while scaling down the grid-powered central UPF.",
        "ground_truth": {"action": "SCALE", "trigger": "energy_cost", "target": "upf", "slice_id": "6", "strategy": "green_shift", "scale_up": "solar_edge", "scale_down": "grid_central"}
    },
    {
        "intent_id": "SCAL-040",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Set up predictive scaling for Slice ID 99 based on historical traffic patterns — the ML model predicts 3x traffic every Friday 7PM-11PM.",
        "ground_truth": {"action": "SCALE", "target": "all", "slice_id": "99", "strategy": "predictive_ml", "predicted_multiplier": 3, "schedule": "Friday_19-23"}
    },
    {
        "intent_id": "SCAL-041",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Implement a scaling cascade for Slice ID 53 — when UPF reaches 70% capacity scale it up first, then if AMF signaling reaches 80% scale AMF too.",
        "ground_truth": {"action": "SCALE", "target": "cascade", "slice_id": "53", "cascade_rules": [{"condition": "upf_capacity > 70%", "action": "scale_upf"}, {"condition": "amf_signaling > 80%", "action": "scale_amf"}]}
    },
    {
        "intent_id": "SCAL-042",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Scale Slice ID 26 during a content delivery network cache refresh — 5-minute aggressive bandwidth boost to 50 Gbps, then back to normal.",
        "ground_truth": {"action": "SCALE", "trigger": "cdn_refresh", "target": "bandwidth", "slice_id": "26", "bandwidth_gbps": 50, "duration_min": 5, "strategy": "burst"}
    },
    {
        "intent_id": "SCAL-043",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Slice ID 11 needs cross-slice resource borrowing — temporarily scale up using idle resources from Slice IDs 12 and 13, return them after 2 hours.",
        "ground_truth": {"action": "SCALE", "target": "cross_slice", "slice_id": "11", "borrow_from": ["12", "13"], "duration_hours": 2}
    },
    {
        "intent_id": "SCAL-044",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Apply a canary scaling strategy for Slice ID 35 — scale up only 10% of new SMF replicas first, monitor for 5 minutes, then proceed to full scale if no errors.",
        "ground_truth": {"action": "SCALE", "target": "smf", "slice_id": "35", "strategy": "canary", "canary_percent": 10, "monitor_min": 5, "condition": "no_errors"}
    },
    {
        "intent_id": "SCAL-045",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "For Slice ID 78's URLLC traffic, implement always-on over-provisioning at 150% of peak demand for the next 24 hours during a major industrial operation.",
        "ground_truth": {"action": "SCALE", "target": "all", "slice_id": "78", "overprovision_percent": 150, "duration_hours": 24, "slice_type": "URLLC"}
    },
    {
        "intent_id": "SCAL-046",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Slice ID 2 serves smart meters. Scale down during summer months (June-August) when solar power self-supply reduces metering needs to 30% of normal.",
        "ground_truth": {"action": "SCALE_DOWN", "target": "all", "slice_id": "2", "scale_percent": 30, "months": ["June", "July", "August"], "reason": "solar_self_supply"}
    },
    {
        "intent_id": "SCAL-047",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Container image for UPF v2.3 is available — scale Slice ID 70's UPF fleet with a blue-green deployment, switching all traffic to new version after health checks pass.",
        "ground_truth": {"action": "SCALE", "target": "upf", "slice_id": "70", "strategy": "blue_green", "new_version": "v2.3", "switch_condition": "health_check_pass"}
    },
    {
        "intent_id": "SCAL-048",
        "domain": "Scaling Request",
        "complexity": "Complex",
        "input_text": "Slice ID 9 is a network slicing trial for a new customer. Scale with a quota-based approach: start at 10% capacity, increase to 50% after 1 week of stable operation, 100% after 1 month.",
        "ground_truth": {"action": "SCALE", "target": "all", "slice_id": "9", "strategy": "quota_based", "stages": [{"pct": 10, "trigger": "start"}, {"pct": 50, "trigger": "1_week_stable"}, {"pct": 100, "trigger": "1_month"}]}
    },
]

SCAL_AMBIGUOUS = [
    {
        "intent_id": "SCAL-049",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "The automated factory floor is reporting high packet drops, optimize the network to fix this latency issue.",
        "ground_truth": {"action": "SCALE", "target": "bandwidth", "slice_type": "URLLC", "optimize": "latency"}
    },
    {
        "intent_id": "SCAL-050",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "The network feels slow today for my video calls — can you boost it?",
        "ground_truth": {"action": "SCALE", "target": "bandwidth", "service": "video_call", "optimize": "quality"}
    },
    {
        "intent_id": "SCAL-051",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "We're going viral on social media — our app is going to get hammered.",
        "ground_truth": {"action": "SCALE", "target": "all", "strategy": "burst", "scale_factor": 5}
    },
    {
        "intent_id": "SCAL-052",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "The factory robots are stuttering — make the network more responsive.",
        "ground_truth": {"action": "SCALE", "target": "all", "slice_type": "URLLC", "optimize": "latency", "priority": "high"}
    },
    {
        "intent_id": "SCAL-053",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "Black Friday is coming — our e-commerce platform needs all the network it can get.",
        "ground_truth": {"action": "SCALE", "target": "all", "strategy": "scheduled_surge", "event": "black_friday", "scale_factor": 10}
    },
    {
        "intent_id": "SCAL-054",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "The IoT sensors are flooding the network with data — can you handle it better?",
        "ground_truth": {"action": "SCALE", "target": "network_capacity", "slice_type": "mMTC", "optimize": "throughput"}
    },
    {
        "intent_id": "SCAL-055",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "My video surveillance feed keeps buffering — need more bandwidth or something.",
        "ground_truth": {"action": "SCALE", "target": "bandwidth", "slice_type": "eMBB", "bandwidth_gbps": 2}
    },
    {
        "intent_id": "SCAL-056",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "The core is running hot — let's redistribute some load before it crashes.",
        "ground_truth": {"action": "SCALE", "target": "load_distribution", "strategy": "rebalance", "priority": "high"}
    },
    {
        "intent_id": "SCAL-057",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "We're adding 20 more stores to our retail network — make sure the system can handle them.",
        "ground_truth": {"action": "SCALE", "target": "all", "scale_factor": 1.5, "new_locations": 20}
    },
    {
        "intent_id": "SCAL-058",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "The weather forecast says a typhoon is coming — make our emergency network ready for anything.",
        "ground_truth": {"action": "SCALE", "target": "all", "slice_type": "URLLC", "priority": "emergency", "scale_factor": 3}
    },
    {
        "intent_id": "SCAL-059",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "Students are back from break — the campus network is going to be busy again.",
        "ground_truth": {"action": "SCALE", "target": "all", "strategy": "scheduled", "event": "semester_start", "scale_factor": 2}
    },
    {
        "intent_id": "SCAL-060",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "Our AI inference pipeline is bottlenecked on network I/O — can you clear the pipe?",
        "ground_truth": {"action": "SCALE", "target": "bandwidth", "optimize": "throughput", "application": "ai_inference"}
    },
    {
        "intent_id": "SCAL-061",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "The stadium just scored a goal — traffic is exploding for the next 10 minutes!",
        "ground_truth": {"action": "SCALE", "target": "bandwidth", "strategy": "instant_burst", "scale_factor": 8, "duration_estimate_min": 10}
    },
    {
        "intent_id": "SCAL-062",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "We're decommissioning the old factory wing — scale down whatever was serving that area.",
        "ground_truth": {"action": "SCALE_DOWN", "target": "all", "reason": "decommission", "location": "factory_wing_old"}
    },
    {
        "intent_id": "SCAL-063",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "Electricity prices just doubled — can we save some power on the network side?",
        "ground_truth": {"action": "SCALE_DOWN", "target": "all", "strategy": "energy_saving", "power_reduction_pct": 40}
    },
    {
        "intent_id": "SCAL-064",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "The VR training simulator needs zero lag — do whatever it takes.",
        "ground_truth": {"action": "SCALE", "target": "all", "slice_type": "URLLC", "latency_ms": 1, "priority": "maximum"}
    },
    {
        "intent_id": "SCAL-065",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "Our drone delivery service is expanding to a new city — triple the capacity.",
        "ground_truth": {"action": "SCALE", "target": "all", "scale_factor": 3, "reason": "city_expansion"}
    },
    {
        "intent_id": "SCAL-066",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "The medical imaging department says transfers are taking forever — speed it up.",
        "ground_truth": {"action": "SCALE", "target": "bandwidth", "slice_type": "eMBB", "bandwidth_gbps": 10, "sector": "healthcare"}
    },
    {
        "intent_id": "SCAL-067",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "Something's wrong with the smart grid — meters are reporting with huge delays.",
        "ground_truth": {"action": "SCALE", "target": "all", "slice_type": "mMTC", "optimize": "latency"}
    },
    {
        "intent_id": "SCAL-068",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "Nighttime now — most factories are idle, let's save resources until morning.",
        "ground_truth": {"action": "SCALE_DOWN", "target": "all", "strategy": "time_of_day", "scale_percent": 25, "until": "morning"}
    },
    {
        "intent_id": "SCAL-069",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "The stock market just crashed — our trading platform needs maximum network priority RIGHT NOW.",
        "ground_truth": {"action": "SCALE", "target": "all", "priority": "maximum", "scale_factor": 10, "sector": "finance"}
    },
    {
        "intent_id": "SCAL-070",
        "domain": "Scaling Request",
        "complexity": "Ambiguous",
        "input_text": "The FIFA World Cup final is streaming — we need to handle more viewers than ever before.",
        "ground_truth": {"action": "SCALE", "target": "all", "strategy": "planned_mega_event", "scale_factor": 20, "slice_type": "eMBB"}
    },
]

# =============================================================================
# DOMAIN 3: Conflict Resolution (60 intents: 20 Simple + 20 Complex + 20 Ambiguous)
# =============================================================================

CONF_SIMPLE = [
    {
        "intent_id": "CONF-001",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Disable Slice ID 12 to free up memory for the higher-priority Slice ID 8.",
        "ground_truth": {"action": "RESOLVE", "terminate_slice": "12", "prioritize_slice": "8"}
    },
    {
        "intent_id": "CONF-002",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Terminate Slice ID 45 because Slice ID 10 needs its allocated spectrum immediately.",
        "ground_truth": {"action": "RESOLVE", "terminate_slice": "45", "prioritize_slice": "10"}
    },
    {
        "intent_id": "CONF-003",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Preempt all non-critical slices to give Slice ID 7 exclusive access to the UPF resources.",
        "ground_truth": {"action": "PREEMPT", "target": "non_critical_slices", "priority_to": "7", "resource": "upf"}
    },
    {
        "intent_id": "CONF-004",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Slice ID 99 has violated its SLA bandwidth cap — throttle it back to the agreed 2 Gbps.",
        "ground_truth": {"action": "RESOLVE", "target_slice": "99", "action_type": "throttle", "bandwidth_gbps": 2, "reason": "sla_violation"}
    },
    {
        "intent_id": "CONF-005",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "2 critical slices (ID 3 and ID 5) are competing for the same GPU node — allocate it to ID 5 based on higher priority tag.",
        "ground_truth": {"action": "RESOLVE", "conflict_between": ["3", "5"], "resource": "gpu_node", "allocate_to": "5", "criterion": "priority_tag"}
    },
    {
        "intent_id": "CONF-006",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Release Slice ID 22's reserved but unused bandwidth back to the shared pool.",
        "ground_truth": {"action": "REALLOCATE", "target_slice": "22", "resource": "bandwidth", "destination": "shared_pool"}
    },
    {
        "intent_id": "CONF-007",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Evict the lowest-priority tenant from Slice ID 88 to make room for a new enterprise customer.",
        "ground_truth": {"action": "RESOLVE", "target_slice": "88", "action_type": "evict_lowest_priority_tenant", "reason": "new_enterprise"}
    },
    {
        "intent_id": "CONF-008",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Slice IDs 30 and 31 have overlapping frequency allocations — assign Slice 30 the upper band, Slice 31 the lower band.",
        "ground_truth": {"action": "RESOLVE", "conflict_type": "frequency_overlap", "slices": {"30": "upper_band", "31": "lower_band"}}
    },
    {
        "intent_id": "CONF-009",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Slice ID 55 is causing interference with neighboring Slice 56 — isolate Slice 55 with a guard band.",
        "ground_truth": {"action": "RESOLVE", "conflict_type": "interference", "source_slice": "55", "affected_slice": "56", "solution": "guard_band"}
    },
    {
        "intent_id": "CONF-010",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Pause all mMTC slices temporarily to free control plane capacity for an emergency URLLC deployment.",
        "ground_truth": {"action": "PREEMPT", "target": "all_mmtc", "priority_to": "emergency_urllc", "resource": "control_plane"}
    },
    {
        "intent_id": "CONF-011",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "De-prioritize entertainment slices during an ongoing natural disaster to ensure emergency services get full capacity.",
        "ground_truth": {"action": "PREEMPT", "target": "entertainment_slices", "priority_to": "emergency_services", "reason": "natural_disaster"}
    },
    {
        "intent_id": "CONF-012",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Slice ID 18 is using 3x its allocated memory — force it back to the allocation limit.",
        "ground_truth": {"action": "RESOLVE", "target_slice": "18", "resource": "memory", "action_type": "enforce_limit"}
    },
    {
        "intent_id": "CONF-013",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Two network slices request the same VLAN ID 100 — reassign Slice ID 40 to VLAN 200.",
        "ground_truth": {"action": "RESOLVE", "conflict_type": "vlan_id", "resolve_by": "reassign", "slice_id": "40", "new_vlan": 200}
    },
    {
        "intent_id": "CONF-014",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Slice ID 66 must yield its dedicated UPF to Slice ID 1 — redirect its traffic through the shared UPF pool.",
        "ground_truth": {"action": "RESOLVE", "yielding_slice": "66", "resource": "dedicated_upf", "receiving_slice": "1", "reroute": "shared_pool"}
    },
    {
        "intent_id": "CONF-015",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Terminate all expired trial slices — they're consuming resources without valid SLA.",
        "ground_truth": {"action": "TERMINATE", "target": "expired_trial_slices", "reason": "no_valid_sla"}
    },
    {
        "intent_id": "CONF-016",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Apply strict network isolation between Slice 23 (banking) and Slice 24 (public Wi-Fi) — they must not share any infrastructure.",
        "ground_truth": {"action": "RESOLVE", "conflict_type": "isolation", "slices": ["23", "24"], "solution": "strict_isolation"}
    },
    {
        "intent_id": "CONF-017",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Slice ID 8 has been idle for 30 minutes with no active PDU sessions — terminate it to reclaim resources.",
        "ground_truth": {"action": "TERMINATE", "target_slice": "8", "reason": "idle_30min"}
    },
    {
        "intent_id": "CONF-018",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Preempt Slice ID 33's non-guaranteed-bit-rate traffic to accommodate Slice ID 34's GBR requirement.",
        "ground_truth": {"action": "PREEMPT", "source_slice": "33", "traffic_type": "non_gbr", "destination_slice": "34", "requirement": "gbr"}
    },
    {
        "intent_id": "CONF-019",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Resolve the IP address collision between Slice 50 and Slice 51 subnet allocation.",
        "ground_truth": {"action": "RESOLVE", "conflict_type": "ip_collision", "slices": ["50", "51"], "solution": "subnet_reassignment"}
    },
    {
        "intent_id": "CONF-020",
        "domain": "Conflict Resolution",
        "complexity": "Simple",
        "input_text": "Suspend Slice ID 77 for 2 hours as penalty for exceeding the fair-use policy limit.",
        "ground_truth": {"action": "RESOLVE", "target_slice": "77", "action_type": "suspend", "duration_hours": 2, "reason": "fair_use_violation"}
    },
]

CONF_COMPLEX = [
    {
        "intent_id": "CONF-021",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "The core network CPU is at 98%. We have an incoming critical request for a smart grid emergency protocol, but the existing eMBB slices are consuming all bandwidth. Resolve this.",
        "ground_truth": {"action": "RESOLVE", "priority": "smart_grid", "preempt": "eMBB"}
    },
    {
        "intent_id": "CONF-022",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "Three slices are competing for limited RAN resources: Slice 10 (URLLC, priority 10), Slice 20 (eMBB, priority 5), Slice 30 (mMTC, priority 3). Total capacity is 100 units. Allocate proportionally but ensure URLLC gets minimum 40 units.",
        "ground_truth": {"action": "REALLOCATE", "resource": "ran", "total_capacity": 100, "allocations": {"10": 40, "20": 40, "30": 20}, "priority_based": True, "urllc_minimum": 40}
    },
    {
        "intent_id": "CONF-023",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "A natural disaster just occurred. Ensure emergency responder communications have absolute priority over regular mobile users, even if the system is overloaded.",
        "ground_truth": {"action": "RESOLVE", "priority": "emergency_responders", "preempt": "regular_users", "slice_type": "URLLC"}
    },
    {
        "intent_id": "CONF-024",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "Load balancer detected uneven traffic: Slice 60 has 4x the load of Slice 70 on the same UPF pool. Rebalance by migrating 25% of Slice 60 sessions to Slice 70's capacity.",
        "ground_truth": {"action": "REALLOCATE", "target": "upf_pool", "source_slice": "60", "destination_slice": "70", "migrate_percent": 25, "reason": "load_imbalance"}
    },
    {
        "intent_id": "CONF-025",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "Slice ID 38 has a strict SLA penalty clause — it's consuming 120% of its contract. Rather than hard-terminate, degrade its non-essential traffic to best-effort and notify the tenant.",
        "ground_truth": {"action": "RESOLVE", "target_slice": "38", "action_type": "degrade_non_essential", "notify": True, "reason": "sla_overuse_120%"}
    },
    {
        "intent_id": "CONF-026",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "A rogue mMTC slice is flooding the AMF with 10x normal registration requests — rate-limit it and isolate its signaling to prevent cascading failure to other slices.",
        "ground_truth": {"action": "RESOLVE", "target_slice": "rogue_mmtc", "action_type": "rate_limit_and_isolate", "target_nf": "amf", "rate_limit_factor": 0.1}
    },
    {
        "intent_id": "CONF-027",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "Maintenance window for the transport network affects Slice 5, 7, and 9. Reroute Slice 5 (critical healthcare) to backup path with full QoS, degrade Slice 7 and 9 to reduced capacity during the window.",
        "ground_truth": {"action": "RESOLVE", "trigger": "transport_maintenance", "affected_slices": ["5", "7", "9"], "actions": {"5": "reroute_full_qos", "7": "degrade_reduced", "9": "degrade_reduced"}}
    },
    {
        "intent_id": "CONF-028",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "A new URLLC slice request conflicts with existing eMBB allocation on the same spectrum. Apply spectrum refarming — shift eMBB to a higher frequency band and give the lower band to URLLC for its latency requirements.",
        "ground_truth": {"action": "RESOLVE", "conflict_type": "spectrum", "solution": "refarming", "emmb_action": "shift_to_higher_band", "urllc_action": "allocate_lower_band"}
    },
    {
        "intent_id": "CONF-029",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "Slice ID 14 and Slice ID 16 share a data network name (DNN). Implement DNN aliasing — both can use 'enterprise.local' but with separate traffic steering policies based on slice ID.",
        "ground_truth": {"action": "RESOLVE", "conflict_type": "dnn_collision", "dnn": "enterprise.local", "solution": "alias_with_traffic_steering", "slices": ["14", "16"]}
    },
    {
        "intent_id": "CONF-030",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "The energy budget for the network node is exceeded. Selectively power-down redundant NFs in non-critical slices while maintaining full operation for emergency and healthcare slices.",
        "ground_truth": {"action": "RESOLVE", "trigger": "energy_budget_exceeded", "action_type": "selective_power_down", "protect_slices": ["emergency", "healthcare"], "target_slices": "non_critical"}
    },
    {
        "intent_id": "CONF-031",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "A security breach is detected on Slice ID 42. Immediately isolate it from the rest of the network, quarantine its traffic, and spin up a clean replacement slice with the same configuration.",
        "ground_truth": {"action": "RESOLVE", "trigger": "security_breach", "target_slice": "42", "actions": ["isolate", "quarantine", "replace_with_clean"]}
    },
    {
        "intent_id": "CONF-032",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "The NRF is hitting a scaling limit — 6 slices are waiting for NF discovery. Implement a priority queue: admit URLLC slices first (Slices 3, 6), then eMBB (Slices 10, 15), then mMTC (Slices 20, 25).",
        "ground_truth": {"action": "RESOLVE", "trigger": "nrf_scaling_limit", "queue": {"priority_1": ["3", "6"], "priority_2": ["10", "15"], "priority_3": ["20", "25"]}, "strategy": "priority_queue"}
    },
    {
        "intent_id": "CONF-033",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "Slice 80 is a multi-operator shared slice. Operator A is exceeding its 60% quota — enforce quota, redirect overflow to Operator A's dedicated fallback slice, and compensate Operator B for the brief overage period.",
        "ground_truth": {"action": "RESOLVE", "target_slice": "80", "multi_operator": True, "violator": "Operator_A", "action": "enforce_quota_with_compensation", "quota_pct": 60}
    },
    {
        "intent_id": "CONF-034",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "A fiber cut has severed the primary path for Slice 55 and Slice 56. Slice 55 (URLLC) gets the sole backup path. Slice 56 (eMBB) must wait or use satellite fallback with degraded performance.",
        "ground_truth": {"action": "RESOLVE", "trigger": "fiber_cut", "slices": {"55": "backup_path_exclusive", "56": "wait_or_satellite_degraded"}}
    },
    {
        "intent_id": "CONF-035",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "A UE is simultaneously requesting service from Slice 28 (consumer, low priority) and Slice 29 (enterprise, high priority) — resolve the subscriber profile conflict by honoring the enterprise slice.",
        "ground_truth": {"action": "RESOLVE", "conflict_type": "subscriber_profile", "ue_requesting_slices": ["28", "29"], "resolution": "honor_enterprise"}
    },
    {
        "intent_id": "CONF-036",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "The orchestrator detects that Slice 59's UPF is silently dropping packets while still reporting healthy. Perform a preemptive failover to the standby UPF and flag the primary for investigation.",
        "ground_truth": {"action": "RESOLVE", "trigger": "silent_failure", "target_slice": "59", "component": "upf", "action": "preemptive_failover", "flag_for_investigation": True}
    },
    {
        "intent_id": "CONF-037",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "The SLA for Slice ID 33 mandates 99.999% availability, but a planned core upgrade requires 30 minutes of downtime. Schedule it for 3 AM, notify the tenant 48 hours in advance, and provide a temporary fallback slice at 50% capacity free of charge.",
        "ground_truth": {"action": "RESOLVE", "target_slice": "33", "trigger": "planned_upgrade", "downtime_min": 30, "schedule": "3AM", "notification_hours": 48, "fallback": {"capacity_pct": 50, "charge": "free"}}
    },
    {
        "intent_id": "CONF-038",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "All three SMF instances for Slice 91 are in the same failure domain (rack). Redistribute by migrating one SMF to rack B, one to rack C, keeping one in rack A — but maintain session continuity.",
        "ground_truth": {"action": "RESOLVE", "target_slice": "91", "component": "smf", "reason": "failure_domain", "action": "redistribute", "target_racks": ["A", "B", "C"], "maintain_session": True}
    },
    {
        "intent_id": "CONF-039",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "The NSSF selection algorithm is creating an imbalance — slice 65 is pinned to congested AMF set 1 while AMF set 2 is idle. Update NSSF policy to prefer AMF set 2 for slice 65.",
        "ground_truth": {"action": "RESOLVE", "trigger": "nssf_imbalance", "target_slice": "65", "action": "update_nssf_policy", "prefer_amf_set": "2"}
    },
    {
        "intent_id": "CONF-040",
        "domain": "Conflict Resolution",
        "complexity": "Complex",
        "input_text": "A regulatory order requires that all government traffic on Slice 74 be routed exclusively through domestic network nodes. Reconfigure routing to exclude all international nodes and verify data sovereignty compliance.",
        "ground_truth": {"action": "RESOLVE", "target_slice": "74", "trigger": "regulatory", "requirement": "data_sovereignty", "action": "route_domestic_only"}
    },
]

CONF_AMBIGUOUS = [
    {
        "intent_id": "CONF-041",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "There's a fire at the data center — save what you can!",
        "ground_truth": {"action": "RESOLVE", "trigger": "data_center_fire", "priority": "evacuate_critical_workloads", "strategy": "emergency_failover"}
    },
    {
        "intent_id": "CONF-042",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "The network is fighting itself — some slices are eating all the resources and others are starving.",
        "ground_truth": {"action": "RESOLVE", "trigger": "resource_starvation", "action": "enforce_fairness_policy"}
    },
    {
        "intent_id": "CONF-043",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "Our biggest customer is complaining about slow service — fix it even if it means hurting smaller customers.",
        "ground_truth": {"action": "RESOLVE", "priority": "vip_customer", "action": "degrade_small_customers"}
    },
    {
        "intent_id": "CONF-044",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "The president is about to give a live address — make sure the broadcast network gets everything it needs.",
        "ground_truth": {"action": "RESOLVE", "priority": "presidential_broadcast", "action": "allocate_all_resources", "preempt": "non_broadcast"}
    },
    {
        "intent_id": "CONF-045",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "Something weird is happening — traffic patterns don't look normal and latencies are spiking everywhere.",
        "ground_truth": {"action": "RESOLVE", "trigger": "anomaly_detected", "action": "investigate_and_mitigate", "strategy": "root_cause_analysis"}
    },
    {
        "intent_id": "CONF-046",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "We're running out of IPv4 addresses in this segment — need to make room for the new customer trial.",
        "ground_truth": {"action": "RESOLVE", "trigger": "ip_exhaustion", "action": "reclaim_unused_or_nat"}
    },
    {
        "intent_id": "CONF-047",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "The self-driving car network absolutely cannot fail — if it means shutting down the entertainment network, do it.",
        "ground_truth": {"action": "RESOLVE", "priority": "autonomous_vehicles", "preempt": "entertainment", "slice_type": "URLLC"}
    },
    {
        "intent_id": "CONF-048",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "I don't care how, but reduce the overall network congestion — it's affecting everyone.",
        "ground_truth": {"action": "RESOLVE", "trigger": "network_congestion", "action": "global_congestion_control"}
    },
    {
        "intent_id": "CONF-049",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "A competitor is launching their service today — we need to show our network is superior. Prioritize our demo showcase.",
        "ground_truth": {"action": "RESOLVE", "priority": "demo_showcase", "action": "maximum_qos", "reason": "competitive"}
    },
    {
        "intent_id": "CONF-050",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "The hospital's tele-ICU just went offline — this is life or death, drop everything else.",
        "ground_truth": {"action": "RESOLVE", "priority": "tele_icu", "action": "emergency_restore", "preempt": "all_non_medical"}
    },
    {
        "intent_id": "CONF-051",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "Too many IoT devices are trying to connect at once — it's a connection storm.",
        "ground_truth": {"action": "RESOLVE", "trigger": "connection_storm", "action": "rate_limit_and_stagger", "slice_type": "mMTC"}
    },
    {
        "intent_id": "CONF-052",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "The mobile payment system is lagging — millions in transactions are being delayed.",
        "ground_truth": {"action": "RESOLVE", "priority": "payment_system", "action": "prioritize_financial_traffic"}
    },
    {
        "intent_id": "CONF-053",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "A celebrity just arrived at the airport — the fan live-streaming is crushing the cell tower.",
        "ground_truth": {"action": "RESOLVE", "trigger": "localized_surge", "action": "deploy_temporary_capacity"}
    },
    {
        "intent_id": "CONF-054",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "The energy grid is unstable — network infrastructure must reduce power consumption drastically or risk blackout.",
        "ground_truth": {"action": "RESOLVE", "trigger": "power_grid_emergency", "action": "emergency_power_reduction"}
    },
    {
        "intent_id": "CONF-055",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "A zero-day vulnerability was just announced — secure all network slices before it's exploited.",
        "ground_truth": {"action": "RESOLVE", "trigger": "zero_day", "action": "emergency_security_lockdown"}
    },
    {
        "intent_id": "CONF-056",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "The submarine cable to Europe is damaged — reroute what you can through the Pacific.",
        "ground_truth": {"action": "RESOLVE", "trigger": "submarine_cable_damage", "action": "reroute_pacific", "affected_route": "europe"}
    },
    {
        "intent_id": "CONF-057",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "Our cloud provider just had a major outage — migrate all dependent slices to the backup provider.",
        "ground_truth": {"action": "RESOLVE", "trigger": "cloud_provider_outage", "action": "migrate_to_backup"}
    },
    {
        "intent_id": "CONF-058",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "The AI model training job is hogging all the GPU nodes — push it to low priority, researchers can wait.",
        "ground_truth": {"action": "RESOLVE", "target": "ai_training", "action": "lower_priority"}
    },
    {
        "intent_id": "CONF-059",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "A cyberattack is targeting our 5G core — isolate affected NFs and keep the network alive.",
        "ground_truth": {"action": "RESOLVE", "trigger": "cyberattack", "action": "isolate_and_protect", "target": "5g_core"}
    },
    {
        "intent_id": "CONF-060",
        "domain": "Conflict Resolution",
        "complexity": "Ambiguous",
        "input_text": "The World Cup penalty shootout is happening — the whole country is streaming. Don't let it crash.",
        "ground_truth": {"action": "RESOLVE", "trigger": "national_streaming_event", "action": "maximum_overprovision", "slice_type": "eMBB"}
    },
]


# =============================================================================
# Assemble and validate
# =============================================================================

all_intents = (
    PROV_SIMPLE + PROV_COMPLEX + PROV_AMBIGUOUS +
    SCAL_SIMPLE + SCAL_COMPLEX + SCAL_AMBIGUOUS +
    CONF_SIMPLE + CONF_COMPLEX + CONF_AMBIGUOUS
)

def validate_intent(item, idx):
    """Validate a single intent entry."""
    assert "intent_id" in item, f"Missing intent_id at index {idx}"
    assert "domain" in item, f"Missing domain at index {idx}"
    assert item["domain"] in ["Slicing Provisioning", "Scaling Request", "Conflict Resolution"], f"Invalid domain at {item['intent_id']}"
    assert "complexity" in item, f"Missing complexity at {item['intent_id']}"
    assert item["complexity"] in ["Simple", "Complex", "Ambiguous"], f"Invalid complexity at {item['intent_id']}"
    assert "input_text" in item and item["input_text"], f"Empty input_text at {item['intent_id']}"
    assert "ground_truth" in item, f"Missing ground_truth at {item['intent_id']}"
    assert "action" in item["ground_truth"], f"Missing action in ground_truth at {item['intent_id']}"

def main():
    # Validate all intents
    for i, item in enumerate(all_intents):
        validate_intent(item, i)
    
    # Check for duplicate intent_ids
    ids = [item["intent_id"] for item in all_intents]
    dupes = [id for id in ids if ids.count(id) > 1]
    if dupes:
        raise ValueError(f"Duplicate intent_ids: {set(dupes)}")
    
    # Count by domain and complexity
    from collections import Counter
    domain_counts = Counter(item["domain"] for item in all_intents)
    complexity_counts = Counter((item["domain"], item["complexity"]) for item in all_intents)
    
    print("=" * 60)
    print("Dataset Generation Summary")
    print("=" * 60)
    print(f"\nTotal intents: {len(all_intents)}")
    print(f"\nDomain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")
    
    print(f"\nComplexity distribution:")
    for (domain, complexity), count in sorted(complexity_counts.items()):
        print(f"  {domain} / {complexity}: {count}")
    
    # Write JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in all_intents:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Written to: {OUTPUT_PATH}")
    print(f"   File size: {OUTPUT_PATH.stat().st_size:,} bytes")
    
    # Print sample
    print(f"\nSample entry (first):")
    print(json.dumps(all_intents[0], indent=2, ensure_ascii=False))
    print(f"\nSample entry (last):")
    print(json.dumps(all_intents[-1], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
