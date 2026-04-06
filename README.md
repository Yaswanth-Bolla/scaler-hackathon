---
title: SRE Incident Response Simulator
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

# 🚨 SRE Incident Response Simulator

An OpenEnv environment where AI agents must diagnose and remediate production incidents across a simulated microservices architecture.

## Why This Environment Matters

This is a **POMDP** (Partially Observable Markov Decision Process). The agent never sees the root cause — it sees _symptoms_: climbing memory metrics, cascading error logs, firing alerts. It must gather evidence, form hypotheses, and act — exactly like a real SRE at 3 AM.

| Dimension | Detail |
|-----------|--------|
| **Observation** | Alerts, metric timeseries, structured logs, dependency graphs, deploy history |
| **Action space** | 10 hierarchical action types × 7 target services = rich combinatorics |
| **Difficulty** | Easy (single-service leak) → Medium (cascading failure) → Hard (distributed deadlock) |
| **Reward** | Oracle-shaped per-step signal for training + oracle-independent grader for evaluation |
| **Realism** | Reactive simulation — memory climbs over time, cascades propagate, restarts don't fix root causes |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    SIMULATED INFRASTRUCTURE                      │
│                                                                  │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐ │
│   │ API GW  │────►│ Auth    │────►│ Orders  │────►│ Payment │ │
│   └────┬────┘     └─────────┘     └────┬────┘     └────┬────┘ │
│        │                               │                │      │
│        ▼                               ▼                ▼      │
│   ┌─────────┐                    ┌─────────┐     ┌─────────┐  │
│   │ Cache   │                    │   DB    │     │ Queue   │  │
│   └─────────┘                    └─────────┘     └─────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

7 services with reactive metrics, logs, alerts, and dependency-aware cascade propagation.

---

## Action Space (Hierarchical)

### Level 1: Action Type

| Action | Category | Description |
|--------|----------|------------|
| `view_alerts` | Diagnostic | See all firing alerts |
| `query_logs` | Diagnostic | Query service logs (with level/keyword filters) |
| `check_metrics` | Diagnostic | Get 30-minute metric timeseries |
| `check_dependencies` | Diagnostic | View upstream/downstream dependency map |
| `check_deploy_history` | Diagnostic | Recent deploys for a service |
| `run_health_check` | Diagnostic | Ping a service for status |
| `restart_service` | Remediation | Restart (fixes symptoms temporarily, not root cause) |
| `rollback_deploy` | Remediation | Rollback to previous deploy version |
| `scale_service` | Remediation | Scale replicas up/down |
| `declare_root_cause` | Terminal | Submit diagnosis — ends episode |

### Level 2: Target Service + Parameters
Targeted actions require `target_service` from: `api_gateway`, `auth`, `orders`, `payment`, `cache`, `database`, `queue`.

### Action Masking
The observation includes `valid_actions[]` — illegal actions (e.g., rollback on a service with no deploy history) are rejected with a penalty.

---

## Observation Space (POMDP)

The agent **never** sees: `fault_type`, `is_bad` deploy flag, or internal simulation state.

It **does** see:
- **Incident summary** and severity
- **Service statuses** (healthy/degraded/down)
- **Active alert count**
- **Action result** (data from the last action: logs, metrics, alerts, etc.)
- **Valid actions** (action mask)
- **Time elapsed / budget** (SLA pressure)
- **Cumulative reward** and step count

---

## Tasks

| Task | Description | Difficulty | Root Cause |
|------|-------------|-----------|------------|
| `memory_leak` | Orders service OOM from bad deploy | Easy | Rollback orders deploy v2.3.1 |
| `cascading_failure` | Auth config change cascading to API GW + orders | Medium | Rollback auth deploy, restart dependents |
| `distributed_deadlock` | Payment retry change creates circular wait | Hard | Rollback payment, scale queue, restart orders |

---

## Reward Design (Two-Layer)

### Layer 1: Per-Step Training Rewards (Oracle-Shaped)
These rewards peek at hidden state to guide RL training:

| Action Category | Condition | Reward |
|----------------|-----------|--------|
| Diagnostic | Investigating involved service | +0.15 |
| Diagnostic | Investigating uninvolved service | +0.05 |
| Any | Repeating a previous action | -0.05 |
| Remediation | Correct target (root cause service) | +0.30 |
| Remediation | Helpful (affected, not root cause) | +0.10 |
| Remediation | Harmful (healthy service) | -0.15 |
| Declaration | Correct root cause | +0.40 |
| Declaration | Wrong root cause | -0.20 |
| Any | Per-step efficiency penalty | -0.02 |
| Completion | All services healthy | +0.20 |
| Completion | Time budget exceeded | -0.10 |

### Layer 2: Evaluation Grader (Oracle-Independent)
The grader scores only the trajectory — no hidden state access:

| Criterion | Weight | What it measures |
|-----------|--------|-----------------|
| Root cause accuracy | 40% | Did the agent declare the correct root cause? |
| Remediation quality | 30% | Did the agent take the right fix actions? |
| Diagnostic efficiency | 20% | Fewer steps to diagnosis = better |
| Service restoration | 10% | Are all services healthy at episode end? |

---

## Quick Start

### Local Development

```bash
# Install dependencies
cd incident_env
pip install -e .

# Start server
uvicorn incident_env.server.app:app --host 0.0.0.0 --port 8000

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_name": "memory_leak"}'
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"action_type": "view_alerts"}'
```

### Run Inference

```bash
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=http://localhost:8000

python inference.py
```

### Docker

```bash
docker build -t incident-env -f server/Dockerfile .
docker run -p 8000:8000 incident-env
```

---

## Example Agent Interaction

```
Agent: POST /reset {"task_name": "memory_leak"}
  → Incident triggered: "Orders service experiencing failures..."
  → Services: orders=degraded, rest=healthy

Agent: POST /step {"action_type": "view_alerts"}
  → 3 alerts: orders HighMemoryUsage (critical), orders HighErrorRate, orders HighLatencyP99
  → reward = +0.13

Agent: POST /step {"action_type": "check_metrics", "target_service": "orders"}
  → 30 data points: memory climbing from 35% → 78% over 20 minutes
  → reward = +0.13

Agent: POST /step {"action_type": "check_deploy_history", "target_service": "orders"}
  → 2 deploys: v2.3.1 (20 min ago, "batch order processing") and v1.2.0
  → reward = +0.13

Agent: POST /step {"action_type": "rollback_deploy", "target_service": "orders"}
  → "Rolled back orders from v2.3.1 to v1.2.0 — service recovering"
  → reward = +0.28

Agent: POST /step {"action_type": "declare_root_cause", "parameters": {"root_cause": "memory leak in orders caused by bad deploy v2.3.1"}}
  → Episode done. Final grade: 0.97
```
