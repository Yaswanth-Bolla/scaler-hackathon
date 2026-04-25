"""
Phase-B scenario 2 — Misleading severity inversion.

The highest-severity alert is on a service that is the *downstream victim*,
not the cause.  A retry-storm in `orders` (caused by an aggressive auth
client config) floods `auth` with traffic; `auth` falls over with
CRITICAL alerts, while `orders` itself only shows WARN-level "elevated
retry counter" alerts.

A base model prompted to "follow the highest-severity alert" goes straight
to `auth` and finds nothing — auth's own deploy history is clean.

The RL signal: in retry-storm scenarios, the *quietest degraded service*
is the culprit and the loudest is the victim.  This is a scenario-class
specific reasoning rule no fixed prompt can encode.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set

from .base import BaseScenario
from .code_context_builder import SEVERITY_INVERSION_CODE_CONTEXT
from ..models import CodeContext
from ..simulation.infrastructure import Infrastructure
from ..simulation.service import Deploy
from ..simulation.metrics import (
    generate_error_spike_history,
    generate_healthy_history,
)


class SeverityInversionScenario(BaseScenario):

    @property
    def task_name(self) -> str:
        return "severity_inversion"

    @property
    def code_context(self) -> CodeContext:
        return SEVERITY_INVERSION_CODE_CONTEXT

    @property
    def fault_class(self) -> str:
        return "retry_storm"

    @property
    def display_name(self) -> str:
        return "Severity Inversion — Auth Drowning, Orders Is the Culprit"

    @property
    def incident_summary(self) -> str:
        return (
            "INCIDENT: Auth service is firing CRITICAL alerts (HighErrorRate, "
            "ServiceUnreachable, LatencyP99>5000ms). Customer logins are failing. "
            "Orders service shows a WARN-level alert: 'Elevated retry counter'."
        )

    @property
    def severity(self) -> str:
        return "SEV1"

    @property
    def correct_root_cause(self) -> str:
        return "orders auth-client retry storm overwhelming auth — orders deploy is root cause"

    @property
    def root_cause_keywords(self) -> List[str]:
        return ["orders", "retry", "storm", "auth-client"]

    @property
    def involved_services(self) -> Set[str]:
        return {"orders", "auth"}

    @property
    def root_cause_service(self) -> str:
        return "orders"

    @property
    def correct_remediation_actions(self) -> List[Dict[str, str]]:
        return [
            {"action_type": "rollback_deploy", "target_service": "orders"},
            {"action_type": "restart_service", "target_service": "auth"},
        ]

    def inject(self, infra: Infrastructure) -> None:
        orders = infra.get_service("orders")
        auth   = infra.get_service("auth")
        if orders is None or auth is None:
            return

        # --- The actual bad deploy: orders changed auth-client retry policy ---
        orders.deploy_history.append(Deploy(
            version="v2.5.1", timestamp_minutes=-12, author="carol",
            commit_hash="f8c9b13",
            description="Resilience: increase auth-client retries 3 → 20",
            is_bad=True,
        ))

        # --- Auth has a CLEAN recent deploy history (no smoking gun there) ---
        auth.deploy_history.append(Deploy(
            version="v1.7.4", timestamp_minutes=-86400, author="bob",
            commit_hash="a01122", description="Routine: bump TLS cert",
            is_bad=False,
        ))

        # --- Auth is the loud victim: high error rate, big latency ---
        auth.inject_fault("high_error_rate", rate=72.0)
        auth.error_rate_percent = 72.0 + random.gauss(0, 3)
        auth.latency_p95_ms     = 4200 + random.gauss(0, 200)
        auth.latency_p99_ms     = 7100 + random.gauss(0, 400)
        auth.requests_per_sec   = 4500 + random.gauss(0, 200)   # ← anomalously high RPS!
        auth.status             = "down"
        auth.metric_history     = generate_error_spike_history(
            minutes=30, start_minute=0, spike_start_offset=12, error_rate_target=72.0)
        auth.logs = [
            {"timestamp": "2025-01-15T14:14:00Z", "level": "WARN", "service": "auth",
             "message": "Request throughput tripled in last 4 minutes (1500 → 4400 RPS)",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:18:00Z", "level": "ERROR", "service": "auth",
             "message": "Token validation queue overflow — dropping requests",
             "trace_id": "trace-991201"},
            {"timestamp": "2025-01-15T14:21:00Z", "level": "ERROR", "service": "auth",
             "message": "Health check failed: auth returned HTTP 500 (overload)",
             "trace_id": None},
        ]

        # --- Orders looks almost healthy; only subtle clue is the retry counter ---
        orders.metric_history = generate_healthy_history(minutes=30, start_minute=0)
        orders.cpu_percent          = 38.0 + random.gauss(0, 3)
        orders.memory_percent       = 47.0 + random.gauss(0, 2)
        orders.error_rate_percent   = 1.5  + random.gauss(0, 0.3)
        orders.latency_p95_ms       = 110.0 + random.gauss(0, 8)
        orders.latency_p99_ms       = 240.0 + random.gauss(0, 15)
        orders.status               = "degraded"   # ← quiet degradation, easy to miss
        orders.logs = [
            {"timestamp": "2025-01-15T14:13:00Z", "level": "INFO", "service": "orders",
             "message": "Deploy v2.5.1 complete — auth-client retry policy updated",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:17:00Z", "level": "WARN", "service": "orders",
             "message": "auth-client retry counter elevated: avg 12 retries per validation",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:22:00Z", "level": "WARN", "service": "orders",
             "message": "auth-client circuit-breaker NOT tripped (retries policy ignores it)",
             "trace_id": None},
        ]
