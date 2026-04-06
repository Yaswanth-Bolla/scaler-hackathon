"""
Task 2 — Medium: Cascading Failure.

A bad configuration change to the auth service causes it to return 500s
on every token validation request. This cascades to api_gateway and orders
(both depend on auth). Multiple alerts fire across services — the agent
must trace the dependency graph to find the ROOT cause is auth, not the
downstream services showing symptoms.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set

from .base import BaseScenario
from ..simulation.infrastructure import Infrastructure
from ..simulation.service import Deploy
from ..simulation.metrics import generate_error_spike_history, generate_healthy_history


class CascadingFailureScenario(BaseScenario):

    @property
    def task_name(self) -> str:
        return "cascading_failure"

    @property
    def display_name(self) -> str:
        return "Cascading Failure — Auth Service Configuration"

    @property
    def incident_summary(self) -> str:
        return (
            "INCIDENT: Multiple services are experiencing elevated error rates. "
            "API Gateway is returning 5xx errors to external clients. "
            "Orders and Auth services both show failures. "
            "Customer-facing impact confirmed — payments not processing."
        )

    @property
    def severity(self) -> str:
        return "SEV1"

    @property
    def correct_root_cause(self) -> str:
        return "auth service bad config deploy caused authentication failures cascading to dependents"

    @property
    def root_cause_keywords(self) -> List[str]:
        return ["auth", "config", "deploy", "cascade"]

    @property
    def involved_services(self) -> Set[str]:
        return {"auth", "api_gateway", "orders"}

    @property
    def root_cause_service(self) -> str:
        return "auth"

    @property
    def correct_remediation_actions(self) -> List[Dict[str, str]]:
        return [
            {"action_type": "rollback_deploy", "target_service": "auth"},
            {"action_type": "restart_service", "target_service": "api_gateway"},
            {"action_type": "restart_service", "target_service": "orders"},
        ]

    def inject(self, infra: Infrastructure) -> None:
        """
        Set up cascading failure:
        1. Auth gets a bad config deploy → high_error_rate fault
        2. Downstream services (api_gateway, orders) get dependency_degraded
        3. Red herrings: orders also shows some unrelated log noise
        """
        auth = infra.get_service("auth")
        api_gw = infra.get_service("api_gateway")
        orders = infra.get_service("orders")

        if not all([auth, api_gw, orders]):
            return

        # --- Bad deploy on auth (the root cause) ---
        bad_deploy = Deploy(
            version="v1.8.0",
            timestamp_minutes=-15,
            author="bob",
            commit_hash="d4e5f6",
            description="Config update: rotate JWT signing secret",
            is_bad=True,
        )
        auth.deploy_history.append(bad_deploy)

        # --- Inject auth fault ---
        auth.inject_fault("high_error_rate", rate=65.0)
        auth.metric_history = generate_error_spike_history(
            minutes=30, start_minute=0, spike_start_offset=15, error_rate_target=65.0)
        auth.error_rate_percent = 65.0 + random.gauss(0, 3)
        auth.status = "down"
        auth.latency_p95_ms = 500 + random.gauss(0, 50)
        auth.latency_p99_ms = 1500 + random.gauss(0, 100)

        # Auth-specific logs showing the config error
        auth.logs = [
            {"timestamp": "2025-01-15T14:15:00Z", "level": "INFO", "service": "auth",
             "message": "Deploy v1.8.0 started — config update: JWT secret rotation",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:15:30Z", "level": "INFO", "service": "auth",
             "message": "Deploy v1.8.0 complete — all replicas updated", "trace_id": None},
            {"timestamp": "2025-01-15T14:16:00Z", "level": "ERROR", "service": "auth",
             "message": "NullPointerException: configuration key 'auth.jwt.secret' is null",
             "trace_id": "trace-110001"},
            {"timestamp": "2025-01-15T14:16:15Z", "level": "ERROR", "service": "auth",
             "message": "Token validation failed: cannot sign with null key — returning 500",
             "trace_id": "trace-110002"},
            {"timestamp": "2025-01-15T14:17:00Z", "level": "ERROR", "service": "auth",
             "message": "Health check failed: auth returned HTTP 500", "trace_id": None},
            {"timestamp": "2025-01-15T14:20:00Z", "level": "ERROR", "service": "auth",
             "message": "Authentication failed for 180 requests in last 60s — returning HTTP 500",
             "trace_id": None},
        ]

        # --- Cascaded impact on api_gateway ---
        api_gw.inject_fault("dependency_degraded", upstream="auth")
        api_gw.error_rate_percent = 45.0 + random.gauss(0, 5)
        api_gw.latency_p95_ms = 2000 + random.gauss(0, 200)
        api_gw.status = "degraded"
        api_gw.logs = [
            {"timestamp": "2025-01-15T14:17:00Z", "level": "ERROR", "service": "api_gateway",
             "message": "Call to auth-service failed: HTTP 500 Internal Server Error — "
                        "retrying (1/3)", "trace_id": "trace-220001"},
            {"timestamp": "2025-01-15T14:17:30Z", "level": "ERROR", "service": "api_gateway",
             "message": "All retry attempts to auth-service exhausted — returning 502 to client",
             "trace_id": "trace-220002"},
            {"timestamp": "2025-01-15T14:18:00Z", "level": "WARN", "service": "api_gateway",
             "message": "Circuit breaker for auth-service: state=OPEN, failures=25, threshold=10",
             "trace_id": None},
        ]

        # --- Cascaded impact on orders ---
        orders.inject_fault("dependency_degraded", upstream="auth")
        orders.error_rate_percent = 30.0 + random.gauss(0, 4)
        orders.latency_p95_ms = 1800 + random.gauss(0, 150)
        orders.status = "degraded"
        orders.logs = [
            {"timestamp": "2025-01-15T14:17:00Z", "level": "ERROR", "service": "orders",
             "message": "Call to auth-service failed: HTTP 500 — cannot validate order token",
             "trace_id": "trace-330001"},
            {"timestamp": "2025-01-15T14:18:00Z", "level": "ERROR", "service": "orders",
             "message": "Order creation failed: authentication service unavailable",
             "trace_id": "trace-330002"},
            # Red herring
            {"timestamp": "2025-01-15T14:19:00Z", "level": "WARN", "service": "orders",
             "message": "Database connection pool: 18/20 active connections — approaching limit",
             "trace_id": None},
        ]
