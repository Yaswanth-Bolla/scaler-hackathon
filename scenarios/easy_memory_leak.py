"""
Task 1 — Easy: Memory Leak.

Single-service failure with clear metric signal and obvious recent deploy.
Agent should: view alerts → check metrics(orders) → check deploy history →
rollback deploy → declare root cause.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set

from .base import BaseScenario
from ..simulation.infrastructure import Infrastructure
from ..simulation.service import Deploy
from ..simulation.metrics import generate_memory_leak_history


class MemoryLeakScenario(BaseScenario):

    @property
    def task_name(self) -> str:
        return "memory_leak"

    @property
    def display_name(self) -> str:
        return "Memory Leak — Orders Service"

    @property
    def incident_summary(self) -> str:
        return (
            "INCIDENT: Orders service is experiencing intermittent failures "
            "and restarts. Customers are reporting failed checkout attempts. "
            "The on-call SRE has been paged."
        )

    @property
    def severity(self) -> str:
        return "SEV2"

    @property
    def correct_root_cause(self) -> str:
        return "memory leak in orders service caused by bad deploy v2.3.1"

    @property
    def root_cause_keywords(self) -> List[str]:
        return ["memory", "leak", "orders", "deploy"]

    @property
    def involved_services(self) -> Set[str]:
        return {"orders"}

    @property
    def root_cause_service(self) -> str:
        return "orders"

    @property
    def correct_remediation_actions(self) -> List[Dict[str, str]]:
        return [
            {"action_type": "rollback_deploy", "target_service": "orders"},
        ]

    def inject(self, infra: Infrastructure) -> None:
        """
        Set up the memory leak scenario:
        1. Add a bad deploy to orders service
        2. Inject memory_leak fault
        3. Populate metric history showing the leak pattern
        4. Pre-populate some logs showing OOM symptoms
        """
        orders = infra.get_service("orders")
        if not orders:
            return

        # --- Bad deploy (the root cause) ---
        bad_deploy = Deploy(
            version="v2.3.1",
            timestamp_minutes=-20,  # 20 minutes ago
            author="alice",
            commit_hash="a1b2c3",
            description="Feature: batch order processing with in-memory cache",
            is_bad=True,
        )
        orders.deploy_history.append(bad_deploy)

        # --- Inject the memory leak fault ---
        orders.inject_fault("memory_leak", rate=1.2)

        # --- Pre-populate metric history showing the leak ---
        orders.metric_history = generate_memory_leak_history(
            minutes=30,
            start_minute=0,
            leak_start_offset=10,  # leak started 20 min into the history
            rate=1.2,
        )

        # --- Set current metrics to reflect ~20 minutes of leak ---
        orders.memory_percent = 78.0 + random.gauss(0, 2)
        orders.cpu_percent = 35.0 + random.gauss(0, 3)
        orders.error_rate_percent = 12.0 + random.gauss(0, 2)
        orders.latency_p95_ms = 350.0 + random.gauss(0, 30)
        orders.latency_p99_ms = 800.0 + random.gauss(0, 50)
        orders.status = "degraded"

        # --- Pre-populated logs showing symptoms ---
        base_logs = [
            {"timestamp": "2025-01-15T14:10:00Z", "level": "INFO", "service": "orders",
             "message": "Deploy v2.3.1 started — rolling update initiated", "trace_id": None},
            {"timestamp": "2025-01-15T14:11:00Z", "level": "INFO", "service": "orders",
             "message": "Deploy v2.3.1 complete — all replicas updated", "trace_id": None},
            {"timestamp": "2025-01-15T14:18:00Z", "level": "WARN", "service": "orders",
             "message": "GC pressure: heap usage at 62%, GC pause 340ms", "trace_id": None},
            {"timestamp": "2025-01-15T14:22:00Z", "level": "WARN", "service": "orders",
             "message": "GC pressure: heap usage at 71%, GC pause 580ms", "trace_id": None},
            {"timestamp": "2025-01-15T14:25:00Z", "level": "ERROR", "service": "orders",
             "message": "Memory allocation failed: unable to allocate 128MB for batch cache",
             "trace_id": "trace-442918"},
            {"timestamp": "2025-01-15T14:27:00Z", "level": "WARN", "service": "orders",
             "message": "GC overhead limit exceeded: spent 87% of time in GC",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:28:00Z", "level": "ERROR", "service": "orders",
             "message": "Request processing failed: OutOfMemoryError in batch order handler",
             "trace_id": "trace-553201"},
        ]
        orders.logs = base_logs
