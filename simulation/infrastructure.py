"""
Infrastructure state machine.

Manages the full service topology, cascade propagation, and the
validate → mutate → tick execution ordering.

This is the central coordinator that scenarios inject faults into
and the environment executes actions against.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from .service import Deploy, ServiceState
from .alerts import evaluate_alerts
from .logs import (
    generate_noise_logs,
    generate_red_herring_logs,
)


# ------------------------------------------------------------------
# Service topology — the dependency graph
# ------------------------------------------------------------------

SERVICE_NAMES = [
    "api_gateway", "auth", "orders", "payment",
    "cache", "database", "queue",
]

# depends_on: service → list of services it calls
DEPENDENCY_GRAPH: Dict[str, List[str]] = {
    "api_gateway": ["auth", "orders", "cache"],
    "auth":        ["database"],
    "orders":      ["database", "payment", "auth"],
    "payment":     ["queue", "database"],
    "cache":       [],
    "database":    [],
    "queue":       [],
}


def _depended_by_graph() -> Dict[str, List[str]]:
    """Invert the dependency graph: service → who depends on it."""
    inv: Dict[str, List[str]] = {name: [] for name in SERVICE_NAMES}
    for service, deps in DEPENDENCY_GRAPH.items():
        for dep in deps:
            inv[dep].append(service)
    return inv


DEPENDED_BY = _depended_by_graph()


class Infrastructure:
    """
    Virtual infrastructure state machine.

    Owns all services, handles cascade propagation, tracks simulation
    time, and enforces action validation.
    """

    def __init__(self) -> None:
        self.services: Dict[str, ServiceState] = {}
        self.current_minute: int = 0
        self.time_budget_minutes: int = 30
        self._actions_taken: List[Tuple[str, Optional[str]]] = []  # (action_type, target)
        self._all_logs: List[Dict[str, Any]] = []
        self._setup_services()

    def _setup_services(self) -> None:
        """Create all seven services with their dependency graphs."""
        for name in SERVICE_NAMES:
            svc = ServiceState(
                name=name,
                dependencies=list(DEPENDENCY_GRAPH.get(name, [])),
            )
            # Give each service a "good" deploy history baseline
            svc.deploy_history = [
                Deploy(
                    version=f"v1.{random.randint(0, 9)}.{random.randint(0, 9)}",
                    timestamp_minutes=-120,
                    author=random.choice(["alice", "bob", "charlie", "deploy-bot"]),
                    commit_hash=f"{random.randint(0, 0xFFFFFF):06x}",
                    description="Routine release — bug fixes and performance improvements",
                ),
            ]
            # Populate 30 minutes of healthy metric history
            from .metrics import generate_healthy_history
            svc.metric_history = generate_healthy_history(30, start_minute=0)
            svc._reset_metrics_healthy()
            self.services[name] = svc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_service(self, name: str) -> Optional[ServiceState]:
        return self.services.get(name)

    def get_all_statuses(self) -> Dict[str, str]:
        return {name: svc.status for name, svc in self.services.items()}

    def get_alerts(self) -> List[Dict[str, Any]]:
        return evaluate_alerts(self.services, self.current_minute)

    def record_action(self, action_type: str, target: Optional[str]) -> None:
        self._actions_taken.append((action_type, target))

    def was_action_taken(self, action_type: str, target: Optional[str] = None) -> bool:
        """Check if this exact action was already taken (for repeat detection)."""
        return (action_type, target) in self._actions_taken

    def action_count(self) -> int:
        return len(self._actions_taken)

    # ------------------------------------------------------------------
    # Tick — advances simulation by one minute (called after every step)
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """
        Advance the simulation by one minute.
        Order: propagate cascades → tick all services → generate noise logs.
        """
        self.current_minute += 1
        self._propagate_cascades()

        for name, svc in self.services.items():
            new_logs = svc.tick(self.current_minute)
            self._all_logs.extend(new_logs)

            # Mix in noise and red herrings
            if random.random() < 0.4:
                noise = generate_noise_logs(name, self.current_minute, count=1)
                svc.logs.extend(noise)
                self._all_logs.extend(noise)
            if random.random() < 0.15:
                herrings = generate_red_herring_logs(name, self.current_minute, count=1)
                svc.logs.extend(herrings)
                self._all_logs.extend(herrings)

    # ------------------------------------------------------------------
    # Cascade propagation
    # ------------------------------------------------------------------

    def _propagate_cascades(self) -> None:
        """
        If a service is DOWN or DEGRADED, its downstream dependents
        should accumulate dependency_degraded faults.
        If a service recovers, clear the cascaded faults.
        """
        for name, svc in self.services.items():
            dependents = DEPENDED_BY.get(name, [])
            if svc.status in ("down", "degraded") and svc.active_faults:
                # Cascade to dependents
                for dep_name in dependents:
                    dep_svc = self.services[dep_name]
                    if not dep_svc.has_fault("dependency_degraded"):
                        dep_svc.inject_fault("dependency_degraded", upstream=name)
            elif svc.status == "healthy" and not svc.active_faults:
                # Service recovered — clear cascade on dependents
                for dep_name in dependents:
                    dep_svc = self.services[dep_name]
                    params = dep_svc.fault_params.get("dependency_degraded", {})
                    if params.get("upstream") == name:
                        dep_svc.recover_from_dependency(self.current_minute)

    # ------------------------------------------------------------------
    # Validation — Layer 5: validate BEFORE mutating
    # ------------------------------------------------------------------

    def validate_action(
        self,
        action_type: str,
        target_service: Optional[str],
    ) -> Tuple[bool, str]:
        """
        Validate that an action is legal in the current state.
        Returns (is_valid, error_message).
        """
        from ..models import ActionType, TARGETED_ACTIONS

        try:
            at = ActionType(action_type)
        except ValueError:
            return False, f"Unknown action type: {action_type}"

        if at in TARGETED_ACTIONS:
            if not target_service:
                return False, f"Action {action_type} requires a target_service"
            if target_service not in self.services:
                return False, f"Unknown service: {target_service}"

        # Specific validations (action masking)
        if at == ActionType.ROLLBACK_DEPLOY:
            svc = self.services.get(target_service, None)
            if svc and len(svc.deploy_history) < 2:
                return False, f"No previous deploy to rollback to for {target_service}"

        if at == ActionType.SCALE_SERVICE:
            svc = self.services.get(target_service, None)
            if svc and svc.status == "down":
                return False, f"Cannot scale {target_service} — service is DOWN"

        return True, ""

    def get_valid_actions(self) -> List[str]:
        """
        Return list of valid (action_type, target) descriptions.
        Used to populate valid_actions[] in the observation.
        """
        from ..models import ActionType, TARGETED_ACTIONS
        valid = []
        for at in ActionType:
            if at in TARGETED_ACTIONS:
                for svc_name in self.services:
                    is_valid, _ = self.validate_action(at.value, svc_name)
                    if is_valid:
                        valid.append(f"{at.value}:{svc_name}")
            else:
                valid.append(at.value)
        return valid

    # ------------------------------------------------------------------
    # Service queries (observation builders)
    # ------------------------------------------------------------------

    def get_logs_for_service(
        self,
        service_name: str,
        level_filter: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Query logs for a service with optional filtering."""
        svc = self.services.get(service_name)
        if not svc:
            return []

        logs = list(svc.logs)
        if level_filter:
            logs = [l for l in logs if l.get("level", "").upper() == level_filter.upper()]
        if keyword:
            kw = keyword.lower()
            logs = [l for l in logs if kw in l.get("message", "").lower()]
        return logs[-limit:]

    def get_metrics_for_service(self, service_name: str) -> List[Dict[str, float]]:
        svc = self.services.get(service_name)
        return list(svc.metric_history) if svc else []

    def get_dependencies_for_service(self, service_name: str) -> Dict[str, List[str]]:
        return {
            "depends_on": list(DEPENDENCY_GRAPH.get(service_name, [])),
            "depended_by": list(DEPENDED_BY.get(service_name, [])),
        }

    def get_deploy_history_for_service(self, service_name: str) -> List[Dict[str, Any]]:
        svc = self.services.get(service_name)
        if not svc:
            return []
        return [
            {
                "version": d.version,
                "timestamp": f"2025-01-15T{14 + d.timestamp_minutes // 60:02d}:"
                             f"{d.timestamp_minutes % 60:02d}:00Z"
                             if d.timestamp_minutes >= 0
                             else f"2025-01-15T{12 + (d.timestamp_minutes + 120) // 60:02d}:"
                                  f"{(d.timestamp_minutes + 120) % 60:02d}:00Z",
                "author": d.author,
                "commit_hash": d.commit_hash,
                "description": d.description,
            }
            for d in svc.deploy_history
        ]

    def run_health_check(self, service_name: str) -> Dict[str, Any]:
        svc = self.services.get(service_name)
        if not svc:
            return {"status": "unknown", "response_time_ms": 0}
        response_time = {
            "healthy": random.randint(5, 50),
            "degraded": random.randint(200, 2000),
            "down": 30000,  # timeout
        }.get(svc.status, 0)
        return {
            "status": svc.status,
            "response_time_ms": response_time,
            "replicas": svc.replica_count,
            "active_faults_count": len(svc.active_faults),  # agent can see SOMETHING is wrong
        }

    def all_services_healthy(self) -> bool:
        return all(svc.status == "healthy" for svc in self.services.values())
