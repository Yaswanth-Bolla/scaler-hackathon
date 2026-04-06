"""
Individual service simulator.

Each service is a stateful entity with health, metrics, logs, deploy history,
and fault injection points.  When faults are injected, metrics respond
reactively — memory climbs, error rates spike, latency degrades — and the
service produces appropriate log entries automatically.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Deploy:
    """A single deploy record."""
    version: str
    timestamp_minutes: int      # simulation minutes since epoch
    author: str
    commit_hash: str
    description: str
    is_bad: bool = False        # hidden — grader never sees this directly


@dataclass
class ServiceState:
    """
    Full mutable state for one service.

    The agent NEVER sees this directly. It can only observe symptoms
    through the five observation modalities (alerts, metrics, logs, deps, deploys).
    """
    name: str
    status: str = "healthy"                         # healthy | degraded | down
    dependencies: List[str] = field(default_factory=list)

    # --- Metrics (reactive) ---
    cpu_percent: float = 15.0
    memory_percent: float = 35.0
    error_rate_percent: float = 0.1
    latency_p50_ms: float = 12.0
    latency_p95_ms: float = 45.0
    latency_p99_ms: float = 120.0
    requests_per_sec: float = 500.0

    # --- Metric history (last 30 data points = 30 minutes) ---
    metric_history: List[Dict[str, float]] = field(default_factory=list)

    # --- Logs (circular buffer, last 50) ---
    logs: List[Dict[str, Any]] = field(default_factory=list)

    # --- Deploy history ---
    deploy_history: List[Deploy] = field(default_factory=list)

    # --- Fault state (hidden — drives reactive behavior) ---
    active_faults: List[str] = field(default_factory=list)
    fault_params: Dict[str, Any] = field(default_factory=dict)

    # --- Operational ---
    replica_count: int = 3
    restarts_since_fault: int = 0
    ticks_in_degraded: int = 0
    ticks_in_down: int = 0
    was_rolled_back: bool = False

    # ---------------------------------------------------------------
    # Fault injection — called by scenarios at setup time
    # ---------------------------------------------------------------

    def inject_fault(self, fault_type: str, **params: Any) -> None:
        """Inject a named fault. Metrics will react on subsequent ticks."""
        self.active_faults.append(fault_type)
        self.fault_params[fault_type] = params

    def clear_fault(self, fault_type: str) -> None:
        """Remove a fault (e.g. after rollback fixes the root cause)."""
        if fault_type in self.active_faults:
            self.active_faults.remove(fault_type)
            self.fault_params.pop(fault_type, None)

    def clear_all_faults(self) -> None:
        self.active_faults.clear()
        self.fault_params.clear()

    def has_fault(self, fault_type: str) -> bool:
        return fault_type in self.active_faults

    # ---------------------------------------------------------------
    # Tick — advance one simulation minute.  Metrics react to faults.
    # ---------------------------------------------------------------

    def tick(self, current_minute: int) -> List[Dict[str, Any]]:
        """
        Advance the service by one simulation minute.
        Returns any new log entries generated this tick.
        """
        new_logs: List[Dict[str, Any]] = []
        noise = lambda: random.gauss(0, 1)

        # --- Memory leak: memory climbs steadily ---
        if "memory_leak" in self.active_faults:
            rate = self.fault_params.get("memory_leak", {}).get("rate", 1.5)
            self.memory_percent = min(99.0, self.memory_percent + rate + noise() * 0.3)
            self.cpu_percent = min(95.0, self.cpu_percent + 0.3 + noise() * 0.2)
            if self.memory_percent > 90:
                self.status = "down"
                self.error_rate_percent = min(100.0, 85.0 + noise() * 5)
                new_logs.append(self._log(current_minute, "FATAL",
                    f"OutOfMemoryError: Java heap space — service {self.name} killed by OOM killer"))
                new_logs.append(self._log(current_minute, "ERROR",
                    f"Container {self.name}-0 exited with code 137 (OOMKilled)"))
            elif self.memory_percent > 75:
                self.status = "degraded"
                self.error_rate_percent = min(50.0, 15.0 + (self.memory_percent - 75) * 1.5 + noise() * 2)
                self.latency_p95_ms = max(self.latency_p95_ms, 200 + noise() * 20)
                self.latency_p99_ms = max(self.latency_p99_ms, 500 + noise() * 30)
                new_logs.append(self._log(current_minute, "WARN",
                    f"GC pressure: heap usage at {self.memory_percent:.0f}%, "
                    f"GC pause {random.randint(200, 800)}ms"))

        # --- High error rate (e.g. bad config) ---
        if "high_error_rate" in self.active_faults:
            target_rate = self.fault_params.get("high_error_rate", {}).get("rate", 60.0)
            self.error_rate_percent = min(100.0, target_rate + noise() * 5)
            if self.error_rate_percent > 50:
                self.status = "down"
                new_logs.append(self._log(current_minute, "ERROR",
                    f"Health check failed: {self.name} returned HTTP 500"))
            elif self.error_rate_percent > 20:
                self.status = "degraded"
            new_logs.append(self._log(current_minute, "ERROR",
                f"Internal Server Error: configuration key 'auth.token.secret' is null"))

        # --- High latency (e.g. deadlock / contention) ---
        if "high_latency" in self.active_faults:
            target_p99 = self.fault_params.get("high_latency", {}).get("p99", 5000)
            self.latency_p50_ms = min(2000, 300 + noise() * 30)
            self.latency_p95_ms = min(8000, target_p99 * 0.7 + noise() * 100)
            self.latency_p99_ms = min(15000, target_p99 + noise() * 200)
            self.error_rate_percent = min(40.0, 10.0 + noise() * 3)
            self.status = "degraded"
            new_logs.append(self._log(current_minute, "WARN",
                f"Request timeout: upstream call to dependency exceeded 5000ms"))

        # --- Dependency degradation (cascaded from upstream) ---
        if "dependency_degraded" in self.active_faults:
            upstream = self.fault_params.get("dependency_degraded", {}).get("upstream", "unknown")
            self.error_rate_percent = min(80.0, 25.0 + noise() * 8)
            self.latency_p95_ms = max(self.latency_p95_ms, 1500 + noise() * 100)
            self.latency_p99_ms = max(self.latency_p99_ms, 3000 + noise() * 200)
            if self.error_rate_percent > 50:
                self.status = "down"
            else:
                self.status = "degraded"
            new_logs.append(self._log(current_minute, "ERROR",
                f"Connection refused: {upstream}:8080 — upstream service unavailable"))

        # --- Circular wait / deadlock ---
        if "circular_wait" in self.active_faults:
            peers = self.fault_params.get("circular_wait", {}).get("peers", [])
            self.latency_p50_ms = min(3000, 500 + noise() * 50)
            self.latency_p95_ms = min(10000, 4000 + noise() * 200)
            self.latency_p99_ms = min(30000, 8000 + noise() * 500)
            self.error_rate_percent = min(30.0, 12.0 + noise() * 3)
            self.requests_per_sec = max(10, self.requests_per_sec * 0.85)
            self.status = "degraded"
            peer = random.choice(peers) if peers else "unknown"
            new_logs.append(self._log(current_minute, "WARN",
                f"Timeout waiting for response from {peer}: "
                f"request {self._trace_id()} blocked for {random.randint(5000, 15000)}ms"))
            if random.random() < 0.3:
                new_logs.append(self._log(current_minute, "ERROR",
                    f"Retry exhausted for {peer}: CircuitBreaker OPEN after 5 consecutive failures"))

        # --- Healthy service noise ---
        if not self.active_faults:
            self._tick_healthy(current_minute)
        else:
            self.ticks_in_degraded += 1 if self.status == "degraded" else 0
            self.ticks_in_down += 1 if self.status == "down" else 0

        # Record metric snapshot
        self.metric_history.append({
            "minute": current_minute,
            "cpu": round(self.cpu_percent, 1),
            "memory": round(self.memory_percent, 1),
            "error_rate": round(self.error_rate_percent, 2),
            "latency_p50": round(self.latency_p50_ms, 1),
            "latency_p95": round(self.latency_p95_ms, 1),
            "latency_p99": round(self.latency_p99_ms, 1),
            "rps": round(self.requests_per_sec, 1),
        })
        # Keep last 30 data points
        if len(self.metric_history) > 30:
            self.metric_history = self.metric_history[-30:]

        # Keep last 50 logs
        self.logs.extend(new_logs)
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]

        return new_logs

    # ---------------------------------------------------------------
    # Remediation actions
    # ---------------------------------------------------------------

    def restart(self, current_minute: int) -> str:
        """
        Restart the service.  Temporarily fixes symptoms but NOT root cause
        unless the fault has been cleared first (e.g. via rollback).
        """
        self.restarts_since_fault += 1

        if not self.active_faults:
            # Service is healthy — restart is unnecessary
            self.status = "healthy"
            self.logs.append(self._log(current_minute, "INFO",
                f"Service {self.name} restarted (was already healthy)"))
            return f"{self.name} restarted (was already healthy)"

        # Reset metrics temporarily — faults will re-corrupt on next tick
        self.memory_percent = 35.0 + random.gauss(0, 3)
        self.cpu_percent = 15.0 + random.gauss(0, 2)
        self.error_rate_percent = max(0.1, self.error_rate_percent * 0.3)
        self.latency_p50_ms = 12.0 + random.gauss(0, 2)
        self.latency_p95_ms = 45.0 + random.gauss(0, 5)
        self.latency_p99_ms = 120.0 + random.gauss(0, 10)
        self.status = "healthy"

        self.logs.append(self._log(current_minute, "INFO",
            f"Service {self.name} restarted — metrics reset. "
            f"NOTE: underlying issue may recur."))
        return f"{self.name} restarted — metrics temporarily reset"

    def rollback_deploy(self, current_minute: int) -> str:
        """
        Roll back to the previous deploy.
        If the active fault was caused by a bad deploy, this FIXES IT.
        """
        if len(self.deploy_history) < 2:
            return f"No previous deploy to rollback to for {self.name}"

        bad_deploy = self.deploy_history[-1]
        prev_deploy = self.deploy_history[-2]

        self.was_rolled_back = True

        # If the bad deploy is what caused the fault, clear it
        if bad_deploy.is_bad:
            self.clear_all_faults()
            self.status = "healthy"
            self._reset_metrics_healthy()
            self.logs.append(self._log(current_minute, "INFO",
                f"Rolled back {self.name} from {bad_deploy.version} to "
                f"{prev_deploy.version} — fault cleared"))
            return (f"Rolled back {self.name} from {bad_deploy.version} to "
                    f"{prev_deploy.version} — service recovering")
        else:
            self.logs.append(self._log(current_minute, "INFO",
                f"Rolled back {self.name} from {bad_deploy.version} to "
                f"{prev_deploy.version} — no change in symptoms"))
            return (f"Rolled back {self.name} to {prev_deploy.version} "
                    f"— symptoms unchanged (likely not the cause)")

    def scale(self, new_replicas: int, current_minute: int) -> str:
        """Scale to new replica count. Helps with load but not root cause."""
        old = self.replica_count
        self.replica_count = max(1, min(10, new_replicas))
        if self.replica_count > old and "circular_wait" not in self.active_faults:
            # Scaling up reduces latency proportionally
            factor = old / self.replica_count
            self.latency_p50_ms *= factor
            self.latency_p95_ms *= factor
            self.latency_p99_ms *= factor
            self.requests_per_sec /= factor
        self.logs.append(self._log(current_minute, "INFO",
            f"Scaled {self.name} from {old} to {self.replica_count} replicas"))
        return f"Scaled {self.name}: {old} -> {self.replica_count} replicas"

    # ---------------------------------------------------------------
    # Recovery after upstream fix
    # ---------------------------------------------------------------

    def recover_from_dependency(self, current_minute: int) -> None:
        """Called when an upstream fault clears — this service should heal."""
        self.clear_fault("dependency_degraded")
        if not self.active_faults:
            self.status = "healthy"
            self._reset_metrics_healthy()
            self.logs.append(self._log(current_minute, "INFO",
                f"Service {self.name} recovering — upstream dependency restored"))

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------

    def _tick_healthy(self, current_minute: int) -> None:
        """Normal baseline metric jitter for healthy services."""
        noise = lambda: random.gauss(0, 1)
        self.cpu_percent = max(5, min(40, 15 + noise() * 3))
        self.memory_percent = max(20, min(55, 35 + noise() * 3))
        self.error_rate_percent = max(0, min(2, 0.1 + abs(noise()) * 0.1))
        self.latency_p50_ms = max(5, 12 + noise() * 2)
        self.latency_p95_ms = max(20, 45 + noise() * 5)
        self.latency_p99_ms = max(50, 120 + noise() * 10)
        self.requests_per_sec = max(200, 500 + noise() * 30)
        self.status = "healthy"

    def _reset_metrics_healthy(self) -> None:
        """Fully reset to healthy baseline."""
        self.cpu_percent = 15.0 + random.gauss(0, 2)
        self.memory_percent = 35.0 + random.gauss(0, 3)
        self.error_rate_percent = 0.1 + abs(random.gauss(0, 0.05))
        self.latency_p50_ms = 12.0 + random.gauss(0, 1)
        self.latency_p95_ms = 45.0 + random.gauss(0, 3)
        self.latency_p99_ms = 120.0 + random.gauss(0, 8)
        self.requests_per_sec = 500.0 + random.gauss(0, 20)
        self.status = "healthy"

    def _log(self, minute: int, level: str, message: str) -> Dict[str, Any]:
        return {
            "timestamp": f"2025-01-15T14:{minute:02d}:00Z",
            "level": level,
            "service": self.name,
            "message": message,
            "trace_id": self._trace_id() if level in ("ERROR", "FATAL") else None,
        }

    @staticmethod
    def _trace_id() -> str:
        return f"trace-{random.randint(100000, 999999)}"
