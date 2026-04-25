"""
Stage-4 entry safety: Pool-B variance gate + r_cross warmup schedule.

The brief's bootstrapping constraint:

  - If `Var[r_code | context(τ_1)]` is too high (the code agent itself is
    unstable on this task class), then `r_cross` is mostly noise and
    Stage 4 will inject that noise into Phase-1 gradients.  Block entry
    until variance falls below a threshold.

  - Even after the gate opens, ramp `r_cross_weight` from 0 → 1 over
    `warmup_steps` so Phase-1 learning isn't suddenly knocked off-track.

This module is small and pure — the curriculum runner queries it on
each Stage-4 step.
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class VarianceGate:
    """Tracks a moving window of `r_code` values per task and gates Stage 4."""

    window:                int      = 64
    max_acceptable_stdev:  float    = 0.15     # standard deviation of r_code
    min_samples:           int      = 16
    history: Dict[str, Deque[float]] = field(default_factory=dict)

    def record(self, task_name: str, r_code: float) -> None:
        if task_name not in self.history:
            self.history[task_name] = deque(maxlen=self.window)
        self.history[task_name].append(float(r_code))

    def stdev(self, task_name: str) -> float:
        h = self.history.get(task_name) or []
        if len(h) < 2:
            return float("inf")
        return statistics.stdev(h)

    def is_open_for(self, task_name: str) -> bool:
        h = self.history.get(task_name) or []
        if len(h) < self.min_samples:
            return False
        return self.stdev(task_name) <= self.max_acceptable_stdev

    def open_tasks(self) -> List[str]:
        return [t for t in self.history if self.is_open_for(t)]

    def status(self) -> Dict[str, Dict[str, float]]:
        """Diagnostic snapshot for logging."""
        return {
            t: {
                "n":     len(h),
                "stdev": round(self.stdev(t), 4),
                "open":  self.is_open_for(t),
            }
            for t, h in self.history.items()
        }


# ──────────────────────────────────────────────────────────────────────
# r_cross warmup schedule
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RCrossWarmup:
    """
    Linear warmup of the r_cross coefficient over the first
    `warmup_steps` Stage-4 optimizer steps.

        weight(step) = min(1.0, step / warmup_steps)
    """
    warmup_steps: int   = 500
    cap:          float = 1.0

    def weight(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return self.cap
        return min(self.cap, step / self.warmup_steps)
