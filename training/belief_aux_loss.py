"""
Belief-state auxiliary loss — supervises the orchestrator's `<belief_state>`
output during Stage 2.

Three sub-losses are combined:

  1. Service classification CE
        cross-entropy over `suspected_service` against ground-truth
        `scenario.root_cause_service`

  2. Fault-class classification CE
        cross-entropy over `suspected_fault_class` against
        `scenario.fault_class`

  3. Confidence calibration MSE
        squared error between declared `service_confidence` and the
        empirical correctness rate of the model on this scenario class.
        (Smoothed to avoid the calibration target moving too fast — we
        track an EMA of accuracy per scenario.)

  4. Evidence-gap consistency
        a soft consistency rule: if `service_confidence < 0.5`,
        `evidence_gaps` should be non-empty.  Penalty otherwise.

This module is framework-agnostic — it returns the loss components as
plain floats / dicts.  The trainer wraps them in tensors.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..models import BeliefState
from ..tasks import get_scenario


# Known fault-class universe (must match scenarios)
FAULT_CLASSES = [
    "memory_leak",
    "config_change",
    "deadlock",
    "retry_storm",
    "cache_thrash",
    "shared_dependency",
    "none",
]

# Known service universe (must match infrastructure)
SERVICES = [
    "api_gateway", "auth", "orders", "payment",
    "cache", "database", "queue",
]


# ──────────────────────────────────────────────────────────────────────
# Component losses (numpy / pure-python; trainer can torch-ify)
# ──────────────────────────────────────────────────────────────────────


def softmax(xs: List[float]) -> List[float]:
    m = max(xs) if xs else 0.0
    es = [math.exp(x - m) for x in xs]
    s = sum(es) or 1.0
    return [e / s for e in es]


def ce_loss(logits: List[float], target_idx: int) -> float:
    """One-sample categorical cross-entropy."""
    probs = softmax(logits)
    p = probs[target_idx] if 0 <= target_idx < len(probs) else 1e-9
    return -math.log(max(p, 1e-9))


def mse_loss(pred: float, target: float) -> float:
    return (pred - target) ** 2


# ──────────────────────────────────────────────────────────────────────
# Calibration target tracker
# ──────────────────────────────────────────────────────────────────────


@dataclass
class CalibrationEMA:
    """
    EMA of (correctness rate) per (task_name, fault_class) bucket.

    Used as the target for `service_confidence` so the orchestrator
    learns to emit confidences that match empirical accuracy on
    similar scenarios.
    """
    alpha: float = 0.05
    means: Dict[Tuple[str, str], float] = field(default_factory=dict)
    counts: Dict[Tuple[str, str], int]  = field(default_factory=lambda: defaultdict(int))

    def update(self, task: str, fault: str, correct: bool) -> None:
        key = (task, fault)
        x = 1.0 if correct else 0.0
        if key not in self.means:
            self.means[key] = x
        else:
            self.means[key] = (1 - self.alpha) * self.means[key] + self.alpha * x
        self.counts[key] += 1

    def target(self, task: str, fault: str) -> float:
        return self.means.get((task, fault), 0.5)


# ──────────────────────────────────────────────────────────────────────
# Composite loss
# ──────────────────────────────────────────────────────────────────────


@dataclass
class BeliefAuxConfig:
    w_service:       float = 1.0
    w_fault:         float = 1.0
    w_calibration:   float = 0.5
    w_consistency:   float = 0.25
    confidence_floor: float = 0.5    # gap-non-empty boundary


def compute_belief_aux(
    belief:          BeliefState,
    task_name:       str,
    service_logits:  Optional[List[float]] = None,
    fault_logits:    Optional[List[float]] = None,
    correctness:     Optional[bool] = None,
    calibration_ema: Optional[CalibrationEMA] = None,
    cfg:             BeliefAuxConfig = BeliefAuxConfig(),
) -> Dict[str, float]:
    """
    Compute the four-component aux loss for ONE belief sample.

    `service_logits` / `fault_logits` are supplied by the language model's
    auxiliary classification heads (or by the trainer projecting hidden
    states onto the service/fault label spaces).  If omitted, the
    classification components are skipped.

    `correctness` is whether the orchestrator's eventual root-cause
    declaration was right; required to update the calibration EMA but
    optional for forward loss computation.

    Returns a dict of component losses (and a `total` field).
    """
    scenario = get_scenario(task_name)
    losses: Dict[str, float] = {}

    # 1. Service CE
    if service_logits is not None:
        try:
            target = SERVICES.index(scenario.root_cause_service)
        except ValueError:
            target = -1
        if target >= 0:
            losses["service_ce"] = cfg.w_service * ce_loss(service_logits, target)

    # 2. Fault CE
    if fault_logits is not None:
        try:
            target = FAULT_CLASSES.index(scenario.fault_class)
        except ValueError:
            target = FAULT_CLASSES.index("none")
        losses["fault_ce"] = cfg.w_fault * ce_loss(fault_logits, target)

    # 3. Calibration MSE
    if calibration_ema is not None:
        calib_target = calibration_ema.target(task_name, scenario.fault_class)
        losses["calibration_mse"] = cfg.w_calibration * mse_loss(
            float(belief.service_confidence), calib_target,
        )
        if correctness is not None:
            calibration_ema.update(task_name, scenario.fault_class, correctness)

    # 4. Consistency
    has_gaps = bool(belief.evidence_gaps and len(belief.evidence_gaps) > 0)
    low_conf = float(belief.service_confidence) < cfg.confidence_floor
    if low_conf and not has_gaps:
        losses["consistency"] = cfg.w_consistency * 1.0
    elif (not low_conf) and has_gaps:
        # not strictly wrong but mild penalty (high conf + gaps = under-stopping)
        losses["consistency"] = cfg.w_consistency * 0.25
    else:
        losses["consistency"] = 0.0

    losses["total"] = round(sum(losses.values()), 4)
    return {k: round(v, 4) for k, v in losses.items()}
