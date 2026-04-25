"""
Task registry and unified grader.

`TASK_REGISTRY` maps task_name → scenario class.  Pools (A/B/C/D) live
in `pools.py` and reuse the same registry — there's no duplication.

The unified grader is oracle-INDEPENDENT: it consumes only step records
plus terminal artefacts (declared patch, declared no-change), so it can
score a saved trajectory file long after the episode ended.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from .models import StepRecord
from .scenarios.base import BaseScenario
from .scenarios.easy_memory_leak import MemoryLeakScenario
from .scenarios.medium_cascading_failure import CascadingFailureScenario
from .scenarios.hard_distributed_deadlock import DistributedDeadlockScenario
from .scenarios.grader_p2 import (
    grade_patch_quality,
    grade_no_change,
    grade_p2_efficiency,
)


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Type[BaseScenario]] = {
    "memory_leak":          MemoryLeakScenario,
    "cascading_failure":    CascadingFailureScenario,
    "distributed_deadlock": DistributedDeadlockScenario,
}

# Phase B scenarios are registered lazily to avoid import cycles
# during the initial Phase A bring-up.  See `_lazy_register_phase_b()`.
_PHASE_B_REGISTERED = False


def _lazy_register_phase_b() -> None:
    """Register Phase B scenarios if importable."""
    global _PHASE_B_REGISTERED
    if _PHASE_B_REGISTERED:
        return
    _PHASE_B_REGISTERED = True

    try:
        from .scenarios.aliased_fault import AliasedFaultScenario
        TASK_REGISTRY["aliased_fault"] = AliasedFaultScenario
    except ImportError:
        pass
    try:
        from .scenarios.severity_inversion import SeverityInversionScenario
        TASK_REGISTRY["severity_inversion"] = SeverityInversionScenario
    except ImportError:
        pass
    try:
        from .scenarios.confidence_inversion import ConfidenceInversionScenario
        TASK_REGISTRY["confidence_inversion"] = ConfidenceInversionScenario
    except ImportError:
        pass
    try:
        from .scenarios.info_ordering import InfoOrderingScenario
        TASK_REGISTRY["info_ordering"] = InfoOrderingScenario
    except ImportError:
        pass
    try:
        from .scenarios.circuit_breaker_noop import CircuitBreakerNoopScenario
        TASK_REGISTRY["circuit_breaker_noop"] = CircuitBreakerNoopScenario
    except ImportError:
        pass
    # Pool D held-out compounds
    try:
        from .scenarios.heldout import (
            HeldoutAliasedSeverityScenario,
            HeldoutConfidenceOrderingScenario,
        )
        TASK_REGISTRY["heldout_aliased_severity"]    = HeldoutAliasedSeverityScenario
        TASK_REGISTRY["heldout_confidence_ordering"] = HeldoutConfidenceOrderingScenario
    except ImportError:
        pass


_lazy_register_phase_b()
TASK_NAMES = list(TASK_REGISTRY.keys())


def get_scenario(task_name: str) -> BaseScenario:
    cls = TASK_REGISTRY.get(task_name)
    if cls is None:
        raise ValueError(
            f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY)}")
    return cls()


# ──────────────────────────────────────────────────────────────────────
# P1-only grader (legacy)
# ──────────────────────────────────────────────────────────────────────

def grade_trajectory(task_name: str, trajectory: List[StepRecord]) -> float:
    """Score a P1-only trajectory in [0.01, 0.99]."""
    scenario = get_scenario(task_name)
    return float(scenario.grade(trajectory))


# ──────────────────────────────────────────────────────────────────────
# Unified grader
# ──────────────────────────────────────────────────────────────────────

#  Component weights (must sum to 1.0)
W_P1_RCA            = 0.25
W_P1_EFFICIENCY     = 0.15
W_PATCH_QUALITY     = 0.35
W_NO_CHANGE         = 0.25
# Note: weights sum to 1.0; for is_valid_issue scenarios the "no_change"
# slot is reallocated to P2 efficiency.
W_P2_EFFICIENCY     = 0.25


def grade_trajectory_unified(
    task_name:          str,
    p1_trajectory:      List[StepRecord],
    p2_trajectory:      List[StepRecord],
    declared_patch:     Optional[str],
    declared_no_change: bool,
    p1_belief_history:  Optional[List[dict]] = None,
) -> Dict[str, float]:
    """
    Score a unified P1 + P2 trajectory.

    Returns a breakdown dict with the four weighted component scores
    and the final aggregate.  Each component is in [0, 1] *before*
    weighting; the final is also in [0, 1].
    """
    scenario = get_scenario(task_name)
    ctx      = scenario.code_context

    # ---- P1 components (always evaluated) ----
    p1_rca_raw = scenario.grade_p1_rca(p1_trajectory)
    p1_eff_raw = scenario.grade_p1_efficiency(p1_trajectory)

    # ---- P2 components (only if scenario has a code_context) ----
    if ctx is None:
        # P1-only scenario — entire P2 budget goes to P1 RCA & efficiency
        return {
            "final":               round(p1_rca_raw * 0.5 + p1_eff_raw * 0.5, 4),
            "p1_rca":              round(p1_rca_raw,  4),
            "p1_efficiency":       round(p1_eff_raw,  4),
            "patch_quality":       0.0,
            "no_change_detection": 0.0,
            "p2_efficiency":       0.0,
        }

    if ctx.is_valid_issue:
        patch_raw     = grade_patch_quality(declared_patch or "", ctx)
        no_change_raw = 0.0
        p2_eff_raw    = grade_p2_efficiency(
            p2_steps      = sum(1 for r in p2_trajectory if r.phase == 2),
            expected_steps= ctx.expected_p2_steps,
        )
    else:
        # No-change scenario: declared_no_change is the right answer,
        # any patch is wrong.  We grade `no_change` and keep efficiency.
        patch_raw     = 0.0
        no_change_raw = grade_no_change(declared_no_change, ctx)
        p2_eff_raw    = grade_p2_efficiency(
            p2_steps      = sum(1 for r in p2_trajectory if r.phase == 2),
            expected_steps= ctx.expected_p2_steps,
        )

    # Weighted sum
    final = (
        W_P1_RCA        * p1_rca_raw +
        W_P1_EFFICIENCY * p1_eff_raw +
        W_PATCH_QUALITY * patch_raw +
        W_NO_CHANGE     * no_change_raw
    )
    # If no_change wasn't applicable (valid-issue scenario), reallocate
    # its weight to P2 efficiency so weights still sum to 1.0
    if ctx.is_valid_issue:
        final += W_P2_EFFICIENCY * p2_eff_raw - W_NO_CHANGE * no_change_raw

    return {
        "final":               round(final, 4),
        "p1_rca":              round(p1_rca_raw,  4),
        "p1_efficiency":       round(p1_eff_raw,  4),
        "patch_quality":       round(patch_raw,   4),
        "no_change_detection": round(no_change_raw, 4),
        "p2_efficiency":       round(p2_eff_raw,  4),
    }


# ──────────────────────────────────────────────────────────────────────
# Counterfactual r_cross
# ──────────────────────────────────────────────────────────────────────

def compute_r_cross(
    task_name:      str,
    declared_patch: Optional[str],
    declared_no_change: bool,
    p2_trajectory:  List[StepRecord],
) -> float:
    """
    Counterfactual cross-phase reward:
        r_cross = max(0, r_code(τ_2 | context(τ_1)) - r_code(τ_2 | context(∅)))

    The null-context baseline lives on `CodeContext.null_context_p2_score`
    (filled in by `training/run_pool_b_baseline.py`).  We clamp to ≥0 so
    Phase 1 is never punished for inherently hard bugs that no context
    could have helped.
    """
    scenario = get_scenario(task_name)
    ctx      = scenario.code_context
    if ctx is None:
        return 0.0

    if ctx.is_valid_issue:
        with_ctx = grade_patch_quality(declared_patch or "", ctx)
    else:
        with_ctx = grade_no_change(declared_no_change, ctx)

    return max(0.0, with_ctx - ctx.null_context_p2_score)
