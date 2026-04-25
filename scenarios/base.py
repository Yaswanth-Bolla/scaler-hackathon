"""
Base scenario class.

A scenario is *static config*:
  - How to inject faults into the infrastructure (`inject()`)
  - The correct root cause string + keywords (for grading)
  - Which services are involved (for reward shaping)
  - The optional code-attribution context (`code_context` property)
  - The oracle-independent grader (`grade()`)

Per-episode mutable state (phase, p1/p2 trajectories, declared patch,
code workspace) lives on `IncidentEnvironment`, NOT here. A scenario
instance is therefore safe to share across episodes.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from ..models import StepRecord, CodeContext, BeliefState
from ..simulation.infrastructure import Infrastructure


class BaseScenario(ABC):
    """
    Abstract scenario.  Subclasses implement `inject()` and the static
    config properties below.

    Optional Phase-2 support: subclasses override `code_context` to point
    at a bundled mini-repo + ground-truth diff. If `code_context` returns
    `None`, the scenario is Phase-1 only (legacy).
    """

    # ------------------------------------------------------------------
    # Static config (must be overridden by subclasses)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Machine-readable task identifier."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for display."""
        ...

    @property
    @abstractmethod
    def incident_summary(self) -> str:
        """Opening summary shown to the agent at reset."""
        ...

    @property
    @abstractmethod
    def severity(self) -> str:
        """SEV1 / SEV2 / SEV3."""
        ...

    @property
    @abstractmethod
    def correct_root_cause(self) -> str:
        """The canonical root cause string (for grading)."""
        ...

    @property
    @abstractmethod
    def involved_services(self) -> Set[str]:
        """Services that are actually part of the incident (for reward shaping)."""
        ...

    @property
    @abstractmethod
    def root_cause_service(self) -> str:
        """The primary service where the fault originates."""
        ...

    @property
    @abstractmethod
    def correct_remediation_actions(self) -> List[Dict[str, str]]:
        """List of {action_type, target_service} that constitute correct remediation."""
        ...

    @property
    def root_cause_keywords(self) -> List[str]:
        """Keywords that must appear in a correct root cause declaration."""
        return []

    @property
    def time_budget_minutes(self) -> int:
        return 30

    @property
    def max_steps(self) -> int:
        return 20

    # ---- Phase-2 hook (optional) ------------------------------------

    @property
    def code_context(self) -> Optional[CodeContext]:
        """
        Override to enable Phase 2 (code attribution).
        Default: scenario is P1-only.
        """
        return None

    @property
    def fault_class(self) -> str:
        """
        Ground-truth fault class for belief-state aux loss in Stage 2.
        One of: memory_leak | config_change | deadlock | resource_exhaustion |
                cascading | none
        """
        return "none"

    # ---- P2 handoff: synthetic issue text -----------------------------

    def build_p2_issue(self, belief: Optional[BeliefState] = None) -> str:
        """
        Build the synthetic GitHub-issue-style text the code agent reads at
        handoff. Combines the incident summary with whatever runtime evidence
        Phase 1 surfaced. The agent uses this to seed its codebase search.
        """
        lines = [
            f"## Incident: {self.display_name}",
            "",
            self.incident_summary,
            "",
        ]
        if belief is not None:
            lines.append("## Phase-1 diagnosis (handed off)")
            lines.append(f"- Suspected service: **{belief.suspected_service or 'unknown'}**")
            lines.append(f"- Suspected fault class: **{belief.suspected_fault_class or 'unknown'}**")
            lines.append(f"- Service confidence: {belief.service_confidence:.2f}")
            lines.append(f"- Fault confidence:   {belief.fault_confidence:.2f}")
            if belief.evidence_gaps:
                gaps = belief.evidence_gaps if isinstance(belief.evidence_gaps, list) \
                    else [belief.evidence_gaps]
                lines.append(f"- Outstanding evidence gaps: {', '.join(map(str, gaps))}")
            if belief.reasoning:
                lines.append(f"- Reasoning: {belief.reasoning}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Fault injection (must be implemented)
    # ------------------------------------------------------------------

    @abstractmethod
    def inject(self, infra: Infrastructure) -> None:
        """
        Inject faults into the infrastructure.
        Called once at reset time.
        """
        ...

    # ==================================================================
    # Grading — oracle-independent, trajectory-only
    # ==================================================================

    def grade(self, trajectory: List[StepRecord]) -> float:
        """
        P1-only grader (legacy). Returns float in [0.01, 0.99].
        Component breakdown: 40% RCA + 30% remediation + 20% efficiency + 10% restoration.
        """
        score = 0.0
        score += self._grade_root_cause(trajectory)        # 0.00 – 0.40
        score += self._grade_remediation(trajectory)        # 0.00 – 0.30
        score += self._grade_efficiency(trajectory)         # 0.00 – 0.20
        score += self._grade_restoration(trajectory)        # 0.00 – 0.10

        if not math.isfinite(score):
            score = 0.0

        # OpenEnv validator requires strict (0, 1)
        adjusted = 0.01 + (min(max(float(score), 0.0), 1.0) * 0.98)
        if adjusted <= 0.001:
            return 0.01
        if adjusted >= 0.999:
            return 0.99
        return float(round(adjusted, 4))

    # ---- Component graders (used directly by unified grader) ---------

    def grade_p1_rca(self, p1_trajectory: List[StepRecord]) -> float:
        """RCA component in [0, 1] (independent of weight)."""
        return self._grade_root_cause(p1_trajectory) / 0.40

    def grade_p1_efficiency(self, p1_trajectory: List[StepRecord]) -> float:
        """Efficiency component in [0, 1]: 1.0 at 0 steps to declare, 0 at max_steps."""
        declare_step = next(
            (r.step_number for r in p1_trajectory
             if r.action.action_type == "declare_root_cause"),
            self.max_steps,
        )
        return max(0.0, 1.0 - (declare_step / max(self.max_steps, 1)))

    # ---- Internal raw component graders -------------------------------

    def _grade_root_cause(self, trajectory: List[StepRecord]) -> float:
        """0.40 if last declaration matches keywords ≥60%, 0.20 ≥30%, else 0."""
        declarations = [
            s for s in trajectory
            if s.action.action_type == "declare_root_cause"
        ]
        if not declarations:
            return 0.0

        declared = declarations[-1].action.parameters.get("root_cause", "").lower()
        keywords = self.root_cause_keywords or self.correct_root_cause.lower().split()

        matched = sum(1 for kw in keywords if kw in declared)
        match_ratio = matched / len(keywords) if keywords else 0.0

        if match_ratio >= 0.6:
            return 0.40
        if match_ratio >= 0.3:
            return 0.20
        return 0.0

    def _grade_remediation(self, trajectory: List[StepRecord]) -> float:
        """Fraction of correct (action, target) pairs taken, scaled to 0.30."""
        correct = self.correct_remediation_actions
        if not correct:
            return 0.0

        taken = [
            (s.action.action_type, s.action.target_service)
            for s in trajectory
            if s.action.action_type in ("restart_service", "rollback_deploy", "scale_service")
        ]

        matched = sum(
            1 for ca in correct
            if (ca["action_type"], ca["target_service"]) in taken
        )
        return round((matched / len(correct)) * 0.30, 3)

    def _grade_efficiency(self, trajectory: List[StepRecord]) -> float:
        """Step-count tier credit, max 0.20."""
        n = len(trajectory)
        if n == 0:
            return 0.0
        if n <= 6:
            return 0.20
        if n <= 10:
            return 0.15
        if n <= 14:
            return 0.10
        if n <= 17:
            return 0.05
        return 0.02

    def _grade_restoration(self, trajectory: List[StepRecord]) -> float:
        """Final-step service health, max 0.10."""
        if not trajectory:
            return 0.0
        final = trajectory[-1].service_statuses_after
        if not final:
            return 0.0
        if all(s == "healthy" for s in final.values()):
            return 0.10
        healthy = sum(1 for s in final.values() if s == "healthy")
        return round(0.10 * (healthy / len(final)), 3)
