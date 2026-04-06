"""
Base scenario class.

Each scenario defines:
  - How to inject faults into the infrastructure
  - The correct root cause string
  - Which services are involved (for reward shaping)
  - The oracle grader (trajectory-only, no hidden state)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set

from ..simulation.infrastructure import Infrastructure
from ..models import StepRecord


class BaseScenario(ABC):
    """
    Abstract scenario.  Subclasses implement inject() and grade().

    inject() mutates the infrastructure to set up the incident.
    grade() evaluates a complete trajectory WITHOUT access to hidden state.
    """

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
        """SEV1/SEV2/SEV3."""
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

    @abstractmethod
    def inject(self, infra: Infrastructure) -> None:
        """
        Inject faults into the infrastructure.
        Called once at reset time.
        """
        ...

    # ---------------------------------------------------------------
    # Grading — oracle-independent, trajectory-only (Layer 6)
    # ---------------------------------------------------------------

    def grade(self, trajectory: List[StepRecord]) -> float:
        """
        Grade the complete trajectory.
        Returns float in [0.0, 1.0].

        This function receives ONLY the step records — no hidden state,
        no infrastructure reference. This is critical: the evaluation
        harness must be able to call this on a saved trajectory.
        """
        score = 0.0
        score += self._grade_root_cause(trajectory)        # 0.00 – 0.40
        score += self._grade_remediation(trajectory)        # 0.00 – 0.30
        score += self._grade_efficiency(trajectory)         # 0.00 – 0.20
        score += self._grade_restoration(trajectory)        # 0.00 – 0.10
        return max(0.0, min(1.0, score))

    def _grade_root_cause(self, trajectory: List[StepRecord]) -> float:
        """
        Did the agent correctly declare the root cause?
        Full credit (0.40) for correct, partial credit for close.
        """
        declarations = [
            s for s in trajectory
            if s.action.action_type == "declare_root_cause"
        ]
        if not declarations:
            return 0.0  # Never declared — 0 points

        # Use the LAST declaration
        declared = declarations[-1].action.parameters.get("root_cause", "").lower()

        # Check keyword match
        keywords = self.root_cause_keywords
        if not keywords:
            keywords = self.correct_root_cause.lower().split()

        matched = sum(1 for kw in keywords if kw in declared)
        match_ratio = matched / len(keywords) if keywords else 0

        if match_ratio >= 0.6:
            return 0.40  # Close enough — full credit
        elif match_ratio >= 0.3:
            return 0.20  # Partial credit
        else:
            return 0.0

    def _grade_remediation(self, trajectory: List[StepRecord]) -> float:
        """
        Did the agent take the correct fix actions?
        """
        correct_actions = self.correct_remediation_actions
        if not correct_actions:
            return 0.0

        taken_remediations = [
            (s.action.action_type, s.action.target_service)
            for s in trajectory
            if s.action.action_type in ("restart_service", "rollback_deploy", "scale_service")
        ]

        matched = 0
        for ca in correct_actions:
            needed = (ca["action_type"], ca["target_service"])
            if needed in taken_remediations:
                matched += 1

        ratio = matched / len(correct_actions)
        return round(ratio * 0.30, 3)

    def _grade_efficiency(self, trajectory: List[StepRecord]) -> float:
        """
        Fewer steps to reach correct diagnosis = more points.
        Optimal path (for the scenario) gets full credit.
        """
        total_steps = len(trajectory)
        if total_steps == 0:
            return 0.0

        # Generous: < 8 steps is excellent, 8-12 is good, 13-16 is okay, 17+ is bad
        if total_steps <= 6:
            return 0.20
        elif total_steps <= 10:
            return 0.15
        elif total_steps <= 14:
            return 0.10
        elif total_steps <= 17:
            return 0.05
        else:
            return 0.02

    def _grade_restoration(self, trajectory: List[StepRecord]) -> float:
        """
        Are all services healthy at the end of the episode?
        Check the LAST step's service_statuses_after.
        """
        if not trajectory:
            return 0.0

        final_statuses = trajectory[-1].service_statuses_after
        if all(s == "healthy" for s in final_statuses.values()):
            return 0.10
        # Partial credit: how many are healthy
        healthy_count = sum(1 for s in final_statuses.values() if s == "healthy")
        total = len(final_statuses) if final_statuses else 1
        return round(0.10 * (healthy_count / total), 3)
