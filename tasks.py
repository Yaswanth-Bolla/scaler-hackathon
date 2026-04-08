"""
Task registry and evaluation graders.

Each task maps to a scenario.  The grader is oracle-independent —
it takes only List[StepRecord] and returns a float in [0, 1].
"""

from __future__ import annotations

from typing import Dict, List, Type

from .models import StepRecord
from .scenarios.base import BaseScenario
from .scenarios.easy_memory_leak import MemoryLeakScenario
from .scenarios.medium_cascading_failure import CascadingFailureScenario
from .scenarios.hard_distributed_deadlock import DistributedDeadlockScenario


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Type[BaseScenario]] = {
    "memory_leak": MemoryLeakScenario,
    "cascading_failure": CascadingFailureScenario,
    "distributed_deadlock": DistributedDeadlockScenario,
}

TASK_NAMES = list(TASK_REGISTRY.keys())


def get_scenario(task_name: str) -> BaseScenario:
    """Instantiate a scenario by task name."""
    cls = TASK_REGISTRY.get(task_name)
    if cls is None:
        raise ValueError(
            f"Unknown task: {task_name}. Available: {TASK_NAMES}")
    return cls()


def grade_trajectory(task_name: str, trajectory: List[StepRecord]) -> float:
    """
    Grade a trajectory for a given task.
    This is the evaluation entry point — standalone, no hidden state.
    """
    scenario = get_scenario(task_name)
    score = scenario.grade(trajectory)
    return max(0.01, min(0.99, float(score)))
