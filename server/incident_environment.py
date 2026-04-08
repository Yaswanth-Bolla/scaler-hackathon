"""
Core Environment implementation.

Execution order per step:  validate → mutate → tick → observe → reward.

The environment uses oracle-shaped rewards for training (they peek at hidden
state to compute whether the agent investigated the right service) but the
grader used for evaluation is oracle-independent (trajectory-only).
"""

from __future__ import annotations

import uuid
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from ..models import (
    ActionType,
    IncidentAction,
    IncidentObservation,
    IncidentState,
    StepRecord,
    DIAGNOSTIC_ACTIONS,
    REMEDIATION_ACTIONS,
    TARGETED_ACTIONS,
)
from ..simulation.infrastructure import Infrastructure, SERVICE_NAMES
from ..tasks import get_scenario, TASK_NAMES
from ..scenarios.base import BaseScenario


class IncidentEnvironment:
    """
    SRE Incident Response Environment.

    Implements the three OpenEnv methods:
      - reset(task_name) → IncidentObservation
      - step(action)     → dict with observation, reward, done
      - state()          → IncidentState
    """

    def __init__(self) -> None:
        self._infra: Optional[Infrastructure] = None
        self._scenario: Optional[BaseScenario] = None
        self._state = IncidentState()
        self._trajectory: List[StepRecord] = []
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._root_cause_declared: bool = False

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Initialize a new incident episode.

        Args:
            task_name: One of "memory_leak", "cascading_failure", "distributed_deadlock".
                       If None, picks randomly.
            seed: Optional random seed for reproducibility.

        Returns:
            Dict with observation, reward=0.0, done=False.
        """
        if seed is not None:
            random.seed(seed)

        if task_name is None:
            task_name = random.choice(TASK_NAMES)

        # Create fresh infrastructure
        self._infra = Infrastructure()
        self._scenario = get_scenario(task_name)
        self._infra.time_budget_minutes = self._scenario.time_budget_minutes

        # Inject scenario faults
        self._scenario.inject(self._infra)

        # Run a few ticks to let cascades propagate
        for _ in range(3):
            self._infra.tick()

        # Reset episode state
        self._state = IncidentState(
            episode_id=str(uuid.uuid4()),
            task_name=task_name,
            step_count=0,
            time_elapsed_minutes=self._infra.current_minute,
            done=False,
            cumulative_reward=0.0,
        )
        self._trajectory = []
        self._cumulative_reward = 0.0
        self._done = False
        self._root_cause_declared = False

        obs = self._build_observation(
            action_result={"message": "Incident triggered. Begin investigation."},
            action_success=True,
            action_message="Episode started",
            reward=0.0,
        )

        return {
            "observation": obs,
            "reward": 0.01,
            "done": False,
        }

    # ------------------------------------------------------------------
    # step()  —  validate → mutate → tick → observe → reward
    # ------------------------------------------------------------------

    def step(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one agent action.

        Args:
            action_data: Dict with action_type, target_service, parameters.

        Returns:
            Dict with observation, reward, done, info.
        """
        if self._done:
            obs = self._build_observation(
                action_result={"error": "Episode is already done."},
                action_success=False,
                action_message="Episode already finished",
                reward=0.0,
            )
            final_grade = self._scenario.grade(self._trajectory) if self._scenario else 0.01
            return {"observation": obs, "reward": 0.01, "done": True, "info": {"score": final_grade}}

        if self._infra is None or self._scenario is None:
            obs = self._build_observation(
                action_result={"error": "Environment not initialized. Call reset() first."},
                action_success=False,
                action_message="Not initialized",
                reward=0.0,
            )
            return {"observation": obs, "reward": 0.01, "done": False, "info": {}}

        # Parse action
        action = IncidentAction(
            action_type=action_data.get("action_type", ""),
            target_service=action_data.get("target_service"),
            parameters=action_data.get("parameters", {}),
        )

        # ---- VALIDATE ----
        is_valid, error_msg = self._infra.validate_action(
            action.action_type, action.target_service)

        if not is_valid:
            reward = -0.05
            self._cumulative_reward += reward
            self._state.step_count += 1
            obs = self._build_observation(
                action_result={"error": error_msg},
                action_success=False,
                action_message=f"Invalid action: {error_msg}",
                reward=reward,
            )
            self._record_step(action, reward, obs)
            return {"observation": obs, "reward": reward, "done": False, "info": {"error": error_msg}}

        # ---- MUTATE ----
        action_result, action_msg = self._execute_action(action)

        # ---- TICK ----
        self._infra.tick()
        self._state.step_count += 1
        self._state.time_elapsed_minutes = self._infra.current_minute

        # ---- REWARD (oracle-shaped for training) ----
        # Must compute BEFORE recording — so repeat detection doesn't
        # flag the current action as already taken.
        reward = self._compute_reward(action)
        self._infra.record_action(action.action_type, action.target_service)
        self._cumulative_reward += reward
        self._state.cumulative_reward = self._cumulative_reward

        # ---- CHECK DONE ----
        done = self._check_done(action)
        self._done = done
        self._state.done = done

        # ---- OBSERVE ----
        obs = self._build_observation(
            action_result=action_result,
            action_success=True,
            action_message=action_msg,
            reward=reward,
        )

        self._record_step(action, reward, obs)

        info: Dict[str, Any] = {}
        if done:
            # Compute final grade (oracle-independent)
            final_grade = self._scenario.grade(self._trajectory)
            info["score"] = final_grade
            info["task_name"] = self._scenario.task_name
            info["steps_taken"] = self._state.step_count
            info["trajectory_length"] = len(self._trajectory)

        return {"observation": obs, "reward": reward, "done": done, "info": info}

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    @property
    def state(self) -> IncidentState:
        return self._state

    def get_state(self) -> Dict[str, Any]:
        return {
            "episode_id": self._state.episode_id,
            "task_name": self._state.task_name,
            "step_count": self._state.step_count,
            "time_elapsed_minutes": self._state.time_elapsed_minutes,
            "done": self._state.done,
            "cumulative_reward": round(self._state.cumulative_reward, 3),
            "declared_root_cause": self._state.declared_root_cause,
        }

    # ------------------------------------------------------------------
    # Action execution handlers
    # ------------------------------------------------------------------

    def _execute_action(
        self, action: IncidentAction
    ) -> Tuple[Dict[str, Any], str]:
        """Execute a validated action. Returns (result_dict, message)."""
        at = action.parsed_type()
        target = action.target_service

        if at == ActionType.VIEW_ALERTS:
            alerts = self._infra.get_alerts()
            return {"alerts": alerts, "count": len(alerts)}, f"Viewing {len(alerts)} active alerts"

        elif at == ActionType.QUERY_LOGS:
            level_filter = action.parameters.get("level")
            keyword = action.parameters.get("keyword")
            limit = action.parameters.get("limit", 15)
            logs = self._infra.get_logs_for_service(target, level_filter, keyword, limit)
            return {"logs": logs, "count": len(logs), "service": target}, \
                   f"Queried {len(logs)} logs from {target}"

        elif at == ActionType.CHECK_METRICS:
            metrics = self._infra.get_metrics_for_service(target)
            return {"metrics": metrics, "service": target, "data_points": len(metrics)}, \
                   f"Retrieved {len(metrics)} metric data points for {target}"

        elif at == ActionType.CHECK_DEPENDENCIES:
            deps = self._infra.get_dependencies_for_service(target)
            return {"dependencies": deps, "service": target}, \
                   f"Retrieved dependency map for {target}"

        elif at == ActionType.CHECK_DEPLOY_HISTORY:
            deploys = self._infra.get_deploy_history_for_service(target)
            return {"deploys": deploys, "service": target, "count": len(deploys)}, \
                   f"Retrieved {len(deploys)} deploys for {target}"

        elif at == ActionType.RUN_HEALTH_CHECK:
            health = self._infra.run_health_check(target)
            return {"health_check": health, "service": target}, \
                   f"Health check for {target}: {health['status']}"

        elif at == ActionType.RESTART_SERVICE:
            svc = self._infra.get_service(target)
            msg = svc.restart(self._infra.current_minute) if svc else "Service not found"
            return {"result": msg, "service": target}, msg

        elif at == ActionType.ROLLBACK_DEPLOY:
            svc = self._infra.get_service(target)
            msg = svc.rollback_deploy(self._infra.current_minute) if svc else "Service not found"
            return {"result": msg, "service": target}, msg

        elif at == ActionType.SCALE_SERVICE:
            svc = self._infra.get_service(target)
            new_replicas = action.parameters.get("replicas", 5)
            msg = svc.scale(new_replicas, self._infra.current_minute) if svc else "Service not found"
            return {"result": msg, "service": target}, msg

        elif at == ActionType.DECLARE_ROOT_CAUSE:
            root_cause = action.parameters.get("root_cause", "")
            self._state.declared_root_cause = root_cause
            self._root_cause_declared = True
            return {
                "declared": root_cause,
                "message": "Root cause declaration registered. Episode will end after this step.",
            }, f"Root cause declared: {root_cause}"

        else:
            return {"error": f"Unhandled action type: {at}"}, "Unknown action"

    # ------------------------------------------------------------------
    # Reward computation (oracle-shaped — Layer 6)
    # ------------------------------------------------------------------

    def _compute_reward(self, action: IncidentAction) -> float:
        """
        Compute per-step reward using oracle-shaped signal.
        The training reward has access to hidden state (involved_services,
        root_cause_service) — this is necessary for learning.
        The GRADER does NOT use this; it scores trajectory-only.
        """
        at = action.parsed_type()
        target = action.target_service
        scenario = self._scenario
        reward = 0.0

        # --- Step penalty (efficiency pressure) ---
        reward -= 0.02

        # --- Repeat detection ---
        if self._infra.was_action_taken(action.action_type, target):
            reward -= 0.05
            return round(reward, 3)

        # --- Diagnostic actions ---
        if at in DIAGNOSTIC_ACTIONS:
            if target and target in scenario.involved_services:
                reward += 0.15  # Investigating a relevant service
            elif target and target not in scenario.involved_services:
                reward += 0.05  # Exploring — not penalized heavily
            elif at == ActionType.VIEW_ALERTS:
                reward += 0.15  # Always good to view alerts

        # --- Remediation actions ---
        elif at in REMEDIATION_ACTIONS:
            if target == scenario.root_cause_service:
                reward += 0.30  # Correct remediation target
            elif target and target in scenario.involved_services:
                reward += 0.10  # Helpful but not the root cause
            else:
                reward -= 0.15  # Remediating healthy/uninvolved service

        # --- Root cause declaration ---
        elif at == ActionType.DECLARE_ROOT_CAUSE:
            declared = action.parameters.get("root_cause", "").lower()
            keywords = scenario.root_cause_keywords
            if keywords:
                matched = sum(1 for kw in keywords if kw in declared)
                ratio = matched / len(keywords)
                if ratio >= 0.6:
                    reward += 0.40  # Correct
                elif ratio >= 0.3:
                    reward += 0.15  # Partial
                else:
                    reward -= 0.20  # Wrong
            else:
                reward -= 0.20

        # --- Episode completion bonus/penalty ---
        if self._root_cause_declared:
            if self._infra.all_services_healthy():
                reward += 0.20  # All services restored
            if self._infra.current_minute > self._infra.time_budget_minutes:
                reward -= 0.10  # Exceeded time budget

        return round(reward, 3)

    # ------------------------------------------------------------------
    # Done check
    # ------------------------------------------------------------------

    def _check_done(self, action: IncidentAction) -> bool:
        """Episode ends when root cause is declared or max steps reached."""
        if self._root_cause_declared:
            return True
        if self._state.step_count >= self._scenario.max_steps:
            return True
        return False

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        action_result: Dict[str, Any],
        action_success: bool,
        action_message: str,
        reward: float,
    ) -> Dict[str, Any]:
        """Build the POMDP observation dict (no hidden state exposed)."""
        statuses = self._infra.get_all_statuses() if self._infra else {}
        alerts = self._infra.get_alerts() if self._infra else []
        valid_actions = self._infra.get_valid_actions() if self._infra else []

        return {
            "incident_summary": self._scenario.incident_summary if self._scenario else "",
            "severity": self._scenario.severity if self._scenario else "SEV3",
            "time_elapsed_minutes": self._infra.current_minute if self._infra else 0,
            "time_budget_minutes": self._infra.time_budget_minutes if self._infra else 30,
            "action_result": action_result,
            "action_success": action_success,
            "action_message": action_message,
            "service_statuses": statuses,
            "active_alerts_count": len(alerts),
            "valid_actions": valid_actions,
            "available_services": list(SERVICE_NAMES),
            "current_reward": reward,
            "cumulative_reward": round(self._cumulative_reward, 3),
            "steps_taken": self._state.step_count,
            "max_steps": self._scenario.max_steps if self._scenario else 20,
            "done": self._done,
        }

    # ------------------------------------------------------------------
    # Trajectory recording
    # ------------------------------------------------------------------

    def _record_step(
        self,
        action: IncidentAction,
        reward: float,
        observation: Dict[str, Any],
    ) -> None:
        """Record step for trajectory-based grading."""
        record = StepRecord(
            step_number=self._state.step_count,
            action=action,
            reward=reward,
            observation_summary={
                "action_message": observation.get("action_message", ""),
                "active_alerts_count": observation.get("active_alerts_count", 0),
            },
            service_statuses_after=dict(observation.get("service_statuses", {})),
            timestamp_minutes=self._infra.current_minute if self._infra else 0,
        )
        self._trajectory.append(record)

    # ------------------------------------------------------------------
    # Trajectory access (for external grading)
    # ------------------------------------------------------------------

    def get_trajectory(self) -> List[StepRecord]:
        return list(self._trajectory)
