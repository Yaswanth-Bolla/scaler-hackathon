"""
Core Environment implementation.

Per-step execution order:  validate → mutate → tick → observe → reward.

Two-phase architecture:
  Phase 1 — ops/SRE diagnostic loop (existing behavior).
  Phase 2 — code attribution loop, sandboxed under a CodeWorkspace.

Mode selection is automatic per scenario:
  - Scenario with `code_context = None`  → legacy P1-only episode
                                            (declare_root_cause terminates)
  - Scenario with `code_context != None` → unified P1 → P2 episode
                                            (declare_root_cause is silent;
                                             transition_to_phase2 switches phase;
                                             propose_patch / declare_no_change
                                             terminate the episode)

The environment uses oracle-shaped per-step rewards for training.  The
oracle-INDEPENDENT graders live on `BaseScenario` and `scenarios.grader_p2`.
"""

from __future__ import annotations

import uuid
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from ..models import (
    ActionType,
    IncidentAction,
    IncidentState,
    StepRecord,
    BeliefState,
    DIAGNOSTIC_ACTIONS,
    REMEDIATION_ACTIONS,
    TARGETED_ACTIONS,
    PHASE1_ACTIONS,
    PHASE2_ACTIONS,
    PHASE2_DIAGNOSTIC_ACTIONS,
    PHASE2_TERMINAL_ACTIONS,
)
from ..simulation.infrastructure import Infrastructure, SERVICE_NAMES
from ..tasks import get_scenario, TASK_NAMES
from ..scenarios.base import BaseScenario
from ..pools import POOLS, get_pool, sample_task, oracle_belief
from .code_workspace import CodeWorkspace, CodeWorkspaceError


# Per-step reward constants ------------------------------------------------
_STEP_PENALTY     = -0.02
_REPEAT_PENALTY   = -0.05
_INVALID_PENALTY  = -0.05

# Phase 2 shaping (small — terminal patch quality is graded post-hoc)
_P2_DIAG_REWARD   = +0.05
_P2_TERMINAL_BONUS = +0.10


class IncidentEnvironment:
    """
    SRE Incident Response Environment.

    Implements the three OpenEnv methods:
      - reset(task_name)  → initial observation + info
      - step(action)      → dict with observation, reward, done, info
      - state()           → IncidentState for monitoring

    Plus two extras used by the unified evaluator:
      - get_trajectory()        → P1 + P2 step records
      - score_unified(...)      → component scores for unified grader
    """

    def __init__(self) -> None:
        self._infra: Optional[Infrastructure] = None
        self._scenario: Optional[BaseScenario] = None
        self._state = IncidentState()

        # ---- Per-episode mutable state ----
        self._phase: int = 1
        self._workspace: Optional[CodeWorkspace] = None
        self._belief_at_transition: Optional[BeliefState] = None
        self._p1_trajectory: List[StepRecord] = []
        self._p2_trajectory: List[StepRecord] = []
        self._declared_patch: Optional[str] = None
        self._declared_no_change: bool = False
        self._declared_root_cause: Optional[str] = None
        self._cumulative_reward: float = 0.0
        self._done: bool = False

        # ---- Pool / mode (set by reset, drives episode semantics) ----
        # mode in {"joint" (default), "p1_only" (Pool A), "p2_only" (Pool B)}
        self._pool: Optional[str] = None
        self._mode: str = "joint"
        self._inject_oracle_belief: bool = False

        # P2-only tracking (for repeat detection inside P2)
        self._p2_actions_taken: List[Tuple[str, str]] = []   # (atype, primary_param)

    # ==================================================================
    # reset()
    # ==================================================================

    def reset(
        self,
        task_name: Optional[str] = None,
        seed:      Optional[int] = None,
        pool:      Optional[str] = None,
        mode:      Optional[str] = None,
        **kwargs:  Any,
    ) -> Dict[str, Any]:
        """
        Initialize a new incident episode.

        `pool`  selects training pool A/B/C/D (overrides default mode).
        `mode`  forces episode semantics ("p1_only"|"p2_only"|"joint").
                Explicit `mode` always wins over pool defaults.
        """
        if seed is not None:
            random.seed(seed)

        # ---- Pool / task selection ----
        pool_obj = None
        if pool:
            pool_obj = get_pool(pool)
            if task_name is None:
                task_name = sample_task(pool, rng=random)
            self._pool = pool_obj.name
            self._mode = pool_obj.mode
            self._inject_oracle_belief = pool_obj.inject_oracle_belief
        else:
            self._pool = None
            self._mode = "joint"
            self._inject_oracle_belief = False

        if mode:
            self._mode = mode
            if mode == "p2_only":
                self._inject_oracle_belief = True

        if task_name is None:
            task_name = random.choice(TASK_NAMES)

        self._infra    = Infrastructure()
        self._scenario = get_scenario(task_name)
        self._infra.time_budget_minutes = self._scenario.time_budget_minutes
        self._scenario.inject(self._infra)

        # Let cascades propagate a few minutes
        for _ in range(3):
            self._infra.tick()

        self._state = IncidentState(
            episode_id           = str(uuid.uuid4()),
            task_name            = task_name,
            step_count           = 0,
            time_elapsed_minutes = self._infra.current_minute,
            done                 = False,
            cumulative_reward    = 0.0,
        )
        self._phase                 = 1
        self._workspace             = None
        self._belief_at_transition  = None
        self._p1_trajectory         = []
        self._p2_trajectory         = []
        self._declared_patch        = None
        self._declared_no_change    = False
        self._declared_root_cause   = None
        self._cumulative_reward     = 0.0
        self._done                  = False
        self._p2_actions_taken      = []

        # ---- Pool B (p2_only) auto-handoff with oracle belief --------
        # The agent never sees Phase 1; we synthesise a perfect handoff and
        # immediately switch the env into Phase 2.
        if self._mode == "p2_only" and self._scenario.code_context is not None:
            belief = oracle_belief(self._scenario)
            self._handle_transition(IncidentAction(
                action_type    = ActionType.TRANSITION_TO_PHASE2.value,
                target_service = None,
                parameters     = {"belief": asdict(belief)},
            ))
            # _handle_transition already returned; we just consume its
            # observation as the reset observation so caller sees Phase 2.
            obs = self._build_observation(
                action_result   = {
                    "message":          "[Pool B] Auto-handoff with oracle Phase-1 belief.",
                    "issue":            self._scenario.build_p2_issue(belief),
                    "file_tree":        (self._workspace.file_tree(max_depth=4)
                                         if self._workspace else []),
                    "bad_commit_sha":   self._scenario.code_context.bad_commit_sha,
                    "bad_commit":       (self._workspace.bad_commit_metadata()
                                         if self._workspace else None),
                },
                action_success  = True,
                action_message  = "Episode started in Pool B (P2-only) mode",
                reward          = 0.0,
            )
            return {
                "observation": obs,
                "reward":      0.01,
                "done":        False,
                "info":        {"task_name": task_name,
                                "pool":      self._pool,
                                "mode":      self._mode,
                                "has_phase2": True,
                                "phase":     2},
            }

        obs = self._build_observation(
            action_result   = {"message": "Incident triggered. Begin investigation."},
            action_success  = True,
            action_message  = "Episode started",
            reward          = 0.0,
        )
        return {
            "observation": obs,
            "reward":      0.01,
            "done":        False,
            "info":        {"task_name":  task_name,
                            "pool":       self._pool,
                            "mode":       self._mode,
                            "has_phase2": self._scenario.code_context is not None},
        }

    # ==================================================================
    # step()
    # ==================================================================

    def step(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one agent action — phase-aware dispatch."""
        if self._done:
            return self._final_step_response()

        if self._infra is None or self._scenario is None:
            return self._not_initialized_response()

        action = IncidentAction(
            action_type    = action_data.get("action_type", ""),
            target_service = action_data.get("target_service"),
            parameters     = action_data.get("parameters", {}) or {},
        )

        # ---- Type validation ----------------------------------------
        try:
            atype = ActionType(action.action_type)
        except ValueError:
            return self._invalid_action_response(
                f"Unknown action type: {action.action_type!r}",
                action,
            )

        # ---- Phase-aware dispatch -----------------------------------
        if atype == ActionType.TRANSITION_TO_PHASE2:
            return self._handle_transition(action)

        if self._phase == 1:
            if atype not in PHASE1_ACTIONS:
                return self._invalid_action_response(
                    f"Action {atype.value!r} not allowed in Phase 1", action,
                )
            return self._step_phase1(action, atype)

        # Phase 2
        if atype not in PHASE2_ACTIONS:
            return self._invalid_action_response(
                f"Action {atype.value!r} not allowed in Phase 2", action,
            )
        return self._step_phase2(action, atype)

    # ------------------------------------------------------------------
    # Phase 1 step
    # ------------------------------------------------------------------

    def _step_phase1(
        self,
        action: IncidentAction,
        atype:  ActionType,
    ) -> Dict[str, Any]:
        # Validate target / preconditions via Infrastructure
        is_valid, err = self._infra.validate_action(
            action.action_type, action.target_service)
        if not is_valid:
            return self._invalid_action_response(err, action)

        # Mutate
        action_result, action_msg = self._execute_p1_action(action, atype)

        # Tick simulation
        self._infra.tick()
        self._state.step_count           += 1
        self._state.time_elapsed_minutes = self._infra.current_minute

        # Reward (compute BEFORE recording so repeat-detection sees prior actions)
        reward = self._compute_p1_reward(action, atype)
        self._infra.record_action(action.action_type, action.target_service)

        self._cumulative_reward += reward
        self._state.cumulative_reward = self._cumulative_reward

        # Done check
        done = self._check_done_p1(atype)
        self._done = done
        self._state.done = done

        obs = self._build_observation(
            action_result   = action_result,
            action_success  = True,
            action_message  = action_msg,
            reward          = reward,
        )

        record = StepRecord(
            step_number             = self._state.step_count,
            action                  = action,
            reward                  = reward,
            observation_summary     = {
                "action_message":      obs.get("action_message", ""),
                "active_alerts_count": obs.get("active_alerts_count", 0),
            },
            service_statuses_after  = dict(obs.get("service_statuses", {})),
            timestamp_minutes       = self._infra.current_minute,
            phase                   = 1,
        )
        self._p1_trajectory.append(record)

        info: Dict[str, Any] = {}
        if done:
            info["score"]            = self._scenario.grade(self._p1_trajectory)
            info["task_name"]        = self._scenario.task_name
            info["steps_taken"]      = self._state.step_count
            info["trajectory_length"] = len(self._p1_trajectory)

        return {"observation": obs, "reward": reward, "done": done, "info": info}

    # ------------------------------------------------------------------
    # Phase 2 step
    # ------------------------------------------------------------------

    def _step_phase2(
        self,
        action: IncidentAction,
        atype:  ActionType,
    ) -> Dict[str, Any]:
        if self._workspace is None:
            return self._invalid_action_response(
                "Phase 2 not initialised — must transition_to_phase2 first.",
                action,
            )

        params = action.parameters or {}

        # ---- Execute action ----
        try:
            if atype == ActionType.LIST_DIR:
                result = self._workspace.list_dir(params.get("path", "."))
                msg = f"Listed {result.get('count', 0)} entries in {result.get('path', '.')}"
            elif atype == ActionType.READ_FILE:
                result = self._workspace.read_file(params.get("path", ""))
                msg = f"Read {result.get('path')} ({result.get('size', 0)} bytes)"
            elif atype == ActionType.SEARCH_CODE:
                result = self._workspace.search_code(
                    query        = params.get("query", ""),
                    file_pattern = params.get("file_pattern", "*.py"),
                    max_hits     = params.get("max_hits"),
                )
                msg = f"Found {result.get('count', 0)} hit(s) for {params.get('query')!r}"
            elif atype == ActionType.GET_GIT_LOG:
                result = self._workspace.get_git_log(
                    path      = params.get("path", ""),
                    n_commits = int(params.get("n_commits", 10)),
                )
                msg = f"Returned {result.get('count', 0)} commit(s)"
            elif atype == ActionType.GET_FILE_DIFF:
                result = self._workspace.get_file_diff(
                    commit_sha = params.get("commit_sha", ""),
                    path       = params.get("path", ""),
                )
                msg = f"Diff for {result.get('commit_sha')[:8]} ({len(result.get('diff', ''))} bytes)"
            elif atype == ActionType.PROPOSE_PATCH:
                diff = params.get("diff", "")
                self._declared_patch = diff
                result = {"accepted": True, "patch_bytes": len(diff)}
                msg = "Patch proposal accepted — episode terminating."
            elif atype == ActionType.DECLARE_NO_CHANGE:
                self._declared_no_change = True
                reason = params.get("reason", "")
                result = {"accepted": True, "reason": reason}
                msg = "no-change declaration accepted — episode terminating."
            else:
                return self._invalid_action_response(
                    f"Unhandled P2 action type: {atype.value!r}", action,
                )
            success = True
        except CodeWorkspaceError as e:
            result  = {"error": str(e)}
            msg     = f"Workspace error: {e}"
            success = False

        # ---- Tick (simulation time still advances during P2) ----
        self._infra.tick()
        self._state.step_count           += 1
        self._state.time_elapsed_minutes = self._infra.current_minute

        # ---- Reward ----
        reward = self._compute_p2_reward(action, atype, success)
        self._cumulative_reward += reward
        self._state.cumulative_reward = self._cumulative_reward

        # ---- Done ----
        done = (atype in PHASE2_TERMINAL_ACTIONS) or self._exceeded_step_budget()
        self._done = done
        self._state.done = done

        obs = self._build_observation(
            action_result   = result,
            action_success  = success,
            action_message  = msg,
            reward          = reward,
        )

        # Record
        record = StepRecord(
            step_number             = self._state.step_count,
            action                  = action,
            reward                  = reward,
            observation_summary     = {
                "action_message": obs.get("action_message", ""),
                "p2_action":      atype.value,
            },
            service_statuses_after  = dict(obs.get("service_statuses", {})),
            timestamp_minutes       = self._infra.current_minute,
            phase                   = 2,
        )
        self._p2_trajectory.append(record)

        # Track repeats inside P2
        prim_param = self._p2_primary_param(atype, params)
        self._p2_actions_taken.append((atype.value, prim_param))

        info: Dict[str, Any] = {}
        if done:
            info["score"]            = self._compute_unified_final_score()
            info["task_name"]        = self._scenario.task_name
            info["steps_taken"]      = self._state.step_count
            info["trajectory_length"] = len(self._p1_trajectory) + len(self._p2_trajectory)

        return {"observation": obs, "reward": reward, "done": done, "info": info}

    # ------------------------------------------------------------------
    # transition_to_phase2 handler
    # ------------------------------------------------------------------

    def _handle_transition(self, action: IncidentAction) -> Dict[str, Any]:
        if self._phase != 1:
            return self._invalid_action_response(
                "Already in Phase 2 — cannot transition again.", action,
            )
        if self._scenario is None or self._scenario.code_context is None:
            return self._invalid_action_response(
                "Scenario has no code_context — Phase 2 unavailable.", action,
            )

        ctx = self._scenario.code_context

        # Construct workspace
        try:
            self._workspace = CodeWorkspace(
                snapshot_root  = ctx.repo_snapshot_path,
                bad_commit_sha = ctx.bad_commit_sha,
            )
        except CodeWorkspaceError as e:
            return self._invalid_action_response(
                f"Cannot open snapshot: {e}", action,
            )

        # Capture handoff belief
        belief_dict = (action.parameters or {}).get("belief") or {}
        self._belief_at_transition = self._coerce_belief(belief_dict)

        # Switch phase
        self._phase                       = 2
        self._state.step_count           += 1
        self._infra.tick()
        self._state.time_elapsed_minutes  = self._infra.current_minute

        # Initial P2 obs
        issue_text = self._scenario.build_p2_issue(self._belief_at_transition)
        file_tree  = self._workspace.file_tree(max_depth=4)
        commit_meta = self._workspace.bad_commit_metadata()

        action_result = {
            "phase":            2,
            "issue":            issue_text,
            "file_tree":        file_tree,
            "bad_commit_sha":   ctx.bad_commit_sha,
            "bad_commit":       commit_meta,
            "snapshot_root":    str(self._workspace.tree_root),
        }

        # Reward: small handoff bonus only when belief is non-trivial
        reward = 0.0
        if self._belief_at_transition.suspected_service:
            reward += 0.05
        self._cumulative_reward += reward
        self._state.cumulative_reward = self._cumulative_reward

        obs = self._build_observation(
            action_result   = action_result,
            action_success  = True,
            action_message  = "Transitioned to Phase 2 (code attribution).",
            reward          = reward,
        )

        record = StepRecord(
            step_number             = self._state.step_count,
            action                  = action,
            reward                  = reward,
            observation_summary     = {
                "action_message": "transition_to_phase2",
                "transition":     True,
            },
            service_statuses_after  = dict(obs.get("service_statuses", {})),
            timestamp_minutes       = self._infra.current_minute,
            phase                   = 2,
            belief_state_snapshot   = asdict(self._belief_at_transition),
        )
        self._p2_trajectory.append(record)

        return {"observation": obs, "reward": reward, "done": False, "info": {}}

    @staticmethod
    def _coerce_belief(d: Dict[str, Any]) -> BeliefState:
        """Best-effort: turn an inference-side dict into the canonical BeliefState."""
        gaps = d.get("evidence_gaps", [])
        if isinstance(gaps, str):
            gaps = [g.strip() for g in gaps.split(",") if g.strip() and g.strip() != "none"]
        return BeliefState(
            suspected_service     = d.get("suspected_service") or None,
            suspected_fault_class = d.get("suspected_fault_class") or None,
            service_confidence    = float(d.get("service_confidence") or 0.0),
            fault_confidence      = float(d.get("fault_confidence") or 0.0),
            evidence_gaps         = list(gaps),
            estimated_p2_cost     = d.get("estimated_p2_cost") or "unknown",
            decision              = d.get("decision") or "transition",
            reasoning             = d.get("reasoning") or "",
        )

    # ==================================================================
    # state()
    # ==================================================================

    @property
    def state(self) -> IncidentState:
        return self._state

    def get_state(self) -> Dict[str, Any]:
        return {
            "episode_id":           self._state.episode_id,
            "task_name":            self._state.task_name,
            "step_count":           self._state.step_count,
            "time_elapsed_minutes": self._state.time_elapsed_minutes,
            "done":                 self._state.done,
            "cumulative_reward":    round(self._state.cumulative_reward, 3),
            "declared_root_cause":  self._declared_root_cause,
            "declared_patch":       self._declared_patch,
            "declared_no_change":   self._declared_no_change,
            "phase":                self._phase,
            "phase_transition_at":  next(
                (r.step_number for r in self._p2_trajectory
                 if r.action.action_type == ActionType.TRANSITION_TO_PHASE2.value),
                None,
            ),
        }

    # ==================================================================
    # Phase 1 action execution
    # ==================================================================

    def _execute_p1_action(
        self,
        action: IncidentAction,
        atype:  ActionType,
    ) -> Tuple[Dict[str, Any], str]:
        target = action.target_service
        params = action.parameters or {}

        if atype == ActionType.VIEW_ALERTS:
            alerts = self._infra.get_alerts()
            return {"alerts": alerts, "count": len(alerts)}, \
                   f"Viewing {len(alerts)} active alerts"

        if atype == ActionType.QUERY_LOGS:
            level   = params.get("level")
            keyword = params.get("keyword")
            limit   = params.get("limit", 15)
            logs = self._infra.get_logs_for_service(target, level, keyword, limit)
            return {"logs": logs, "count": len(logs), "service": target}, \
                   f"Queried {len(logs)} logs from {target}"

        if atype == ActionType.CHECK_METRICS:
            metrics = self._infra.get_metrics_for_service(target)
            return {"metrics": metrics, "service": target,
                    "data_points": len(metrics)}, \
                   f"Retrieved {len(metrics)} metric points for {target}"

        if atype == ActionType.CHECK_DEPENDENCIES:
            deps = self._infra.get_dependencies_for_service(target)
            return {"dependencies": deps, "service": target}, \
                   f"Retrieved dependency map for {target}"

        if atype == ActionType.CHECK_DEPLOY_HISTORY:
            deploys = self._infra.get_deploy_history_for_service(target)
            return {"deploys": deploys, "service": target,
                    "count": len(deploys)}, \
                   f"Retrieved {len(deploys)} deploys for {target}"

        if atype == ActionType.RUN_HEALTH_CHECK:
            h = self._infra.run_health_check(target)
            return {"health_check": h, "service": target}, \
                   f"Health check for {target}: {h['status']}"

        if atype == ActionType.RESTART_SERVICE:
            svc = self._infra.get_service(target)
            msg = svc.restart(self._infra.current_minute) if svc else "Service not found"
            return {"result": msg, "service": target}, msg

        if atype == ActionType.ROLLBACK_DEPLOY:
            svc = self._infra.get_service(target)
            msg = svc.rollback_deploy(self._infra.current_minute) \
                  if svc else "Service not found"
            return {"result": msg, "service": target}, msg

        if atype == ActionType.SCALE_SERVICE:
            svc = self._infra.get_service(target)
            new_replicas = params.get("replicas", 5)
            msg = svc.scale(new_replicas, self._infra.current_minute) \
                  if svc else "Service not found"
            return {"result": msg, "service": target}, msg

        if atype == ActionType.DECLARE_ROOT_CAUSE:
            rc = params.get("root_cause", "")
            self._declared_root_cause = rc
            self._state.declared_root_cause = rc
            return {
                "declared": rc,
                "message":  ("Root cause declared. " +
                             ("Episode continues — Phase 2 awaits."
                              if self._scenario.code_context
                              else "Episode will end after this step.")),
            }, f"Root cause declared: {rc[:120]}"

        return {"error": f"Unhandled action type: {atype.value}"}, "Unknown action"

    # ==================================================================
    # Reward computation
    # ==================================================================

    def _compute_p1_reward(
        self,
        action: IncidentAction,
        atype:  ActionType,
    ) -> float:
        scenario = self._scenario
        target   = action.target_service
        reward   = _STEP_PENALTY

        if self._infra.was_action_taken(action.action_type, target):
            return round(reward + _REPEAT_PENALTY, 3)

        if atype in DIAGNOSTIC_ACTIONS:
            if target and target in scenario.involved_services:
                reward += 0.15
            elif target and target not in scenario.involved_services:
                reward += 0.05
            elif atype == ActionType.VIEW_ALERTS:
                reward += 0.15
        elif atype in REMEDIATION_ACTIONS:
            if target == scenario.root_cause_service:
                reward += 0.30
            elif target and target in scenario.involved_services:
                reward += 0.10
            else:
                reward -= 0.15
        elif atype == ActionType.DECLARE_ROOT_CAUSE:
            declared = (action.parameters or {}).get("root_cause", "").lower()
            kws = scenario.root_cause_keywords
            if kws:
                ratio = sum(1 for k in kws if k in declared) / len(kws)
                if ratio >= 0.6:
                    reward += 0.40
                elif ratio >= 0.3:
                    reward += 0.15
                else:
                    reward -= 0.20
            else:
                reward -= 0.20

        # Completion bonus when episode terminates
        if self._declared_root_cause and not scenario.code_context:
            if self._infra.all_services_healthy():
                reward += 0.20
            if self._infra.current_minute > self._infra.time_budget_minutes:
                reward -= 0.10

        return round(reward, 3)

    def _compute_p2_reward(
        self,
        action:  IncidentAction,
        atype:   ActionType,
        success: bool,
    ) -> float:
        params = action.parameters or {}
        prim   = self._p2_primary_param(atype, params)
        reward = _STEP_PENALTY

        if not success:
            return round(reward + _INVALID_PENALTY, 3)

        if (atype.value, prim) in self._p2_actions_taken:
            return round(reward + _REPEAT_PENALTY, 3)

        if atype in PHASE2_DIAGNOSTIC_ACTIONS:
            reward += _P2_DIAG_REWARD
        elif atype in PHASE2_TERMINAL_ACTIONS:
            reward += _P2_TERMINAL_BONUS

        return round(reward, 3)

    @staticmethod
    def _p2_primary_param(atype: ActionType, params: Dict[str, Any]) -> str:
        if atype == ActionType.LIST_DIR:
            return params.get("path", ".")
        if atype == ActionType.READ_FILE:
            return params.get("path", "")
        if atype == ActionType.SEARCH_CODE:
            return params.get("query", "")
        if atype == ActionType.GET_GIT_LOG:
            return params.get("path", "")
        if atype == ActionType.GET_FILE_DIFF:
            return f'{params.get("commit_sha", "")}:{params.get("path", "")}'
        return ""

    # ==================================================================
    # Done logic
    # ==================================================================

    def _check_done_p1(self, atype: ActionType) -> bool:
        # Pool A / explicit p1_only mode: declare_root_cause always terminates,
        # regardless of whether the scenario could otherwise transition to P2.
        if atype == ActionType.DECLARE_ROOT_CAUSE:
            if self._mode == "p1_only" or self._scenario.code_context is None:
                return True
        if self._exceeded_step_budget():
            return True
        return False

    def _exceeded_step_budget(self) -> bool:
        budget = self._scenario.max_steps if self._scenario else 20
        # When code_context exists, allow a bit more headroom for P2 exploration
        if self._scenario and self._scenario.code_context is not None:
            budget = budget + 15
        return self._state.step_count >= budget

    # ==================================================================
    # Observation builder
    # ==================================================================

    def _build_observation(
        self,
        action_result:  Dict[str, Any],
        action_success: bool,
        action_message: str,
        reward:         float,
    ) -> Dict[str, Any]:
        statuses      = self._infra.get_all_statuses() if self._infra else {}
        alerts        = self._infra.get_alerts() if self._infra else []
        valid_actions = self._valid_actions_for_phase()

        return {
            "incident_summary":     self._scenario.incident_summary if self._scenario else "",
            "severity":             self._scenario.severity if self._scenario else "SEV3",
            "time_elapsed_minutes": self._infra.current_minute if self._infra else 0,
            "time_budget_minutes":  self._infra.time_budget_minutes if self._infra else 30,
            "action_result":        action_result,
            "action_success":       action_success,
            "action_message":       action_message,
            "service_statuses":     statuses,
            "active_alerts_count":  len(alerts),
            "valid_actions":        valid_actions,
            "available_services":   list(SERVICE_NAMES),
            "current_phase":        self._phase,
            "current_reward":       reward,
            "cumulative_reward":    round(self._cumulative_reward, 3),
            "steps_taken":          self._state.step_count,
            "max_steps":            self._scenario.max_steps if self._scenario else 20,
            "done":                 self._done,
            # Convenience field surfaced after transition (so the inference loop
            # can grab it without re-issuing a step) — only meaningful after
            # transition_to_phase2 has been called.
            "bad_commit_sha":       (self._scenario.code_context.bad_commit_sha
                                     if self._scenario and self._scenario.code_context else None),
        }

    def _valid_actions_for_phase(self) -> List[str]:
        if self._phase == 1:
            base = self._infra.get_valid_actions() if self._infra else []
            # Filter to only P1 + (optionally) transition_to_phase2
            valid = [a for a in base
                     if a.split(":", 1)[0] in {at.value for at in PHASE1_ACTIONS}]
            if self._scenario and self._scenario.code_context is not None:
                valid.append(ActionType.TRANSITION_TO_PHASE2.value)
            return valid
        # Phase 2
        return [at.value for at in PHASE2_ACTIONS]

    # ==================================================================
    # Trajectory access (used by /score endpoint and Pool runners)
    # ==================================================================

    def get_trajectory(self) -> List[StepRecord]:
        return list(self._p1_trajectory) + list(self._p2_trajectory)

    def get_p1_trajectory(self) -> List[StepRecord]:
        return list(self._p1_trajectory)

    def get_p2_trajectory(self) -> List[StepRecord]:
        return list(self._p2_trajectory)

    def get_belief_at_transition(self) -> Optional[BeliefState]:
        return self._belief_at_transition

    # ==================================================================
    # Final unified scoring
    # ==================================================================

    def _compute_unified_final_score(self) -> float:
        """Quick wrapper for the in-step `info.score` field."""
        from ..tasks import grade_trajectory_unified
        if self._scenario is None:
            return 0.01
        breakdown = grade_trajectory_unified(
            task_name           = self._scenario.task_name,
            p1_trajectory       = self._p1_trajectory,
            p2_trajectory       = self._p2_trajectory,
            declared_patch      = self._declared_patch,
            declared_no_change  = self._declared_no_change,
            p1_belief_history   = [],
        )
        return float(breakdown.get("final", 0.01))

    def score_unified(
        self,
        belief_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Public wrapper exposed by the /score endpoint."""
        from ..tasks import grade_trajectory_unified
        if self._scenario is None:
            return {"final": 0.01}
        return grade_trajectory_unified(
            task_name           = self._scenario.task_name,
            p1_trajectory       = self._p1_trajectory,
            p2_trajectory       = self._p2_trajectory,
            declared_patch      = self._declared_patch,
            declared_no_change  = self._declared_no_change,
            p1_belief_history   = belief_history or [],
        )

    # ==================================================================
    # Error / fallback responses
    # ==================================================================

    def _invalid_action_response(
        self,
        msg:    str,
        action: IncidentAction,
    ) -> Dict[str, Any]:
        reward = _INVALID_PENALTY
        self._cumulative_reward += reward
        self._state.step_count  += 1
        obs = self._build_observation(
            action_result   = {"error": msg},
            action_success  = False,
            action_message  = f"Invalid action: {msg}",
            reward          = reward,
        )
        # Still record the failed attempt so trajectory analysis sees it
        record = StepRecord(
            step_number             = self._state.step_count,
            action                  = action,
            reward                  = reward,
            observation_summary     = {"action_message": f"invalid: {msg}"},
            service_statuses_after  = dict(obs.get("service_statuses", {})),
            timestamp_minutes       = self._infra.current_minute if self._infra else 0,
            phase                   = self._phase,
        )
        if self._phase == 1:
            self._p1_trajectory.append(record)
        else:
            self._p2_trajectory.append(record)

        return {"observation": obs, "reward": reward, "done": False,
                "info": {"error": msg}}

    def _final_step_response(self) -> Dict[str, Any]:
        obs = self._build_observation(
            action_result   = {"error": "Episode is already done."},
            action_success  = False,
            action_message  = "Episode already finished",
            reward          = 0.0,
        )
        score = (self._compute_unified_final_score()
                 if self._scenario and self._scenario.code_context
                 else (self._scenario.grade(self._p1_trajectory)
                       if self._scenario else 0.01))
        return {"observation": obs, "reward": 0.01, "done": True,
                "info": {"score": score}}

    def _not_initialized_response(self) -> Dict[str, Any]:
        obs = self._build_observation(
            action_result   = {"error": "Environment not initialized. Call reset() first."},
            action_success  = False,
            action_message  = "Not initialized",
            reward          = 0.0,
        )
        return {"observation": obs, "reward": 0.01, "done": False, "info": {}}
