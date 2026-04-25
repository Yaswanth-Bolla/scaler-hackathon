"""
Typed models for the SRE Incident Response environment.

Action space: hierarchical — select action_type first, then target + params.
Observation space: POMDP — agent never sees fault_type, only symptoms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Action Space (Layer 4 — Hierarchical + Masked)
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Level-1 action categories — what kind of operation."""
    # ---- Phase 1: ops / SRE ---------------------------------------
    VIEW_ALERTS = "view_alerts"
    QUERY_LOGS = "query_logs"
    CHECK_METRICS = "check_metrics"
    CHECK_DEPENDENCIES = "check_dependencies"
    CHECK_DEPLOY_HISTORY = "check_deploy_history"
    RUN_HEALTH_CHECK = "run_health_check"
    RESTART_SERVICE = "restart_service"
    ROLLBACK_DEPLOY = "rollback_deploy"
    SCALE_SERVICE = "scale_service"
    DECLARE_ROOT_CAUSE = "declare_root_cause"

    # ---- Cross-phase control -------------------------------------
    TRANSITION_TO_PHASE2 = "transition_to_phase2"

    # ---- Phase 2: codebase exploration ---------------------------
    READ_FILE = "read_file"
    SEARCH_CODE = "search_code"
    LIST_DIR = "list_dir"
    GET_GIT_LOG = "get_git_log"
    GET_FILE_DIFF = "get_file_diff"

    # ---- Phase 2: terminal --------------------------------------
    PROPOSE_PATCH = "propose_patch"
    DECLARE_NO_CHANGE = "declare_no_change"


# Actions that require a target_service (Level 2 — where to apply)
TARGETED_ACTIONS = {
    ActionType.QUERY_LOGS,
    ActionType.CHECK_METRICS,
    ActionType.CHECK_DEPENDENCIES,
    ActionType.CHECK_DEPLOY_HISTORY,
    ActionType.RUN_HEALTH_CHECK,
    ActionType.RESTART_SERVICE,
    ActionType.ROLLBACK_DEPLOY,
    ActionType.SCALE_SERVICE,
}

# Actions that are diagnostic (information-gathering, no state mutation)
DIAGNOSTIC_ACTIONS = {
    ActionType.VIEW_ALERTS,
    ActionType.QUERY_LOGS,
    ActionType.CHECK_METRICS,
    ActionType.CHECK_DEPENDENCIES,
    ActionType.CHECK_DEPLOY_HISTORY,
    ActionType.RUN_HEALTH_CHECK,
}

# Actions that mutate infrastructure state
REMEDIATION_ACTIONS = {
    ActionType.RESTART_SERVICE,
    ActionType.ROLLBACK_DEPLOY,
    ActionType.SCALE_SERVICE,
}

# ---- Phase classification ------------------------------------------
PHASE1_ACTIONS = (
    DIAGNOSTIC_ACTIONS
    | REMEDIATION_ACTIONS
    | {ActionType.DECLARE_ROOT_CAUSE}
)

PHASE2_DIAGNOSTIC_ACTIONS = {
    ActionType.LIST_DIR,
    ActionType.READ_FILE,
    ActionType.SEARCH_CODE,
    ActionType.GET_GIT_LOG,
    ActionType.GET_FILE_DIFF,
}

PHASE2_TERMINAL_ACTIONS = {
    ActionType.PROPOSE_PATCH,
    ActionType.DECLARE_NO_CHANGE,
}

PHASE2_ACTIONS = PHASE2_DIAGNOSTIC_ACTIONS | PHASE2_TERMINAL_ACTIONS

# Cross-phase control action — only legal when transitioning P1 → P2
CONTROL_ACTIONS = {ActionType.TRANSITION_TO_PHASE2}

@dataclass
class BeliefState:
    """
    Structured scratchpad the orchestrator emits before each transition decision.
    Making this explicit (not implicit in hidden states) lets us:
      - supervise confidence calibration with auxiliary loss
      - audit stopping criterion decisions
      - compute consistency losses (e.g. empty gaps + low confidence = incoherent)
    """
    suspected_service: Optional[str] = None
    suspected_fault_class: Optional[str] = None   # "memory_leak" | "config_change" | "deadlock" | "none"
    service_confidence: float = 0.0               # [0, 1] — calibrated against ground truth in Stage 2
    fault_confidence: float = 0.0                 # [0, 1]
    evidence_gaps: List[str] = field(default_factory=list)  # e.g. ["deploy_history_unchecked"]
    estimated_p2_cost: str = "unknown"            # "low" | "medium" | "high"
    decision: str = "continue"                    # "continue" | "transition" | "abort"
    reasoning: str = ""                           # free-text, used for consistency loss

@dataclass
class CodeContext:
    """
    Hidden code-layer state — injected into Phase 2 at transition.
    Agent never sees this directly; it must be inferred by exploration.
    """
    repo_snapshot_path: str                       # path to bundled mini-repo
    bad_commit_sha: str
    ground_truth_files: List[str]                 # files touched by real PR
    ground_truth_diff: str                        # unified diff string
    is_valid_issue: bool = True                   # False = user confusion, no-change correct
    expected_p2_steps: int = 8                    # baseline for efficiency normalization
    null_context_p2_score: float = 0.0            # filled in during Stage 3 (Pool B baseline)

@dataclass
class IncidentAction:
    """
    Agent action — hierarchical: action_type → target_service → parameters.

    The LLM emits JSON with these three fields. The action mask in the
    observation tells it which (action_type, target_service) pairs are legal.
    """
    action_type: str                            # ActionType value
    target_service: Optional[str] = None        # Required for TARGETED_ACTIONS
    parameters: Dict[str, Any] = field(default_factory=dict)
    # Phase tracking
    current_phase: int = 1                            # 1 = ops, 2 = code
    belief_state: Optional[BeliefState] = None        # orchestrator scratchpad output

    def parsed_type(self) -> ActionType:
        return ActionType(self.action_type)


# ---------------------------------------------------------------------------
# Observation Space (Layer 2 — POMDP, partial views only)
# ---------------------------------------------------------------------------

@dataclass
class AlertInfo:
    """A single firing alert — what fired, not why."""
    alert_id: str
    severity: str           # "critical" | "warning" | "info"
    source_service: str
    description: str
    firing_since: str       # ISO timestamp

@dataclass
class MetricSnapshot:
    """Time-series metrics for a single service — temporal pattern visible."""
    service_name: str
    timestamps: List[str]
    cpu_percent: List[float]
    memory_percent: List[float]
    error_rate_percent: List[float]
    latency_p50_ms: List[float]
    latency_p95_ms: List[float]
    latency_p99_ms: List[float]
    requests_per_sec: List[float]

@dataclass
class LogEntry:
    """A single structured log entry — error semantics visible."""
    timestamp: str
    level: str              # "DEBUG" | "INFO" | "WARN" | "ERROR" | "FATAL"
    service: str
    message: str
    trace_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeployRecord:
    """A single deploy — evidence trail for rollback decisions."""
    version: str
    timestamp: str
    author: str
    commit_hash: str
    description: str

@dataclass
class DependencyInfo:
    """Upstream/downstream dependency map for a service."""
    service_name: str
    depends_on: List[str]       # services this one calls
    depended_by: List[str]      # services that call this one


@dataclass
class IncidentObservation:
    """
    What the agent sees after each step.

    This is a PARTIAL observation — the agent never sees fault_type,
    fault_target, or the internal simulation state. It must infer
    the root cause from the five observation modalities.
    """
    # --- Incident context (always visible) ---
    incident_summary: str
    severity: str                               # "SEV1" | "SEV2" | "SEV3"
    time_elapsed_minutes: int
    time_budget_minutes: int

    # --- Result of last action ---
    action_result: Dict[str, Any] = field(default_factory=dict)
    action_success: bool = True
    action_message: str = ""

    # --- Dashboard (always visible) ---
    service_statuses: Dict[str, str] = field(default_factory=dict)  # name → "healthy"|"degraded"|"down"
    active_alerts_count: int = 0

    # --- Action mask (Layer 4 — prevents illegal actions) ---
    valid_actions: List[str] = field(default_factory=list)
    available_services: List[str] = field(default_factory=list)

    # --- Episode progress ---
    current_reward: float = 0.0
    cumulative_reward: float = 0.0
    steps_taken: int = 0
    max_steps: int = 20
    done: bool = False


# ---------------------------------------------------------------------------
# State (internal tracking — exposed via state() for debugging)
# ---------------------------------------------------------------------------

@dataclass
class IncidentState:
    """Episode metadata — returned by state() for monitoring/debugging."""
    episode_id: str = ""
    task_name: str = ""
    step_count: int = 0
    time_elapsed_minutes: int = 0
    done: bool = False
    cumulative_reward: float = 0.0
    declared_root_cause: Optional[str] = None


# ---------------------------------------------------------------------------
# Step record — stored per step for trajectory-based grading
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """
    Immutable record of a single step — used by the grader.
    The grader receives List[StepRecord] and scores WITHOUT hidden state.
    """
    step_number: int
    action: IncidentAction
    reward: float
    observation_summary: Dict[str, Any]     # key fields from observation
    service_statuses_after: Dict[str, str]  # service health after this step
    timestamp_minutes: int                  # simulation time
    phase: int = 1
    belief_state_snapshot: Optional[dict] = None     # serialized BeliefState for behavioral analysis
