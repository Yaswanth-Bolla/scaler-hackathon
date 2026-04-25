"""
Thin FastAPI server — marshals JSON in/out.
No simulation logic lives here.

Endpoints:
  GET  /health                      health check
  GET  /tasks                       list available scenarios
  POST /reset    {task_name, seed}  start a new episode
  POST /step     {action_type, ...} execute one action (phase-aware)
  GET  /state                       per-episode metadata
  GET  /trajectory                  full P1+P2 step records
  POST /score    {declared_patch, declared_no_change, belief_history}
                                    unified grader breakdown
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .incident_environment import IncidentEnvironment


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title       = "SRE Incident Response Environment",
    description = "Two-phase OpenEnv environment (P1 ops + P2 code attribution).",
    version     = "0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

env = IncidentEnvironment()


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------

class StepRequest(BaseModel):
    action_type:    str
    target_service: Optional[str]      = None
    parameters:     Dict[str, Any]     = {}


class ScoreRequest(BaseModel):
    declared_patch:     Optional[str]   = None
    declared_no_change: bool            = False
    belief_history:     List[Dict[str, Any]] = []


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset")
async def reset(request: Request) -> Dict[str, Any]:
    """
    Initialize a new incident episode.

    Accepts (all optional):
        task_name : str       specific scenario, otherwise sampled from pool
        seed      : int       RNG seed for deterministic replay
        pool      : "A"|"B"|"C"|"D"   selects training pool (sets default mode)
        mode      : "p1_only"|"p2_only"|"joint"   force episode mode
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    return env.reset(
        task_name = body.get("task_name"),
        seed      = body.get("seed"),
        pool      = body.get("pool"),
        mode      = body.get("mode"),
    )


@app.get("/pools")
def list_pools() -> Dict[str, Any]:
    """Pool registry — used by training runners to discover task names."""
    from ..pools import POOLS
    return {
        name: {
            "name":        p.name,
            "description": p.description,
            "task_names":  list(p.task_names),
            "mode":        p.mode,
            "inject_oracle_belief": p.inject_oracle_belief,
        }
        for name, p in POOLS.items()
    }


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    """Execute one agent action — phase-aware dispatch."""
    return env.step({
        "action_type":    request.action_type,
        "target_service": request.target_service,
        "parameters":     request.parameters or {},
    })


@app.get("/state")
def state() -> Dict[str, Any]:
    return env.get_state()


@app.get("/trajectory")
def trajectory() -> Dict[str, Any]:
    """Return the current episode's full P1 + P2 trajectory."""
    return {
        "p1": [_serialize_step(r) for r in env.get_p1_trajectory()],
        "p2": [_serialize_step(r) for r in env.get_p2_trajectory()],
    }


@app.post("/score")
def score(req: ScoreRequest) -> Dict[str, Any]:
    """
    Unified grader breakdown + counterfactual r_cross.

    Returns:
        final, p1_rca, p1_efficiency, patch_quality, no_change_detection,
        p2_efficiency, r_cross, null_context_p2_score
    """
    from ..tasks import compute_r_cross
    breakdown = env.score_unified(belief_history=req.belief_history)
    state = env.get_state()
    task = state.get("task_name")
    r_cross = 0.0
    null_baseline = 0.0
    if task:
        try:
            r_cross = compute_r_cross(
                task_name          = task,
                declared_patch     = state.get("declared_patch"),
                declared_no_change = bool(state.get("declared_no_change")),
                p2_trajectory      = env.get_p2_trajectory(),
            )
            from ..tasks import get_scenario
            ctx = get_scenario(task).code_context
            if ctx is not None:
                null_baseline = float(ctx.null_context_p2_score)
        except Exception:
            pass
    return {
        **breakdown,
        "r_cross":               round(r_cross, 4),
        "null_context_p2_score": round(null_baseline, 4),
    }


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    from ..tasks import TASK_REGISTRY
    out: Dict[str, Any] = {}
    for name, cls in TASK_REGISTRY.items():
        scenario = cls()
        out[name] = {
            "display_name":        scenario.display_name,
            "severity":            scenario.severity,
            "max_steps":           scenario.max_steps,
            "time_budget_minutes": scenario.time_budget_minutes,
            "has_phase2":          scenario.code_context is not None,
            "fault_class":         scenario.fault_class,
        }
    return {"tasks": out}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _serialize_step(r) -> Dict[str, Any]:
    """Convert a StepRecord into a JSON-safe dict."""
    return {
        "step_number":           r.step_number,
        "phase":                 r.phase,
        "action": {
            "action_type":    r.action.action_type,
            "target_service": r.action.target_service,
            "parameters":     r.action.parameters,
        },
        "reward":                r.reward,
        "observation_summary":   r.observation_summary,
        "service_statuses_after": r.service_statuses_after,
        "timestamp_minutes":     r.timestamp_minutes,
        "belief_state_snapshot": r.belief_state_snapshot,
    }


def main() -> None:
    import uvicorn
    uvicorn.run("incident_env.server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
