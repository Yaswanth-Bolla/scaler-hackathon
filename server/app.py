"""
Thin FastAPI server — marshals JSON in/out.
No simulation logic lives here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .incident_environment import IncidentEnvironment

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="SRE Incident Response Environment",
    description="An OpenEnv environment for training AI agents on production incident response.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = IncidentEnvironment()


# ------------------------------------------------------------------
# Request / Response models (thin wrappers)
# ------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str
    target_service: Optional[str] = None
    parameters: Dict[str, Any] = {}


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """Health check — the validator pings this first."""
    return {"status": "healthy"}


@app.post("/reset")
def reset(request: ResetRequest) -> Dict[str, Any]:
    """
    Initialize a new incident episode.
    POST /reset {"task_name": "memory_leak", "seed": 42}
    """
    result = env.reset(
        task_name=request.task_name,
        seed=request.seed,
    )
    return result


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    """
    Execute one agent action.
    POST /step {"action_type": "view_alerts"}
    """
    action_data = {
        "action_type": request.action_type,
        "target_service": request.target_service,
        "parameters": request.parameters,
    }
    result = env.step(action_data)
    return result


@app.get("/state")
def state() -> Dict[str, Any]:
    """
    Get current episode metadata.
    GET /state
    """
    return env.get_state()


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """List available tasks with descriptions."""
    from ..tasks import TASK_REGISTRY
    tasks = {}
    for name, cls in TASK_REGISTRY.items():
        scenario = cls()
        tasks[name] = {
            "display_name": scenario.display_name,
            "severity": scenario.severity,
            "max_steps": scenario.max_steps,
            "time_budget_minutes": scenario.time_budget_minutes,
        }
    return {"tasks": tasks}
