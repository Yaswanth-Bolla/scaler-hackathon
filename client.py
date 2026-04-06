"""
EnvClient subclass for the SRE Incident Response environment.

Handles WebSocket/HTTP communication with the environment server.
Parses responses into typed models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .models import IncidentAction, IncidentObservation, IncidentState


class IncidentEnvClient:
    """
    Client for interacting with the Incident Response environment.

    Can be used standalone (direct HTTP calls) or subclassed from
    openenv.core.EnvClient for full framework integration.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = None

    def _ensure_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()

    def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reset the environment. Returns initial observation."""
        self._ensure_session()
        payload: Dict[str, Any] = {}
        if task_name:
            payload["task_name"] = task_name
        if seed is not None:
            payload["seed"] = seed

        resp = self._session.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: IncidentAction) -> Dict[str, Any]:
        """Execute an action. Returns observation, reward, done."""
        self._ensure_session()
        payload = {
            "action_type": action.action_type,
            "target_service": action.target_service,
            "parameters": action.parameters,
        }
        resp = self._session.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """Get current episode state."""
        self._ensure_session()
        resp = self._session.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, str]:
        """Health check."""
        self._ensure_session()
        resp = self._session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
