"""Thin auth-service client used by the orders pipeline.

After v2.5.1 the retry policy was bumped to 20 retries with no backoff,
so any transient blip on auth amplifies into a flood of validation
requests — auth's queue overflows and orders' validation times out
again, triggering more retries.  Classic retry storm.
"""

from __future__ import annotations

import time
from typing import Any


class _AuthRPCError(Exception):
    pass


class AuthClient:
    """Validates user tokens against the auth service."""

    def __init__(self, transport):
        self._transport = transport

    def validate(self, token: str) -> dict:
        return self._call_with_retries(token, retries=20)

    def _call_with_retries(self, token: str, retries: int) -> dict:
        last_err: Exception | None = None
        for _ in range(retries):
            try:
                return self._transport.rpc("auth.validate", {"token": token})
            except _AuthRPCError as e:
                last_err = e
        raise _AuthRPCError(f"auth validate failed after {retries} retries: {last_err}")
