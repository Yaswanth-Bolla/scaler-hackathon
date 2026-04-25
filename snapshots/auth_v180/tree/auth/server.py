from __future__ import annotations

from .token import TokenService


class AuthServer:
    """HTTP entry-point for the auth service."""

    def __init__(self) -> None:
        self._tokens = TokenService()

    def login(self, user: str, password: str) -> str:
        if not self._authenticate(user, password):
            raise PermissionError("invalid credentials")
        return self._tokens.issue(user, ["read", "write"])

    def validate(self, token: str) -> bool:
        return self._tokens.validate(token)

    def _authenticate(self, user: str, password: str) -> bool:
        return bool(user) and bool(password)
