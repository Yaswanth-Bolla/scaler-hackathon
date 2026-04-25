from __future__ import annotations

import time
from typing import Optional

from .config import AUTH_CONFIG


class TokenService:
    """Issues + validates short-lived JWT-style tokens."""

    def __init__(self, secret: Optional[str] = None) -> None:
        self._secret = secret or AUTH_CONFIG.jwt_secret

    def issue(self, subject: str, scopes: list[str]) -> str:
        payload = f"{subject}|{','.join(scopes)}|{int(time.time())}"
        return f"{payload}.{self._sign(payload)}"

    def validate(self, token: str) -> bool:
        try:
            payload, sig = token.rsplit(".", 1)
        except ValueError:
            return False
        return sig == self._sign(payload)

    def _sign(self, payload: str) -> str:
        return f"sig-{abs(hash((payload, self._secret))) & 0xFFFFFFFF:08x}"
