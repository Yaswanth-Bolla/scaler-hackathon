"""
Auth service configuration.

`JWT_SECRET` is loaded from the environment.  v1.8.0 added a fallback to
a baked-in dev secret to "make local testing easier" — this is the change
shipped on commit b8e2d44.

The fallback was *intended* to apply only when `ENV=dev`, but the env
check was forgotten — so production now happily falls back too if the
real secret is missing or unreadable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


_DEFAULT_DEV_SECRET = "dev-only-do-not-use-in-prod-zZ3kf81P"


@dataclass(frozen=True)
class AuthConfig:
    jwt_secret: str
    token_ttl_seconds: int
    issuer: str


JWT_SECRET = os.environ.get("JWT_SECRET") or _DEFAULT_DEV_SECRET

AUTH_CONFIG = AuthConfig(
    jwt_secret        = JWT_SECRET,
    token_ttl_seconds = 3600,
    issuer            = "auth.example.com",
)
