"""Thin wrapper around the upstream shared-serializer library.

Each downstream service imports SchemaCodec.encode/decode.  The schema
contract changed in shared-serializer 1.4.x (added optional
`event_v2.idempotency_key`).  When `requirements.txt` was downgraded
from 1.4.2 → 1.3.0 (PR 9d2e7af), every downstream container that gets
rebuilt now silently fails to deserialise events emitted by services
still running 1.4.x.
"""

from __future__ import annotations

from typing import Any


class SchemaCodec:
    """Per-service codec with a hard-coded supported schema range."""

    SUPPORTED = "1.3.x"

    def encode(self, payload: Any) -> bytes:
        # imported library handles the actual wire format
        from shared_serializer import encode  # type: ignore
        return encode(payload)

    def decode(self, raw: bytes) -> Any:
        from shared_serializer import decode  # type: ignore
        return decode(raw)
