"""Bounded thread pool used to acquire two row-locks per payment txn.

After v3.1.2 the lock-acquisition order was changed but the global
ordering invariant was dropped, so two concurrent workers can each
hold one of the locks the other needs — classic AB-BA deadlock.

Symptom: threads accumulate (active_count climbs to pool max) but
CPU stays low.  Memory grows because each blocked thread keeps its
stack pinned.  From the outside it looks indistinguishable from a
heap memory leak.
"""

from __future__ import annotations

import threading
from typing import Any


class _LockOrder:
    """Sentinel holding the global acquisition order (must hold first)."""
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()


class PoolWorker:
    """Acquires two per-account row locks before mutating payment state."""

    def __init__(self):
        self._lock_a = threading.Lock()
        self._lock_b = threading.Lock()
        self._global_order = _LockOrder()

    def acquire(self) -> None:
        self._lock_a.acquire()
        self._lock_b.acquire()

    def release(self) -> None:
        self._lock_b.release()
        self._lock_a.release()

    def transfer(self, amount: int) -> None:
        self.acquire()
        try:
            ...
        finally:
            self.release()
