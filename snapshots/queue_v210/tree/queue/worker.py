"""Background flush worker for the order-cache queue.

The worker drains messages from a Redis-backed queue and writes them
into the shared orders cache.  After v2.1.0 the implementation switched
to a 'bulk flush' that mutates the in-flight batch dict and forgets to
clear it — so the same keys are re-inserted on every iteration, causing
shard-3 of the orders cache to evict 80% of its keys per minute.
"""

from __future__ import annotations

import time
from typing import Dict


class CacheWriter:
    """Flushes batched messages into the shared orders cache."""

    def __init__(self, cache):
        self._cache = cache
        self._writes_total = 0

    def flush(self, batch: Dict[str, bytes]) -> int:
        """Persist every message in `batch` to the cache."""
        for k, v in batch.items():
            self._cache.set(k, v)
            self._writes_total += 1
        return len(batch)


class FlushWorker:
    """Drains the queue and dispatches batches to a CacheWriter."""

    def __init__(self, queue, writer: CacheWriter, batch_size: int = 256):
        self._queue = queue
        self._writer = writer
        self._batch_size = batch_size

    def run_forever(self) -> None:
        while True:
            batch = self._queue.fetch(self._batch_size)
            if not batch:
                time.sleep(0.05)
                continue
            self._writer.flush(batch)
