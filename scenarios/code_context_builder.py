"""
CodeContext registry for each Phase-2-enabled scenario.

Snapshots live under <repo_root>/snapshots/<name>/, bundled in the repo —
no live GitHub API calls.  Each context provides the snapshot path, the bad
commit SHA, the ground-truth files/diff (used by `grader_p2.grade_patch_quality`)
and a slot for the Pool-B null-context baseline (filled in by the
`training/run_pool_b_baseline.py` runner — re-imported on demand).

The `null_context_p2_score` field starts at a hand-tuned prior; it is
overwritten in-place by the baseline runner once we have measurements.
"""

from __future__ import annotations

from pathlib import Path

from ..models import CodeContext


# ──────────────────────────────────────────────────────────────────────
# Snapshot-root resolution
# ──────────────────────────────────────────────────────────────────────

_PKG_ROOT       = Path(__file__).resolve().parent.parent          # incident_env/
_REPO_ROOT      = _PKG_ROOT.parent                                # project/
SNAPSHOTS_ROOT  = _REPO_ROOT / "snapshots"


def _snap(name: str) -> str:
    """Absolute path to a snapshot directory under <repo_root>/snapshots/."""
    return str(SNAPSHOTS_ROOT / name)


# ──────────────────────────────────────────────────────────────────────
# Memory leak (easy)
# ──────────────────────────────────────────────────────────────────────

MEMORY_LEAK_CODE_CONTEXT = CodeContext(
    repo_snapshot_path  = _snap("orders_v231"),
    bad_commit_sha      = "a3f7c91",
    ground_truth_files  = ["orders/handlers/batch.py"],
    ground_truth_diff   = """--- a/orders/handlers/batch.py
+++ b/orders/handlers/batch.py
@@ -41,6 +41,7 @@ class BatchProcessor:
         for order in orders:
             self._cache[order.id] = order
+        self._cache.clear()
         self._notify(orders)
""",
    is_valid_issue          = True,
    expected_p2_steps       = 5,
    null_context_p2_score   = 0.21,
)

# ──────────────────────────────────────────────────────────────────────
# Cascading failure (medium)
# ──────────────────────────────────────────────────────────────────────

CASCADING_FAILURE_CODE_CONTEXT = CodeContext(
    repo_snapshot_path  = _snap("auth_v180"),
    bad_commit_sha      = "b8e2d44",
    ground_truth_files  = ["auth/config.py"],
    ground_truth_diff   = """--- a/auth/config.py
+++ b/auth/config.py
@@ -10,3 +10,3 @@
-JWT_SECRET = os.environ.get("JWT_SECRET")
+JWT_SECRET = os.environ.get("JWT_SECRET") or _DEFAULT_DEV_SECRET
""",
    is_valid_issue          = True,
    expected_p2_steps       = 4,
    null_context_p2_score   = 0.18,
)

# ──────────────────────────────────────────────────────────────────────
# Distributed deadlock (hard)
# ──────────────────────────────────────────────────────────────────────

DISTRIBUTED_DEADLOCK_CODE_CONTEXT = CodeContext(
    repo_snapshot_path  = _snap("payment_v310"),
    bad_commit_sha      = "c5a1f77",
    ground_truth_files  = ["payment/processor.py"],
    ground_truth_diff   = """--- a/payment/processor.py
+++ b/payment/processor.py
@@ -85,5 +85,7 @@ class PaymentProcessor:
     def retry(self, txn):
+        delay = min(2 ** self.retry_count, 30)
+        time.sleep(delay)
         self._queue.enqueue(txn)
""",
    is_valid_issue          = True,
    expected_p2_steps       = 10,
    null_context_p2_score   = 0.09,
)

# ──────────────────────────────────────────────────────────────────────
# Circuit breaker — no-change scenario (the patch grader should reject any
# proposed diff and reward `declare_no_change`).
# ──────────────────────────────────────────────────────────────────────

CIRCUIT_BREAKER_CODE_CONTEXT = CodeContext(
    repo_snapshot_path  = _snap("orders_v300"),
    bad_commit_sha      = "d2b9c11",
    ground_truth_files  = [],
    ground_truth_diff   = "",
    is_valid_issue      = False,
    expected_p2_steps   = 6,
    null_context_p2_score = 0.15,
)


# ──────────────────────────────────────────────────────────────────────
# Phase-B scenarios (RL-discoverable, the brief's four types)
# ──────────────────────────────────────────────────────────────────────

ALIASED_FAULT_CODE_CONTEXT = CodeContext(
    repo_snapshot_path  = _snap("queue_v210"),
    bad_commit_sha      = "e1f4a02",
    ground_truth_files  = ["queue/worker.py"],
    ground_truth_diff   = """--- a/queue/worker.py
+++ b/queue/worker.py
@@ -22,4 +22,5 @@ class CacheWriter:
     def flush(self, batch):
-        for k, v in batch.items():
+        for k, v in list(batch.items()):
             self._cache.set(k, v)
+        batch.clear()
""",
    is_valid_issue          = True,
    expected_p2_steps       = 7,
    null_context_p2_score   = 0.16,
)

SEVERITY_INVERSION_CODE_CONTEXT = CodeContext(
    repo_snapshot_path  = _snap("orders_retry_storm"),
    bad_commit_sha      = "f8c9b13",
    ground_truth_files  = ["orders/auth_client.py"],
    ground_truth_diff   = """--- a/orders/auth_client.py
+++ b/orders/auth_client.py
@@ -15,5 +15,6 @@ class AuthClient:
     def validate(self, token):
-        return self._call_with_retries(token, retries=20)
+        return self._call_with_retries(token, retries=2,
+                                        backoff_seconds=0.5)
""",
    is_valid_issue          = True,
    expected_p2_steps       = 8,
    null_context_p2_score   = 0.12,
)

CONFIDENCE_INVERSION_CODE_CONTEXT = CodeContext(
    repo_snapshot_path  = _snap("payment_threadpool"),
    bad_commit_sha      = "11abf04",
    ground_truth_files  = ["payment/threadpool.py"],
    ground_truth_diff   = """--- a/payment/threadpool.py
+++ b/payment/threadpool.py
@@ -8,4 +8,5 @@ class PoolWorker:
     def acquire(self):
-        self._lock_a.acquire()
-        self._lock_b.acquire()
+        with self._global_order:
+            self._lock_a.acquire()
+            self._lock_b.acquire()
""",
    is_valid_issue          = True,
    expected_p2_steps       = 9,
    null_context_p2_score   = 0.10,
)

INFO_ORDERING_CODE_CONTEXT = CodeContext(
    repo_snapshot_path  = _snap("shared_libs_dep"),
    bad_commit_sha      = "9d2e7af",
    ground_truth_files  = ["requirements.txt", "shared/serializer.py"],
    ground_truth_diff   = """--- a/requirements.txt
+++ b/requirements.txt
@@ -3,1 +3,1 @@
-shared-serializer==1.4.2
+shared-serializer==1.3.0
""",
    is_valid_issue          = True,
    expected_p2_steps       = 9,
    null_context_p2_score   = 0.11,
)


# ──────────────────────────────────────────────────────────────────────
# Pool-D held-out scenarios
# ──────────────────────────────────────────────────────────────────────

HELDOUT_ALIASED_SEVERITY_CODE_CONTEXT = CodeContext(
    repo_snapshot_path  = _snap("orders_retry_storm"),
    bad_commit_sha      = "d09a4f1",
    ground_truth_files  = ["orders/auth_client.py"],
    ground_truth_diff   = """--- a/orders/auth_client.py
+++ b/orders/auth_client.py
@@ -15,5 +15,6 @@ class AuthClient:
     def validate(self, token):
-        return self._call_with_retries(token, retries=25)
+        return self._call_with_retries(token, retries=2,
+                                        backoff_seconds=0.5)
""",
    is_valid_issue          = True,
    expected_p2_steps       = 9,
    null_context_p2_score   = 0.10,
)

HELDOUT_CONFIDENCE_ORDERING_CODE_CONTEXT = CodeContext(
    repo_snapshot_path  = _snap("shared_libs_dep"),
    bad_commit_sha      = "9d2e7af",
    ground_truth_files  = ["requirements.txt"],
    ground_truth_diff   = """--- a/requirements.txt
+++ b/requirements.txt
@@ -3,1 +3,1 @@
-shared-serializer==1.4.2
+shared-serializer==1.3.0
""",
    is_valid_issue          = True,
    expected_p2_steps       = 10,
    null_context_p2_score   = 0.10,
)


# ──────────────────────────────────────────────────────────────────────
# Lookup helpers (used by Pool-B baseline runner to write baselines back)
# ──────────────────────────────────────────────────────────────────────

CODE_CONTEXTS = {
    "memory_leak":                   MEMORY_LEAK_CODE_CONTEXT,
    "cascading_failure":             CASCADING_FAILURE_CODE_CONTEXT,
    "distributed_deadlock":          DISTRIBUTED_DEADLOCK_CODE_CONTEXT,
    "circuit_breaker_noop":          CIRCUIT_BREAKER_CODE_CONTEXT,
    "aliased_fault":                 ALIASED_FAULT_CODE_CONTEXT,
    "severity_inversion":            SEVERITY_INVERSION_CODE_CONTEXT,
    "confidence_inversion":          CONFIDENCE_INVERSION_CODE_CONTEXT,
    "info_ordering":                 INFO_ORDERING_CODE_CONTEXT,
    "heldout_aliased_severity":      HELDOUT_ALIASED_SEVERITY_CODE_CONTEXT,
    "heldout_confidence_ordering":   HELDOUT_CONFIDENCE_ORDERING_CODE_CONTEXT,
}


def update_null_baseline(task_name: str, score: float) -> None:
    """Mutate the in-process null-context baseline for `task_name`."""
    ctx = CODE_CONTEXTS.get(task_name)
    if ctx is None:
        raise KeyError(f"No code context for task {task_name}")
    ctx.null_context_p2_score = float(score)
