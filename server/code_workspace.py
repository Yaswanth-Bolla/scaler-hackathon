"""
CodeWorkspace — read-only filesystem + fake-git layer for Phase 2 exploration.

Each scenario points `CodeContext.repo_snapshot_path` at a directory under
`snapshots/`. That directory contains:

  snapshots/<name>/
      tree/             ← actual source files the agent reads
          <pkg>/<file>.py
          ...
      git_log.json      ← list of commits (sha, author, date, message, files[])
      diffs/<sha>.patch ← unified diff for that commit (any file path)

This is a "fake git" by design — it's tighter, deterministic, and trivially
serializable for trajectory replay.  No subprocess, no real .git directory.

CodeWorkspace is constructed at the start of Phase 2 and lives for the rest
of the episode.  It exposes safe, sandboxed file access (no `..`, no absolute
paths, no symlinks).
"""

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CommitRecord:
    sha: str
    author: str
    date: str            # ISO-ish string
    message: str
    files: List[str]


class CodeWorkspaceError(Exception):
    """Raised on illegal path access or missing files (returned to agent)."""


class CodeWorkspace:
    """
    Sandboxed read-only view over a snapshot.

    All paths are interpreted as relative to `tree/` inside the snapshot.
    Access to anything outside `tree/` raises CodeWorkspaceError.
    """

    MAX_FILE_BYTES   = 64 * 1024     # truncate large files in read_file()
    MAX_LIST_ENTRIES = 200
    MAX_SEARCH_HITS  = 50
    MAX_SEARCH_BYTES = 5 * 1024 * 1024  # don't grep monsters

    def __init__(self, snapshot_root: str, bad_commit_sha: str = ""):
        root = Path(snapshot_root).resolve()
        if not root.exists():
            raise CodeWorkspaceError(f"Snapshot not found: {snapshot_root}")
        self.root            = root
        self.tree_root       = (root / "tree").resolve()
        if not self.tree_root.exists():
            raise CodeWorkspaceError(
                f"Snapshot {snapshot_root} missing tree/ subdir")
        self.bad_commit_sha  = bad_commit_sha
        self._git_log: Optional[List[CommitRecord]] = None
        self._diffs_root     = (root / "diffs").resolve()

    # ------------------------------------------------------------------
    # Public file-system API (1 method per agent action_type)
    # ------------------------------------------------------------------

    def list_dir(self, path: str = ".") -> Dict[str, Any]:
        """List files + subdirs at a relative path under tree/."""
        target = self._resolve_tree(path)
        if not target.is_dir():
            raise CodeWorkspaceError(f"Not a directory: {path}")

        entries = []
        for child in sorted(target.iterdir()):
            if child.name.startswith("."):
                continue
            entries.append({
                "name":   child.name,
                "type":   "dir" if child.is_dir() else "file",
                "size":   child.stat().st_size if child.is_file() else None,
            })
            if len(entries) >= self.MAX_LIST_ENTRIES:
                break
        return {
            "path":    self._rel_to_tree(target),
            "entries": entries,
            "count":   len(entries),
        }

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file under tree/. Truncates if larger than MAX_FILE_BYTES."""
        target = self._resolve_tree(path)
        if not target.is_file():
            raise CodeWorkspaceError(f"Not a file: {path}")
        data = target.read_bytes()
        truncated = False
        if len(data) > self.MAX_FILE_BYTES:
            data = data[: self.MAX_FILE_BYTES]
            truncated = True
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        return {
            "path":      self._rel_to_tree(target),
            "content":   text,
            "size":      target.stat().st_size,
            "truncated": truncated,
        }

    def search_code(
        self,
        query:        str,
        file_pattern: str = "*.py",
        max_hits:     Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Substring search across files matching `file_pattern` under tree/.
        Returns up to `max_hits` (or MAX_SEARCH_HITS) hits with line context.
        """
        if not query:
            return {"query": query, "hits": [], "count": 0}
        cap = min(max_hits or self.MAX_SEARCH_HITS, self.MAX_SEARCH_HITS)
        hits: List[Dict[str, Any]] = []
        bytes_scanned = 0

        for fp in self._iter_tree_files(file_pattern):
            try:
                text = fp.read_text("utf-8", errors="replace")
            except OSError:
                continue
            bytes_scanned += len(text)
            if bytes_scanned > self.MAX_SEARCH_BYTES:
                break
            for ln, line in enumerate(text.splitlines(), 1):
                if query in line:
                    hits.append({
                        "path":  self._rel_to_tree(fp),
                        "line":  ln,
                        "match": line.strip()[:240],
                    })
                    if len(hits) >= cap:
                        return {"query": query, "hits": hits, "count": len(hits),
                                "truncated": True}
        return {"query": query, "hits": hits, "count": len(hits),
                "truncated": False}

    def get_git_log(
        self,
        path:      str = "",
        n_commits: int = 10,
    ) -> Dict[str, Any]:
        """
        Return up to `n_commits` commits from the snapshot's pre-baked git_log.
        If `path` is provided, filters to commits that touched a file matching
        that exact path (or that path's directory).
        """
        log = self._load_git_log()
        if path:
            target = path.strip("/")
            log = [c for c in log if any(
                f == target or f.startswith(target.rstrip("/") + "/") for f in c.files
            )]
        log = log[: max(1, n_commits)]
        return {
            "path":    path or ".",
            "commits": [
                {"sha": c.sha, "author": c.author, "date": c.date,
                 "message": c.message, "files": list(c.files)}
                for c in log
            ],
            "count":   len(log),
        }

    def get_file_diff(
        self,
        commit_sha: str,
        path:       str = "",
    ) -> Dict[str, Any]:
        """
        Return the unified diff for `commit_sha`, optionally filtered to
        hunks touching files matching `path`.
        """
        diff_path = (self._diffs_root / f"{commit_sha}.patch")
        try:
            diff_path = diff_path.resolve()
            if not str(diff_path).startswith(str(self._diffs_root)):
                raise CodeWorkspaceError(f"Illegal diff path: {commit_sha}")
        except OSError:
            raise CodeWorkspaceError(f"Diff not found for {commit_sha}")

        if not diff_path.exists():
            raise CodeWorkspaceError(f"Diff not found for {commit_sha}")

        text = diff_path.read_text("utf-8", errors="replace")
        if path:
            text = self._filter_diff_by_path(text, path)
        return {
            "commit_sha": commit_sha,
            "path":       path or "*",
            "diff":       text,
        }

    # ------------------------------------------------------------------
    # Lightweight introspection (used to seed the code agent at handoff)
    # ------------------------------------------------------------------

    def file_tree(self, max_depth: int = 3) -> List[str]:
        """Flat list of files under tree/, capped to a sane depth."""
        out: List[str] = []
        for fp in self._iter_tree_files("*", max_depth=max_depth):
            out.append(self._rel_to_tree(fp))
            if len(out) >= self.MAX_LIST_ENTRIES:
                break
        return sorted(out)

    def bad_commit_metadata(self) -> Optional[Dict[str, Any]]:
        """Return commit metadata for `bad_commit_sha` (without the diff)."""
        if not self.bad_commit_sha:
            return None
        for c in self._load_git_log():
            if c.sha.startswith(self.bad_commit_sha) or self.bad_commit_sha.startswith(c.sha):
                return {"sha": c.sha, "author": c.author, "date": c.date,
                        "message": c.message, "files": list(c.files)}
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_git_log(self) -> List[CommitRecord]:
        if self._git_log is not None:
            return self._git_log
        path = self.root / "git_log.json"
        if not path.exists():
            self._git_log = []
            return self._git_log
        raw = json.loads(path.read_text("utf-8"))
        self._git_log = [
            CommitRecord(
                sha     = c["sha"],
                author  = c.get("author", "unknown"),
                date    = c.get("date", ""),
                message = c.get("message", ""),
                files   = list(c.get("files", [])),
            )
            for c in raw
        ]
        return self._git_log

    def _resolve_tree(self, path: str) -> Path:
        """Resolve a user-supplied relative path under tree/, blocking escapes."""
        cleaned = (path or ".").lstrip("/").lstrip(os.sep)
        if cleaned in ("", "."):
            return self.tree_root
        target = (self.tree_root / cleaned).resolve()
        if not str(target).startswith(str(self.tree_root)):
            raise CodeWorkspaceError(f"Illegal path (escapes sandbox): {path}")
        if not target.exists():
            raise CodeWorkspaceError(f"Path not found: {path}")
        return target

    def _rel_to_tree(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.tree_root)) or "."
        except ValueError:
            return str(p)

    def _iter_tree_files(self, pattern: str, max_depth: int = 16):
        """Yield Paths under tree/ matching pattern (glob-style)."""
        for dirpath, dirnames, filenames in os.walk(self.tree_root):
            depth = Path(dirpath).relative_to(self.tree_root).parts
            if len(depth) > max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fname in filenames:
                if fname.startswith("."):
                    continue
                if pattern == "*" or fnmatch.fnmatch(fname, pattern):
                    yield Path(dirpath) / fname

    @staticmethod
    def _filter_diff_by_path(diff: str, path: str) -> str:
        """Return only diff hunks where the +++ b/<file> matches `path`."""
        out: List[str] = []
        keep = False
        target = path.strip("/")
        for line in diff.split("\n"):
            if line.startswith("diff --git ") or line.startswith("--- a/") or line.startswith("+++ b/"):
                if line.startswith("+++ b/"):
                    file_in_hunk = line[6:].strip()
                    keep = (file_in_hunk == target
                            or file_in_hunk.startswith(target.rstrip("/") + "/"))
                if line.startswith("diff --git "):
                    # carry through; we'll re-evaluate at +++
                    pass
            if keep or line.startswith("diff --git "):
                out.append(line)
        return "\n".join(out)
