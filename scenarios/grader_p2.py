"""
Phase 2 grader — oracle-independent.
Takes only the P2 trajectory (List[StepRecord]) and the declared patch/no-change.
Scores patch quality using three tiers.
"""

import ast
import difflib
from typing import List, Optional
from ..models import StepRecord, CodeContext


def grade_patch_quality(proposed_diff: str, ctx: CodeContext) -> float:
    """
    Three-tier patch scoring:
      Tier 1 (40%): file overlap — did they touch the right files?
      Tier 2 (30%): AST hunk similarity — do the changed functions match?
      Tier 3 (30%): syntax validity — does the patch parse cleanly?
    """
    proposed_files = _extract_files_from_diff(proposed_diff)
    ground_truth_files = set(ctx.ground_truth_files)

    # Tier 1
    if not ground_truth_files:
        file_score = 0.0
    else:
        intersection = proposed_files & ground_truth_files
        union = proposed_files | ground_truth_files
        file_score = len(intersection) / len(union) if union else 0.0

    # Tier 2
    hunk_score = _ast_hunk_similarity(proposed_diff, ctx.ground_truth_diff)

    # Tier 3
    syntax_score = 1.0 if _patch_parses_cleanly(proposed_diff) else 0.0

    return 0.4 * file_score + 0.3 * hunk_score + 0.3 * syntax_score


def grade_no_change(declared: bool, ctx: CodeContext) -> float:
    """1.0 if agent correctly identified spurious issue, 0.0 otherwise."""
    if not ctx.is_valid_issue and declared:
        return 1.0
    if ctx.is_valid_issue and not declared:
        return 0.0
    return 0.0   # wrong in either direction


def grade_p2_efficiency(p2_steps: int, expected_steps: int) -> float:
    """
    Normalized efficiency — doesn't penalize inherently hard bugs.
    Score = 1.0 at expected_steps, decays to 0 at 2x expected.
    """
    ratio = p2_steps / max(expected_steps, 1)
    return max(0.0, 1.0 - max(0.0, ratio - 1.0))


def _extract_files_from_diff(diff: str) -> set:
    files = set()
    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            files.add(line[6:].strip())
    return files


def _ast_hunk_similarity(proposed: str, ground_truth: str) -> float:
    """
    Extract (file, function_name) pairs from both diffs.
    Score = Jaccard overlap of those sets.
    """
    proposed_fns = _extract_changed_functions(proposed)
    truth_fns = _extract_changed_functions(ground_truth)
    if not truth_fns:
        return 1.0  # trivial patch, full credit if syntax valid
    intersection = proposed_fns & truth_fns
    union = proposed_fns | truth_fns
    return len(intersection) / len(union) if union else 0.0


def _extract_changed_functions(diff: str) -> set:
    """Parse diff hunks, extract function names via simple @@ line parsing."""
    fns = set()
    current_file = ""
    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
        elif line.startswith("@@") and "def " in line:
            # hunk header often contains function context
            parts = line.split("def ")
            if len(parts) > 1:
                fn_name = parts[1].split("(")[0].strip()
                fns.add(f"{current_file}:{fn_name}")
    return fns


def _patch_parses_cleanly(diff: str) -> bool:
    """Extract added lines, try to parse as Python."""
    added = [l[1:] for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++")]
    try:
        ast.parse("\n".join(added))
        return True
    except SyntaxError:
        return False