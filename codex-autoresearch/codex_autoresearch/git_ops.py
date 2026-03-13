from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .session import AUTORESEARCH_IDEAS, AUTORESEARCH_JSONL, AUTORESEARCH_MD, AUTORESEARCH_SH


class GitError(RuntimeError):
    """Raised for git command failures."""


def run_git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise GitError(result.stderr.strip() or result.stdout.strip() or "git command failed")
    return result


def ensure_git_repo(cwd: Path) -> None:
    run_git(cwd, "rev-parse", "--show-toplevel")


def working_tree_dirty(cwd: Path) -> bool:
    result = run_git(cwd, "status", "--porcelain", check=True)
    return bool(result.stdout.strip())


def current_branch(cwd: Path) -> str:
    return run_git(cwd, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def checkout_branch(cwd: Path, branch: str) -> None:
    exists = run_git(cwd, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False)
    if exists.returncode == 0:
        run_git(cwd, "switch", branch)
        return
    run_git(cwd, "switch", "-c", branch)


def head_commit(cwd: Path) -> str:
    return run_git(cwd, "rev-parse", "--short=7", "HEAD").stdout.strip()


def commit_keep(cwd: Path, description: str, primary_metric_name: str, metric: float, metrics: dict[str, float], scope: list[str]) -> tuple[str, str]:
    pathspecs = _pathspecs_for_commit(cwd, scope)
    if not pathspecs:
        raise GitError("Nothing is available to commit for this run.")

    run_git(cwd, "add", "--", *pathspecs)

    cached = run_git(cwd, "diff", "--cached", "--quiet", check=False)
    if cached.returncode == 0:
        return head_commit(cwd), "nothing to commit"

    result_payload = {"status": "keep", primary_metric_name: metric, **metrics}
    commit_message = f"{description}\n\nResult: {json.dumps(result_payload, sort_keys=True)}"
    run_git(cwd, "commit", "-m", commit_message)
    return head_commit(cwd), "committed"


def restore_scope(cwd: Path, scope: list[str]) -> None:
    if not scope:
        return
    run_git(cwd, "restore", "--staged", "--worktree", "--source=HEAD", "--", *scope)
    run_git(cwd, "clean", "-fd", "--", *scope, check=False)


def _pathspecs_for_commit(cwd: Path, scope: list[str]) -> list[str]:
    candidates = list(scope)
    for name in (AUTORESEARCH_MD, AUTORESEARCH_SH, AUTORESEARCH_IDEAS, AUTORESEARCH_JSONL):
        if (cwd / name).exists():
            candidates.append(name)

    ordered: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
