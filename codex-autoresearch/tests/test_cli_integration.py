from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from codex_autoresearch.models import RunStatus
from codex_autoresearch.session import load_session

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing else f"{PROJECT_ROOT}{os.pathsep}{existing}"
    return env


def _run_cli(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "codex_autoresearch", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_env(),
        check=False,
    )


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "user.email", "test@example.com")
    (repo / "src").mkdir()
    (repo / "src" / "app.txt").write_text("baseline\n", encoding="utf-8")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "initial")
    return repo


def _write_benchmark(repo: Path) -> None:
    (repo / "autoresearch.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

value="$(cat src/metric.txt)"
if [[ "${value}" == "crash" ]]; then
  echo "boom" >&2
  exit 3
fi

echo "METRIC score=${value}"
echo "METRIC allocs=1"
""",
        encoding="utf-8",
    )
    (repo / "autoresearch.sh").chmod(0o755)


def test_init_creates_session_files_and_branch(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    result = _run_cli(
        repo,
        "init",
        "--name",
        "Demo Experiment",
        "--metric-name",
        "score",
        "--direction",
        "lower",
        "--scope",
        "src",
    )
    assert result.returncode == 0, result.stderr
    assert (repo / "autoresearch.md").exists()
    assert (repo / "autoresearch.sh").exists()
    assert (repo / "autoresearch.jsonl").exists()
    branch = _run_git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    assert branch == f"codex/autoresearch/demo-experiment-{time.strftime('%Y%m%d')}"


def test_trial_keeps_best_and_discards_regressions(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run_cli(
        repo,
        "init",
        "--name",
        "Score Loop",
        "--metric-name",
        "score",
        "--direction",
        "lower",
        "--scope",
        "src",
    )
    _write_benchmark(repo)
    (repo / "src" / "metric.txt").write_text("10\n", encoding="utf-8")

    baseline = _run_cli(repo, "trial", "--description", "baseline")
    assert baseline.returncode == 0, baseline.stderr

    state = load_session(repo)
    assert len(state.runs) == 1
    assert state.runs[0].status is RunStatus.KEEP
    assert state.best_run is not None and state.best_run.metric == 10

    (repo / "src" / "metric.txt").write_text("15\n", encoding="utf-8")
    (repo / "src" / "app.txt").write_text("worse\n", encoding="utf-8")
    worse = _run_cli(repo, "trial", "--description", "worse")
    assert worse.returncode == 0, worse.stderr

    state = load_session(repo)
    assert len(state.runs) == 2
    assert state.runs[-1].status is RunStatus.DISCARD
    assert (repo / "src" / "metric.txt").read_text(encoding="utf-8") == "10\n"
    assert (repo / "src" / "app.txt").read_text(encoding="utf-8") == "baseline\n"
    status = _run_git(repo, "status", "--short")
    assert "src/app.txt" not in status
    assert "src/metric.txt" not in status


def test_trial_marks_crashes_and_restores_scope(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run_cli(
        repo,
        "init",
        "--name",
        "Crash Loop",
        "--metric-name",
        "score",
        "--direction",
        "lower",
        "--scope",
        "src",
    )
    _write_benchmark(repo)
    (repo / "src" / "metric.txt").write_text("10\n", encoding="utf-8")
    kept = _run_cli(repo, "trial", "--description", "baseline")
    assert kept.returncode == 0, kept.stderr

    (repo / "src" / "metric.txt").write_text("crash\n", encoding="utf-8")
    (repo / "src" / "app.txt").write_text("broken\n", encoding="utf-8")
    crashed = _run_cli(repo, "trial", "--description", "crash case")
    assert crashed.returncode == 0, crashed.stderr

    state = load_session(repo)
    assert state.runs[-1].status is RunStatus.CRASH
    assert (repo / "src" / "metric.txt").read_text(encoding="utf-8") == "10\n"
    assert (repo / "src" / "app.txt").read_text(encoding="utf-8") == "baseline\n"
