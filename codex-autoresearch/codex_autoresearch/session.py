from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import ConfigRecord, Direction, LastRun, RunRecord, RunStatus

AUTORESEARCH_MD = "autoresearch.md"
AUTORESEARCH_SH = "autoresearch.sh"
AUTORESEARCH_JSONL = "autoresearch.jsonl"
AUTORESEARCH_IDEAS = "autoresearch.ideas.md"

METRIC_PATTERN = re.compile(
    r"^\s*METRIC\s+(?P<name>[A-Za-z0-9_.:-]+)=(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)

DEFAULT_MD_TEMPLATE = """# Autoresearch: {name}

## Objective
Describe exactly what is being optimized and why it matters.

## Metrics
- **Primary**: {metric_name} ({metric_unit_display}, {direction_text})
- **Secondary**: Add any extra metrics that are important to watch.

## How to Run
`./autoresearch.sh` — emits `METRIC name=value` lines.

## Files in Scope
{scope_block}

## Off Limits
{off_limits_block}

## Constraints
{constraints_block}

## What's Been Tried
- Baseline not recorded yet.
"""

DEFAULT_SH_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

# Replace this with the real benchmark for the project.
# The script must emit one line for the primary metric:
#   METRIC {metric_name}=123.45

echo "TODO: implement benchmark command for {metric_name}" >&2
echo "METRIC {metric_name}=0"
"""


class SessionError(RuntimeError):
    """Raised for session state errors."""


@dataclass(slots=True)
class SessionState:
    cwd: Path
    configs: list[ConfigRecord]
    runs: list[RunRecord]

    @property
    def current_config(self) -> ConfigRecord | None:
        return self.configs[-1] if self.configs else None

    @property
    def current_segment(self) -> int:
        return self.current_config.segment if self.current_config else 0

    @property
    def current_runs(self) -> list[RunRecord]:
        segment = self.current_segment
        return [run for run in self.runs if run.segment == segment]

    @property
    def baseline_run(self) -> RunRecord | None:
        for run in self.current_runs:
            if run.status is RunStatus.KEEP:
                return run
        return None

    @property
    def best_run(self) -> RunRecord | None:
        config = self.current_config
        if config is None:
            return None
        kept = [run for run in self.current_runs if run.status is RunStatus.KEEP]
        if not kept:
            return None
        key = min if config.direction is Direction.LOWER else max
        return key(kept, key=lambda run: run.metric)

    @property
    def next_run_number(self) -> int:
        return len(self.runs) + 1


def session_jsonl_path(cwd: Path) -> Path:
    return cwd / AUTORESEARCH_JSONL


def session_md_path(cwd: Path) -> Path:
    return cwd / AUTORESEARCH_MD


def session_sh_path(cwd: Path) -> Path:
    return cwd / AUTORESEARCH_SH


def session_ideas_path(cwd: Path) -> Path:
    return cwd / AUTORESEARCH_IDEAS


def git_state_dir(cwd: Path) -> Path:
    return cwd / ".git" / "codex-autoresearch"


def last_run_path(cwd: Path) -> Path:
    return git_state_dir(cwd) / "last-run.json"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "experiment"


def default_branch_name(name: str, stamp: time.struct_time | None = None) -> str:
    stamp = stamp or time.localtime()
    return f"codex/autoresearch/{slugify(name)}-{time.strftime('%Y%m%d', stamp)}"


def parse_metric_lines(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in METRIC_PATTERN.finditer(output):
        metrics[match.group("name")] = float(match.group("value"))
    return metrics


def render_markdown_template(config: ConfigRecord) -> str:
    return DEFAULT_MD_TEMPLATE.format(
        name=config.name,
        metric_name=config.metric_name,
        metric_unit_display=config.metric_unit or "unitless",
        direction_text="lower is better" if config.direction is Direction.LOWER else "higher is better",
        scope_block=_render_bullets(config.scope),
        off_limits_block=_render_bullets(config.off_limits, default="(none yet)"),
        constraints_block=_render_bullets(config.constraints, default="(none yet)"),
    )


def render_script_template(config: ConfigRecord) -> str:
    return DEFAULT_SH_TEMPLATE.format(metric_name=config.metric_name)


def _render_bullets(values: Iterable[str], default: str = "(fill this in)") -> str:
    items = [str(value) for value in values if str(value).strip()]
    if not items:
        return default
    return "\n".join(f"- `{item}`" for item in items)


def ensure_session_files(cwd: Path, config: ConfigRecord) -> None:
    md_path = session_md_path(cwd)
    if not md_path.exists():
        md_path.write_text(render_markdown_template(config), encoding="utf-8")

    sh_path = session_sh_path(cwd)
    if not sh_path.exists():
        sh_path.write_text(render_script_template(config), encoding="utf-8")
        sh_path.chmod(0o755)


def ensure_git_state_dir(cwd: Path) -> Path:
    path = git_state_dir(cwd)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_session(cwd: Path) -> SessionState:
    path = session_jsonl_path(cwd)
    if not path.exists():
        return SessionState(cwd=cwd, configs=[], runs=[])

    configs: list[ConfigRecord] = []
    runs: list[RunRecord] = []
    segment = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("type") == "config":
            segment += 1
            config = ConfigRecord.from_json_record(payload, segment=segment)
            if config.segment != segment:
                segment = config.segment
            configs.append(config)
            continue

        effective_segment = int(payload.get("segment", segment if segment > 0 else 1))
        runs.append(RunRecord.from_json_record(payload, effective_segment))

    return SessionState(cwd=cwd, configs=configs, runs=runs)


def append_config(cwd: Path, config: ConfigRecord) -> None:
    with session_jsonl_path(cwd).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(config.to_json_record(), sort_keys=True) + "\n")


def append_run(cwd: Path, run: RunRecord) -> None:
    with session_jsonl_path(cwd).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run.to_json_record(), sort_keys=True) + "\n")


def write_last_run(cwd: Path, last_run: LastRun) -> Path:
    ensure_git_state_dir(cwd)
    path = last_run_path(cwd)
    path.write_text(json.dumps(last_run.to_json_record(), sort_keys=True), encoding="utf-8")
    return path


def load_last_run(cwd: Path) -> LastRun:
    path = last_run_path(cwd)
    if not path.exists():
        raise SessionError("No pending run found. Execute `codex-autoresearch run` first.")
    return LastRun.from_json_record(json.loads(path.read_text(encoding="utf-8")))


def delete_last_run(cwd: Path) -> None:
    path = last_run_path(cwd)
    if path.exists():
        path.unlink()


def choose_status(config: ConfigRecord, session: SessionState, last_run: LastRun) -> RunStatus:
    if not last_run.succeeded:
        return RunStatus.CRASH

    best_run = session.best_run
    if best_run is None:
        return RunStatus.KEEP

    assert last_run.primary_metric is not None
    if config.direction is Direction.LOWER:
        return RunStatus.KEEP if last_run.primary_metric < best_run.metric else RunStatus.DISCARD
    return RunStatus.KEEP if last_run.primary_metric > best_run.metric else RunStatus.DISCARD

