from __future__ import annotations

import datetime as dt
from pathlib import Path

from .models import Direction, RunRecord, RunStatus
from .session import SessionState


def format_metric(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    if value == int(value):
        rendered = f"{int(value):,}"
    else:
        rendered = f"{value:,.3f}".rstrip("0").rstrip(".")
    return f"{rendered}{unit}"


def render_status(state: SessionState) -> str:
    config = state.current_config
    if config is None:
        return "No autoresearch session initialized yet."

    runs = state.current_runs
    keeps = sum(1 for run in runs if run.status is RunStatus.KEEP)
    discards = sum(1 for run in runs if run.status is RunStatus.DISCARD)
    crashes = sum(1 for run in runs if run.status is RunStatus.CRASH)
    baseline = state.baseline_run.metric if state.baseline_run else None
    best = state.best_run.metric if state.best_run else None
    return (
        f"autoresearch {len(runs)} runs {keeps} kept {discards} discarded {crashes} crashed"
        f" | baseline: {format_metric(baseline, config.metric_unit)}"
        f" | best: {format_metric(best, config.metric_unit)}"
        f" | branch: {config.branch or '(current)'}"
    )


def render_dashboard(state: SessionState, tail: int = 20, cwd: Path | None = None) -> str:
    config = state.current_config
    if config is None:
        return "No autoresearch session initialized yet."

    runs = state.current_runs[-tail:] if tail > 0 else state.current_runs
    lines = [
        f"Autoresearch: {config.name}",
        f"Metric: {config.metric_name} ({config.metric_unit or 'unitless'}, {_direction_label(config.direction)})",
        f"Branch: {config.branch or '(current)'}",
        f"Scope: {', '.join(config.scope) if config.scope else '(none)'}",
        f"Baseline: {format_metric(state.baseline_run.metric if state.baseline_run else None, config.metric_unit)}",
        f"Best: {format_metric(state.best_run.metric if state.best_run else None, config.metric_unit)}",
        "",
        "Recent Runs",
        "run  status   metric      commit   description",
        "---  -------  ----------  -------  -----------",
    ]

    best_metric = state.best_run.metric if state.best_run else None
    for run in runs:
        metric_text = format_metric(run.metric, config.metric_unit).ljust(10)
        description = run.description[:60]
        lines.append(
            f"{str(run.run).rjust(3)}  {run.status.value.ljust(7)}  {metric_text}  {run.commit[:7].ljust(7)}  {description}"
        )
        if run.metrics:
            extras = " ".join(f"{key}={value:g}" for key, value in sorted(run.metrics.items()))
            lines.append(f"     metrics  {extras}")
        if best_metric is not None:
            delta = _delta_text(run.metric, best_metric, config.direction)
            stamp = dt.datetime.fromtimestamp(run.timestamp / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            lines.append(f"     delta    {delta} | {stamp}")

    if cwd is not None:
        lines.extend(["", f"CWD: {cwd}"])

    return "\n".join(lines)


def _direction_label(direction: Direction) -> str:
    return "lower is better" if direction is Direction.LOWER else "higher is better"


def _delta_text(value: float, best: float, direction: Direction) -> str:
    if best == 0:
        return "n/a"
    delta = value - best
    pct = (delta / best) * 100
    if direction is Direction.HIGHER:
        pct *= -1
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}% vs best"

