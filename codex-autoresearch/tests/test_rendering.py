from __future__ import annotations

import json
from pathlib import Path

from codex_autoresearch.rendering import render_dashboard, render_status
from codex_autoresearch.session import load_session


def _build_state(tmp_path: Path):
    rows = [
        {"type": "config", "name": "demo", "metricName": "latency", "metricUnit": "ms", "bestDirection": "lower", "scope": ["src"], "branch": "codex/autoresearch/demo-20260313"},
        {"run": 1, "segment": 1, "metric": 10, "metrics": {"allocs": 1}, "status": "keep", "description": "baseline", "timestamp": 1, "commit": "aaaaaaa", "duration_seconds": 1.1, "exit_code": 0},
        {"run": 2, "segment": 1, "metric": 12, "metrics": {}, "status": "discard", "description": "worse", "timestamp": 2, "commit": "aaaaaaa", "duration_seconds": 1.2, "exit_code": 0},
    ]
    (tmp_path / "autoresearch.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return load_session(tmp_path)


def test_render_status_reports_counts(tmp_path: Path) -> None:
    text = render_status(_build_state(tmp_path))
    assert "autoresearch 2 runs 1 kept 1 discarded 0 crashed" in text
    assert "baseline: 10ms" in text
    assert "best: 10ms" in text


def test_render_dashboard_contains_recent_runs(tmp_path: Path) -> None:
    text = render_dashboard(_build_state(tmp_path), tail=10, cwd=tmp_path)
    assert "Autoresearch: demo" in text
    assert "Recent Runs" in text
    assert "baseline" in text
    assert "worse" in text

