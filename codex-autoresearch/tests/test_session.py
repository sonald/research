from __future__ import annotations

import json
from pathlib import Path

from codex_autoresearch.models import RunStatus
from codex_autoresearch.session import load_session, parse_metric_lines


def test_parse_metric_lines_extracts_multiple_values() -> None:
    output = """
    hello
    METRIC latency_ms=12.5
    METRIC allocs=42
    METRIC ratio=1.2e+01
    """
    metrics = parse_metric_lines(output)
    assert metrics == {"latency_ms": 12.5, "allocs": 42.0, "ratio": 12.0}


def test_load_session_reads_pi_compatible_segments(tmp_path: Path) -> None:
    rows = [
        {"type": "config", "name": "one", "metricName": "latency", "metricUnit": "ms", "bestDirection": "lower", "scope": ["src"]},
        {"run": 1, "metric": 10, "metrics": {"allocs": 1}, "status": "keep", "description": "baseline", "timestamp": 1, "commit": "aaaaaaa", "duration_seconds": 1.1, "exit_code": 0},
        {"type": "config", "name": "two", "metricName": "score", "metricUnit": "", "bestDirection": "higher", "scope": ["pkg"]},
        {"run": 2, "metric": 50, "metrics": {}, "status": "keep", "description": "reinit", "timestamp": 2, "commit": "bbbbbbb", "duration_seconds": 2.2, "exit_code": 0},
    ]
    (tmp_path / "autoresearch.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    state = load_session(tmp_path)

    assert len(state.configs) == 2
    assert state.current_config is not None
    assert state.current_config.name == "two"
    assert state.current_segment == 2
    assert [run.segment for run in state.runs] == [1, 2]
    assert state.current_runs[0].status is RunStatus.KEEP

