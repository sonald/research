from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Direction(str, Enum):
    LOWER = "lower"
    HIGHER = "higher"


class RunStatus(str, Enum):
    KEEP = "keep"
    DISCARD = "discard"
    CRASH = "crash"


@dataclass(slots=True)
class ConfigRecord:
    name: str
    metric_name: str
    metric_unit: str
    direction: Direction
    scope: list[str]
    off_limits: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    branch: str | None = None
    segment: int = 1

    def to_json_record(self) -> dict[str, Any]:
        return {
            "type": "config",
            "name": self.name,
            "metricName": self.metric_name,
            "metricUnit": self.metric_unit,
            "bestDirection": self.direction.value,
            "scope": self.scope,
            "off_limits": self.off_limits,
            "constraints": self.constraints,
            "branch": self.branch,
            "segment": self.segment,
        }

    @classmethod
    def from_json_record(cls, record: dict[str, Any], segment: int) -> "ConfigRecord":
        return cls(
            name=str(record.get("name", "Autoresearch")),
            metric_name=str(record.get("metricName", record.get("metric_name", "metric"))),
            metric_unit=str(record.get("metricUnit", record.get("metric_unit", ""))),
            direction=Direction(str(record.get("bestDirection", record.get("direction", "lower")))),
            scope=[str(item) for item in record.get("scope", [])],
            off_limits=[str(item) for item in record.get("off_limits", record.get("offLimits", []))],
            constraints=[str(item) for item in record.get("constraints", [])],
            branch=str(record["branch"]) if record.get("branch") else None,
            segment=int(record.get("segment", segment)),
        )


@dataclass(slots=True)
class RunRecord:
    run: int
    segment: int
    metric: float
    metrics: dict[str, float]
    status: RunStatus
    description: str
    timestamp: int
    commit: str
    duration_seconds: float
    exit_code: int | None

    def to_json_record(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run": self.run,
            "segment": self.segment,
            "metric": self.metric,
            "metrics": self.metrics,
            "status": self.status.value,
            "description": self.description,
            "timestamp": self.timestamp,
            "commit": self.commit,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
        }
        return payload

    @classmethod
    def from_json_record(cls, record: dict[str, Any], segment: int) -> "RunRecord":
        return cls(
            run=int(record.get("run", 0)),
            segment=int(record.get("segment", segment)),
            metric=float(record.get("metric", 0.0)),
            metrics={str(key): float(value) for key, value in dict(record.get("metrics", {})).items()},
            status=RunStatus(str(record.get("status", "crash"))),
            description=str(record.get("description", "")),
            timestamp=int(record.get("timestamp", 0)),
            commit=str(record.get("commit", "")),
            duration_seconds=float(record.get("duration_seconds", record.get("durationSeconds", 0.0))),
            exit_code=int(record["exit_code"]) if record.get("exit_code") is not None else (
                int(record["exitCode"]) if record.get("exitCode") is not None else None
            ),
        )


@dataclass(slots=True)
class LastRun:
    command: str
    duration_seconds: float
    exit_code: int | None
    timed_out: bool
    parse_error: str | None
    primary_metric: float | None
    metrics: dict[str, float]
    tail_output: str
    timestamp: int

    @property
    def succeeded(self) -> bool:
        return (
            not self.timed_out
            and self.exit_code == 0
            and self.parse_error is None
            and self.primary_metric is not None
        )

    def to_json_record(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "parse_error": self.parse_error,
            "primary_metric": self.primary_metric,
            "metrics": self.metrics,
            "tail_output": self.tail_output,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_json_record(cls, record: dict[str, Any]) -> "LastRun":
        return cls(
            command=str(record.get("command", "./autoresearch.sh")),
            duration_seconds=float(record.get("duration_seconds", 0.0)),
            exit_code=int(record["exit_code"]) if record.get("exit_code") is not None else None,
            timed_out=bool(record.get("timed_out", False)),
            parse_error=str(record["parse_error"]) if record.get("parse_error") else None,
            primary_metric=float(record["primary_metric"]) if record.get("primary_metric") is not None else None,
            metrics={str(key): float(value) for key, value in dict(record.get("metrics", {})).items()},
            tail_output=str(record.get("tail_output", "")),
            timestamp=int(record.get("timestamp", 0)),
        )

