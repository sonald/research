from __future__ import annotations

from ssd_impl.config import FilteringConfig
from ssd_impl.filters import evaluate_completion


def test_filter_rejects_empty_output() -> None:
    decision = evaluate_completion("   ", FilteringConfig())
    assert not decision.accepted
    assert decision.reason == "empty_completion"


def test_filter_rejects_empty_code_block() -> None:
    decision = evaluate_completion("```python\n```", FilteringConfig())
    assert not decision.accepted
    assert decision.reason == "empty_code_block"


def test_filter_rejects_single_line_stub() -> None:
    decision = evaluate_completion("pass", FilteringConfig())
    assert not decision.accepted
    assert decision.reason == "single_line_stub"


def test_filter_accepts_multi_line_completion() -> None:
    decision = evaluate_completion("```python\nprint(1)\nprint(2)\n```", FilteringConfig())
    assert decision.accepted
    assert decision.reason is None

