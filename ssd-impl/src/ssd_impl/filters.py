from __future__ import annotations

from dataclasses import dataclass

from ssd_impl.config import FilteringConfig


@dataclass
class FilterDecision:
    accepted: bool
    reason: str | None = None


def _is_empty_code_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "```" not in stripped:
        return False
    residual = stripped.replace("```python", "").replace("```", "").strip()
    return not residual


def evaluate_completion(text: str, filtering: FilteringConfig) -> FilterDecision:
    stripped = (text or "").strip()
    if filtering.reject_empty_completion and not stripped:
        return FilterDecision(False, "empty_completion")

    if filtering.reject_empty_code_block and _is_empty_code_block(stripped):
        return FilterDecision(False, "empty_code_block")

    if filtering.reject_single_line_stub:
        nonempty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(nonempty_lines) == 1 and len(nonempty_lines[0]) <= filtering.single_line_stub_max_chars:
            return FilterDecision(False, "single_line_stub")

    return FilterDecision(True, None)

