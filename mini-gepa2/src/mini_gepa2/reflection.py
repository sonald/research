from __future__ import annotations

import json
import re
from typing import Protocol

from .adapter import Candidate, ReflectiveRecord


class ReflectionLM(Protocol):
    """Minimal callable interface for the reflection model."""

    def __call__(self, prompt: str) -> str:
        """Return a rewritten component text."""


_CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", re.DOTALL)


def render_reflection_prompt(
    candidate: Candidate,
    component_name: str,
    reflective_records: list[ReflectiveRecord],
) -> str:
    """Render the prompt shown to the reflection LM.

    The prompt is deliberately plain:

    - show the current text
    - show structured evidence collected from failed and successful rollouts
    - ask for one improved replacement

    This mirrors the heart of GEPA: it learns from rich textual side information,
    not only from scalar rewards.
    """

    current_text = candidate[component_name]
    records_json = json.dumps(reflective_records, indent=2, ensure_ascii=False)
    return (
        "You are improving one text component inside a larger system.\n\n"
        f"Component name: {component_name}\n\n"
        "Current text:\n"
        "```text\n"
        f"{current_text}\n"
        "```\n\n"
        "Reflective records gathered from recent rollouts:\n"
        "```json\n"
        f"{records_json}\n"
        "```\n\n"
        "Write a better replacement for the current text.\n"
        "Return only the new text. A single fenced code block is also acceptable.\n"
    )


def extract_new_text(raw_lm_output: str) -> str:
    """Parse the reflection LM output into the actual replacement text.

    We support both of the simple output styles that are common in practice:

    1. the model returns plain text directly
    2. the model wraps the answer in one fenced code block
    """

    match = _CODE_BLOCK_RE.search(raw_lm_output)
    if match is not None:
        return match.group(1).strip()
    return raw_lm_output.strip()


def propose_component_update(
    candidate: Candidate,
    component_name: str,
    reflective_records: list[ReflectiveRecord],
    reflection_lm: ReflectionLM,
) -> tuple[str, str, str]:
    """Run one reflection step for one component.

    Returning the rendered prompt and raw output makes the algorithm easier to inspect.
    A teaching implementation should make the "what did we show the LM?" question easy
    to answer.
    """

    rendered_prompt = render_reflection_prompt(
        candidate=candidate,
        component_name=component_name,
        reflective_records=reflective_records,
    )
    raw_lm_output = reflection_lm(rendered_prompt)
    new_text = extract_new_text(raw_lm_output)
    return new_text, rendered_prompt, raw_lm_output
