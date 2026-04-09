from __future__ import annotations

from pathlib import Path

import torch

from ssd_impl.config import load_config
from ssd_impl.io_utils import read_jsonl, write_jsonl
from ssd_impl.synthesize import generate_records, select_generation_backend
from tests.helpers import FakeTokenizer


class StubBackend:
    def __init__(self, grouped_outputs):
        self.grouped_outputs = grouped_outputs

    def generate(self, prompt_texts, decode):
        return self.grouped_outputs[: len(prompt_texts)]


def test_select_generation_backend_falls_back_to_transformers(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_root: .",
                "model:",
                "  name_or_path: test-model",
                "sample:",
                "  preset: post_ssd",
                "  presets:",
                "    post_ssd:",
                "      temperature: 1.1",
                "    frozen:",
                "      temperature: 0.7",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    class FakeTransformersBackend:
        def __init__(self, *args, **kwargs):
            self.kind = "transformers"

    monkeypatch.setattr("ssd_impl.synthesize.TransformersGenerationBackend", FakeTransformersBackend)
    monkeypatch.setattr("ssd_impl.synthesize.is_vllm_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    backend = select_generation_backend(config, config.selected_sample_decode("post_ssd"))
    assert backend.kind == "transformers"


def test_generate_records_applies_filter_and_writes_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_root: .",
                "model:",
                "  name_or_path: test-model",
                "paths:",
                "  prompts_path: artifacts/prompts/prompts.jsonl",
                "  synth_train_path: artifacts/synth/train.jsonl",
                "  synth_rejected_path: artifacts/synth/rejected.jsonl",
                "sample:",
                "  preset: post_ssd",
                "  presets:",
                "    post_ssd:",
                "      temperature: 1.1",
                "    frozen:",
                "      temperature: 0.7",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    write_jsonl(
        tmp_path / "artifacts/prompts/prompts.jsonl",
        [
            {"problem_id": "1", "question": "A", "starter_code": "", "normalized_question": "A"},
            {"problem_id": "2", "question": "B", "starter_code": "", "normalized_question": "B"},
        ],
    )

    accepted, rejected = generate_records(
        config,
        decode=config.selected_sample_decode("post_ssd"),
        output_path=tmp_path / "artifacts/synth/train.jsonl",
        rejected_path=tmp_path / "artifacts/synth/rejected.jsonl",
        backend=StubBackend([["```python\nprint(1)\nprint(2)\n```"], ["pass"]]),
        tokenizer=FakeTokenizer(),
        batch_size=2,
    )

    assert len(accepted) == 1
    assert len(rejected) == 1
    assert read_jsonl(tmp_path / "artifacts/synth/train.jsonl")[0]["problem_id"] == "1"
    assert read_jsonl(tmp_path / "artifacts/synth/rejected.jsonl")[0]["reject_reason"] == "single_line_stub"

