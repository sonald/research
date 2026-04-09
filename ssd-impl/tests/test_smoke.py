from __future__ import annotations

from pathlib import Path

from accelerate import Accelerator

from ssd_impl.config import load_config
from ssd_impl.io_utils import read_jsonl
from ssd_impl.prompts import prepare_prompts
from ssd_impl.synthesize import synthesize_dataset
from ssd_impl.train import train
from tests.helpers import FakeTokenizer, ToyLM


class SmokeBackend:
    def generate(self, prompt_texts, decode):
        return [["```python\nprint(1)\nprint(2)\n```"] for _ in prompt_texts]


def test_mock_end_to_end_pipeline(tmp_path: Path) -> None:
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
                "  checkpoints_dir: artifacts/checkpoints",
                "training:",
                "  per_device_batch_size: 1",
                "  gradient_accumulation_steps: 1",
                "  max_steps: 2",
                "  warmup_steps: 0",
                "  learning_rate: 0.001",
                "  save_every_steps: 1",
                "  log_every_steps: 1",
                "  gradient_checkpointing: false",
                "  adapter:",
                "    target_modules: [q_proj]",
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

    dataset_items = [
        {"question_id": "1", "question": "Write hello world", "starter_code": ""},
        {"question_id": "2", "question": "Write fibonacci", "starter_code": ""},
    ]
    prepare_prompts(config, dataset_loader=lambda *args, **kwargs: dataset_items)
    synthesize_dataset(config, backend=SmokeBackend(), tokenizer=FakeTokenizer())

    result = train(
        config,
        tokenizer=FakeTokenizer(),
        model=ToyLM(),
        accelerator=Accelerator(cpu=True),
    )

    assert result["steps"] == 2
    assert len(read_jsonl(tmp_path / "artifacts/synth/train.jsonl")) == 2
    assert Path(result["final_checkpoint"]).exists()

