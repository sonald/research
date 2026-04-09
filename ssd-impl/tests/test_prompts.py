from __future__ import annotations

from pathlib import Path

from ssd_impl.config import load_config
from ssd_impl.io_utils import read_jsonl
from ssd_impl.prompts import dedupe_prompt_records, normalize_question, prepare_prompts


def test_normalize_question_compacts_whitespace() -> None:
    assert normalize_question("a  \n  b\tc") == "a b c"


def test_dedupe_prompt_records_keeps_first_normalized_question() -> None:
    records = dedupe_prompt_records(
        [
            {"question_id": "1", "question": "print(1)\n", "starter_code": ""},
            {"question_id": "2", "question": "print(1)", "starter_code": "ignored"},
            {"question_id": "3", "question": "print(2)", "starter_code": ""},
        ]
    )
    assert [record["problem_id"] for record in records] == ["1", "3"]


def test_prepare_prompts_writes_expected_jsonl(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_root: .",
                "model:",
                "  name_or_path: test-model",
                "paths:",
                "  prompts_path: artifacts/prompts/prompts.jsonl",
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
        {"question_id": "1", "question": "A  B", "starter_code": ""},
        {"question_id": "2", "question": "A B", "starter_code": "pass"},
        {"question_id": "3", "question": "C", "starter_code": ""},
    ]
    records = prepare_prompts(config, dataset_loader=lambda *args, **kwargs: dataset_items)
    output_records = read_jsonl(tmp_path / "artifacts/prompts/prompts.jsonl")

    assert len(records) == 2
    assert output_records[0]["normalized_question"] == "A B"
    assert output_records[1]["problem_id"] == "3"

