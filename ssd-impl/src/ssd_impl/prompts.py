from __future__ import annotations

import argparse
import re
from typing import Any, Iterable

from ssd_impl.config import add_config_arguments, load_config_from_args, SsdConfig
from ssd_impl.io_utils import write_jsonl


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def prompt_record_from_item(item: dict[str, Any]) -> dict[str, Any]:
    problem_id = item.get("question_id", item.get("id", ""))
    question = str(item.get("question", "")).strip()
    starter_code = str(item.get("starter_code", "") or "")
    return {
        "problem_id": str(problem_id),
        "question": question,
        "starter_code": starter_code,
        "normalized_question": normalize_question(question),
    }


def dedupe_prompt_records(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        record = prompt_record_from_item(item)
        normalized = record["normalized_question"]
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(record)
    return deduped


def load_seed_dataset(config: SsdConfig, dataset_loader=None) -> list[dict[str, Any]]:
    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    dataset = dataset_loader(
        config.dataset.name,
        config.dataset.subset,
        split=config.dataset.split,
        cache_dir=config.dataset.cache_dir,
    )
    return [dict(item) for item in dataset]


def prepare_prompts(config: SsdConfig, dataset_loader=None) -> list[dict[str, Any]]:
    dataset_items = load_seed_dataset(config, dataset_loader=dataset_loader)
    prompt_records = dedupe_prompt_records(dataset_items)
    output_path = config.resolve_path(config.paths.prompts_path)
    write_jsonl(output_path, prompt_records)
    return prompt_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deduplicated SSD prompt records.")
    add_config_arguments(parser)
    args = parser.parse_args()

    config = load_config_from_args(args)
    records = prepare_prompts(config)
    print(f"Wrote {len(records)} prompt records to {config.resolve_path(config.paths.prompts_path)}")

