from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ssd_impl.io_utils import read_jsonl


@dataclass
class TokenizedConversation:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def build_training_texts(tokenizer: Any, messages: list[dict[str, str]], completion: str) -> tuple[str, str]:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": completion}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_text, full_text


def tokenize_conversation(
    tokenizer: Any,
    messages: list[dict[str, str]],
    completion: str,
    max_seq_len: int,
) -> TokenizedConversation | None:
    prompt_text, full_text = build_training_texts(tokenizer, messages, completion)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = list(full_ids)
    for index in range(prompt_len):
        labels[index] = -100

    if all(value == -100 for value in labels):
        return None

    return TokenizedConversation(
        input_ids=list(full_ids),
        attention_mask=[1] * len(full_ids),
        labels=labels,
    )


class SsdTrainingDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> None:
        self.examples: list[TokenizedConversation] = []
        self.dropped = 0

        for record in records:
            example = tokenize_conversation(
                tokenizer=tokenizer,
                messages=list(record["messages"]),
                completion=str(record["completion"]),
                max_seq_len=max_seq_len,
            )
            if example is None:
                self.dropped += 1
                continue
            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
        }


def collate_examples(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []

    for item in batch:
        padding = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        attention_mask.append(item["attention_mask"] + [0] * padding)
        labels.append(item["labels"] + [-100] * padding)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_train_dataloader(
    dataset: SsdTrainingDataset,
    batch_size: int,
    pad_token_id: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_examples(batch, pad_token_id=pad_token_id),
    )


def load_synth_records(path: str) -> list[dict]:
    return read_jsonl(path)

