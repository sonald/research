from __future__ import annotations

from ssd_impl.dataset import SsdTrainingDataset, collate_examples
from tests.helpers import FakeTokenizer


def test_training_dataset_masks_prompt_tokens() -> None:
    tokenizer = FakeTokenizer()
    records = [
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
            "completion": "```python\nprint(1)\nprint(2)\n```",
        }
    ]

    dataset = SsdTrainingDataset(records, tokenizer=tokenizer, max_seq_len=512)
    example = dataset[0]

    assert any(label == -100 for label in example["labels"])
    assert any(label != -100 for label in example["labels"])


def test_collate_examples_pads_to_longest_sequence() -> None:
    batch = [
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        {"input_ids": [1], "attention_mask": [1], "labels": [-100]},
    ]
    collated = collate_examples(batch, pad_token_id=0)

    assert tuple(collated["input_ids"].shape) == (2, 2)
    assert collated["input_ids"][1, 1].item() == 0
    assert collated["labels"][1, 1].item() == -100

