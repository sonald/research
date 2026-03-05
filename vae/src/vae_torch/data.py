from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset, load_from_disk
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms


def read_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class HFDataManager:
    """Load and optionally persist Hugging Face dataset splits."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def load_splits(self, from_disk: bool, data_dir: str | Path) -> dict[str, Dataset]:
        output: dict[str, Dataset] = {}
        dataset_path = self.config["path"]

        for split in self.config["splits"]:
            if from_disk:
                split_path = Path(data_dir) / dataset_path / split
                output[split] = load_from_disk(str(split_path))
            else:
                output[split] = load_dataset(dataset_path, split=split)

        return output

    def save_splits(self, splits: dict[str, Dataset], output_root: str | Path) -> None:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        for split, ds in splits.items():
            split_path = output_root / split
            ds.save_to_disk(str(split_path))


def make_image_transform(height: int, width: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ]
    )


def build_hf_transform(height: int, width: int):
    image_transform = make_image_transform(height, width)

    def _transform(example: dict[str, Any]) -> dict[str, Any]:
        images = example["image"]
        if isinstance(images, list):
            processed = [image_transform(img.convert("RGB")) for img in images]
            example["image"] = processed
        else:
            example["image"] = image_transform(images.convert("RGB"))
        return example

    return _transform


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    if not batch:
        raise ValueError("Received empty batch in collate_fn.")

    images = [item["image"] for item in batch]
    output: dict[str, Tensor] = {"image": torch.stack(images, dim=0)}

    if "label" in batch[0]:
        labels = [item["label"] for item in batch]
        output["label"] = torch.as_tensor(labels)

    return output


def build_train_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    height: int,
    width: int,
) -> DataLoader:
    transformed = dataset.with_transform(build_hf_transform(height=height, width=width))
    return DataLoader(
        transformed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and save Hugging Face dataset splits.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    manager = HFDataManager(config)

    splits = manager.load_splits(from_disk=False, data_dir=config.get("data_dir", "data"))

    base_data_dir = args.data_dir if args.data_dir is not None else config.get("data_dir", "data")
    output_root = Path(base_data_dir) / config["path"]

    manager.save_splits(splits, output_root)
    print(f"Saved splits {config['splits']} to {output_root}")


if __name__ == "__main__":
    main()
