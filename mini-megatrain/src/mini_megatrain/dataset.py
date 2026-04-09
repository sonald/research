"""Synthetic datasets for system validation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config


class RandomTokenDataset(Dataset):
    """Deterministic random tokens.

    We intentionally default to synthetic data because the point of this
    teaching project is to validate the training *system* rather than language
    quality. That makes the project fully offline and easy to reproduce.
    """

    def __init__(self, vocab_size: int, seq_len: int, num_sequences: int, base_seed: int = 0):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.base_seed + index)
        tokens = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,),
            generator=generator,
            dtype=torch.long,
        )
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
        }


def build_dataloader(config: Config) -> DataLoader:
    """Build the dataloader used by the CLI entrypoint."""
    num_sequences = max(config.training.total_steps, config.data.num_batches) * config.training.batch_size
    dataset = RandomTokenDataset(
        vocab_size=config.model.vocab_size,
        seq_len=config.training.seq_len,
        num_sequences=num_sequences,
        base_seed=config.data.base_seed,
    )
    return DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
