"""Configuration helpers for the teaching implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml


def parse_dtype(name: str) -> torch.dtype:
    """Map a human-readable dtype name to a PyTorch dtype."""
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported dtype '{name}'. Valid options: {valid}") from exc


@dataclass
class ModelConfig:
    """Architecture knobs for the custom dense decoder-only model."""

    vocab_size: int = 32768
    hidden_size: int = 1536
    num_layers: int = 24
    num_heads: int = 12
    mlp_hidden_size: int = 6144
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass
class DataConfig:
    """Data configuration.

    The teaching version intentionally defaults to synthetic data so the system
    can be validated without any external downloads.
    """

    dataset_kind: str = "random"
    num_batches: int = 64
    base_seed: int = 1234

    def __post_init__(self) -> None:
        if self.dataset_kind != "random":
            raise ValueError("The teaching version currently only supports dataset_kind='random'")


@dataclass
class TrainingConfig:
    """Runtime and optimization settings."""

    batch_size: int = 1
    seq_len: int = 512
    total_steps: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    checkpoint_group_size: int = 2
    compute_dtype: str = "bfloat16"
    master_dtype: str = "float32"
    device: str = "auto"
    log_interval: int = 1
    seed: int = 42
    save_group_inputs_on_cpu: bool = False
    pin_group_inputs: bool = False
    use_double_buffer: bool = True

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if self.total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if self.checkpoint_group_size < 1:
            raise ValueError("checkpoint_group_size must be >= 1")

    def resolve_device(self) -> torch.device:
        """Resolve `auto` into a concrete device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def resolve_compute_dtype(self, device: torch.device) -> torch.dtype:
        """Pick the runtime compute dtype.

        CPU execution stays in FP32 because several operations in this teaching
        implementation are meant to be used for correctness checks on machines
        that may not support BF16/FP16 efficiently.
        """

        requested = parse_dtype(self.compute_dtype)
        if device.type == "cpu":
            return torch.float32
        return requested

    def resolve_master_dtype(self) -> torch.dtype:
        """Resolve the CPU master dtype.

        We keep the master weights in FP32 to match the "CPU authoritative
        store" idea from the paper.
        """

        resolved = parse_dtype(self.master_dtype)
        if resolved != torch.float32:
            raise ValueError("master_dtype must stay float32 in the teaching implementation")
        return resolved


@dataclass
class Config:
    """Top-level configuration object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def estimate_parameter_count(model: ModelConfig) -> int:
    """Estimate the total parameter count of the custom architecture.

    This formula mirrors the actual modules in `model.py`:

    - input embedding: token + position embeddings
    - each block: q/k/v/o + gate/up/down + two RMSNorm scales
    - output: final RMSNorm + lm_head
    """

    per_block = (
        4 * model.hidden_size * model.hidden_size
        + 3 * model.hidden_size * model.mlp_hidden_size
        + 2 * model.hidden_size
    )
    embedding = model.vocab_size * model.hidden_size + model.max_seq_len * model.hidden_size
    final_norm = model.hidden_size
    lm_head = 0 if model.tie_word_embeddings else model.hidden_size * model.vocab_size
    return embedding + model.num_layers * per_block + final_norm + lm_head


def _merge_dataclass(dataclass_type: type[Any], raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    return dataclass_type(**raw)


def load_config(path: str | Path) -> Config:
    """Load YAML into the strongly-typed config tree."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return Config(
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        data=_merge_dataclass(DataConfig, raw.get("data")),
        training=_merge_dataclass(TrainingConfig, raw.get("training")),
    )
