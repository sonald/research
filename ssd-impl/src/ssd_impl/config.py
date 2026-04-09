from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is malformed."""


@dataclass
class ModelConfig:
    name_or_path: str
    trust_remote_code: bool = False
    dtype: str = "auto"


@dataclass
class DatasetConfig:
    name: str = "microsoft/rStar-Coder"
    subset: str = "seed_sft"
    split: str = "train"
    cache_dir: str | None = None


@dataclass
class PathsConfig:
    prompts_path: str = "artifacts/prompts/prompts.jsonl"
    synth_train_path: str = "artifacts/synth/train.jsonl"
    synth_rejected_path: str = "artifacts/synth/rejected.jsonl"
    checkpoints_dir: str = "artifacts/checkpoints"
    merged_model_dir: str = "artifacts/merged"
    sample_output_dir: str = "artifacts/samples"


@dataclass
class PromptingConfig:
    system_prompt: str = "请给出 Python 解法，并把最终答案放进单个 markdown code block"


@dataclass
class DecodeConfig:
    backend: str = "auto"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    max_model_len: int = 32768
    n: int = 1
    seed: int = 1234


@dataclass
class SynthesisConfig:
    batch_size: int = 1
    max_records: int | None = None
    decode: DecodeConfig = field(default_factory=DecodeConfig)


@dataclass
class FilteringConfig:
    reject_empty_completion: bool = True
    reject_empty_code_block: bool = True
    reject_single_line_stub: bool = True
    single_line_stub_max_chars: int = 120


@dataclass
class AdapterConfig:
    type: str = "lora"
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 100
    warmup_steps: int = 0
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    lr_scheduler_type: str = "cosine"
    max_seq_len: int = 8192
    save_every_steps: int = 250
    log_every_steps: int = 10
    num_workers: int = 0
    seed: int = 1234
    resume_from: str | None = None
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    adapter: AdapterConfig = field(default_factory=AdapterConfig)


@dataclass
class SampleConfig:
    batch_size: int = 1
    max_records: int | None = None
    preset: str = "post_ssd"
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class SsdConfig:
    project_root: Path
    config_path: Path
    model: ModelConfig
    dataset: DatasetConfig
    paths: PathsConfig
    prompting: PromptingConfig
    synthesis: SynthesisConfig
    filtering: FilteringConfig
    training: TrainingConfig
    sample: SampleConfig
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["config_path"] = str(self.config_path)
        return payload

    def resolve_path(self, path_value: str | Path) -> Path:
        target = Path(path_value)
        if target.is_absolute():
            return target
        return (self.project_root / target).resolve()

    def selected_sample_decode(self, preset: str | None = None) -> DecodeConfig:
        active = preset or self.sample.preset
        if active not in self.sample.presets:
            raise ConfigError(f"Unknown sample preset: {active}")
        return _decode_from_dict(self.sample.presets[active])


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = current.get(part)
        if node is None:
            node = {}
            current[part] = node
        if not isinstance(node, dict):
            raise ConfigError(f"Cannot override non-mapping key: {dotted_key}")
        current = node
    current[parts[-1]] = value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return loaded


def load_raw_config(path: str | Path, overrides: list[str] | None = None) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    data = _load_yaml(config_path)

    extends = data.pop("extends", None)
    if extends:
        parent_path = (config_path.parent / extends).resolve()
        parent_data, _ = load_raw_config(parent_path)
        data = _merge_dicts(parent_data, data)

    for override in overrides or []:
        if "=" not in override:
            raise ConfigError(f"Override must use key=value form: {override}")
        key, raw_value = override.split("=", 1)
        _set_dotted(data, key.strip(), yaml.safe_load(raw_value))

    return data, config_path


def _decode_from_dict(data: dict[str, Any]) -> DecodeConfig:
    return DecodeConfig(
        backend=str(data.get("backend", "auto")),
        temperature=float(data.get("temperature", 1.0)),
        top_k=int(data.get("top_k", 0)),
        top_p=float(data.get("top_p", 1.0)),
        max_new_tokens=int(data.get("max_new_tokens", 2048)),
        max_model_len=int(data.get("max_model_len", 32768)),
        n=int(data.get("n", 1)),
        seed=int(data.get("seed", 1234)),
    )


def load_config(path: str | Path, overrides: list[str] | None = None) -> SsdConfig:
    data, config_path = load_raw_config(path, overrides=overrides)
    project_root_value = data.get("project_root", ".")
    project_root = (config_path.parent / project_root_value).resolve()

    model_data = data.get("model", {})
    if "name_or_path" not in model_data:
        raise ConfigError("model.name_or_path is required")

    sample_data = data.get("sample", {})
    sample_presets = sample_data.get("presets", {})
    if not isinstance(sample_presets, dict) or not sample_presets:
        raise ConfigError("sample.presets must be a non-empty mapping")

    return SsdConfig(
        project_root=project_root,
        config_path=config_path,
        model=ModelConfig(
            name_or_path=str(model_data["name_or_path"]),
            trust_remote_code=bool(model_data.get("trust_remote_code", False)),
            dtype=str(model_data.get("dtype", "auto")),
        ),
        dataset=DatasetConfig(**data.get("dataset", {})),
        paths=PathsConfig(**data.get("paths", {})),
        prompting=PromptingConfig(**data.get("prompting", {})),
        synthesis=SynthesisConfig(
            batch_size=int(data.get("synthesis", {}).get("batch_size", 1)),
            max_records=data.get("synthesis", {}).get("max_records"),
            decode=_decode_from_dict(data.get("synthesis", {}).get("decode", {})),
        ),
        filtering=FilteringConfig(**data.get("filtering", {})),
        training=TrainingConfig(
            **{
                **{k: v for k, v in data.get("training", {}).items() if k != "adapter"},
                "adapter": AdapterConfig(**data.get("training", {}).get("adapter", {})),
            }
        ),
        sample=SampleConfig(
            batch_size=int(sample_data.get("batch_size", 1)),
            max_records=sample_data.get("max_records"),
            preset=str(sample_data.get("preset", "post_ssd")),
            presets=sample_presets,
        ),
        raw=data,
    )


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values with dotted key=value syntax",
    )


def load_config_from_args(args: argparse.Namespace) -> SsdConfig:
    return load_config(args.config, overrides=list(args.overrides))

