from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ssd_impl.config import add_config_arguments, load_config_from_args, SsdConfig
from ssd_impl.synthesize import (
    DecodeConfig,
    TransformersGenerationBackend,
    VllmGenerationBackend,
    generate_records,
    is_vllm_available,
    load_tokenizer,
)


def select_sampling_backend(
    config: SsdConfig,
    decode: DecodeConfig,
    model_name_or_path: str,
    *,
    tokenizer: Any | None = None,
) -> Any:
    requested = decode.backend
    if requested == "auto":
        requested = "vllm" if torch.cuda.is_available() and is_vllm_available() else "transformers"

    if requested == "vllm":
        return VllmGenerationBackend(
            model_name_or_path,
            trust_remote_code=config.model.trust_remote_code,
            max_model_len=decode.max_model_len,
        )
    if requested == "transformers":
        return TransformersGenerationBackend(
            model_name_or_path,
            trust_remote_code=config.model.trust_remote_code,
            tokenizer=tokenizer,
        )
    raise ValueError(f"Unknown sampling backend: {requested}")


def sample_completions(
    config: SsdConfig,
    *,
    model_name_or_path: str | None = None,
    preset: str | None = None,
    tokenizer: Any | None = None,
) -> Path:
    active_preset = preset or config.sample.preset
    decode = config.selected_sample_decode(active_preset)
    tokenizer = tokenizer or load_tokenizer(
        model_name_or_path or config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )

    output_dir = config.resolve_path(config.paths.sample_output_dir)
    output_path = output_dir / f"{active_preset}.jsonl"
    backend = select_sampling_backend(
        config,
        decode,
        model_name_or_path=model_name_or_path or str(config.resolve_path(config.paths.merged_model_dir)),
        tokenizer=tokenizer,
    )
    generate_records(
        config,
        decode=decode,
        output_path=output_path,
        rejected_path=None,
        backend=backend,
        tokenizer=tokenizer,
        batch_size=config.sample.batch_size,
        max_records=config.sample.max_records,
        apply_filter=False,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample completions from the frozen or post-SSD model.")
    add_config_arguments(parser)
    parser.add_argument("--model", type=str, default=None, help="Model or local merged checkpoint to sample from")
    parser.add_argument("--preset", type=str, default=None, help="Sampling preset to use")
    args = parser.parse_args()

    config = load_config_from_args(args)
    model_name_or_path = args.model or (
        str(config.resolve_path(config.paths.merged_model_dir))
        if config.resolve_path(config.paths.merged_model_dir).exists()
        else config.model.name_or_path
    )
    output_path = sample_completions(config, model_name_or_path=model_name_or_path, preset=args.preset)
    print(f"Samples written to {output_path}")
