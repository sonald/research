from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ssd_impl.checkpoint import resolve_resume_checkpoint
from ssd_impl.config import add_config_arguments, load_config_from_args, SsdConfig


def resolve_adapter_path(checkpoint: str | Path) -> Path:
    target = Path(checkpoint)
    if (target / "adapter_config.json").exists():
        return target
    if (target / "adapter").exists():
        return target / "adapter"
    raise FileNotFoundError(f"Could not locate adapter weights under {target}")


def merge_adapter(
    config: SsdConfig,
    *,
    checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_root = checkpoint or resolve_resume_checkpoint(config.resolve_path(config.paths.checkpoints_dir))
    if checkpoint_root is None:
        raise FileNotFoundError("No checkpoint found to merge.")

    adapter_path = resolve_adapter_path(checkpoint_root)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=torch.float32,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()

    save_dir = Path(output_dir or config.resolve_path(config.paths.merged_model_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge the latest SSD LoRA adapter into a full model checkpoint.")
    add_config_arguments(parser)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint step directory or adapter directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for the merged model")
    args = parser.parse_args()

    config = load_config_from_args(args)
    output_dir = merge_adapter(config, checkpoint=args.checkpoint, output_dir=args.output_dir)
    print(f"Merged adapter written to {output_dir}")

