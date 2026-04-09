from __future__ import annotations

import argparse
import math
import random
from typing import Any

import numpy as np
import torch

from ssd_impl.checkpoint import load_training_checkpoint, resolve_resume_checkpoint, save_training_checkpoint
from ssd_impl.config import SsdConfig, add_config_arguments, load_config_from_args
from ssd_impl.dataset import SsdTrainingDataset, build_train_dataloader, load_synth_records


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_tokenizer(config: SsdConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(config: SsdConfig) -> Any:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch_dtype = resolve_torch_dtype(config.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    if config.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    adapter_cfg = config.training.adapter
    lora_config = LoraConfig(
        r=adapter_cfg.r,
        lora_alpha=adapter_cfg.alpha,
        lora_dropout=adapter_cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(adapter_cfg.target_modules),
    )
    return get_peft_model(model, lora_config)


def create_scheduler(optimizer: Any, config: SsdConfig):
    from transformers import get_scheduler

    return get_scheduler(
        name=config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.max_steps,
    )


def count_trainable_parameters(model: Any) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def train(
    config: SsdConfig,
    *,
    tokenizer: Any | None = None,
    model: Any | None = None,
    accelerator: Any | None = None,
) -> dict[str, Any]:
    from accelerate import Accelerator

    set_random_seed(config.training.seed)

    tokenizer = tokenizer or load_tokenizer(config)
    model = model or build_model(config)
    pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)

    records = load_synth_records(str(config.resolve_path(config.paths.synth_train_path)))
    train_dataset = SsdTrainingDataset(records, tokenizer=tokenizer, max_seq_len=config.training.max_seq_len)
    if len(train_dataset) == 0:
        raise RuntimeError("No valid training examples after tokenization.")

    dataloader = build_train_dataloader(
        train_dataset,
        batch_size=config.training.per_device_batch_size,
        pad_token_id=pad_token_id,
        num_workers=config.training.num_workers,
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay,
    )
    scheduler = create_scheduler(optimizer, config)

    accelerator = accelerator or Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    checkpoints_root = config.resolve_path(config.paths.checkpoints_dir)
    resume_target = resolve_resume_checkpoint(
        config.resolve_path(config.training.resume_from) if config.training.resume_from else checkpoints_root
    )
    completed_steps = 0
    if resume_target is not None:
        resume_meta = load_training_checkpoint(
            resume_target,
            model=accelerator.unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu",
        )
        completed_steps = int(resume_meta["step"])
        accelerator.print(f"Resumed training from step {completed_steps} at {resume_target}")

    trainable, total = count_trainable_parameters(accelerator.unwrap_model(model))
    accelerator.print(f"Trainable parameters: {trainable}/{total}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    data_iter = iter(dataloader)
    micro_steps = completed_steps * config.training.gradient_accumulation_steps
    last_loss = math.nan

    while completed_steps < config.training.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            last_loss = float(loss.detach().item())
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1

                if completed_steps % config.training.log_every_steps == 0 or completed_steps == 1:
                    current_lr = scheduler.get_last_lr()[0]
                    accelerator.print(
                        f"step={completed_steps}/{config.training.max_steps} "
                        f"loss={last_loss:.4f} lr={current_lr:.6g}"
                    )

                if completed_steps % config.training.save_every_steps == 0:
                    save_dir = None
                    if accelerator.is_main_process:
                        save_dir = save_training_checkpoint(
                            checkpoints_root,
                            model=accelerator.unwrap_model(model),
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=completed_steps,
                            config=config.to_dict(),
                        )
                    accelerator.wait_for_everyone()
                    if save_dir is not None:
                        accelerator.print(f"Saved checkpoint to {save_dir}")
            micro_steps += 1

    final_dir = None
    if accelerator.is_main_process:
        final_dir = save_training_checkpoint(
            checkpoints_root,
            model=accelerator.unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            step=completed_steps,
            config=config.to_dict(),
        )
    accelerator.wait_for_everyone()
    if final_dir is not None:
        accelerator.print(f"Saved final checkpoint to {final_dir}")
    return {
        "steps": completed_steps,
        "final_checkpoint": str(final_dir or checkpoint_root_for_return(checkpoints_root, completed_steps)),
        "dataset_size": len(train_dataset),
        "dropped_examples": train_dataset.dropped,
        "last_loss": last_loss,
    }


def checkpoint_root_for_return(checkpoints_root, completed_steps):
    return checkpoints_root / "last" / f"step-{completed_steps:04d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a portable SSD LoRA adapter.")
    add_config_arguments(parser)
    args = parser.parse_args()

    config = load_config_from_args(args)
    result = train(config)
    print(
        f"Training finished after {result['steps']} steps with dataset_size={result['dataset_size']} "
        f"and final checkpoint {result['final_checkpoint']}"
    )
