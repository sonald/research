from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import torch
from tqdm import tqdm

from ssd_impl.config import DecodeConfig, SsdConfig, add_config_arguments, load_config_from_args
from ssd_impl.filters import evaluate_completion
from ssd_impl.io_utils import read_jsonl, write_jsonl


@dataclass
class PromptJob:
    prompt_record: dict[str, Any]
    messages: list[dict[str, str]]
    prompt_text: str


class GenerationBackend(Protocol):
    def generate(self, prompt_texts: list[str], decode: DecodeConfig) -> list[list[str]]:
        ...


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def build_messages(question: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = False) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_prompt_jobs(
    prompt_records: Iterable[dict[str, Any]],
    tokenizer: Any,
    system_prompt: str,
) -> list[PromptJob]:
    jobs: list[PromptJob] = []
    for record in prompt_records:
        messages = build_messages(record["question"], system_prompt=system_prompt)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        jobs.append(PromptJob(prompt_record=record, messages=messages, prompt_text=prompt_text))
    return jobs


class TransformersGenerationBackend:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        trust_remote_code: bool = False,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.device = device or resolve_device()
        self.tokenizer = tokenizer or load_tokenizer(model_name_or_path, trust_remote_code=trust_remote_code)
        if model is None:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
        self.model = model.to(self.device)
        self.model.eval()

    def generate(self, prompt_texts: list[str], decode: DecodeConfig) -> list[list[str]]:
        if not prompt_texts:
            return []

        torch.manual_seed(decode.seed)
        encoded = self.tokenizer(
            prompt_texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        padded_prompt_length = int(encoded["input_ids"].shape[1])
        temperature = decode.temperature if decode.temperature > 0 else 1.0

        outputs = self.model.generate(
            **encoded,
            do_sample=decode.temperature > 0,
            temperature=temperature,
            top_k=decode.top_k,
            top_p=decode.top_p,
            max_new_tokens=decode.max_new_tokens,
            num_return_sequences=decode.n,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        grouped: list[list[str]] = []
        offset = 0
        for _ in prompt_texts:
            completions: list[str] = []
            for _ in range(decode.n):
                sequence = outputs[offset]
                generated_ids = sequence[padded_prompt_length:]
                completions.append(self.tokenizer.decode(generated_ids, skip_special_tokens=True))
                offset += 1
            grouped.append(completions)
        return grouped


class VllmGenerationBackend:
    def __init__(self, model_name_or_path: str, *, trust_remote_code: bool = False, max_model_len: int = 32768) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=model_name_or_path,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
        )

    def generate(self, prompt_texts: list[str], decode: DecodeConfig) -> list[list[str]]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=decode.temperature,
            top_k=decode.top_k,
            top_p=decode.top_p,
            max_tokens=decode.max_new_tokens,
            n=decode.n,
            seed=decode.seed,
        )
        outputs = self.llm.generate(prompt_texts, sampling_params=sampling_params)
        return [[candidate.text for candidate in item.outputs] for item in outputs]


def is_vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
    except ImportError:
        return False
    return True


def select_generation_backend(
    config: SsdConfig,
    decode: DecodeConfig,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> GenerationBackend:
    requested = decode.backend
    if requested == "auto":
        if torch.cuda.is_available() and is_vllm_available():
            requested = "vllm"
        else:
            requested = "transformers"

    if requested == "vllm":
        return VllmGenerationBackend(
            config.model.name_or_path,
            trust_remote_code=config.model.trust_remote_code,
            max_model_len=decode.max_model_len,
        )
    if requested == "transformers":
        return TransformersGenerationBackend(
            config.model.name_or_path,
            trust_remote_code=config.model.trust_remote_code,
            model=model,
            tokenizer=tokenizer,
        )
    raise ValueError(f"Unknown generation backend: {requested}")


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def generate_records(
    config: SsdConfig,
    *,
    decode: DecodeConfig,
    output_path: str | Path,
    rejected_path: str | Path | None,
    backend: GenerationBackend | None = None,
    tokenizer: Any | None = None,
    batch_size: int = 1,
    prompt_records_path: str | Path | None = None,
    max_records: int | None = None,
    apply_filter: bool = True,
) -> tuple[list[dict], list[dict]]:
    prompt_records = read_jsonl(prompt_records_path or config.resolve_path(config.paths.prompts_path))
    if max_records is not None:
        prompt_records = prompt_records[:max_records]

    tokenizer = tokenizer or load_tokenizer(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    jobs = prepare_prompt_jobs(prompt_records, tokenizer=tokenizer, system_prompt=config.prompting.system_prompt)
    backend = backend or select_generation_backend(config, decode, tokenizer=tokenizer)

    accepted: list[dict] = []
    rejected: list[dict] = []
    decode_payload = asdict(decode)

    for batch_index, batch_jobs in enumerate(tqdm(list(batched(jobs, batch_size)), desc="Generating", leave=False)):
        prompt_texts = [job.prompt_text for job in batch_jobs]
        grouped_completions = backend.generate(prompt_texts, decode=decode)
        for item_index, (job, completions) in enumerate(zip(batch_jobs, grouped_completions)):
            for sample_index, completion in enumerate(completions):
                seed = decode.seed + batch_index * batch_size + item_index + sample_index
                record = {
                    **job.prompt_record,
                    "messages": job.messages,
                    "prompt_text": job.prompt_text,
                    "completion": completion,
                    "seed": seed,
                    "decode": decode_payload,
                }
                if apply_filter:
                    decision = evaluate_completion(completion, config.filtering)
                    if decision.accepted:
                        accepted.append(record)
                    else:
                        rejected.append({**record, "reject_reason": decision.reason})
                else:
                    accepted.append(record)

    write_jsonl(output_path, accepted)
    if rejected_path is not None:
        write_jsonl(rejected_path, rejected)
    return accepted, rejected


def synthesize_dataset(
    config: SsdConfig,
    *,
    backend: GenerationBackend | None = None,
    tokenizer: Any | None = None,
) -> tuple[list[dict], list[dict]]:
    return generate_records(
        config,
        decode=config.synthesis.decode,
        output_path=config.resolve_path(config.paths.synth_train_path),
        rejected_path=config.resolve_path(config.paths.synth_rejected_path),
        backend=backend,
        tokenizer=tokenizer,
        batch_size=config.synthesis.batch_size,
        max_records=config.synthesis.max_records,
        apply_filter=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SSD self-distillation samples.")
    add_config_arguments(parser)
    args = parser.parse_args()

    config = load_config_from_args(args)
    accepted, rejected = synthesize_dataset(config)
    print(
        f"Wrote {len(accepted)} accepted samples to {config.resolve_path(config.paths.synth_train_path)} "
        f"and {len(rejected)} rejected samples to {config.resolve_path(config.paths.synth_rejected_path)}"
    )
