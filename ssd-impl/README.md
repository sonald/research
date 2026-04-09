# ssd-impl

`ssd-impl` is a portable, single-machine reproduction of the training pipeline from
[Embarrassingly Simple Self-Distillation Improves Code Generation](https://arxiv.org/abs/2604.01193).

It keeps the paper's core loop intact:

1. Prepare prompts from `microsoft/rStar-Coder`
2. Sample raw self-distillation completions from a frozen model
3. Apply only minimal degeneracy filtering
4. Fine-tune with supervised cross-entropy on the raw outputs
5. Merge the adapter and sample with post-SSD decoding settings

The default config targets `Qwen/Qwen3-4B-Instruct-2507` with a LoRA-based
training setup that is easier to run on a single machine than the paper's
large-scale training stack.

## Quick Start

```bash
uv sync
uv run ssd-prepare-prompts --config configs/qwen3_4b_instruct_portable.yaml
uv run ssd-synthesize --config configs/qwen3_4b_instruct_portable.yaml
uv run ssd-train --config configs/qwen3_4b_instruct_portable.yaml
uv run ssd-merge-adapter --config configs/qwen3_4b_instruct_portable.yaml
uv run ssd-sample --config configs/qwen3_4b_instruct_portable.yaml --preset post_ssd
```

## Notes

- `vLLM` is optional. If it is importable and CUDA is available, synthesis and
  sampling will prefer it automatically. Otherwise the code falls back to
  `transformers.generate`.
- The training loop intentionally does not use verifiers, testcase filters, or
  reward shaping, matching the paper's raw-output SSD setup.
- The `paperish` config exposes a longer context and paper-leaning optimizer
  settings, but still uses the same portable code path.

