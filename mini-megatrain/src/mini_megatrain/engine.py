"""Streaming training engine.

This file contains the most important ideas in the teaching implementation:

1. The CPU model is the only authoritative copy of the trainable weights.
2. GPU modules are throwaway *working copies* built on demand.
3. Each block group saves only its input, not the full activation stack.
4. Backward works by recomputing the group and then calling `autograd.grad`.

This is not the full MegaTrain runtime, but it preserves the same mental model.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .model import MiniTransformerLM


def _chunk_blocks(blocks: nn.ModuleList, group_size: int) -> list[list[nn.Module]]:
    return [list(blocks[index:index + group_size]) for index in range(0, len(blocks), group_size)]


def _materialize_module(cpu_module: nn.Module, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """Create a throwaway working copy on the target device.

    The production MegaTrain runtime avoids Python `deepcopy()` by using
    stateless templates and flat buffers. We deliberately keep the teaching
    version simpler so the parameter lifecycle is easy to inspect.
    """

    gpu_module = copy.deepcopy(cpu_module)
    gpu_module.train(cpu_module.training)
    return gpu_module.to(device=device, dtype=dtype)


def _stash_tensor_for_backward(
    tensor: torch.Tensor,
    save_on_cpu: bool,
    pin_memory: bool,
) -> torch.Tensor:
    """Save a block input for backward recomputation."""

    saved = tensor.detach()
    if save_on_cpu:
        saved = saved.to("cpu")
        if pin_memory and torch.cuda.is_available():
            saved = saved.pin_memory()
    else:
        saved = saved.clone()
    return saved


def _accumulate_parameter_grad(cpu_param: nn.Parameter, grad: torch.Tensor | None) -> None:
    """Add a gradient tensor into the CPU master parameter."""

    if grad is None:
        return
    grad_cpu = grad.detach().to(device="cpu", dtype=torch.float32)
    if cpu_param.grad is None:
        cpu_param.grad = grad_cpu.clone()
    else:
        cpu_param.grad.add_(grad_cpu)


def _accumulate_module_grads(cpu_module: nn.Module, working_module: nn.Module) -> None:
    """Copy gradients from the transient working copy back to the CPU master."""

    for cpu_param, working_param in zip(cpu_module.parameters(), working_module.parameters()):
        _accumulate_parameter_grad(cpu_param, working_param.grad)


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token prediction loss."""

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def _compute_peak_gpu_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)


@dataclass
class StepMetrics:
    loss: float
    grad_norm: float
    tokens: int
    step_time_sec: float
    peak_gpu_gb: float

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / max(self.step_time_sec, 1e-9)


class BlockRuntime:
    """Owns the block groups and knows how to run them on demand."""

    def __init__(
        self,
        block_groups: list[list[nn.Module]],
        device: torch.device,
        compute_dtype: torch.dtype,
        save_inputs_on_cpu: bool,
        pin_inputs: bool,
        use_double_buffer: bool,
    ):
        self.block_groups = block_groups
        self.device = device
        self.compute_dtype = compute_dtype
        self.save_inputs_on_cpu = save_inputs_on_cpu
        self.pin_inputs = pin_inputs
        self.use_double_buffer = use_double_buffer and device.type == "cuda"

    def _forward_group_no_grad(self, group_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run a block group without building the full autograd graph.

        We only need the *output* for downstream layers. The backward pass will
        rebuild this group from its saved input.
        """

        cpu_group = self.block_groups[group_index]
        return self._run_sequential_layers(cpu_group, hidden_states, track_autograd=False, keep_layers=False)

    def _backward_group(
        self,
        group_index: int,
        saved_input: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute a block group and extract gradients with `autograd.grad`."""

        cpu_group = self.block_groups[group_index]
        hidden_input = saved_input.to(device=self.device, dtype=self.compute_dtype)
        hidden_input.requires_grad_(True)

        # During recompute we *do* keep the working copies alive, because
        # `autograd.grad` needs their parameters to stay around until the local
        # backward for this group is complete.
        group_output, working_layers = self._run_sequential_layers(
            cpu_group,
            hidden_input,
            track_autograd=True,
            keep_layers=True,
        )
        params: list[nn.Parameter] = []
        for layer in working_layers:
            params.extend(list(layer.parameters()))

        grads = torch.autograd.grad(group_output, [hidden_input] + params, grad_output)
        grad_input = grads[0]

        grad_offset = 1
        for cpu_layer, working_layer in zip(cpu_group, working_layers):
            for cpu_param, _ in zip(cpu_layer.parameters(), working_layer.parameters()):
                _accumulate_parameter_grad(cpu_param, grads[grad_offset])
                grad_offset += 1

        return grad_input

    def _run_sequential_layers(
        self,
        cpu_layers: Sequence[nn.Module],
        hidden_states: torch.Tensor,
        *,
        track_autograd: bool,
        keep_layers: bool,
    ) -> tuple[torch.Tensor, list[nn.Module]] | torch.Tensor:
        """Execute a sequence of layers, optionally with a teaching double buffer."""

        if not cpu_layers:
            if keep_layers:
                return hidden_states, []
            return hidden_states

        default_stream = None
        prefetch_stream = None
        if self.use_double_buffer and len(cpu_layers) > 1:
            default_stream = torch.cuda.current_stream(device=self.device)
            prefetch_stream = torch.cuda.Stream(device=self.device)

        working_layers: list[nn.Module] = []

        def load_layer(layer: nn.Module, stream: torch.cuda.Stream | None = None) -> nn.Module:
            if stream is None:
                return _materialize_module(layer, self.device, self.compute_dtype)
            with torch.cuda.stream(stream):
                return _materialize_module(layer, self.device, self.compute_dtype)

        current_layer = load_layer(cpu_layers[0])

        def maybe_append(layer: nn.Module) -> None:
            if keep_layers:
                working_layers.append(layer)

        if track_autograd:
            context = torch.enable_grad()
        else:
            context = torch.no_grad()

        with context:
            hidden = hidden_states
            for index, _ in enumerate(cpu_layers):
                next_layer = None
                if index + 1 < len(cpu_layers) and prefetch_stream is not None:
                    # This is the teaching approximation of the paper's
                    # double-buffered prefetch logic: while we are about to
                    # compute the current layer, we ask another CUDA stream to
                    # start constructing the next working copy.
                    next_layer = load_layer(cpu_layers[index + 1], prefetch_stream)

                hidden = current_layer(hidden)
                maybe_append(current_layer)

                if index + 1 < len(cpu_layers) and next_layer is None:
                    # CPU / no-prefetch fallback: still run the same lifecycle,
                    # just without the overlap.
                    next_layer = load_layer(cpu_layers[index + 1])

                if next_layer is not None:
                    if prefetch_stream is not None:
                        default_stream.wait_stream(prefetch_stream)
                    current_layer = next_layer

        if keep_layers:
            return hidden, working_layers
        return hidden


class StreamingBlockFunction(torch.autograd.Function):
    """Custom autograd node for a whole block group."""

    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, runtime: BlockRuntime, group_index: int) -> torch.Tensor:
        ctx.runtime = runtime
        ctx.group_index = group_index
        saved_input = _stash_tensor_for_backward(
            hidden_states,
            save_on_cpu=runtime.save_inputs_on_cpu,
            pin_memory=runtime.pin_inputs,
        )
        ctx.save_for_backward(saved_input)
        return runtime._forward_group_no_grad(group_index, hidden_states)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (saved_input,) = ctx.saved_tensors
        grad_input = ctx.runtime._backward_group(ctx.group_index, saved_input, grad_output)
        return grad_input, None, None


class StreamingTrainer:
    """End-to-end trainer built around a CPU-resident master model."""

    def __init__(self, config: Config, model: MiniTransformerLM | None = None):
        self.config = config
        self.device = config.training.resolve_device()
        self.compute_dtype = config.training.resolve_compute_dtype(self.device)
        self.master_dtype = config.training.resolve_master_dtype()

        if config.model.tie_word_embeddings:
            raise NotImplementedError(
                "The teaching runtime keeps embedding and lm_head as separate working copies; "
                "set tie_word_embeddings=false."
            )

        self.model = model if model is not None else MiniTransformerLM(config.model)
        self.model.to(device="cpu", dtype=self.master_dtype)
        self.model.train()

        self.block_groups = _chunk_blocks(self.model.blocks, config.training.checkpoint_group_size)
        self.runtime = BlockRuntime(
            block_groups=self.block_groups,
            device=self.device,
            compute_dtype=self.compute_dtype,
            save_inputs_on_cpu=config.training.save_group_inputs_on_cpu,
            pin_inputs=config.training.pin_group_inputs,
            use_double_buffer=config.training.use_double_buffer,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> StepMetrics:
        """Run one training step."""

        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        start_time = time.perf_counter()

        embedding = _materialize_module(self.model.embeddings, self.device, self.compute_dtype)
        final_norm = _materialize_module(self.model.final_norm, self.device, self.compute_dtype)
        lm_head = _materialize_module(self.model.lm_head, self.device, self.compute_dtype)

        hidden = embedding(input_ids)
        for group_index in range(len(self.block_groups)):
            hidden = StreamingBlockFunction.apply(hidden, self.runtime, group_index)
        hidden = final_norm(hidden)
        logits = lm_head(hidden).float()
        loss = _causal_lm_loss(logits, labels)
        loss.backward()

        _accumulate_module_grads(self.model.embeddings, embedding)
        _accumulate_module_grads(self.model.final_norm, final_norm)
        if not self.config.model.tie_word_embeddings:
            _accumulate_module_grads(self.model.lm_head, lm_head)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
        self.optimizer.step()

        step_time = time.perf_counter() - start_time
        return StepMetrics(
            loss=loss.item(),
            grad_norm=float(grad_norm),
            tokens=int(input_ids.numel()),
            step_time_sec=step_time,
            peak_gpu_gb=_compute_peak_gpu_gb(self.device),
        )
