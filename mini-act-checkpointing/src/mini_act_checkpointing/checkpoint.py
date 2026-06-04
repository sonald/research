"""A small, teachable implementation of reentrant activation checkpointing.

The production PyTorch implementation has two major variants. This file focuses
on the older reentrant idea because it exposes the core trick in one place:

1. Run the checkpointed forward under ``torch.no_grad()`` so intermediate
   activations are not saved by autograd.
2. Save only the tensor inputs plus enough Python structure to call the function
   again.
3. During backward, detach the saved inputs, restore RNG state, recompute the
   forward with gradients enabled, and use autograd to obtain input/parameter
   gradients.

This is intentionally not a drop-in replacement for every torch.utils.checkpoint
feature, but it is complete for real CPU/MPS training snippets and nested
``args`` / ``kwargs``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True)
class _TreeSpec:
    kind: str
    context: Any = None
    children: tuple["_TreeSpec", ...] = ()


@dataclass
class _OutputHolder:
    spec: _TreeSpec | None = None


def _is_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def _flatten_tensors(value: Any) -> tuple[_TreeSpec, list[torch.Tensor]]:
    """Split a nested Python structure into tensor leaves plus a rebuild spec."""

    tensors: list[torch.Tensor] = []

    def walk(node: Any) -> _TreeSpec:
        if _is_tensor(node):
            index = len(tensors)
            tensors.append(node)
            return _TreeSpec("tensor", index)
        if isinstance(node, tuple) and hasattr(node, "_fields"):
            return _TreeSpec(
                "namedtuple",
                (type(node), node._fields),
                tuple(walk(item) for item in node),
            )
        if isinstance(node, tuple):
            return _TreeSpec("tuple", None, tuple(walk(item) for item in node))
        if isinstance(node, list):
            return _TreeSpec("list", None, tuple(walk(item) for item in node))
        if isinstance(node, dict):
            keys = tuple(node.keys())
            return _TreeSpec("dict", keys, tuple(walk(node[key]) for key in keys))
        return _TreeSpec("const", node)

    return walk(value), tensors


def _unflatten_tensors(spec: _TreeSpec, tensors: Iterable[torch.Tensor]) -> Any:
    iterator = iter(tensors)

    def walk(node: _TreeSpec) -> Any:
        if node.kind == "tensor":
            return next(iterator)
        if node.kind == "const":
            return node.context
        if node.kind == "tuple":
            return tuple(walk(child) for child in node.children)
        if node.kind == "list":
            return [walk(child) for child in node.children]
        if node.kind == "dict":
            return {
                key: walk(child)
                for key, child in zip(node.context, node.children, strict=True)
            }
        if node.kind == "namedtuple":
            typ, _fields = node.context
            return typ(*(walk(child) for child in node.children))
        raise TypeError(f"Unknown tree spec kind: {node.kind}")

    return walk(spec)


def _differentiable(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() or tensor.is_complex()


def _device_types(tensors: Iterable[torch.Tensor]) -> set[str]:
    return {tensor.device.type for tensor in tensors if tensor.device.type != "cpu"}


def _mps_available() -> bool:
    return bool(
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.mps, "set_rng_state")
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )


def _capture_rng_state(device_types: set[str]) -> dict[str, torch.Tensor]:
    states = {"cpu": torch.get_rng_state()}
    if "mps" in device_types and _mps_available():
        states["mps"] = torch.mps.get_rng_state()
    return states


def _set_rng_state(states: dict[str, torch.Tensor]) -> None:
    if "cpu" in states:
        torch.set_rng_state(states["cpu"])
    if "mps" in states and _mps_available():
        torch.mps.set_rng_state(states["mps"])


@contextmanager
def _fork_to_forward_rng(
    forward_states: dict[str, torch.Tensor] | None, device_types: set[str]
):
    """Run a recomputation with forward RNG, then restore caller RNG."""

    if forward_states is None:
        yield
        return

    caller_states = _capture_rng_state(device_types)
    _set_rng_state(forward_states)
    try:
        yield
    finally:
        _set_rng_state(caller_states)


class _CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        run_fn: Callable[..., Any],
        preserve_rng_state: bool,
        input_spec: _TreeSpec,
        output_holder: _OutputHolder,
        *tensor_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.run_fn = run_fn
        ctx.input_spec = input_spec
        ctx.input_requires_grad = tuple(t.requires_grad for t in tensor_inputs)
        ctx.device_types = _device_types(tensor_inputs)
        ctx.forward_rng_state = (
            _capture_rng_state(ctx.device_types) if preserve_rng_state else None
        )
        ctx.save_for_backward(*tensor_inputs)
        ctx.set_materialize_grads(False)

        args, kwargs = _unflatten_tensors(input_spec, tensor_inputs)
        with torch.no_grad():
            outputs = run_fn(*args, **kwargs)

        output_spec, tensor_outputs = _flatten_tensors(outputs)
        if not tensor_outputs:
            raise RuntimeError("checkpointed function must return at least one Tensor")

        output_holder.spec = output_spec
        non_diff = [t for t in tensor_outputs if not _differentiable(t)]
        if non_diff:
            ctx.mark_non_differentiable(*non_diff)
        return tuple(tensor_outputs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None):
        saved_inputs = ctx.saved_tensors
        detached_inputs: list[torch.Tensor] = []

        for tensor, requires_grad in zip(
            saved_inputs, ctx.input_requires_grad, strict=True
        ):
            detached = tensor.detach()
            if requires_grad and _differentiable(detached):
                detached.requires_grad_(True)
            detached_inputs.append(detached)

        args, kwargs = _unflatten_tensors(ctx.input_spec, detached_inputs)

        with _fork_to_forward_rng(ctx.forward_rng_state, ctx.device_types):
            with torch.enable_grad():
                recomputed = ctx.run_fn(*args, **kwargs)

        _output_spec, recomputed_tensors = _flatten_tensors(recomputed)
        targets: list[torch.Tensor] = []
        target_grads: list[torch.Tensor] = []

        for output, grad in zip(recomputed_tensors, grad_outputs, strict=True):
            if grad is not None and output.requires_grad:
                targets.append(output)
                target_grads.append(grad)

        if targets:
            torch.autograd.backward(targets, target_grads)

        input_grads = tuple(tensor.grad for tensor in detached_inputs)
        return (None, None, None, None, *input_grads)


def checkpoint(
    function: Callable[..., Any],
    *args: Any,
    preserve_rng_state: bool = True,
    **kwargs: Any,
) -> Any:
    """Run ``function`` with activation checkpointing.

    Parameters mirror the part of ``torch.utils.checkpoint.checkpoint`` that is
    useful for this teaching implementation. Non-tensor leaves in ``args`` and
    ``kwargs`` are stored as Python constants; tensor leaves are passed through a
    custom autograd Function.
    """

    input_spec, tensor_inputs = _flatten_tensors((args, kwargs))
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in tensor_inputs):
        return function(*args, **kwargs)

    output_holder = _OutputHolder()
    flat_outputs = _CheckpointFunction.apply(
        function, preserve_rng_state, input_spec, output_holder, *tensor_inputs
    )
    if output_holder.spec is None:
        raise RuntimeError("checkpoint forward did not record an output spec")
    if isinstance(flat_outputs, torch.Tensor):
        flat_outputs = (flat_outputs,)
    return _unflatten_tensors(output_holder.spec, flat_outputs)
