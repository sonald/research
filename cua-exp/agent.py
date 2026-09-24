#!/usr/bin/env python3
"""Run one LiteLLM agent turn.

An optional ``--tools tools.py`` module must export ``TOOLS``, a list of
ordinary, type-annotated Python functions. Their names and docstrings become
the model-visible tool definitions.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import inspect
import json
import mimetypes
import os
from pathlib import Path
from string import Template
from typing import Any, get_args, get_origin, get_type_hints

from rich.console import Console


def json_schema(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin is list:
        return {"type": "array", "items": json_schema(get_args(annotation)[0])}
    if origin is dict:
        return {"type": "object"}
    return {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }.get(annotation, {"type": "string"})


def load_tools(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {}
    spec = importlib.util.spec_from_file_location("cua_external_tools", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法导入工具模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    functions = getattr(module, "TOOLS", None)
    if not isinstance(functions, (list, tuple)) or not all(
        callable(f) for f in functions
    ):
        raise ValueError(f"{path} 必须导出由函数组成的 TOOLS 列表")

    schemas = []
    handlers = {}
    for function in functions:
        signature = inspect.signature(function)
        hints = get_type_hints(function)
        properties = {
            name: json_schema(hints.get(name, str)) for name in signature.parameters
        }
        required = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
        ]
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": function.__name__,
                    "description": inspect.getdoc(function) or function.__name__,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
        handlers[function.__name__] = function
    return schemas, handlers


def render(path: Path) -> str:
    return Template(path.read_text(encoding="utf-8")).substitute(os.environ)


def image_part(value: str) -> dict[str, Any]:
    if value.startswith(("http://", "https://", "data:")):
        url = value
    else:
        path = Path(value)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    return {"type": "image_url", "image_url": {"url": url}}


def run_agent(
    completion: Any,
    model: str,
    messages: list[dict[str, Any]],
    schemas: list[dict[str, Any]],
    handlers: dict[str, Any],
    chunk_builder: Any,
    max_steps: int = 2,
    pause_after_tool: bool = True,
    **model_args: Any,
) -> str:
    console = Console(highlight=False, markup=False, soft_wrap=True)
    for _ in range(max_steps):
        chunks = []
        in_thinking = False
        in_content = False
        for chunk in completion(
            model=model,
            messages=messages,
            tools=schemas or None,
            stream=True,
            **model_args,
        ):
            chunks.append(chunk)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = (
                delta.get("content")
                if isinstance(delta, dict)
                else getattr(delta, "content", None)
            )
            reasoning = (
                delta.get("reasoning_content")
                if isinstance(delta, dict)
                else getattr(delta, "reasoning_content", None)
            )
            blocks = (
                delta.get("thinking_blocks")
                if isinstance(delta, dict)
                else getattr(delta, "thinking_blocks", None)
            )
            if not reasoning and blocks:
                reasoning = "".join(
                    block.get("thinking", "")
                    if isinstance(block, dict)
                    else getattr(block, "thinking", "")
                    for block in blocks
                )
            if reasoning:
                if not in_thinking:
                    console.print("thinking: ", style="bold yellow", end="")
                    in_thinking = True
                console.print(reasoning, style="yellow", end="")
            if content:
                if in_thinking:
                    console.print()
                    in_thinking = False
                if not in_content:
                    console.print("content: ", style="bold cyan", end="")
                    in_content = True
                console.print(content, style="cyan", end="")
        if in_thinking:
            console.print()
        response = chunk_builder(chunks, messages=messages)
        if response is None:
            raise RuntimeError("模型没有返回可用响应")
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        messages.append(message.model_dump(exclude_none=True))
        if not tool_calls:
            return message.content or ""
        for call in tool_calls:
            if call.function.name not in handlers:
                raise ValueError(f"模型调用了未知工具: {call.function.name}")
            arguments = json.loads(call.function.arguments or "{}")
            result = handlers[call.function.name](**arguments)
            content = (
                result
                if isinstance(result, str)
                else json.dumps(result, ensure_ascii=False, default=str)
            )
            console.print("tool: ", style="bold magenta", end="")
            console.print(
                f"{call.function.name}({json.dumps(arguments, ensure_ascii=False)})"
                f" -> {content}",
                style="magenta",
            )
            if pause_after_tool:
                console.print(
                    "paused: press Enter to continue, Ctrl+C to exit: ",
                    style="dim magenta",
                    end="",
                )
                input()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": content,
                }
            )
    raise RuntimeError(f"达到最大工具调用轮数: {max_steps}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--system", type=Path, default=Path("system.md"))
    parser.add_argument("--task", type=Path, default=Path("task.md"))
    parser.add_argument(
        "--tools", type=Path, help="导出 TOOLS=[函数, ...] 的 Python 文件"
    )
    parser.add_argument(
        "--image",
        "-i",
        action="append",
        default=[],
        help="本地图片或 HTTP/data URL，可重复",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--no-pause-after-tool",
        action="store_true",
        help="工具执行后不等待 Enter，直接继续",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from dotenv import load_dotenv
    from litellm import completion, stream_chunk_builder

    load_dotenv(args.env_file)
    model = os.getenv("LITELLM_MODEL") or os.environ["MODEL"]
    schemas, handlers = load_tools(args.tools)
    task = render(args.task)
    user_content: str | list[dict[str, Any]] = task
    if args.image:
        user_content = [{"type": "text", "text": task}, *map(image_part, args.image)]
    messages: list[dict[str, Any]] = []
    if args.system.exists():
        messages.append({"role": "system", "content": render(args.system)})
    messages.append({"role": "user", "content": user_content})
    model_args = {
        key: value
        for key, value in {
            "api_base": os.getenv("LITELLM_API_BASE") or os.getenv("API_BASE"),
            "api_key": os.getenv("LITELLM_API_KEY") or os.getenv("API_KEY"),
        }.items()
        if value
    }
    run_agent(
        completion,
        model,
        messages,
        schemas,
        handlers,
        stream_chunk_builder,
        args.max_steps,
        pause_after_tool=not args.no_pause_after_tool,
        **model_args,
    )
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
