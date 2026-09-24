import os
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent import image_part, load_tools, render, run_agent
from cua_tools import click, hover_probe, move_to


class Message(SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return {key: value for key, value in vars(self).items() if value is not None}


def test_single_tool_round_trip():
    with tempfile.TemporaryDirectory() as directory:
        tool_file = Path(directory) / "tools.py"
        tool_file.write_text(
            'def add(a: int, b: int) -> int:\n'
            '    """Add two integers."""\n'
            '    return a + b\n\n'
            'TOOLS = [add]\n',
            encoding="utf-8",
        )
        schemas, handlers = load_tools(tool_file)

    responses = [
        Message(content=None, tool_calls=[SimpleNamespace(
            id="1", function=SimpleNamespace(name="add", arguments='{"a":2,"b":3}')
        )]),
        Message(content="5", tool_calls=[]),
    ]

    def completion(**kwargs):
        assert kwargs["stream"] is True
        if len(responses) == 1:
            assert kwargs["messages"][-1] == {
                "role": "tool", "tool_call_id": "1", "content": "5"
            }
        deltas = (
            [{"reasoning_content": "calculate"}]
            if len(responses) == 2 else
            [{"thinking_blocks": [{"type": "thinking", "thinking": "verify"}]},
             {"content": responses[0].content}]
        )
        return [SimpleNamespace(choices=[SimpleNamespace(delta=delta)]) for delta in deltas]

    def chunk_builder(chunks, messages):
        return SimpleNamespace(choices=[SimpleNamespace(message=responses.pop(0))])

    assert schemas[0]["function"]["parameters"]["required"] == ["a", "b"]
    output = io.StringIO()
    with redirect_stdout(output), patch("builtins.input", return_value="") as paused:
        result = run_agent(
            completion, "fake/model", [{"role": "user", "content": "2+3"}],
            schemas, handlers, chunk_builder
        )
    paused.assert_called_once()
    assert result == "5"
    rendered = output.getvalue()
    assert "thinking: calculate" in rendered
    assert 'tool: add({"a": 2, "b": 3}) -> 5' in rendered
    assert "content: 5" in rendered


def test_template_and_local_image():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        template = directory / "task.md"
        template.write_text("Hello ${WHO}", encoding="utf-8")
        image = directory / "tiny.png"
        image.write_bytes(b"png")
        os.environ["WHO"] = "agent"
        assert render(template) == "Hello agent"
        assert image_part(str(image))["image_url"]["url"].startswith("data:image/png;base64,")


def test_cua_tools():
    debug = io.StringIO()
    with (
        redirect_stdout(debug),
        patch("cua_tools.pyautogui.moveTo") as move,
        patch("cua_tools.pyautogui.click") as mouse_click,
        patch("cua_tools.time.sleep") as sleep,
    ):
        move_to(10, 20)
        click(30, 40)
        hover_probe(50, 60, 250)
    assert move.call_args_list == [((10, 20),), ((50, 60),)]
    mouse_click.assert_called_once_with(x=30, y=40)
    sleep.assert_called_once_with(0.25)
    assert debug.getvalue().splitlines() == [
        "move_to(10, 20)",
        "click(30, 40)",
        "hover_probe(50, 60, 250)",
    ]


if __name__ == "__main__":
    test_single_tool_round_trip()
    test_template_and_local_image()
    test_cua_tools()
    print("ok")
