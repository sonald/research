import time

import pyautogui


def hover_probe(x: float, y: float, duration: float) -> None:
    """Move to screen coordinate (x, y) and hover there for duration milliseconds."""
    print(f"hover_probe({x}, {y}, {duration})")
    pyautogui.moveTo(x, y)
    time.sleep(duration / 1000)


def move_to(x: float, y: float) -> None:
    """Move the pointer to screen coordinate (x, y)."""
    print(f"move_to({x}, {y})")
    pyautogui.moveTo(x, y)


def click(x: float, y: float) -> None:
    """Click screen coordinate (x, y)."""
    print(f"click({x}, {y})")
    pyautogui.click(x=x, y=y)


# TOOLS = [hover_probe, move_to, click]
TOOLS = [move_to, click]
