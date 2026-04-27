from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import message_bus_group_chat


class GroupChatTests(unittest.TestCase):
    def test_group_chat_uses_shared_topic(self) -> None:
        result = message_bus_group_chat.run_demo()
        self.assertEqual(result["group_topic"], "group_chat")
        self.assertEqual(len(result["transcript"]), 4)
        self.assertIn("planner:", result["transcript"][0])
        self.assertIn("Approved.", result["transcript"][-1])


if __name__ == "__main__":
    unittest.main()
