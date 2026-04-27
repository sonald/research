from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import shared_state_langgraph_api


class LangGraphStateGraphTests(unittest.TestCase):
    def test_graph_accumulates_shared_messages(self) -> None:
        result = shared_state_langgraph_api.run_demo()
        final_state = result["final_state"]
        self.assertIn("answer", final_state)
        self.assertEqual(len(final_state["messages"]), 4)
        self.assertGreaterEqual(len(result["trace"]), 3)


if __name__ == "__main__":
    unittest.main()
