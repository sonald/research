from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import shared_state


class SharedStateTests(unittest.TestCase):
    def test_shared_state_reaches_termination_condition(self) -> None:
        result = shared_state.run_demo()
        categories = {item["category"] for item in result["findings"]}
        self.assertTrue(result["done"])
        self.assertGreater(result["version"], 0)
        self.assertTrue({"academia", "industry", "patent", "news"}.issubset(categories))


if __name__ == "__main__":
    unittest.main()
