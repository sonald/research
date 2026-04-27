from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import agent_teams_camel_workforce


class CamelWorkforceTests(unittest.TestCase):
    def test_workforce_can_create_dynamic_worker(self) -> None:
        result = agent_teams_camel_workforce.run_demo()
        self.assertEqual(len(result["completed_tasks"]), 3)
        self.assertEqual(len(result["dynamic_workers"]), 1)
        self.assertIn("dynamic-design-worker", result["dynamic_workers"])


if __name__ == "__main__":
    unittest.main()
