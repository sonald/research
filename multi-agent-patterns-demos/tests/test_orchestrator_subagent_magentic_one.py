from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import orchestrator_subagent_magentic_one


class MagenticOneTests(unittest.TestCase):
    def test_orchestrator_replans_and_finishes(self) -> None:
        result = orchestrator_subagent_magentic_one.run_demo()
        self.assertEqual(result["progress_ledger"]["status"], "done")
        self.assertGreaterEqual(result["progress_ledger"]["stalled_steps"], 1)
        self.assertGreaterEqual(result["outer_loops"], 2)


if __name__ == "__main__":
    unittest.main()
