from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import orchestrator_subagent


class OrchestratorSubagentTests(unittest.TestCase):
    def test_orchestrator_collects_ephemeral_results(self) -> None:
        result = orchestrator_subagent.run_demo()
        context_ids = [item["context_id"] for item in result["subagent_results"]]
        self.assertEqual(len(context_ids), 3)
        self.assertEqual(len(set(context_ids)), 3)
        self.assertIn("billing", result["summary"])


if __name__ == "__main__":
    unittest.main()
