from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import agent_teams


class AgentTeamsTests(unittest.TestCase):
    def test_workers_keep_local_context_and_detect_conflict(self) -> None:
        result = agent_teams.run_demo()
        frontend = result["worker_state"]["frontend"]
        self.assertGreaterEqual(frontend["handled_count"], 2)
        self.assertIn("landing-refresh", frontend["memory"])
        self.assertEqual(len(result["round_two"]["conflicts"]), 1)


if __name__ == "__main__":
    unittest.main()
