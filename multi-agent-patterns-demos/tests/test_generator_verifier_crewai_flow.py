from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import generator_verifier_crewai_flow


class CrewAISelfEvaluationLoopTests(unittest.TestCase):
    def test_flow_retries_until_valid(self) -> None:
        result = generator_verifier_crewai_flow.run_demo()
        self.assertTrue(result["valid"])
        self.assertGreater(result["attempts"], 1)
        self.assertLessEqual(result["attempts"], 3)


if __name__ == "__main__":
    unittest.main()
