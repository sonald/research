from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import generator_verifier


class GeneratorVerifierTests(unittest.TestCase):
    def test_strict_verifier_converges(self) -> None:
        result = generator_verifier.run_demo()
        self.assertTrue(result["strict_verifier"]["accepted"])
        self.assertFalse(result["capped_strict_verifier"]["accepted"])
        self.assertTrue(result["vague_verifier"]["accepted"])


if __name__ == "__main__":
    unittest.main()
