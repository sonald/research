from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import message_bus


class MessageBusTests(unittest.TestCase):
    def test_message_bus_routes_by_topic(self) -> None:
        result = message_bus.run_demo()
        trace = "\n".join(result["trace"])
        self.assertEqual(result["deliveries_on_entry_topic"], 2)
        self.assertIn("alerts.network", trace)
        self.assertIn("fire-and-forget", result["response_style"])


if __name__ == "__main__":
    unittest.main()
