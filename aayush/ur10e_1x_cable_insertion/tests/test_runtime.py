"""Host-side smoke tests for the cable-insertion behaviour tree JSON."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TANISH_DIR = REPO_ROOT / "tanish"
AAYUSH_DIR = REPO_ROOT / "aayush"
for path in (str(TANISH_DIR), str(AAYUSH_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from behaviour_tree_insertion import BehaviourTreeRuntime, load_task_intelligence


class CableInsertionTreeTests(unittest.TestCase):
    def test_task_intelligence_renders_grasp_lift_tree(self) -> None:
        json_path = Path(__file__).resolve().parents[1] / "task_intelligence.json"
        payload = load_task_intelligence(json_path)
        tree = BehaviourTreeRuntime(payload, {}, logger=lambda _: None)
        rendered = tree.render_tree()
        self.assertIn("Grasp and lift ethernet cable head", rendered)
        self.assertIn("Detect E_part006_44", rendered)
        self.assertIn("Grasp and lift E_part006_44", rendered)


if __name__ == "__main__":
    unittest.main()
