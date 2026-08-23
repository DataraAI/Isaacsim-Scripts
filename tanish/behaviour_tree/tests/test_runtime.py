from __future__ import annotations

import unittest
from pathlib import Path

from behaviour_tree.isaac_adapters import controller_primitive
from behaviour_tree.runtime import (
    BehaviourTreeRuntime,
    PrimitiveContext,
    Status,
    load_task_intelligence,
    normalize_task_intelligence,
)


class CountingPrimitive:
    def __init__(self, ticks: int) -> None:
        self.remaining = ticks

    def initialise(self, context: PrimitiveContext) -> None:
        pass

    def tick(self, context: PrimitiveContext) -> Status:
        self.remaining -= 1
        return Status.SUCCESS if self.remaining <= 0 else Status.RUNNING

    def terminate(self, context: PrimitiveContext, new_status: Status) -> None:
        pass


PAYLOAD = {
    "tasks": [{
        "task_name": "Insert connector",
        "subtasks": [{
            "subtask_name": "Pick connector",
            "primitive": "grasp_object",
            "preconditions": ["connector_visible"],
            "success_conditions": ["connector_grasped"],
        }, {
            "subtask_name": "Insert connector",
            "primitive": "manipulate_object",
            "preconditions": ["connector_grasped"],
            "success_conditions": ["connector_inserted"],
        }],
    }]
}


class FakeMotionController:
    def __init__(self) -> None:
        self.remaining = 0

    def clear_queue(self) -> None:
        self.remaining = 0

    def is_done(self) -> bool:
        return self.remaining == 0

    def forward(self, _joints):
        self.remaining -= 1
        return "action"


class FakeArticulationController:
    def __init__(self) -> None:
        self.actions = []

    def apply_action(self, action) -> None:
        self.actions.append(action)


class FakeRobot:
    def get_joint_positions(self):
        return [0.0] * 9


class BehaviourTreeRuntimeTests(unittest.TestCase):
    def test_bundled_isaac_demo_json_is_valid(self) -> None:
        payload = load_task_intelligence(Path(__file__).parents[1] / "demo_task_intelligence.json")
        tree = BehaviourTreeRuntime(payload, {}, logger=lambda _: None)
        rendered = tree.render_tree()
        self.assertIn("Physical block pick-and-place", rendered)
        self.assertIn("Find the block", rendered)
        self.assertIn("Retry physical grasp", rendered)
        self.assertIn("Final checks in parallel", rendered)

    def test_selector_retry_and_parallel_nodes_execute(self) -> None:
        payload = {
            "tree": {
                "type": "sequence",
                "name": "root",
                "children": [{
                    "type": "selector",
                    "name": "fallback",
                    "children": [
                        {"type": "condition", "name": "cached", "fact": "cached"},
                        {
                            "type": "retry",
                            "name": "retry flaky action",
                            "max_attempts": 2,
                            "child": {
                                "type": "action",
                                "name": "flaky",
                                "primitive": "flaky",
                                "success_conditions": ["recovered"],
                            },
                        },
                    ],
                }, {
                    "type": "parallel",
                    "name": "parallel checks",
                    "children": [
                        {"type": "condition", "name": "check one", "fact": "recovered"},
                        {"type": "condition", "name": "check two", "fact": "ready"},
                    ],
                }],
            }
        }
        attempts = {"count": 0}

        class FlakyPrimitive(CountingPrimitive):
            def __init__(self) -> None:
                super().__init__(1)

            def tick(self, context: PrimitiveContext) -> Status:
                attempts["count"] += 1
                return Status.FAILURE if attempts["count"] == 1 else Status.SUCCESS

        messages = []
        tree = BehaviourTreeRuntime(
            payload,
            {"flaky": lambda _step: FlakyPrimitive()},
            initial_facts={"ready"},
            logger=messages.append,
        )
        self.assertIs(tree.tick(), Status.RUNNING)
        self.assertIs(tree.tick(), Status.SUCCESS)
        self.assertEqual(attempts["count"], 2)
        self.assertIn("recovered", tree.blackboard)
        self.assertTrue(any("BT SELECTOR" in message for message in messages))
        self.assertTrue(any("BT RETRY" in message for message in messages))

    def test_generated_sequence_runs_across_frames(self) -> None:
        tree = BehaviourTreeRuntime(
            PAYLOAD,
            {"grasp_object": lambda _step: CountingPrimitive(2),
             "manipulate_object": lambda _step: CountingPrimitive(1)},
            initial_facts={"connector_visible"},
            logger=lambda _message: None,
        )
        self.assertIs(tree.tick(), Status.RUNNING)
        self.assertIs(tree.tick(), Status.SUCCESS)
        self.assertIn("connector_grasped", tree.blackboard)
        self.assertIn("connector_inserted", tree.blackboard)

    def test_missing_precondition_fails_before_action(self) -> None:
        tree = BehaviourTreeRuntime(PAYLOAD, {}, logger=lambda _message: None)
        self.assertIs(tree.tick(), Status.FAILURE)
        self.assertIn("connector_visible", tree.feedback)

    def test_unregistered_primitive_fails_explicitly(self) -> None:
        tree = BehaviourTreeRuntime(
            PAYLOAD,
            {},
            initial_facts={"connector_visible"},
            logger=lambda _: None,
        )
        self.assertIs(tree.tick(), Status.FAILURE)
        self.assertIn("unregistered primitive", tree.feedback)

    def test_isaac_adapter_advances_one_controller_frame_per_tick(self) -> None:
        motion = FakeMotionController()
        articulation = FakeArticulationController()

        def queue(_context: PrimitiveContext) -> None:
            motion.remaining = 2

        tree = BehaviourTreeRuntime(
            [{"action": "move", "primitive": "navigate_to_workspace"}],
            {"navigate_to_workspace": controller_primitive(queue)},
            services={"motion_controller": motion, "robot": FakeRobot(),
                      "articulation_controller": articulation},
            logger=lambda _: None,
        )
        self.assertIs(tree.tick(), Status.RUNNING)
        self.assertIs(tree.tick(), Status.RUNNING)
        self.assertIs(tree.tick(), Status.SUCCESS)
        self.assertEqual(articulation.actions, ["action", "action"])


if __name__ == "__main__":
    unittest.main()
