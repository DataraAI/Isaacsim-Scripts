"""UR10e behaviour-tree demo: detect E_part006_44, grasp, and lift.

Run from Isaacsim-Scripts:

    /home/aayush/isaacsim/python.sh aayush/ur10e_1x_cable_insertion/main.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Task-intelligence JSON file")
    parser.add_argument("--headless", action="store_true", help="Run without a GUI")
    parser.add_argument("--max-frames", type=int, default=9000, help="Fail after this many frames")
    parser.add_argument(
        "--initial-fact",
        action="append",
        default=[],
        help="Add a true starting precondition; repeat for multiple facts",
    )
    return parser.parse_args()


ARGS = _parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless})

THIS_DIR = Path(__file__).resolve().parent
AAYUSH_DIR = THIS_DIR.parent
REPO_ROOT = AAYUSH_DIR.parent
TANISH_DIR = REPO_ROOT / "tanish"
CONTROLLER_DIR = REPO_ROOT / "detailedInsertion" / "cable"
for path in (str(TANISH_DIR), str(CONTROLLER_DIR), str(AAYUSH_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from behaviour_tree_insertion import BehaviourTreeRuntime, Status, load_task_intelligence
from behaviour_tree_insertion.isaac_adapters import controller_primitive, function_primitive

from ur10e_1x_cable_insertion.primitives import (
    check_physical_grasp,
    detect_grasp_part,
    inspect_workspace,
    queue_grasp,
    queue_move,
)
from ur10e_1x_cable_insertion.scene import apply_ur10e_home_pose, build_scene


def _hold_gui(seconds: float = 10.0) -> None:
    if ARGS.headless:
        return
    import time

    print(f"[BT CABLE] Holding GUI open for {seconds:.0f}s…")
    deadline = time.time() + seconds
    while simulation_app.is_running() and time.time() < deadline:
        simulation_app.update()


def main() -> int:
    json_path = (ARGS.json or THIS_DIR / "task_intelligence.json").expanduser().resolve()
    print(f"[BT CABLE] Loading: {json_path}")
    payload = load_task_intelligence(json_path)

    try:
        bundle = build_scene(simulation_app)
    except Exception:
        import traceback

        traceback.print_exc()
        print("[BT CABLE FAIL] Scene setup failed (see traceback above).")
        _hold_gui(10.0)
        simulation_app.close()
        return 3

    world = bundle.world
    robot = bundle.robot
    controller = bundle.motion_controller

    registry = {
        "navigate_to_workspace": controller_primitive(queue_move),
        "perceive_objects": function_primitive(detect_grasp_part),
        "grasp_object": controller_primitive(queue_grasp, validate=check_physical_grasp),
        "grasp_tool": controller_primitive(queue_grasp, validate=check_physical_grasp),
        "inspect_workspace": function_primitive(inspect_workspace),
        "execute_subtask": controller_primitive(queue_move),
    }

    tree = BehaviourTreeRuntime(
        payload,
        registry,
        initial_facts={
            "robot_ready",
            "robot_localized",
            "workspace_map_loaded",
            "camera_ready",
            *ARGS.initial_fact,
        },
        services={
            "world": world,
            "stage": bundle.stage,
            "robot": robot,
            "motion_controller": controller,
            "articulation_controller": robot.get_articulation_controller(),
            "grasp_part_path": bundle.grasp_part_path,
            "path45": bundle.path45,
            "block_top_z": bundle.block_top_z,
            "end_effector_path": bundle.end_effector_path,
        },
    )

    print("\n[BT STRUCTURE]\n" + tree.render_tree() + "\n")

    world.play()
    # Wait until the articulation physics view exists — otherwise the BT adapter
    # sees get_joint_positions()=None forever and the arm never moves.
    # Force-play only during this startup warmup; afterward respect Pause/Stop.
    physics_ready = False
    for warm in range(180):
        if not simulation_app.is_running():
            break
        if not world.is_playing():
            world.play()
        world.step(render=not ARGS.headless)
        try:
            joints = robot.get_joint_positions()
        except Exception:
            joints = None
        if joints is not None:
            physics_ready = True
            apply_ur10e_home_pose(robot, apply_live=True)
            print(f"[BT CABLE] Physics view ready after {warm + 1} warmup step(s)")
            break
    if not physics_ready:
        print("[BT CABLE FAIL] Physics Simulation View never became ready for joint reads.")
        _hold_gui(10.0)
        simulation_app.close()
        return 3

    warmup_frames = 30
    frame = 0
    result = Status.RUNNING
    print("[BT CABLE] Starting behaviour tree…")
    print("[BT CABLE] Pause/Stop are respected (script will not force-play).")
    while simulation_app.is_running() and frame < max(1, ARGS.max_frames):
        world.step(render=not ARGS.headless)
        if getattr(world, "is_stopped", lambda: False)():
            print("[BT CABLE] Timeline stopped; exiting run loop.")
            break
        if not world.is_playing():
            continue
        frame += 1
        if frame <= warmup_frames:
            continue
        result = tree.tick()
        if result in (Status.SUCCESS, Status.FAILURE):
            break

    if result is Status.SUCCESS:
        print(f"[BT CABLE PASS] Completed {tree.step_index} generated steps in {frame} frames")
        exit_code = 0
    elif result is Status.FAILURE:
        print(f"[BT CABLE FAIL] {tree.feedback}")
        exit_code = 1
    else:
        print(f"[BT CABLE FAIL] Timed out after {frame} frames; {tree.feedback}")
        exit_code = 2

    for _ in range(180 if not ARGS.headless else 1):
        if not simulation_app.is_running():
            break
        world.step(render=not ARGS.headless)
    simulation_app.close()
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("[BT CABLE FAIL] Unhandled exception (see traceback above).")
        try:
            _hold_gui(10.0)
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass
        raise SystemExit(3)
