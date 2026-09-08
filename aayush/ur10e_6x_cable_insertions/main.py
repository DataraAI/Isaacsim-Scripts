"""Six UR10e behaviour-tree demo: each arm grasps a cable and inserts into an RJ45.

Run from Isaacsim-Scripts:

    /home/aayush/isaacsim/python.sh aayush/ur10e_6x_cable_insertions/main.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Task-intelligence JSON file")
    parser.add_argument(
        "--usd",
        type=Path,
        default=None,
        help="Override DataHall_6r.usd path",
    )
    parser.add_argument("--headless", action="store_true", help="Run without a GUI")
    parser.add_argument("--max-frames", type=int, default=12000, help="Fail after this many frames")
    parser.add_argument(
        "--station",
        action="append",
        default=[],
        help="Only run this station_id (e.g. NegativeY_Top). Repeatable.",
    )
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

from ur10e_6x_cable_insertions import config as cfg
from ur10e_6x_cable_insertions.primitives import (
    check_at_port_insert,
    check_physical_grasp,
    detect_grasp_part,
    inspect_workspace,
    monitor_cable_hold,
    queue_grasp,
    queue_move,
    queue_port_approach,
)
from ur10e_6x_cable_insertions.scene import apply_ur10e_home_pose, build_scene


def _set_datahall_overview() -> None:
    """Start the GUI with the full DataHall asset in view."""

    if ARGS.headless:
        return
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=[800.0, 650.0, 650.0],
            target=[-70.0, -70.0, 260.0],
            camera_prim_path="/OmniverseKit_Persp",
        )
    except Exception as exc:
        print(f"[BT CABLE 6X] DataHall overview warning: {exc}")


def _hold_gui(seconds: float = 10.0) -> None:
    if ARGS.headless:
        return
    import time

    print(f"[BT CABLE 6X] Holding GUI open for {seconds:.0f}s…")
    deadline = time.time() + seconds
    while simulation_app.is_running() and time.time() < deadline:
        simulation_app.update()


def _select_stations():
    if not ARGS.station:
        return cfg.STATIONS
    wanted = {str(name) for name in ARGS.station}
    known = cfg.stations_by_id()
    missing = sorted(wanted - set(known))
    if missing:
        raise SystemExit(
            f"Unknown --station {missing}. Known: {', '.join(spec.station_id for spec in cfg.STATIONS)}"
        )
    return tuple(known[name] for name in (spec.station_id for spec in cfg.STATIONS) if name in wanted)


def _make_registry():
    return {
        "navigate_to_workspace": controller_primitive(queue_move),
        "perceive_objects": function_primitive(detect_grasp_part),
        "grasp_object": controller_primitive(
            queue_grasp, validate=check_physical_grasp, while_running=monitor_cable_hold
        ),
        "grasp_tool": controller_primitive(
            queue_grasp, validate=check_physical_grasp, while_running=monitor_cable_hold
        ),
        "maneuver_to_ports": controller_primitive(
            queue_port_approach,
            validate=check_at_port_insert,
            while_running=monitor_cable_hold,
        ),
        "inspect_workspace": function_primitive(inspect_workspace),
        "execute_subtask": controller_primitive(queue_move),
    }


def _make_tree(payload, station) -> BehaviourTreeRuntime:
    spec = station.spec
    sid = spec.station_id
    return BehaviourTreeRuntime(
        payload,
        _make_registry(),
        initial_facts={
            "robot_ready",
            "robot_localized",
            "workspace_map_loaded",
            "camera_ready",
            *ARGS.initial_fact,
        },
        services={
            "world": None,
            "stage": None,
            "robot": station.robot,
            "motion_controller": station.motion_controller,
            "articulation_controller": station.robot.get_articulation_controller(),
            "grasp_part_path": station.grasp_part_path,
            "path45": station.path45,
            "block_top_z": station.block_top_z,
            "observe_hand": station.observe_hand,
            "port_contacts_path": station.port_contacts_path,
            "end_effector_path": station.end_effector_path,
            "station_id": sid,
            "simulation_app": simulation_app,
            "monitor_cable_hold": False,
            "abort_simulation": False,
        },
        logger=lambda msg, s=sid: print(f"[{s}] {msg}"),
    )


def main() -> int:
    json_path = (ARGS.json or THIS_DIR / "task_intelligence.json").expanduser().resolve()
    print(f"[BT CABLE 6X] Loading: {json_path}")
    payload = load_task_intelligence(json_path)
    selected = _select_stations()
    print(
        f"[BT CABLE 6X] Stations: {', '.join(spec.station_id for spec in selected)}"
    )

    try:
        bundle = build_scene(simulation_app, usd_path=ARGS.usd, stations=selected)
    except Exception:
        import traceback

        traceback.print_exc()
        print("[BT CABLE 6X FAIL] Scene setup failed (see traceback above).")
        _hold_gui(10.0)
        simulation_app.close()
        return 3

    world = bundle.world
    _set_datahall_overview()

    trees: list[BehaviourTreeRuntime] = []
    for station in bundle.stations:
        tree = _make_tree(payload, station)
        tree.services["world"] = world
        tree.services["stage"] = bundle.stage
        trees.append(tree)

    print("\n[BT STRUCTURE]\n" + trees[0].render_tree() + "\n")

    world.play()
    physics_ready = False
    for warm in range(240):
        if not simulation_app.is_running():
            break
        if not world.is_playing():
            world.play()
        world.step(render=not ARGS.headless)
        ready = True
        for station in bundle.stations:
            try:
                joints = station.robot.get_joint_positions()
            except Exception:
                joints = None
            if joints is None:
                ready = False
                break
        if ready:
            for station in bundle.stations:
                apply_ur10e_home_pose(station.robot, apply_live=True)
            physics_ready = True
            print(f"[BT CABLE 6X] Physics view ready after {warm + 1} warmup step(s)")
            break
    if not physics_ready:
        print("[BT CABLE 6X FAIL] Physics Simulation View never became ready for joint reads.")
        _hold_gui(10.0)
        simulation_app.close()
        return 3

    warmup_frames = 30
    frame = 0
    print("[BT CABLE 6X] Starting behaviour trees…")
    print("[BT CABLE 6X] Pause/Stop are respected (script will not force-play).")
    while simulation_app.is_running() and frame < max(1, ARGS.max_frames):
        world.step(render=not ARGS.headless)
        if getattr(world, "is_stopped", lambda: False)():
            print("[BT CABLE 6X] Timeline stopped; exiting run loop.")
            break
        if not world.is_playing():
            continue
        frame += 1
        if frame <= warmup_frames:
            continue
        for tree in trees:
            if tree.status in (Status.SUCCESS, Status.FAILURE):
                continue
            result = tree.tick()
            if tree.services.get("abort_simulation"):
                print(
                    f"[BT CABLE 6X FAIL {tree.services.get('station_id')}] "
                    f"{tree.services.get('abort_reason', 'aborted')}"
                )
                tree.status = Status.FAILURE
            elif result in (Status.SUCCESS, Status.FAILURE):
                sid = tree.services.get("station_id")
                if result is Status.SUCCESS:
                    print(f"[BT CABLE 6X PASS {sid}] {tree.step_index} generated steps")
                else:
                    print(f"[BT CABLE 6X FAIL {sid}] {tree.feedback}")
        if all(tree.status in (Status.SUCCESS, Status.FAILURE) for tree in trees):
            break

    successes = [t for t in trees if t.status is Status.SUCCESS]
    failures = [t for t in trees if t.status is Status.FAILURE]
    running = [t for t in trees if t.status not in (Status.SUCCESS, Status.FAILURE)]
    print(
        f"[BT CABLE 6X] Done frames={frame} "
        f"pass={len(successes)} fail={len(failures)} running={len(running)}"
    )
    for tree in trees:
        sid = tree.services.get("station_id")
        print(f"  {sid}: {tree.status.value} {tree.feedback}")

    if failures or running:
        exit_code = 1 if failures else 2
        if running:
            print(f"[BT CABLE 6X FAIL] Timed out after {frame} frames")
    else:
        print(f"[BT CABLE 6X PASS] All {len(successes)} station(s) completed in {frame} frames")
        exit_code = 0

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
        print("[BT CABLE 6X FAIL] Unhandled exception (see traceback above).")
        try:
            _hold_gui(10.0)
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass
        raise SystemExit(3)
