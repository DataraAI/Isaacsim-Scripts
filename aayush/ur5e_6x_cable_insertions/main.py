"""UR5e demo: grasp cable, lift, maneuver to port offset, then idle.

Run from Isaacsim-Scripts:

    /home/aayush/isaacsim/python.sh aayush/ur5e_6x_cable_insertions/main.py \\
        --station NegativeY_Top
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
        help="Override DataHall_6r_ur5e.usd path",
    )
    parser.add_argument("--headless", action="store_true", help="Run without a GUI")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional safety stop after N frames (0 = run until you stop Isaac)",
    )
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
from behaviour_tree_insertion.isaac_adapters import controller_primitive

from ur5e_6x_cable_insertions import config as cfg
from ur5e_6x_cable_insertions.contact_monitor import DataHallCollisionMonitor
from ur5e_6x_cable_insertions.primitives import (
    check_at_port_offset,
    check_cable_in_gripper,
    check_port_inserted,
    monitor_cable_hold,
    queue_align_and_insert,
    queue_close_grasp,
    queue_descend_to_neck,
    queue_lift_cable,
    queue_maneuver_to_port_offset,
    queue_move,
    queue_orient_tilt,
    while_align_and_insert,
)
from ur5e_6x_cable_insertions.scene import (
    apply_live_arm_gains,
    apply_ur5e_home_pose,
    build_scene,
    freeze_inactive_ur5e_robots,
    hold_idle_ur5e_homes,
    remove_all_ur5e_ros_graphs,
)


def _set_datahall_overview() -> None:
    if ARGS.headless:
        return
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=[800.0, 650.0, 690.0],
            target=[-70.0, -70.0, 300.0],
            camera_prim_path="/OmniverseKit_Persp",
        )
    except Exception as exc:
        print(f"[BT CABLE UR5E 6X] DataHall overview warning: {exc}")


def _hold_gui(seconds: float = 10.0) -> None:
    if ARGS.headless:
        return
    import time

    print(f"[BT CABLE UR5E 6X] Holding GUI open for {seconds:.0f}s…")
    deadline = time.time() + seconds
    while simulation_app.is_running() and time.time() < deadline:
        simulation_app.update()


def _save_maneuver_cache(tree, *, verified: bool) -> None:
    """Persist angled-approach progress so the next run can continue past IK fails."""

    if not bool(cfg.MANEUVER_CACHE_SAVE):
        return
    plan = tree.services.get("maneuver_angled_plan")
    if not plan:
        return
    tip_offset = tree.services.get("maneuver_tip_offset")
    if tip_offset is None:
        tip_offset = tree.services.get("port_offset_tip")
    station_id = str(
        tree.services.get("maneuver_cache_station_id")
        or tree.services.get("station_id")
        or "unknown"
    )
    try:
        from ur5e_6x_cable_insertions import maneuver_cache

        label = str(tree.feedback or "")
        abort_reason = str(tree.services.get("abort_reason") or "")
        label = f"{label} {abort_reason}".strip()
        ctrl = tree.services.get("motion_controller")
        if ctrl is not None and hasattr(ctrl, "failure_reason"):
            label = f"{label} {ctrl.failure_reason()}"
        label_l = label.lower()

        # TipOffset already reached — insert/slip failures must not trim the
        # successful lift→offset path (that wiped a verified cache).
        post_offset = bool(tree.services.get("maneuver_reached_offset")) or any(
            key in label_l
            for key in (
                "align",
                "insert",
                "cable left",
                "cable lost",
                "cable_between",
            )
        )
        if not verified and post_offset:
            print(
                f"[MANEUVER CACHE] keep TipOffset path station={station_id} "
                f"(post-offset failure; not trimming)"
            )
            verified = True

        if verified:
            progress = 1.0
            keep = list(plan)
        else:
            failed_t = maneuver_cache.parse_angled_t_from_label(label)
            planned_ts = [float(w["t"]) for w in plan]
            if failed_t is None:
                progress = float(
                    max((t for t in planned_ts if t < 1.0 - 1e-9), default=0.0)
                )
                if "port-offset" in label and planned_ts:
                    progress = maneuver_cache.progress_before_failure(1.0, planned_ts)
                keep = [w for w in plan if float(w["t"]) <= progress + 1e-9]
            else:
                progress = maneuver_cache.progress_before_failure(failed_t, planned_ts)
                keep = [w for w in plan if float(w["t"]) <= progress + 1e-9]

            # DataHall / rack hit: drop any tips that already entered the rack
            # and push tip_offset further +X so the next run does not replay it.
            if "collision" in label_l or "datahall" in label_l or "rack_core" in label_l:
                min_x = float(cfg.MANEUVER_CACHE_SAFE_TIP_X_M)
                safe_progress = maneuver_cache.last_safe_progress_before_rack(
                    plan, min_tip_x=min_x
                )
                if safe_progress + 1e-9 < progress:
                    progress = safe_progress
                keep = [w for w in plan if float(w["t"]) <= progress + 1e-9]
                tip_offset = maneuver_cache.retract_tip_offset_x(
                    tip_offset, float(cfg.MANEUVER_CACHE_COLLISION_X_RETRACT_M)
                )
                print(
                    f"[MANEUVER CACHE] collision trim station={station_id} "
                    f"safe_progress={progress:.2f} min_tip_x={min_x:.3f} "
                    f"tip_offset_x="
                    f"{None if tip_offset is None else float(tip_offset[0]):.3f}"
                )
            else:
                print(
                    f"[MANEUVER CACHE] IK progress trim station={station_id} "
                    f"failed_t={failed_t} keep_progress={progress:.2f} "
                    f"waypoints={len(keep)}"
                )
        maneuver_cache.save_station(
            station_id,
            angled_progress=progress,
            waypoints=keep,
            tip_offset=tip_offset,
            verified=verified,
            path=cfg.MANEUVER_CACHE_PATH,
        )
    except Exception as exc:
        print(f"[MANEUVER CACHE] save failed: {exc}")

def _select_stations():
    if not ARGS.station:
        return (cfg.stations_by_id()["NegativeY_Top"],)
    wanted = {str(name) for name in ARGS.station}
    known = cfg.stations_by_id()
    missing = sorted(wanted - set(known))
    if missing:
        raise SystemExit(
            f"Unknown --station {missing}. Known: "
            f"{', '.join(spec.station_id for spec in cfg.STATIONS)}"
        )
    return tuple(
        known[name]
        for name in (spec.station_id for spec in cfg.STATIONS)
        if name in wanted
    )


def _make_registry():
    hold = dict(while_running=monitor_cable_hold)
    return {
        "navigate_to_workspace": controller_primitive(queue_move),
        "orient_gripper": controller_primitive(queue_orient_tilt),
        "descend_to_neck": controller_primitive(queue_descend_to_neck),
        "grasp_object": controller_primitive(
            queue_close_grasp,
            validate=check_cable_in_gripper,
            while_running=monitor_cable_hold,
        ),
        "lift_cable": controller_primitive(queue_lift_cable, **hold),
        "maneuver_to_ports": controller_primitive(
            queue_maneuver_to_port_offset,
            validate=check_at_port_offset,
            while_running=monitor_cable_hold,
        ),
        "align_and_insert": controller_primitive(
            queue_align_and_insert,
            validate=check_port_inserted,
            while_running=while_align_and_insert,
        ),
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
            "cable_root_path": station.cable_root_path,
            "grasp_part_path": spec.grasp_part_path,
            "end_effector_path": station.end_effector_path,
            "station_id": sid,
            "station_spec": spec,
            "simulation_app": simulation_app,
            "abort_simulation": False,
            "monitor_cable_hold": False,
        },
        logger=lambda msg, s=sid: print(f"[{s}] {msg}"),
    )


def main() -> int:
    json_path = (ARGS.json or THIS_DIR / "task_intelligence.json").expanduser().resolve()
    print(f"[BT CABLE UR5E 6X] Loading: {json_path}")
    payload = load_task_intelligence(json_path)
    selected = _select_stations()
    print(
        f"[BT CABLE UR5E 6X] Stations: {', '.join(spec.station_id for spec in selected)}"
    )

    try:
        bundle = build_scene(simulation_app, usd_path=ARGS.usd, stations=selected)
    except Exception:
        import traceback

        traceback.print_exc()
        print("[BT CABLE UR5E 6X FAIL] Scene setup failed (see traceback above).")
        _hold_gui(10.0)
        simulation_app.close()
        return 3

    world = bundle.world
    _set_datahall_overview()

    trees: list[BehaviourTreeRuntime] = []
    collision_monitors: list[DataHallCollisionMonitor] = []
    for station in bundle.stations:
        tree = _make_tree(payload, station)
        tree.services["world"] = world
        tree.services["stage"] = bundle.stage
        watched = (station.spec.cable_root_path,)
        skip = (
            (station.spec.robot_prim_path,)
            if bool(getattr(cfg, "INSERT_CONTACT_SKIP_ROBOT", True))
            else ()
        )
        try:
            monitor = DataHallCollisionMonitor(watched, skip_prefixes=skip)
            tree.services["collision_monitor"] = monitor
            collision_monitors.append(monitor)
        except Exception as exc:
            print(
                f"[BT CABLE UR5E 6X] Collision monitor unavailable "
                f"({station.spec.station_id}): {exc}"
            )
            tree.services["collision_monitor"] = None
        trees.append(tree)

    print("\n[BT STRUCTURE]\n" + trees[0].render_tree() + "\n")

    world.play()
    # Play/reset reloads authored ROS graphs — strip again on every arm.
    selected_specs = [station.spec for station in bundle.stations]
    remove_all_ur5e_ros_graphs(bundle.stage, selected_specs)
    freeze_inactive_ur5e_robots(bundle.stage, selected_specs)
    physics_ready = False
    for warm in range(240):
        if not simulation_app.is_running():
            break
        if not world.is_playing():
            world.play()
            remove_all_ur5e_ros_graphs(bundle.stage, selected_specs)
            freeze_inactive_ur5e_robots(bundle.stage, selected_specs)
        world.step(render=not ARGS.headless)
        # Keep unselected arms glued to home from the first physics step.
        hold_idle_ur5e_homes(bundle.idle_stations)
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
                arm = station.home_arm
                if arm is None:
                    from ur5e_6x_cable_insertions.scene import (
                        compute_home_arm_above_work_table,
                    )

                    arm = compute_home_arm_above_work_table(
                        bundle.stage, station.motion_controller, station.spec
                    )
                    station.home_arm = arm
                apply_ur5e_home_pose(
                    station.robot, arm_positions=arm, apply_live=True
                )
                # Re-apply after the physics view exists; earlier reset can leave
                # default soft gains that make the hover transit wobble.
                apply_live_arm_gains(station.robot, station.spec.station_id)
            for idle in bundle.idle_stations:
                apply_ur5e_home_pose(
                    idle.robot, arm_positions=idle.home_arm, apply_live=True
                )
                apply_live_arm_gains(idle.robot, idle.spec.station_id)
            hold_idle_ur5e_homes(bundle.idle_stations)
            remove_all_ur5e_ros_graphs(bundle.stage, selected_specs)
            freeze_inactive_ur5e_robots(bundle.stage, selected_specs)
            physics_ready = True
            print(
                f"[BT CABLE UR5E 6X] Physics view ready after {warm + 1} warmup step(s)"
            )
            break
    if not physics_ready:
        print(
            "[BT CABLE UR5E 6X FAIL] Physics Simulation View never became ready "
            "for joint reads."
        )
        _hold_gui(10.0)
        simulation_app.close()
        return 3

    # Hold home (work-table center X) so the gripper mount can be checked before motion.
    home_hold_s = 5.0
    print(f"[BT CABLE UR5E 6X] Holding home pose for {home_hold_s:.0f}s…")
    import time

    home_deadline = time.time() + home_hold_s
    while simulation_app.is_running() and time.time() < home_deadline:
        world.step(render=not ARGS.headless)
        if getattr(world, "is_stopped", lambda: False)():
            print("[BT CABLE UR5E 6X] Timeline stopped during home hold; exiting.")
            simulation_app.close()
            return 2
        if not world.is_playing():
            continue
        for station in bundle.stations:
            apply_ur5e_home_pose(
                station.robot,
                arm_positions=station.home_arm,
                apply_live=True,
            )
        hold_idle_ur5e_homes(bundle.idle_stations)

    # Re-assert stiff PD + max efforts after the teleport home-hold; that hold
    # masks soft control, and the first Lula move is when gravity folds a weak arm.
    # Do NOT rewrite USD MimicJoint / DriveAPI here — that invalidates the live
    # PhysX articulation cache ("articulation configuration has changed").
    for station in bundle.stations:
        station.robot._ur5e_live_gains_applied = False
        apply_live_arm_gains(station.robot, station.spec.station_id)
    for idle in bundle.idle_stations:
        idle.robot._ur5e_live_gains_applied = False
        apply_live_arm_gains(idle.robot, idle.spec.station_id)

    warmup_frames = 30
    frame = 0
    done_announced = False
    slip_announced = False
    max_frames = int(ARGS.max_frames)
    print("[BT CABLE UR5E 6X] Starting behaviour trees…")
    print("[BT CABLE UR5E 6X] Pause/Stop are respected (script will not force-play).")
    cuda_fail_streak = 0
    while simulation_app.is_running():
        if max_frames > 0 and frame >= max_frames:
            print(f"[BT CABLE UR5E 6X] Reached --max-frames={max_frames}; exiting.")
            break
        try:
            world.step(render=not ARGS.headless)
        except Exception as exc:
            msg = str(exc).lower()
            if "700" in msg or "cuda" in msg:
                print(
                    f"[BT CABLE UR5E 6X FAIL] PhysX/CUDA step failed ({exc}); "
                    "exiting so the next run gets a clean GPU context."
                )
                break
            raise
        if getattr(world, "is_stopped", lambda: False)():
            print("[BT CABLE UR5E 6X] Timeline stopped; exiting run loop.")
            break
        if not world.is_playing():
            continue
        # Detect poisoned PhysX GPU context (CUDA 700) via articulation read failure.
        try:
            _ = bundle.stations[0].robot.get_joint_positions()
            cuda_fail_streak = 0
        except Exception as exc:
            cuda_fail_streak += 1
            if cuda_fail_streak >= 5:
                print(
                    f"[BT CABLE UR5E 6X FAIL] Articulation views dead after PhysX "
                    f"GPU error ({exc}); closing app for a clean relaunch."
                )
                break
        # Default motion for unselected arms: stay at home.
        hold_idle_ur5e_homes(bundle.idle_stations)
        frame += 1
        # Play/reset can revive inactive ROS ArticulationControllers.
        if frame == 1 or frame % 180 == 0:
            remove_all_ur5e_ros_graphs(bundle.stage, selected_specs)
            freeze_inactive_ur5e_robots(bundle.stage, selected_specs)
        if frame <= warmup_frames:
            continue
        for tree in trees:
            if tree.status in (Status.SUCCESS, Status.FAILURE):
                continue
            result = tree.tick()
            if tree.services.get("abort_simulation"):
                print(
                    f"[BT CABLE UR5E 6X FAIL {tree.services.get('station_id')}] "
                    f"{tree.services.get('abort_reason', 'aborted')}"
                )
                tree.status = Status.FAILURE
                _save_maneuver_cache(tree, verified=False)
            elif result in (Status.SUCCESS, Status.FAILURE):
                sid = tree.services.get("station_id")
                if result is Status.SUCCESS:
                    print(
                        f"[BT CABLE UR5E 6X PASS {sid}] "
                        f"grasp→lift→port-offset→insert done "
                        f"({tree.step_index} generated steps)"
                    )
                    _save_maneuver_cache(tree, verified=True)
                else:
                    print(f"[BT CABLE UR5E 6X FAIL {sid}] {tree.feedback}")
                    _save_maneuver_cache(tree, verified=False)
        if (
            not slip_announced
            and any(
                t.status is Status.FAILURE and t.services.get("abort_simulation")
                for t in trees
            )
        ):
            slip_announced = True
            print(
                "[BT CABLE UR5E 6X] Cable-hold postcondition failed — motion stopped. "
                "Simulation stays running; Stop Isaac when done."
            )
        if (
            not done_announced
            and all(tree.status is Status.SUCCESS for tree in trees)
        ):
            done_announced = True
            print(
                "[BT CABLE UR5E 6X] Port-offset sequence complete — leaving "
                "simulation running. Stop Isaac when done."
            )
            # Keep looping; do not exit on BT success.

    successes = [t for t in trees if t.status is Status.SUCCESS]
    failures = [t for t in trees if t.status is Status.FAILURE]
    running = [t for t in trees if t.status not in (Status.SUCCESS, Status.FAILURE)]
    print(
        f"[BT CABLE UR5E 6X] Done frames={frame} "
        f"pass={len(successes)} fail={len(failures)} running={len(running)}"
    )
    for tree in trees:
        sid = tree.services.get("station_id")
        print(f"  {sid}: {tree.status.value} {tree.feedback}")
    for monitor in collision_monitors:
        try:
            monitor.report_summary()
            monitor.close()
        except Exception:
            pass

    exit_code = 0 if successes and not failures and not running else (
        1 if failures else 2
    )
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
        print("[BT CABLE UR5E 6X FAIL] Unhandled exception (see traceback above).")
        try:
            _hold_gui(10.0)
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass
        raise SystemExit(3)
