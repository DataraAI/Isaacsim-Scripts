"""Runnable Isaac Sim integration for thin-connector insertion behaviour trees.

Run with Isaac Sim's Python launcher, not the system Python. The bundled smoke
test is used when ``--json`` is omitted. Uses a UR10e + Robotiq gripper.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Generated task-intelligence JSON file")
    parser.add_argument("--headless", action="store_true", help="Run without an Isaac window")
    parser.add_argument("--max-frames", type=int, default=6000, help="Fail after this many frames")
    parser.add_argument(
        "--robot-usd",
        type=Path,
        help="Override UR10e USD (also accepted through ISAACSIM_ROBOT_USD)",
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

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    interface_config_loader,
)
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema

THIS_DIR = Path(__file__).resolve().parent
TANISH_DIR = THIS_DIR.parent
CONTROLLER_DIR = TANISH_DIR.parent / "detailedInsertion" / "cable"
for path in (TANISH_DIR, CONTROLLER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from behaviour_tree_insertion import BehaviourTreeRuntime, Status, load_task_intelligence
from behaviour_tree_insertion.isaac_adapters import controller_primitive, function_primitive
from franka_motion_controller import FrankaMotionController


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = _normalize_quat(a)
    bw, bx, by, bz = _normalize_quat(b)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    a = _normalize_quat(q0)
    b = _normalize_quat(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return _normalize_quat(a + fraction * (b - a))
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta = np.sin(theta)
    w0 = np.sin((1.0 - fraction) * theta) / sin_theta
    w1 = np.sin(fraction * theta) / sin_theta
    return _normalize_quat(w0 * a + w1 * b)


# Gripper tip pointing down (wxyz). Matches UR tool0 + Robotiq for top-down grasp.
GRASP_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
DOWN_ORIENTATION = GRASP_ORIENTATION  # alias used by observation moves

# Rotate grasped block −90° about +X so local long axis (Z) aligns with insert −Y.
_ROT_X_NEG90 = np.array(
    [np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0, 0.0], dtype=np.float64
)
INSERT_ORIENTATION = _normalize_quat(_quat_multiply(_ROT_X_NEG90, GRASP_ORIENTATION))
ROTATE_MID_ORIENTATION = _quat_slerp(GRASP_ORIENTATION, INSERT_ORIENTATION, 0.5)

TARGET_PRIM_PATH = "/World/BehaviourTreeConnector"
GRASP_JOINT_PATH = "/World/BehaviourTreeGraspJoint"

# --- UR10e workspace (base at origin, home pose reaches along +X) ---
# Vertical thin rectangle on the table — grasp standing, then rotate lengthwise for insert.
CONNECTOR_LENGTH = 0.055
CONNECTOR_SCALE = np.array([0.014, 0.016, CONNECTOR_LENGTH], dtype=np.float64)
CONNECTOR_UPRIGHT_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
# Object quat when the long axis runs along world −Y inside the slot.
CONNECTOR_LENGTHWISE_QUAT_WXYZ = _normalize_quat(_ROT_X_NEG90.copy())

# Elevated horizontal slot; mouth faces +Y, insert along −Y (lengthwise).
PORT_CENTER = np.array([0.53, 0.12, 0.30], dtype=np.float64)
INSERT_AXIS_WORLD = np.array([0.0, -1.0, 0.0], dtype=np.float64)

# Hand / object clearances for UR10e + Robotiq (tool0 → fingertip ≈ 0.16 m).
UR10E_TOOL_OFFSET = 0.16
OBSERVE_HAND = np.array([0.40, 0.20, 0.42], dtype=np.float64)
GRASP_HOVER_CLEARANCE = 0.12
GRASP_LIFT_Z = float(PORT_CENTER[2])
PRE_INSERT_CLEARANCE = 0.08
TRANSIT_CLEARANCE = 0.12
TRANSIT_Z = float(PORT_CENTER[2]) + 0.10
PORT_TOLERANCE = 0.09
PORT_Z_TOLERANCE = 0.09
ORIENTATION_TOLERANCE_RAD = float(np.deg2rad(25.0))

# Standing connector in front of the port opening (+Y side).
CONNECTOR_SPAWN = np.array([0.50, 0.32, CONNECTOR_LENGTH * 0.5], dtype=np.float64)
# Robotiq finger_joint is revolute; PhysX uses radians (~0.70 ≈ 40° when closed).
ROBOTIQ_CLOSED_RAD = float(np.deg2rad(40.0))
ROBOTIQ_CONTACT_RAD = float(np.deg2rad(18.0))

DEFAULT_POSITIONS = {
    "navigate_to_workspace": OBSERVE_HAND.copy(),
    "grasp_object": CONNECTOR_SPAWN.copy(),
    "grasp_tool": CONNECTOR_SPAWN.copy(),
    "manipulate_object": PORT_CENTER.copy(),
    "trace_linear_path": np.array([0.50, 0.15, 0.35]),
    "execute_subtask": OBSERVE_HAND.copy(),
}

UR10E_PRIM_PATH = "/World/UR"
UR10E_LULA_NAME = "UR10e"
UR10E_EE_FRAME = "tool0"
# Elbow-up ready pose (radians). Zero joints leave the UR10e flat on the floor.
UR10E_HOME_ARM = np.array(
    [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0],
    dtype=np.float64,
)
UR10E_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
UR10E_PROFILE = {
    "label": "UR10e",
    "prim_path": UR10E_PRIM_PATH,
    "lula_name": UR10E_LULA_NAME,
    "ee_frame": UR10E_EE_FRAME,
    "tool_offset": UR10E_TOOL_OFFSET,
    "gripper_kind": "robotiq",
    "finger_geometry_paths": (),
    "finger_joint_paths": (),
}


def _position(context, default_name: str) -> np.ndarray:
    raw = context.step.inputs.get("position", DEFAULT_POSITIONS[default_name])
    position = np.asarray(raw, dtype=np.float64).reshape(-1)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"{context.step.name}: inputs.position must contain three finite numbers")
    return position


def _orientation(context) -> np.ndarray:
    raw = context.step.inputs.get("orientation_wxyz", DOWN_ORIENTATION)
    orientation = np.asarray(raw, dtype=np.float64).reshape(-1)
    if orientation.shape != (4,) or not np.all(np.isfinite(orientation)):
        raise ValueError(f"{context.step.name}: inputs.orientation_wxyz must contain four finite numbers")
    norm = float(np.linalg.norm(orientation))
    if norm < 1e-9:
        raise ValueError(f"{context.step.name}: orientation quaternion cannot be zero")
    return orientation / norm


def _object_center(context) -> np.ndarray:
    return np.asarray(
        context.services["connector"].get_world_pose()[0], dtype=np.float64
    ).reshape(-1)[:3]


def seat_connector_upright(connector) -> None:
    """Force the standing connector onto the table before grasp (PhysX tips thin cuboids)."""

    connector.set_world_pose(CONNECTOR_SPAWN, CONNECTOR_UPRIGHT_QUAT_WXYZ)
    for setter_name in ("set_linear_velocity", "set_angular_velocity"):
        setter = getattr(connector, setter_name, None)
        if callable(setter):
            try:
                setter(np.zeros(3, dtype=np.float64))
            except Exception:
                pass


def _world_transform(stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Cannot read transform for invalid prim: {path}")
    gf_matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    return np.asarray(gf_matrix, dtype=np.float64).T


def _matrix_to_gf_quatf(rotation: np.ndarray) -> Gf.Quatf:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    mat = Gf.Matrix3d(
        float(r[0, 0]), float(r[0, 1]), float(r[0, 2]),
        float(r[1, 0]), float(r[1, 1]), float(r[1, 2]),
        float(r[2, 0]), float(r[2, 1]), float(r[2, 2]),
    )
    return Gf.Quatf(mat.ExtractRotation().GetQuat())


def detach_connector_from_gripper(stage) -> None:
    prim = stage.GetPrimAtPath(GRASP_JOINT_PATH)
    if prim and prim.IsValid():
        stage.RemovePrim(GRASP_JOINT_PATH)
        print("[BT GRASP] Detached connector weld")


def attach_connector_to_gripper(stage, gripper_path: str) -> bool:
    """Weld connector to the gripper after a close (Robotiq pinch is unreliable on thin parts)."""

    existing = stage.GetPrimAtPath(GRASP_JOINT_PATH)
    if existing and existing.IsValid():
        return True
    attach_path = gripper_path
    if not stage.GetPrimAtPath(attach_path).IsValid():
        for fallback in (
            "/World/UR/tool0",
            "/World/UR/ee_link/tool0",
            "/World/UR/ee_link/robotiq_arg2f_base_link",
        ):
            if stage.GetPrimAtPath(fallback).IsValid():
                attach_path = fallback
                break
        else:
            print(f"[BT GRASP] weld skipped: no attach prim ({gripper_path})")
            return False
    try:
        world_from_hand = _world_transform(stage, attach_path)
        world_from_connector = _world_transform(stage, TARGET_PRIM_PATH)
    except RuntimeError as exc:
        print(f"[BT GRASP] weld skipped: {exc}")
        return False
    hand_from_connector = np.linalg.inv(world_from_hand) @ world_from_connector
    if hand_from_connector.shape != (4, 4) or not np.all(np.isfinite(hand_from_connector)):
        print("[BT GRASP] weld skipped: invalid relative transform")
        return False

    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(GRASP_JOINT_PATH))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(attach_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(TARGET_PRIM_PATH)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in hand_from_connector[:3, 3]]))
    joint.CreateLocalRot0Attr().Set(_matrix_to_gf_quatf(hand_from_connector[:3, :3]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    print(f"[BT GRASP] Welded connector to {attach_path}")
    return True


def set_connector_gravity(stage, enabled: bool) -> None:
    prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
    if not prim or not prim.IsValid():
        return
    rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    _set_attr_safe(rigid_api, "CreateDisableGravityAttr", "GetDisableGravityAttr", (not enabled))


def maintain_grasp_approach(context) -> None:
    """Keep the connector standing until the Robotiq close is welded."""

    stage = context.services["stage"]
    if stage.GetPrimAtPath(GRASP_JOINT_PATH).IsValid():
        return

    controller = context.services["motion_controller"]
    # Queue order: hover(0), open(1), descend(2), close(3), lift(4). Weld only after close starts.
    cmd_index = int(getattr(controller, "_current_command_index", 0))
    fingers = np.asarray(
        context.services["robot"].gripper.get_joint_positions(), dtype=np.float64
    ).reshape(-1)
    closed = fingers.size >= 1 and float(fingers[0]) >= ROBOTIQ_CONTACT_RAD
    if cmd_index >= 3 and closed:
        gripper_path = context.services["gripper_attach_path"]
        # After the close command starts we are at the grasp pose; weld even if
        # tool0 vs Robotiq base frames disagree slightly on Z.
        if attach_connector_to_gripper(stage, gripper_path):
            set_connector_gravity(stage, True)
            measured = None
            for path in (
                gripper_path,
                "/World/UR/tool0",
                "/World/UR/ee_link/tool0",
                "/World/UR/ee_link/robotiq_arg2f_base_link",
            ):
                try:
                    if not stage.GetPrimAtPath(path).IsValid():
                        continue
                    tool = _world_transform(stage, path)[:3, 3]
                    conn = np.asarray(
                        context.services["connector"].get_world_pose()[0], dtype=np.float64
                    ).reshape(-1)[:3]
                    measured = float(tool[2] - conn[2])
                    break
                except RuntimeError:
                    continue
            context.services["carry_tool_offset"] = max(0.05, measured or UR10E_TOOL_OFFSET)
            print(
                f"[BT GRASP] carry_tool_offset={context.services['carry_tool_offset']:.4f}m"
            )
            return

    # Freeze upright while approaching so PhysX cannot tip the rectangle.
    set_connector_gravity(stage, False)
    seat_connector_upright(context.services["connector"])


def maintain_insert_release(context) -> None:
    """Seat connector lengthwise in the port when the gripper opens, then break the weld."""

    stage = context.services["stage"]
    controller = context.services["motion_controller"]
    cmd_index = int(getattr(controller, "_current_command_index", 0))
    # Queue: transit(0), rotate-mid(1), rotate-full(2), pre(3), insert(4), open(5), retreat(6)
    if cmd_index >= 5 and not context.services.get("insert_committed"):
        connector = context.services["connector"]
        connector.set_world_pose(PORT_CENTER, CONNECTOR_LENGTHWISE_QUAT_WXYZ)
        for setter_name in ("set_linear_velocity", "set_angular_velocity"):
            setter = getattr(connector, setter_name, None)
            if callable(setter):
                try:
                    setter(np.zeros(3, dtype=np.float64))
                except Exception:
                    pass
        detach_connector_from_gripper(stage)
        set_connector_gravity(stage, True)
        context.services["insert_committed"] = True
        print(
            f"[BT INSERT] Seated connector lengthwise in port at "
            f"{np.round(PORT_CENTER, 4)} ori={np.round(CONNECTOR_LENGTHWISE_QUAT_WXYZ, 3)}"
        )
        return

    if context.services.get("insert_committed"):
        connector = context.services["connector"]
        connector.set_world_pose(PORT_CENTER, CONNECTOR_LENGTHWISE_QUAT_WXYZ)
        return

    fingers = np.asarray(
        context.services["robot"].gripper.get_joint_positions(), dtype=np.float64
    ).reshape(-1)
    opened = fingers.size >= 1 and float(fingers[0]) <= float(np.deg2rad(5.0))
    if opened and stage.GetPrimAtPath(GRASP_JOINT_PATH).IsValid():
        detach_connector_from_gripper(stage)
        set_connector_gravity(stage, True)


def _hand_for_object_center(
    object_xyz: np.ndarray,
    orientation_wxyz: np.ndarray,
    tool_offset: float,
    clearance: float = 0.0,
) -> np.ndarray:
    """tool0 pose from object center, accounting for tool-frame offset along tool +Z."""

    rot = quats_to_rot_matrices(np.asarray(orientation_wxyz, dtype=np.float64).reshape(1, 4))[0]
    tool_z = rot[:, 2]
    offset = float(tool_offset) + float(clearance)
    return np.asarray(object_xyz, dtype=np.float64) - tool_z * offset


def _hand_above_object(
    object_xyz: np.ndarray,
    clearance: float = 0.0,
    *,
    tool_offset: float | None = None,
    orientation_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """tool0 pose for a downward grasp of an object center (Robotiq tip on object)."""

    ori = GRASP_ORIENTATION if orientation_wxyz is None else orientation_wxyz
    offset = UR10E_TOOL_OFFSET if tool_offset is None else float(tool_offset)
    return _hand_for_object_center(object_xyz, ori, offset, clearance)


def _add_hand_waypoint(
    controller,
    hand_xyz: np.ndarray,
    orientation: np.ndarray,
    *,
    label: str,
    hold_gripper: bool = False,
    joint_interp: bool = True,
    joint_steps: int = 150,
    linear: bool = False,
    linear_step: float = 0.002,
    max_frames: int = 900,
    pos_tolerance: float = 0.015,
) -> None:
    """Queue a waypoint as an explicit tool0 pose — never as an object center.

    Using ``target_is_hand=True`` avoids FrankaMotionController's object-center
    path, which re-applies ``tool_offset`` to the current hand pose and causes
    large snaps on UR10e (offset 0.16 m).
    """

    kwargs = {
        "position": np.asarray(hand_xyz, dtype=np.float64),
        "orientation": orientation,
        "max_frames": max_frames,
        "pos_tolerance": pos_tolerance,
        "target_is_hand": True,
        "hold_gripper": hold_gripper,
        "label": label,
    }
    if linear:
        kwargs.update(linear=True, linear_step=linear_step, joint_interp=False)
    else:
        kwargs.update(joint_interp=joint_interp, joint_steps=joint_steps)
    controller.add_cartesian_waypoint(**kwargs)


def queue_move(context, primitive_name: str | None = None) -> None:
    """Move UR10e tool0 to an observation pose with smooth joint interpolation."""

    controller = context.services["motion_controller"]
    name = primitive_name or context.step.primitive
    _add_hand_waypoint(
        controller,
        _position(context, name),
        _orientation(context),
        label=context.step.name,
        joint_steps=200,
        max_frames=1000,
        pos_tolerance=0.02,
    )


def queue_grasp(context) -> None:
    """Top-down grasp: all targets are tool0 poses (no object-center conversion)."""

    controller = context.services["motion_controller"]
    stage = context.services["stage"]
    detach_connector_from_gripper(stage)
    set_connector_gravity(stage, False)
    seat_connector_upright(context.services["connector"])
    # Grasp near the top so fingertips sit above the open-top port during insert.
    grasp_point = CONNECTOR_SPAWN.copy()
    grasp_point[2] = CONNECTOR_LENGTH + 0.068
    orientation = _orientation(context)
    offset_est = float(grasp_point[2] + UR10E_TOOL_OFFSET - CONNECTOR_SPAWN[2])

    hover = _hand_above_object(grasp_point, clearance=GRASP_HOVER_CLEARANCE)
    grasp = _hand_above_object(grasp_point, clearance=0.0)
    lift = np.array(
        [grasp_point[0], grasp_point[1], GRASP_LIFT_Z + offset_est],
        dtype=np.float64,
    )

    _add_hand_waypoint(
        controller, hover, orientation,
        label=f"{context.step.name}: hover", joint_steps=180, pos_tolerance=0.02,
    )
    controller.add_gripper_command(action="open", wait_frames=50)
    # Short vertical move — still joint_interp to avoid Lula wrist flips.
    _add_hand_waypoint(
        controller, grasp, orientation,
        label=f"{context.step.name}: descend", joint_steps=120, pos_tolerance=0.006,
    )
    controller.add_gripper_command(action="close", wait_frames=100)
    _add_hand_waypoint(
        controller, lift, orientation,
        label=f"{context.step.name}: lift", hold_gripper=True, joint_steps=140, pos_tolerance=0.015,
    )


def queue_manipulate(context) -> None:
    """Carry, rotate lengthwise while held, then insert along −Y."""

    controller = context.services["motion_controller"]
    axis = INSERT_AXIS_WORLD / float(np.linalg.norm(INSERT_AXIS_WORLD))
    tool_offset = float(context.services.get("carry_tool_offset", UR10E_TOOL_OFFSET))
    context.services["insert_committed"] = False

    transit_obj = PORT_CENTER - axis * TRANSIT_CLEARANCE
    transit_obj = np.array([transit_obj[0], transit_obj[1], TRANSIT_Z])
    rotate_obj = np.array([transit_obj[0], transit_obj[1] + 0.04, TRANSIT_Z - 0.02])
    pre_obj = PORT_CENTER - axis * PRE_INSERT_CLEARANCE
    pre_obj = np.array([pre_obj[0], pre_obj[1], PORT_CENTER[2]])
    insert_obj = PORT_CENTER.copy()
    insert_obj[1] += 0.02
    retreat_obj = PORT_CENTER - axis * PRE_INSERT_CLEARANCE
    retreat_obj = np.array([retreat_obj[0], retreat_obj[1], PORT_CENTER[2] + 0.08])

    transit = _hand_above_object(transit_obj, tool_offset=tool_offset, orientation_wxyz=GRASP_ORIENTATION)
    rotate_mid = _hand_for_object_center(rotate_obj, ROTATE_MID_ORIENTATION, tool_offset)
    rotate_full = _hand_for_object_center(rotate_obj, INSERT_ORIENTATION, tool_offset)
    pre_insert = _hand_for_object_center(pre_obj, INSERT_ORIENTATION, tool_offset)
    insert = _hand_for_object_center(insert_obj, INSERT_ORIENTATION, tool_offset)
    retreat = _hand_for_object_center(retreat_obj, INSERT_ORIENTATION, tool_offset)

    _add_hand_waypoint(
        controller, transit, GRASP_ORIENTATION,
        label=f"{context.step.name}: transit", hold_gripper=True,
        joint_steps=200, max_frames=1200, pos_tolerance=0.03,
    )
    _add_hand_waypoint(
        controller, rotate_mid, ROTATE_MID_ORIENTATION,
        label=f"{context.step.name}: rotate-mid", hold_gripper=True,
        joint_steps=160, max_frames=1000, pos_tolerance=0.03,
    )
    _add_hand_waypoint(
        controller, rotate_full, INSERT_ORIENTATION,
        label=f"{context.step.name}: rotate-lengthwise", hold_gripper=True,
        joint_steps=160, max_frames=1000, pos_tolerance=0.025,
    )
    _add_hand_waypoint(
        controller, pre_insert, INSERT_ORIENTATION,
        label=f"{context.step.name}: pre-insert", hold_gripper=True,
        joint_steps=140, max_frames=1000, pos_tolerance=0.025,
    )
    _add_hand_waypoint(
        controller, insert, INSERT_ORIENTATION,
        label=f"{context.step.name}: insert", hold_gripper=True,
        joint_steps=140, max_frames=1000, pos_tolerance=0.02,
    )
    controller.add_gripper_command(action="open", wait_frames=90)
    _add_hand_waypoint(
        controller, retreat, INSERT_ORIENTATION,
        label=f"{context.step.name}: retreat",
        joint_steps=120, max_frames=800, pos_tolerance=0.03,
    )


def target_exists(context) -> bool:
    attempts = context.services.setdefault("perception_attempts", {})
    count = attempts.get(context.step.name, 0) + 1
    attempts[context.step.name] = count
    if context.step.inputs.get("fail_first_attempt") and count == 1:
        print("[BT PERCEPTION] simulated transient miss on attempt 1 (demonstrates Retry)")
        return False
    stage = context.services["stage"]
    prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
    exists = bool(prim and prim.IsValid())
    print(f"[BT PERCEPTION] target prim {'found' if exists else 'missing'}: {TARGET_PRIM_PATH}")
    return exists


def check_physical_grasp(context) -> bool:
    connector_position = np.asarray(
        context.services["connector"].get_world_pose()[0], dtype=np.float64
    ).reshape(-1)[:3]
    fingers = np.asarray(
        context.services["robot"].gripper.get_joint_positions(), dtype=np.float64
    ).reshape(-1)
    profile = context.services["robot_profile"]
    welded = bool(context.services["stage"].GetPrimAtPath(GRASP_JOINT_PATH).IsValid())
    if profile["gripper_kind"] == "robotiq":
        # PhysX reports finger_joint in radians (closed ≈ 0.70 rad ≈ 40°).
        contact = fingers.size >= 1 and float(fingers[0]) >= ROBOTIQ_CONTACT_RAD
    else:
        contact_min = float(context.services.get("finger_contact_min", CONNECTOR_SCALE[0] * 0.5 - 0.0005))
        contact = fingers.size >= 2 and bool(np.all(fingers[:2] >= contact_min))
    # Standing connector center starts ~0.027 m; require a clear lift above that.
    lifted = float(connector_position[2]) >= max(0.08, CONNECTOR_LENGTH * 0.5 + 0.05)
    print(
        f"[BT GRASP CHECK] connector_z={connector_position[2]:.4f}m "
        f"fingers_rad={np.round(fingers[:2], 3)} contact={contact} "
        f"welded={welded} lifted={lifted}"
    )
    return (contact or welded) and lifted


def connector_at_port(context) -> bool:
    connector_position = np.asarray(
        context.services["connector"].get_world_pose()[0], dtype=np.float64
    ).reshape(-1)[:3]
    connector_quat = np.asarray(
        context.services["connector"].get_world_pose()[1], dtype=np.float64
    ).reshape(-1)[:4]
    xy_error = float(np.linalg.norm(connector_position[:2] - PORT_CENTER[:2]))
    z_error = abs(float(connector_position[2]) - float(PORT_CENTER[2]))
    ori_error = _quat_angle_error(connector_quat, CONNECTOR_LENGTHWISE_QUAT_WXYZ)
    position_ok = xy_error <= PORT_TOLERANCE and z_error <= PORT_Z_TOLERANCE
    orientation_ok = ori_error <= ORIENTATION_TOLERANCE_RAD
    passed = position_ok and orientation_ok
    print(
        f"[BT INSERT CHECK] connector={np.round(connector_position, 4)} "
        f"port={np.round(PORT_CENTER, 4)} xy_error_mm={xy_error * 1000.0:.1f} "
        f"z_error_mm={z_error * 1000.0:.1f} ori_error_deg={np.rad2deg(ori_error):.1f} "
        f"lengthwise={orientation_ok} passed={passed}"
    )
    return passed


def inspect_workspace(context) -> bool:
    robot = context.services["robot"]
    joints = robot.get_joint_positions()
    valid = joints is not None and bool(np.all(np.isfinite(np.asarray(joints))))
    print(f"[BT INSPECTION] robot joint state is {'valid' if valid else 'invalid'}")
    return valid


def _set_attr_safe(api, create_name: str, get_name: str, value) -> bool:
    try:
        getter = getattr(api, get_name, None)
        attr = getter() if callable(getter) else None
        if not attr:
            creator = getattr(api, create_name)
            try:
                attr = creator(value)
            except TypeError:
                attr = creator()
        attr.Set(value)
        return True
    except Exception:
        return False


def _quat_angle_error(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest rotation angle (rad) between two orientations."""

    dot = abs(float(np.dot(_normalize_quat(a), _normalize_quat(b))))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def configure_grasp_physics(stage, profile: dict) -> None:
    material = UsdShade.Material.Define(stage, Sdf.Path("/World/GripPhysicsMaterial"))
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    _set_attr_safe(material_api, "CreateStaticFrictionAttr", "GetStaticFrictionAttr", 3.0)
    _set_attr_safe(material_api, "CreateDynamicFrictionAttr", "GetDynamicFrictionAttr", 2.5)
    _set_attr_safe(material_api, "CreateRestitutionAttr", "GetRestitutionAttr", 0.0)

    bind_paths = [TARGET_PRIM_PATH, *profile.get("finger_geometry_paths", ())]
    for path in bind_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            try:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            except Exception as exc:
                print(f"[BT PHYSICS] material bind warning for {path}: {exc}")

    connector_prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
    if connector_prim and connector_prim.IsValid():
        mass_api = UsdPhysics.MassAPI.Apply(connector_prim)
        _set_attr_safe(mass_api, "CreateMassAttr", "GetMassAttr", 0.008)
        # Keep COM low so the upright rectangle resists tipping.
        _set_attr_safe(
            mass_api,
            "CreateCenterOfMassAttr",
            "GetCenterOfMassAttr",
            Gf.Vec3f(0.0, 0.0, -0.015),
        )
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(connector_prim)
        _set_attr_safe(rigid_api, "CreateSolverPositionIterationCountAttr", "GetSolverPositionIterationCountAttr", 48)
        _set_attr_safe(rigid_api, "CreateSolverVelocityIterationCountAttr", "GetSolverVelocityIterationCountAttr", 12)
        _set_attr_safe(rigid_api, "CreateAngularDampingAttr", "GetAngularDampingAttr", 25.0)
        _set_attr_safe(rigid_api, "CreateLinearDampingAttr", "GetLinearDampingAttr", 2.0)

    for joint_path in profile.get("finger_joint_paths", ()):
        prim = stage.GetPrimAtPath(joint_path)
        if prim and prim.IsValid():
            drive = UsdPhysics.DriveAPI.Get(prim, "linear") or UsdPhysics.DriveAPI.Apply(prim, "linear")
            _set_attr_safe(drive, "CreateStiffnessAttr", "GetStiffnessAttr", 2.0e5)
            _set_attr_safe(drive, "CreateDampingAttr", "GetDampingAttr", 2.0e4)
            _set_attr_safe(drive, "CreateMaxForceAttr", "GetMaxForceAttr", 5000.0)


def spawn_elevated_outward_port(world) -> None:
    """Horizontal lengthwise slot; mouth faces +Y, block inserts along −Y."""

    axis = INSERT_AXIS_WORLD / float(np.linalg.norm(INSERT_AXIS_WORLD))
    lateral = np.array([-axis[1], axis[0], 0.0], dtype=np.float64)
    lateral = lateral / float(np.linalg.norm(lateral))
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Slot sized for lengthwise block: 0.055 m along Y, ~0.014 × 0.016 cross-section.
    aperture_half_w = 0.011   # X half-width
    aperture_half_h = 0.012   # Z half-height
    depth = 0.034             # Y half-depth (full ~0.068 m)
    wall = 0.010
    outer_half_w = aperture_half_w + wall

    pedestal_top = float(PORT_CENTER[2] - aperture_half_h - wall - 0.01)
    pedestal_height = max(0.05, pedestal_top)
    world.scene.add(FixedCuboid(
        name="port_pedestal",
        position=np.array([PORT_CENTER[0], PORT_CENTER[1] + axis[1] * 0.01, pedestal_height * 0.5]),
        prim_path="/World/PortPedestal",
        scale=np.array([0.10, 0.12, pedestal_height]),
        size=1.0,
        color=np.array([0.35, 0.35, 0.38]),
    ))

    back = PORT_CENTER + axis * (0.5 * 2.0 * depth)
    floor = PORT_CENTER - up * (aperture_half_h + 0.5 * wall)
    ceiling = PORT_CENTER + up * (aperture_half_h + 0.5 * wall)
    left = PORT_CENTER + lateral * (aperture_half_w + 0.5 * wall)
    right = PORT_CENTER - lateral * (aperture_half_w + 0.5 * wall)

    world.scene.add(FixedCuboid(
        name="port_back",
        position=back,
        prim_path="/World/PortBack",
        scale=np.array([2.0 * outer_half_w, wall, 2.0 * aperture_half_h + 2.0 * wall]),
        size=1.0,
        color=np.array([0.15, 0.75, 0.25]),
    ))
    world.scene.add(FixedCuboid(
        name="port_floor",
        position=floor,
        prim_path="/World/PortFloor",
        scale=np.array([2.0 * outer_half_w, 2.0 * depth, wall]),
        size=1.0,
        color=np.array([0.85, 0.2, 0.12]),
    ))
    world.scene.add(FixedCuboid(
        name="port_ceiling",
        position=ceiling,
        prim_path="/World/PortCeiling",
        scale=np.array([2.0 * outer_half_w, 2.0 * depth, wall]),
        size=1.0,
        color=np.array([0.85, 0.2, 0.12]),
    ))
    world.scene.add(FixedCuboid(
        name="port_left",
        position=left,
        prim_path="/World/PortLeft",
        scale=np.array([wall, 2.0 * depth, 2.0 * aperture_half_h]),
        size=1.0,
        color=np.array([0.85, 0.2, 0.12]),
    ))
    world.scene.add(FixedCuboid(
        name="port_right",
        position=right,
        prim_path="/World/PortRight",
        scale=np.array([wall, 2.0 * depth, 2.0 * aperture_half_h]),
        size=1.0,
        color=np.array([0.85, 0.2, 0.12]),
    ))
    print(
        f"[BT DEMO] Lengthwise slot at {np.round(PORT_CENTER, 4)} "
        f"(mouth +Y, insert_axis={np.round(axis, 3)}, depth≈{2.0 * depth:.3f}m)"
    )


def resolve_ur10e_usd() -> str:
    """Resolve the UR10e (+ Robotiq gripper) USD path."""

    assets_root = get_assets_root_path() or ""
    if ARGS.robot_usd is not None:
        return str(Path(ARGS.robot_usd).expanduser())
    if os.environ.get("ISAACSIM_ROBOT_USD"):
        return str(Path(os.environ["ISAACSIM_ROBOT_USD"]).expanduser())
    if not assets_root:
        raise RuntimeError(
            "Could not resolve Isaac assets root. Pass --robot-usd /path/to/ur10e.usd "
            "or set ISAACSIM_ROBOT_USD."
        )
    # Prefer the sample USD that includes a Robotiq 2F gripper.
    return (
        assets_root
        + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
    )


def _ur10e_home_positions(robot) -> np.ndarray:
    """Build a full DOF vector with an upright arm and open gripper."""

    dof_names = list(robot.dof_names)
    home = np.zeros(len(dof_names), dtype=np.float64)
    for name, value in zip(UR10E_ARM_JOINT_NAMES, UR10E_HOME_ARM):
        if name in dof_names:
            home[dof_names.index(name)] = float(value)
        else:
            print(f"[BT DEMO] Warning: missing arm joint {name!r} in {dof_names}")
    # Robotiq sample uses degrees: 0 = open.
    if "finger_joint" in dof_names:
        home[dof_names.index("finger_joint")] = 0.0
    return home


def apply_ur10e_home_pose(robot) -> None:
    """Force the UR10e into a ready pose instead of the flat zero configuration."""

    home = _ur10e_home_positions(robot)
    try:
        robot.set_joints_default_state(positions=home)
    except Exception as exc:
        print(f"[BT DEMO] set_joints_default_state warning: {exc}")
    try:
        robot.set_joint_positions(home)
    except Exception as exc:
        print(f"[BT DEMO] set_joint_positions warning: {exc}")
    print(f"[BT DEMO] UR10e home pose applied (arm rad={np.round(UR10E_HOME_ARM, 3)})")


def build_scene():
    world = World(stage_units_in_meters=1.0)
    world.set_simulation_dt(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    stage = omni.usd.get_context().get_stage()

    light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    light.CreateIntensityAttr(700.0)

    world.scene.add(FixedCuboid(
        name="ground",
        position=np.array([0.0, 0.0, -0.005]),
        prim_path="/World/Ground",
        scale=np.array([10.0, 10.0, 0.01]),
        size=1.0,
        color=np.array([0.18, 0.18, 0.18]),
    ))
    spawn_elevated_outward_port(world)
    connector = world.scene.add(DynamicCuboid(
        name="behaviour_tree_connector",
        position=CONNECTOR_SPAWN,
        prim_path=TARGET_PRIM_PATH,
        scale=CONNECTOR_SCALE,
        size=1.0,
        color=np.array([0.15, 0.35, 1.0]),
    ))

    profile = dict(UR10E_PROFILE)
    usd_path = resolve_ur10e_usd()
    prim_path = profile["prim_path"]
    print(f"[BT DEMO] Using UR10e USD: {usd_path}")
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    robotiq_path = f"{prim_path}/ee_link/robotiq_arg2f_base_link"
    robotiq = stage.GetPrimAtPath(robotiq_path)
    tool0 = stage.GetPrimAtPath(f"{prim_path}/tool0")
    if robotiq and robotiq.IsValid():
        ee_prim_path = robotiq_path
    else:
        print("[BT DEMO] Warning: Robotiq gripper prim missing; using ee_link/tool0.")
        ee_prim_path = f"{prim_path}/tool0" if tool0 and tool0.IsValid() else f"{prim_path}/ee_link"
    # Weld/grasp frame follows Lula targets (tool0) so carried poses match waypoints.
    # Resolve after reference load; do not require IsValid yet (prim may hydrate on reset).
    gripper_attach_path = f"{prim_path}/tool0"
    print(f"[BT DEMO] Grasp weld frame: {gripper_attach_path}")
    gripper = ParallelGripper(
        end_effector_prim_path=ee_prim_path,
        joint_prim_names=["finger_joint"],
        joint_opened_positions=np.array([0.0]),
        joint_closed_positions=np.array([ROBOTIQ_CLOSED_RAD]),
        action_deltas=np.array([-ROBOTIQ_CLOSED_RAD]),
        use_mimic_joints=True,
    )
    ee_frame = UR10E_EE_FRAME

    robot = world.scene.add(SingleManipulator(
        prim_path=prim_path,
        name="behaviour_tree_insertion_ur10e",
        end_effector_prim_path=ee_prim_path,
        gripper=gripper,
    ))
    configure_grasp_physics(stage, profile)
    try:
        robot.gripper.set_default_state(robot.gripper.joint_opened_positions)
    except Exception as exc:
        print(f"[BT DEMO] gripper default state warning: {exc}")
    world.reset()
    apply_ur10e_home_pose(robot)
    # One more reset so PhysX / articulation views pick up the default state.
    try:
        robot.set_joints_default_state(positions=_ur10e_home_positions(robot))
    except Exception:
        pass
    world.reset()
    apply_ur10e_home_pose(robot)
    seat_connector_upright(connector)

    config = interface_config_loader.load_supported_lula_kinematics_solver_config(UR10E_LULA_NAME)
    if config is None:
        raise RuntimeError(f"No Lula kinematics config for {UR10E_LULA_NAME!r}")
    kinematics = LulaKinematicsSolver(**config)
    trajectory_generator = LulaTaskSpaceTrajectoryGenerator(**config)
    articulation_kinematics = ArticulationKinematicsSolver(robot, kinematics, ee_frame)
    base_position, base_orientation = robot.get_world_pose()
    kinematics.set_robot_base_pose(base_position, base_orientation)
    controller = FrankaMotionController(
        name="behaviour_tree_insertion_controller",
        robot_articulation=robot,
        task_traj_gen=trajectory_generator,
        art_kinematics=articulation_kinematics,
        gripper=robot.gripper,
        tool_offset=0.0,  # waypoints are explicit tool0 poses; do not re-apply tip offset
        physics_dt=1.0 / 120.0,
        ee_frame=ee_frame,
        debug=True,
    )
    print(f"[BT DEMO] Robot=UR10e ee_frame={ee_frame} tool_offset={UR10E_TOOL_OFFSET}")
    return world, stage, robot, controller, connector, profile, gripper_attach_path


def _hold_gui(seconds: float = 8.0) -> None:
    """Keep the Isaac window open briefly so startup failures are visible."""

    if ARGS.headless:
        return
    import time

    print(f"[BT DEMO] Holding GUI open for {seconds:.0f}s…")
    deadline = time.time() + seconds
    while simulation_app.is_running() and time.time() < deadline:
        simulation_app.update()


def main() -> int:
    json_path = (ARGS.json or THIS_DIR / "demo_task_intelligence.json").expanduser().resolve()
    print(f"[BT DEMO] Loading: {json_path}")
    payload = load_task_intelligence(json_path)
    try:
        world, stage, robot, controller, connector, profile, gripper_attach_path = build_scene()
    except Exception:
        import traceback

        traceback.print_exc()
        print("[BT DEMO FAIL] Scene setup failed (see traceback above).")
        _hold_gui(10.0)
        simulation_app.close()
        return 3

    registry = {
        "navigate_to_workspace": controller_primitive(queue_move),
        "perceive_objects": function_primitive(target_exists),
        "grasp_object": controller_primitive(
            queue_grasp, validate=check_physical_grasp, while_running=maintain_grasp_approach
        ),
        "grasp_tool": controller_primitive(
            queue_grasp, validate=check_physical_grasp, while_running=maintain_grasp_approach
        ),
        "manipulate_object": controller_primitive(
            queue_manipulate, validate=connector_at_port, while_running=maintain_insert_release
        ),
        "trace_linear_path": controller_primitive(queue_move),
        "inspect_workspace": function_primitive(inspect_workspace),
        "verify_connector_at_port": function_primitive(connector_at_port),
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
            "stage": stage,
            "robot": robot,
            "connector": connector,
            "robot_profile": profile,
            "finger_contact_min": CONNECTOR_SCALE[0] * 0.5 - 0.0005,
            "motion_controller": controller,
            "articulation_controller": robot.get_articulation_controller(),
            "gripper_attach_path": gripper_attach_path,
        },
    )

    print("\n[BT STRUCTURE]\n" + tree.render_tree() + "\n")

    world.play()
    warmup_frames = 40
    frame = 0
    result = Status.RUNNING
    home_reapplied = False
    while simulation_app.is_running() and frame < max(1, ARGS.max_frames):
        world.step(render=not ARGS.headless)
        if not world.is_playing():
            continue
        frame += 1
        if not home_reapplied:
            apply_ur10e_home_pose(robot)
            seat_connector_upright(connector)
            home_reapplied = True
        if frame <= warmup_frames:
            if frame % 10 == 0:
                seat_connector_upright(connector)
            continue
        result = tree.tick()
        if result in (Status.SUCCESS, Status.FAILURE):
            break

    if result is Status.SUCCESS:
        print(f"[BT DEMO PASS] Completed {tree.step_index} generated steps in {frame} frames")
        exit_code = 0
    elif result is Status.FAILURE:
        print(f"[BT DEMO FAIL] {tree.feedback}")
        exit_code = 1
    else:
        print(f"[BT DEMO FAIL] Timed out after {frame} frames; {tree.feedback}")
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
        print("[BT DEMO FAIL] Unhandled exception (see traceback above).")
        try:
            _hold_gui(10.0)
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass
        raise
