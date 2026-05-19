from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import time
import typing
import numpy as np
import omni.usd

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    ArticulationMotionPolicy,
    LulaKinematicsSolver,
    RmpFlow,
    interface_config_loader,
)
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config


USD_PATH = "C:/Users/aayus/Desktop/Jonathan_Arun_Test/frana_clean.usd"

# Existing Franka already in the scene.
FRANKA_PRIM_PATH: typing.Optional[str] = None
FRANKA_PRIM_PATH_CANDIDATES = (
    "/World/Franka_01",
    "/World/Franka",
    "/World/panda",
    "/panda",
)

# Set these if you want to override the robot pose from the USD/stage.
ROBOT_WORLD_POSITION: typing.Optional[np.ndarray] = np.array([20.0, -270.0, 140.0], dtype=np.float64)
ROBOT_WORLD_ORIENTATION: typing.Optional[np.ndarray] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # 180 deg about Z, w/x/y/z

# Use the hand frame instead of a single finger link to avoid a built-in TCP offset.
EE_FRAME = "panda_hand"
EE_LINK_PRIM_PATH: typing.Optional[str] = None

# Only the payload cube (the object actually picked up) is authored now.
PAYLOAD_PRIM_PATH = "/World/DemoPayload"
PAYLOAD_CUBE_PATH = "/World/DemoPayload/visual_cube"
PAYLOAD_CUBE_MATERIAL_PATH = "/World/Looks/DemoPayloadBlue"
PAYLOAD_FIXED_JOINT_PATH = "/World/DemoPayloadFixedJoint"

FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

# If you want to place to a real rack port, set USE_REAL_PORT_TARGET = True.
USE_REAL_PORT_TARGET = False
PORT_INDEX = 0

NUM_QUADS = 4
NUM_PAIRS = 4
NUM_CONNECTORS = 2
PORT_BASE_PRIM_PATH = (
    "/World/Equipment/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/"
    "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01"
)

# Relative demo positions when not using a real port.
USE_ABSOLUTE_PICK_PLACE = False
PICK_OFFSET = np.array([0.55, 0.00, 0.33], dtype=np.float64)
PLACE_OFFSET = np.array([-0.45, -0.40, 0.65], dtype=np.float64)

# Small stand-off above the object/port before descending.
APPROACH_OFFSET = np.array([0.0, 0.0, 0.10], dtype=np.float64)

# Visual payload follows just below the hand center during carry.
ATTACHED_PAYLOAD_OFFSET = np.array([0.0, 0.0, -0.035], dtype=np.float64)

# Fixed hand orientation for the demo.
EE_TARGET_EULER_RAD = np.array([0.0, np.pi, 0.0], dtype=np.float64)

TARGET_TOLERANCE = 0.02
GRASP_HOLD_STEPS = 25
RELEASE_HOLD_STEPS = 25
PHASE_MAX_STEPS = 180


def wait_for_stage_load(num_updates: int = 120):
    for _ in range(num_updates):
        simulation_app.update()
        time.sleep(0.01)


def build_port_path_list():
    port_paths = []
    for quad_idx in range(1, NUM_QUADS + 1):
        for pair_idx in range(1, NUM_PAIRS + 1):
            for connector_idx in range(1, NUM_CONNECTORS + 1):
                port_paths.append(
                    PORT_BASE_PRIM_PATH
                    + f"/Connector_Quad_{quad_idx:02d}"
                    + f"/Connector_Pair_{pair_idx:02d}"
                    + f"/QSFP_DD_Connector_A_{connector_idx:02d}"
                )
    return port_paths


def resolve_franka_prim_path(stage: Usd.Stage) -> str:
    if FRANKA_PRIM_PATH is not None:
        prim = stage.GetPrimAtPath(FRANKA_PRIM_PATH)
        if not prim.IsValid():
            raise RuntimeError(f"FRANKA_PRIM_PATH does not exist: {FRANKA_PRIM_PATH}")
        return FRANKA_PRIM_PATH

    roots = []
    for prim_path in FRANKA_PRIM_PATH_CANDIDATES:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid() and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim_path

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            path = str(prim.GetPath())
            roots.append(path)
            if "franka" in path.lower() or "panda" in path.lower():
                return path

    if roots:
        raise RuntimeError(
            "Could not auto-detect Franka articulation root. Found:\n  " + "\n  ".join(roots)
        )
    raise RuntimeError("No articulation roots found on stage.")


def get_world_position(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(world_tf.ExtractTranslation(), dtype=np.float64)


def define_colored_cube(
    stage: Usd.Stage,
    xform_path: str,
    cube_path: str,
    material_path: str,
    position: np.ndarray,
    size: float,
    color: typing.Tuple[float, float, float],
):
    if stage.GetPrimAtPath(xform_path).IsValid():
        stage.RemovePrim(Sdf.Path(xform_path))

    xform_prim = stage.GetPrimAtPath(xform_path)
    if not xform_prim.IsValid():
        stage.DefinePrim(xform_path, "Xform")

    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.CreateSizeAttr(size)
    cube_xformable = UsdGeom.Xformable(cube.GetPrim())
    cube_xformable.ClearXformOpOrder()
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    set_xform_translation(stage, xform_path, position)

    payload_prim = stage.GetPrimAtPath(xform_path)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(payload_prim)
    rigid_body.CreateKinematicEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(payload_prim)
    mass_api.CreateMassAttr(0.02)


def set_xform_translation(stage: Usd.Stage, prim_path: str, position: np.ndarray):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    translate_ops = [
        op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    vec = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    if translate_ops:
        translate_ops[0].Set(vec)
    else:
        xformable.AddTranslateOp().Set(vec)


def move_payload(stage: Usd.Stage, position: np.ndarray):
    set_xform_translation(stage, PAYLOAD_PRIM_PATH, position)
    # Keep the visible child cube local to the payload parent, even if a stale xform was authored previously.
    cube_prim = stage.GetPrimAtPath(PAYLOAD_CUBE_PATH)
    if cube_prim.IsValid():
        cube_xformable = UsdGeom.Xformable(cube_prim)
        cube_xformable.ClearXformOpOrder()


def set_payload_kinematic(stage: Usd.Stage, enabled: bool):
    payload_prim = stage.GetPrimAtPath(PAYLOAD_PRIM_PATH)
    if not payload_prim.IsValid():
        raise RuntimeError(f"Payload prim not found: {PAYLOAD_PRIM_PATH}")
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(payload_prim)
    attr = rigid_body.GetKinematicEnabledAttr()
    if not attr:
        attr = rigid_body.CreateKinematicEnabledAttr()
    attr.Set(bool(enabled))


def resolve_ee_link_prim_path(stage: Usd.Stage, franka_path: str) -> str:
    if EE_LINK_PRIM_PATH is not None:
        prim = stage.GetPrimAtPath(EE_LINK_PRIM_PATH)
        if not prim.IsValid():
            raise RuntimeError(f"EE_LINK_PRIM_PATH does not exist: {EE_LINK_PRIM_PATH}")
        return EE_LINK_PRIM_PATH

    root = stage.GetPrimAtPath(franka_path)
    if not root.IsValid():
        raise RuntimeError(f"Franka root not found: {franka_path}")

    matches = []
    for prim in Usd.PrimRange(root):
        if prim.GetName() == EE_FRAME:
            matches.append(str(prim.GetPath()))

    if not matches:
        raise RuntimeError(f"Could not find end-effector link named {EE_FRAME!r} under {franka_path}")

    return matches[0]


def attach_payload_to_hand(stage: Usd.Stage, hand_link_path: str):
    if stage.GetPrimAtPath(PAYLOAD_FIXED_JOINT_PATH).IsValid():
        return

    set_payload_kinematic(stage, False)
    joint = UsdPhysics.FixedJoint.Define(stage, PAYLOAD_FIXED_JOINT_PATH)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_link_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(PAYLOAD_PRIM_PATH)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    print(f"Attached payload with fixed joint: {hand_link_path} -> {PAYLOAD_PRIM_PATH}")


def detach_payload_from_hand(stage: Usd.Stage, release_position: np.ndarray):
    if stage.GetPrimAtPath(PAYLOAD_FIXED_JOINT_PATH).IsValid():
        stage.RemovePrim(Sdf.Path(PAYLOAD_FIXED_JOINT_PATH))
        print("Detached payload fixed joint")
    move_payload(stage, release_position)
    set_payload_kinematic(stage, True)


def resolve_finger_joint_indices(robot: SingleArticulation):
    dof_names = list(robot.dof_names)
    indices = []
    for joint_name in FINGER_JOINT_NAMES:
        if joint_name not in dof_names:
            raise RuntimeError(
                f"Could not find finger joint '{joint_name}'. Available DOFs: {dof_names}"
            )
        indices.append(dof_names.index(joint_name))
    return np.array(indices, dtype=np.int64)


def set_gripper(robot: SingleArticulation, finger_joint_indices: np.ndarray, width: float):
    action = ArticulationAction(
        joint_positions=np.array([width, width], dtype=np.float64),
        joint_indices=finger_joint_indices,
    )
    robot.apply_action(action)


def build_phase_sequence(pick_position: np.ndarray, place_position: np.ndarray):
    return [
        {
            "name": "pre_pick",
            "target": pick_position + APPROACH_OFFSET,
            "gripper": GRIPPER_OPEN,
            "attach": False,
            "max_steps": PHASE_MAX_STEPS,
        },
        {
            "name": "pick",
            "target": pick_position,
            "gripper": GRIPPER_OPEN,
            "attach": False,
            "max_steps": PHASE_MAX_STEPS,
        },
        {
            "name": "grasp",
            "target": pick_position,
            "gripper": GRIPPER_CLOSED,
            "attach": True,
            "snap_payload": True,
            "hold_steps": GRASP_HOLD_STEPS,
            "max_steps": GRASP_HOLD_STEPS,
        },
        {
            "name": "lift",
            "target": pick_position + APPROACH_OFFSET,
            "gripper": GRIPPER_CLOSED,
            "attach": True,
            "max_steps": PHASE_MAX_STEPS,
        },
        {
            "name": "pre_place",
            "target": place_position + APPROACH_OFFSET,
            "gripper": GRIPPER_CLOSED,
            "attach": True,
            "max_steps": PHASE_MAX_STEPS,
        },
        {
            "name": "place",
            "target": place_position,
            "gripper": GRIPPER_CLOSED,
            "attach": True,
            "max_steps": PHASE_MAX_STEPS,
        },
        {
            "name": "release",
            "target": place_position,
            "gripper": GRIPPER_OPEN,
            "attach": False,
            "hold_steps": RELEASE_HOLD_STEPS,
            "max_steps": RELEASE_HOLD_STEPS,
        },
        {
            "name": "retreat",
            "target": place_position + APPROACH_OFFSET,
            "gripper": GRIPPER_OPEN,
            "attach": False,
            "max_steps": PHASE_MAX_STEPS,
        },
    ]


def build_place_positions(stage: Usd.Stage, base_pos: np.ndarray):
    if USE_REAL_PORT_TARGET:
        port_paths = build_port_path_list()
        if PORT_INDEX < 0 or PORT_INDEX >= len(port_paths):
            raise RuntimeError(f"PORT_INDEX {PORT_INDEX} out of range for {len(port_paths)} ports.")
        print("Using rack port target:", port_paths[PORT_INDEX])
        return [get_world_position(stage, port_paths[PORT_INDEX])]
    return [base_pos + PLACE_OFFSET]


def main():
    print(f"Opening stage: {USD_PATH}")
    omni.usd.get_context().open_stage(USD_PATH)
    wait_for_stage_load()

    stage = omni.usd.get_context().get_stage()
    world = World(stage_units_in_meters=1.0)
    world.reset()

    franka_path = resolve_franka_prim_path(stage)
    print("Using Franka articulation prim:", franka_path)
    hand_link_path = resolve_ee_link_prim_path(stage, franka_path)
    print("Using EE link prim for payload joint:", hand_link_path)

    franka = SingleArticulation(prim_path=franka_path, name="franka")
    franka.initialize()
    finger_joint_indices = resolve_finger_joint_indices(franka)
    print("Finger joint indices:", finger_joint_indices)
    print("DOF names:", list(franka.dof_names))

    if ROBOT_WORLD_POSITION is not None:
        current_position, current_orientation = franka.get_world_pose()
        target_orientation = (
            np.asarray(ROBOT_WORLD_ORIENTATION, dtype=np.float64)
            if ROBOT_WORLD_ORIENTATION is not None
            else np.asarray(current_orientation, dtype=np.float64)
        )
        franka.set_world_pose(
            position=np.asarray(ROBOT_WORLD_POSITION, dtype=np.float64),
            orientation=target_orientation,
        )
        print("Applied robot world pose override:", ROBOT_WORLD_POSITION, target_orientation)

    rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")
    rmp_config["end_effector_frame_name"] = EE_FRAME
    rmpflow = RmpFlow(**rmp_config)
    articulation_rmpflow = ArticulationMotionPolicy(franka, rmpflow)

    ik_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    lula_solver = LulaKinematicsSolver(**ik_config)
    ik_solver = ArticulationKinematicsSolver(franka, lula_solver, EE_FRAME)

    for _ in range(20):
        world.step(render=True)

    world.play()

    base_pos, base_quat = franka.get_world_pose()
    base_pos = np.array(base_pos, dtype=np.float64)
    print("Robot base world position:", base_pos)

    pick_position = base_pos + PICK_OFFSET
    place_positions = build_place_positions(stage, base_pos)
    place_position = place_positions[0]

    print("Pick position:", pick_position)
    print("Place positions:", place_positions)

    # Only the payload cube (the thing being picked up) is authored.
    # The old green RmpFlowTarget and red PlaceMarker debug cubes have been removed.
    define_colored_cube(
        stage,
        PAYLOAD_PRIM_PATH,
        PAYLOAD_CUBE_PATH,
        PAYLOAD_CUBE_MATERIAL_PATH,
        pick_position,
        0.05,
        (0.1, 0.4, 1.0),
    )

    target_orientation = euler_angles_to_quats(EE_TARGET_EULER_RAD)
    phases = build_phase_sequence(pick_position, place_position)
    phase_index = 0
    place_index = 0
    hold_counter = 0
    phase_step_counter = 0
    attached = False
    released = False
    active_phase_name = None
    released_payload_position = pick_position.copy()
    completed = False

    while simulation_app.is_running():
        dt = world.get_physics_dt()
        phase = phases[phase_index]
        desired_target = np.asarray(phase["target"], dtype=np.float64)
        if phase["name"] != active_phase_name:
            active_phase_name = phase["name"]
            print(f"Starting phase: {active_phase_name}")
            phase_step_counter = 0
            if phase.get("snap_payload", False):
                attached = True
                released = False
                move_payload(stage, desired_target + ATTACHED_PAYLOAD_OFFSET)
                attach_payload_to_hand(stage, hand_link_path)

        base_pos, base_quat = franka.get_world_pose()
        base_pos = np.array(base_pos, dtype=np.float64)
        rmpflow.set_robot_base_pose(base_pos, base_quat)
        lula_solver.set_robot_base_pose(base_pos, base_quat)

        rmpflow.set_end_effector_target(desired_target, target_orientation)
        rmpflow.update_world()

        arm_action = articulation_rmpflow.get_next_articulation_action(dt)
        franka.apply_action(arm_action)
        set_gripper(franka, finger_joint_indices, phase["gripper"])

        ee_position, _ = ik_solver.compute_end_effector_pose()
        ee_position = np.asarray(ee_position, dtype=np.float64).reshape(3)

        if phase.get("attach", False):
            attached = True
            released_payload_position = desired_target + ATTACHED_PAYLOAD_OFFSET
        elif phase["name"] == "release":
            attached = False
            released_payload_position = place_position.copy()
            if not released:
                detach_payload_from_hand(stage, released_payload_position)
                released = True

        if attached and not stage.GetPrimAtPath(PAYLOAD_FIXED_JOINT_PATH).IsValid():
            move_payload(stage, desired_target + ATTACHED_PAYLOAD_OFFSET)
        elif not attached:
            move_payload(stage, released_payload_position)

        distance = float(np.linalg.norm(ee_position - desired_target))
        phase_step_counter += 1
        if distance < TARGET_TOLERANCE:
            hold_counter += 1
        else:
            hold_counter = 0

        reached_target = hold_counter >= phase.get("hold_steps", 1)
        timed_out = phase_step_counter >= phase.get("max_steps", PHASE_MAX_STEPS)
        if reached_target or timed_out:
            reason = "target reached" if reached_target else "timed fallback"
            print(f"Completed phase: {phase['name']} ({reason}, distance={distance:.4f})")
            phase_index += 1
            hold_counter = 0
            phase_step_counter = 0
            if phase_index >= len(phases):
                place_index += 1
                if place_index >= len(place_positions):
                    completed = True
                    break

                place_position = place_positions[place_index]
                phases = build_phase_sequence(pick_position, place_position)
                phase_index = 0
                attached = False
                released = False
                active_phase_name = None
                released_payload_position = pick_position.copy()
                detach_payload_from_hand(stage, released_payload_position)
                move_payload(stage, pick_position)
                print(f"Advancing to next place target: {place_position}")

        world.step(render=True)

    if completed:
        print("Pick-and-place sequence complete. Leaving sim open.")

    while simulation_app.is_running():
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        simulation_app.close()
