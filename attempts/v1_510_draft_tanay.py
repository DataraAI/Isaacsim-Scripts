from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import time
import numpy as np
import omni

from pxr import UsdGeom, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver,
    ArticulationKinematicsSolver,
    interface_config_loader,
)

# -----------------------------
# EDIT THESE
# -----------------------------
USD_PATH = r"C:\Users\aayus\Downloads\Datacenter_Files\Assets\DigitalTwin\Assets\Datacenter\Facilities\Stages\Data_Hall\DataHall_Full_01.usd"

FRANKA_PATH = "/World/Franka"
EE_FRAME = "panda_hand"
EE_PRIM_PATH = f"{FRANKA_PATH}/panda_hand"

# Keep the 4.5 baseline-style port enumeration logic.
# Adjust the base path if your 5.1 stage differs.
PORT_BASE_PRIM_PATH = (
    "/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"
)
NUM_QUADS = 4
NUM_PAIRS = 4
NUM_CONN_A = 2

# Uncomment for quick debugging
# MAX_PORTS = 2
MAX_PORTS = None

WIRE_ROOT = "/World/Wires"
TARGET_CUBE_PATH = "/World/TargetCube"

# Wire bank placement relative to robot base
WIRE_BANK_OFFSET = np.array([0.60, -0.80, 0.15], dtype=np.float64)
WIRE_COLS = 4
WIRE_COL_SPACING = 0.28
WIRE_ROW_SPACING = 0.18

# Placeholder wire size
WIRE_LENGTH = 0.22
WIRE_THICKNESS = 0.02

# Motion tuning
PICK_APPROACH_Z = 0.12
PICK_CONTACT_Z = 0.03
LIFT_Z = 0.20

PORT_APPROACH_BACKOFF = 0.18
PORT_TOUCH_BACKOFF = 0.03
PORT_APPROACH_Z_LIFT = 0.03

POSITION_TOL = 0.015
STABLE_FRAMES_REQUIRED = 12

PHASE_DURATIONS = {
    "to_pick_approach": 1.2,
    "to_pick_contact": 0.8,
    "to_lift": 1.0,
    "to_port_approach": 1.6,
    "to_port_touch": 0.9,
    "release_hold": 0.4,
}

# While carrying, keep the wire sligxhtly in front of the hand
CARRY_LOCAL_OFFSET = np.array([0.10, 0.0, 0.0], dtype=np.float64)

# -----------------------------
# HELPERS
# -----------------------------
def wait_for_stage_load():
    for _ in range(100):
        simulation_app.update()
        time.sleep(0.01)

def resolve_path(stage, requested_path: str) -> str:
    prim = stage.GetPrimAtPath(requested_path)
    if prim.IsValid():
        return requested_path

    # common fallback if stage hierarchy shifted
    if requested_path.startswith("/World/"):
        shifted = "/World" + requested_path
        prim = stage.GetPrimAtPath(shifted)
        if prim.IsValid():
            return shifted

    raise RuntimeError(f"Prim path not found: {requested_path}")

def build_port_paths():
    paths = []
    for q in range(1, NUM_QUADS + 1):
        for p in range(1, NUM_PAIRS + 1):
            for a in range(1, NUM_CONN_A + 1):
                suffix = f"/Connector_Quad_{q:02d}/Connector_Pair_{p:02d}/QSFP_DD_Connector_A_{a:02d}"
                paths.append(PORT_BASE_PRIM_PATH + suffix)
    if MAX_PORTS is not None:
        paths = paths[:MAX_PORTS]
    return paths

def quat_rotate_vector(q_wxyz, v_xyz):
    w, x, y, z = q_wxyz
    qvec = np.array([x, y, z], dtype=np.float64)
    v = np.array(v_xyz, dtype=np.float64)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return v / n

def lerp(a, b, alpha):
    return (1.0 - alpha) * a + alpha * b

def create_wire(scene, index: int, position):
    # Demo-grade rigid placeholder wire as an elongated cuboid
    wire = scene.add(
        VisualCuboid(
            prim_path=f"{WIRE_ROOT}/Wire_{index:02d}",
            name=f"wire_{index:02d}",
            position=np.array(position, dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            size=1.0,
            scale=np.array([WIRE_LENGTH, WIRE_THICKNESS, WIRE_THICKNESS], dtype=np.float64),
            color=np.array([0.1, 0.1, 0.1], dtype=np.float64),
        )
    )
    return wire

# -----------------------------
# MAIN STATE MACHINE
# -----------------------------
class Version1WireDemo:
    def __init__(self, world, robot, ee_prim, ik_solver, lula_solver, target_cube, wires, port_prims):
        self.world = world
        self.robot = robot
        self.ee_prim = ee_prim
        self.ik_solver = ik_solver
        self.lula_solver = lula_solver
        self.target_cube = target_cube
        self.wires = wires
        self.port_prims = port_prims

        self.fixed_orientation = None

        self.current_idx = 0
        self.phase = None
        self.phase_elapsed = 0.0
        self.phase_duration = 0.0
        self.phase_start = None
        self.phase_end = None

        self.carrying = False
        self.stable_counter = 0

        self.cached_pick_approach = None
        self.cached_pick_contact = None
        self.cached_lift = None
        self.cached_port_approach = None
        self.cached_port_touch = None

    def initialize(self):
        _, ee_rot = self.ee_prim.get_world_pose()
        self.fixed_orientation = np.array(ee_rot, dtype=np.float64)
        self.start_sequence_for_current_wire()

    def get_robot_base_pose(self):
        pos, quat = self.robot.get_world_pose()
        return np.array(pos, dtype=np.float64), np.array(quat, dtype=np.float64)

    def get_wire_pos(self, idx):
        pos, _ = self.wires[idx].get_world_pose()
        return np.array(pos, dtype=np.float64)

    def get_port_pos(self, idx):
        pos, _ = self.port_prims[idx].get_world_pose()
        return np.array(pos, dtype=np.float64)

    def get_ee_pose(self):
        pos, quat = self.ee_prim.get_world_pose()
        return np.array(pos, dtype=np.float64), np.array(quat, dtype=np.float64)

    def get_target_cube_pos(self):
        pos, _ = self.target_cube.get_world_pose()
        return np.array(pos, dtype=np.float64)

    def set_target_cube_pos(self, pos):
        self.target_cube.set_world_pose(
            position=np.array(pos, dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )

    def compute_wire_points(self, idx):
        wire_pos = self.get_wire_pos(idx)
        pick_approach = wire_pos + np.array([0.0, 0.0, PICK_APPROACH_Z], dtype=np.float64)
        pick_contact = wire_pos + np.array([0.0, 0.0, PICK_CONTACT_Z], dtype=np.float64)
        lift = wire_pos + np.array([0.0, 0.0, LIFT_Z], dtype=np.float64)
        return pick_approach, pick_contact, lift

    def compute_port_points(self, idx):
        robot_base_pos, _ = self.get_robot_base_pose()
        port_pos = self.get_port_pos(idx)

        direction = normalize(port_pos - robot_base_pos)

        port_approach = port_pos - PORT_APPROACH_BACKOFF * direction
        port_approach[2] += PORT_APPROACH_Z_LIFT

        port_touch = port_pos - PORT_TOUCH_BACKOFF * direction
        return port_approach, port_touch

    def begin_phase(self, phase_name, end_pos, duration):
        self.phase = phase_name
        self.phase_elapsed = 0.0
        self.phase_duration = duration
        self.phase_start = self.get_target_cube_pos()
        self.phase_end = np.array(end_pos, dtype=np.float64)
        self.stable_counter = 0
        print(f"Phase -> {phase_name} | end={self.phase_end}")

    def start_sequence_for_current_wire(self):
        if self.current_idx >= len(self.wires):
            print("All wires processed.")
            self.phase = "done"
            return

        self.cached_pick_approach, self.cached_pick_contact, self.cached_lift = self.compute_wire_points(self.current_idx)
        self.cached_port_approach, self.cached_port_touch = self.compute_port_points(self.current_idx)

        self.begin_phase("to_pick_approach", self.cached_pick_approach, PHASE_DURATIONS["to_pick_approach"])

    def advance_phase(self):
        if self.phase == "to_pick_approach":
            self.begin_phase("to_pick_contact", self.cached_pick_contact, PHASE_DURATIONS["to_pick_contact"])

        elif self.phase == "to_pick_contact":
            # fake grasp
            self.carrying = True
            self.begin_phase("to_lift", self.cached_lift, PHASE_DURATIONS["to_lift"])

        elif self.phase == "to_lift":
            self.begin_phase("to_port_approach", self.cached_port_approach, PHASE_DURATIONS["to_port_approach"])

        elif self.phase == "to_port_approach":
            self.begin_phase("to_port_touch", self.cached_port_touch, PHASE_DURATIONS["to_port_touch"])

        elif self.phase == "to_port_touch":
            # fake release: leave wire at the port touch point
            self.carrying = False
            self.wires[self.current_idx].set_world_pose(
                position=np.array(self.cached_port_touch, dtype=np.float64),
                orientation=np.array(self.fixed_orientation, dtype=np.float64),
            )
            self.begin_phase("release_hold", self.cached_port_touch, PHASE_DURATIONS["release_hold"])

        elif self.phase == "release_hold":
            self.current_idx += 1
            self.start_sequence_for_current_wire()

    def update_carrying_wire(self):
        if not self.carrying or self.phase == "done":
            return

        ee_pos, ee_quat = self.get_ee_pose()
        carry_world_offset = quat_rotate_vector(ee_quat, CARRY_LOCAL_OFFSET)
        wire_world_pos = ee_pos + carry_world_offset

        self.wires[self.current_idx].set_world_pose(
            position=np.array(wire_world_pos, dtype=np.float64),
            orientation=np.array(ee_quat, dtype=np.float64),
        )

    def update_target_motion(self, dt):
        if self.phase == "done":
            return

        self.phase_elapsed += dt
        alpha = min(self.phase_elapsed / max(self.phase_duration, 1e-6), 1.0)
        current_target = lerp(self.phase_start, self.phase_end, alpha)
        self.set_target_cube_pos(current_target)

    def step(self, dt):
        if self.phase == "done":
            return

        self.update_target_motion(dt)
        self.update_carrying_wire()

        robot_base_pos, robot_base_quat = self.get_robot_base_pose()
        self.lula_solver.set_robot_base_pose(robot_base_pos, robot_base_quat)

        target_pos = self.get_target_cube_pos()
        ee_pos, _ = self.get_ee_pose()

        action, success = self.ik_solver.compute_inverse_kinematics(
            target_position=np.array(target_pos, dtype=np.float64),
            target_orientation=np.array(self.fixed_orientation, dtype=np.float64),
        )

        if success:
            self.robot.apply_action(action)

        # advance only after target has finished moving and the hand is stably near it
        if np.linalg.norm(ee_pos - target_pos) < POSITION_TOL:
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        if self.phase_elapsed >= self.phase_duration and self.stable_counter >= STABLE_FRAMES_REQUIRED:
            self.advance_phase()

# -----------------------------
# MAIN
# -----------------------------
def main():
    print(f"Opening stage: {USD_PATH}")
    omni.usd.get_context().open_stage(USD_PATH)
    wait_for_stage_load()

    stage = omni.usd.get_context().get_stage()

    # resolve key paths
    resolved_franka_path = resolve_path(stage, FRANKA_PATH)
    resolved_ee_path = resolve_path(stage, EE_PRIM_PATH)

    raw_port_paths = build_port_paths()
    resolved_port_paths = [resolve_path(stage, p) for p in raw_port_paths]

    world = World()

    # Wrap the existing robot in the opened stage
    robot = SingleArticulation(prim_path=resolved_franka_path, name="franka")
    world.scene.add(robot)

    # Create wires near the current robot base, not the origin
    # We need the robot initialized before its world pose is reliable
    world.reset()
    robot.initialize()

    robot_base_pos, _ = robot.get_world_pose()

    # Create a visible target cube, like your 4.5 baseline logic
    target_cube = world.scene.add(
        VisualCuboid(
            prim_path=TARGET_CUBE_PATH,
            name="target_cube",
            position=np.array(robot_base_pos, dtype=np.float64) + np.array([0.2, 0.0, 0.2], dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            size=0.04,
            color=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
    )

    # Build wire bank
    wires = []
    bank_anchor = np.array(robot_base_pos, dtype=np.float64) + WIRE_BANK_OFFSET
    for i in range(len(resolved_port_paths)):
        row = i // WIRE_COLS
        col = i % WIRE_COLS
        wire_pos = bank_anchor + np.array(
            [0.0, col * WIRE_COL_SPACING, -row * WIRE_ROW_SPACING],
            dtype=np.float64,
        )
        wires.append(create_wire(world.scene, i, wire_pos))

    # Port wrappers
    port_prims = [XFormPrim(p) for p in resolved_port_paths]
    ee_prim = XFormPrim(resolved_ee_path)

    # IK setup
    config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    lula_solver = LulaKinematicsSolver(**config)
    ik_solver = ArticulationKinematicsSolver(robot, lula_solver, EE_FRAME)

    demo = Version1WireDemo(
        world=world,
        robot=robot,
        ee_prim=ee_prim,
        ik_solver=ik_solver,
        lula_solver=lula_solver,
        target_cube=target_cube,
        wires=wires,
        port_prims=port_prims,
    )
    demo.initialize()

    world.play()

    while simulation_app.is_running():
        world.step(render=True)
        dt = world.get_physics_dt()
        demo.step(dt)

    simulation_app.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        simulation_app.close()