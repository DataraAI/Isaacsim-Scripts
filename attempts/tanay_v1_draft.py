import numpy as np

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    load_supported_motion_policy_config,
)
import omni.usd
if World.instance():
    World.instance().clear_instance()
# ============================================================
# USER SETUP: CHANGE THESE
# ============================================================

FRANKA_PATH = "/World/Franka"

# One entry per cable -> port assignment
TASKS = [
    {
        "wire_root": "/World/Wires/wire_01",
        "port_preinsert": "/World/Targets/QSFP_DD_Connector_01_preinsert",
        "port_insert": "/World/Targets/QSFP_DD_Connector_01_insert",
    },
]

POSITION_TOL = 0.01         # meters
ORIENTATION_DOT_TOL = 0.995 # quaternion similarity
GRIPPER_SETTLE_STEPS = 20   # frames to wait after close/open
PAUSE_WHEN_DONE = True

# ============================================================
# HELPERS
# ============================================================
from pxr import PhysxSchema

def set_wire_gravity_enabled(wire_root: str, enabled: bool):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(wire_root)

    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, wire_root)
    if not rb:
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

    attr = rb.GetDisableGravityAttr()
    if not attr:
        attr = rb.CreateDisableGravityAttr()

    attr.Set(not enabled)
def _norm_quat(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return q
    return q / n

def _quat_close(q1, q2, dot_tol=ORIENTATION_DOT_TOL):
    q1 = _norm_quat(q1)
    q2 = _norm_quat(q2)
    # q and -q are the same rotation
    return abs(np.dot(q1, q2)) >= dot_tol

def _pose_reached(curr_pos, curr_ori, tgt_pos, tgt_ori):
    return (
        np.linalg.norm(np.asarray(curr_pos) - np.asarray(tgt_pos)) < POSITION_TOL
        and _quat_close(curr_ori, tgt_ori)
    )

def _prim_exists(path: str) -> bool:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    return prim.IsValid()

# ============================================================
# CONTROLLER
# ============================================================

class CablePlugDemo:
    def __init__(self, world: World):
        self.world = world

        # Wrap the Franka already in the stage
        self.franka = world.scene.add(
            Franka(prim_path=FRANKA_PATH, name="franka_robot")
        )

        # Motion policy
        rmp_cfg = load_supported_motion_policy_config("Franka", "RMPflow")
        self.rmp = RmpFlow(**rmp_cfg)
        self.motion = ArticulationMotionPolicy(self.franka, self.rmp)

        self.frame_cache = {}
        self.tasks = TASKS
        self.task_idx = 0
        self.plan_idx = 0
        self.wait_counter = 0
        self.done = False

        self._validate_paths()

        self.plan = self._build_plan(self.tasks[0])

    def _validate_paths(self):
        needed = [FRANKA_PATH]
        for task in self.tasks:
            wr = task["wire_root"]
            ppre = task["port_preinsert"]
            pins = task["port_insert"]

            needed.extend(
                [
                    wr,
                    f"{wr}/pregrasp_frame",
                    f"{wr}/grasp_frame",
                    f"{wr}/lift_frame",
                    ppre,
                    pins,
                ]
            )

        missing = [p for p in needed if not _prim_exists(p)]
        if missing:
            raise RuntimeError(
                "These prim paths do not exist in the stage:\n" + "\n".join(missing)
            )
    def _frame(self, path: str) -> XFormPrim:
        if path not in self.frame_cache:
            self.frame_cache[path] = XFormPrim(path)
        return self.frame_cache[path]

    def _build_plan(self, task):
    	wr = task["wire_root"]
    	ppre = task["port_preinsert"]
    	pins = task["port_insert"]

    	return [
        	("move", f"{wr}/pregrasp_frame"),
        	("wait", 10),
        	("move", f"{wr}/grasp_frame"),
        	("wait", 5),
        	("close", None),
        	("wait", GRIPPER_SETTLE_STEPS),
        	("move", f"{wr}/lift_frame"),
        	("move", ppre),
        	("move", pins),
        	("open", None),
        	("wait", GRIPPER_SETTLE_STEPS),
        	("move", ppre),
    	]
    def reset(self):
        self.rmp.reset()
        self.task_idx = 0
        self.plan_idx = 0
        self.wait_counter = 0
        self.done = False
        self.plan = self._build_plan(self.tasks[0])

        self.franka.gripper.set_joint_positions(
            self.franka.gripper.joint_opened_positions
        )
        for task in self.tasks:
            set_wire_gravity_enabled(task["wire_root"], False)

    def _advance(self):
        self.plan_idx += 1
        self.wait_counter = 0

        if self.plan_idx >= len(self.plan):
            self.task_idx += 1
            if self.task_idx >= len(self.tasks):
                self.done = True
                print("All cable placement tasks finished.")
                if PAUSE_WHEN_DONE:
                    self.world.pause()
                return

            self.plan_idx = 0
            self.plan = self._build_plan(self.tasks[self.task_idx])
            print(f"Starting task {self.task_idx + 1}/{len(self.tasks)}")

    def _update_rmp_base(self):
        self.rmp.update_world()
        base_pos, base_ori = self.franka.get_world_pose()
        self.rmp.set_robot_base_pose(base_pos, base_ori)

    def _move_to_frame(self, frame_path: str, dt: float):
        tgt_pos, tgt_ori = self._frame(frame_path).get_world_pose()

        self.rmp.set_end_effector_target(tgt_pos, tgt_ori)
        action = self.motion.get_next_articulation_action(dt)
        self.franka.apply_action(action)

        ee_pos, ee_ori = self.franka.end_effector.get_world_pose()
        if _pose_reached(ee_pos, ee_ori, tgt_pos, tgt_ori):
            self._advance()

    def on_physics_step(self, dt: float):
        if self.done:
            return

        self._update_rmp_base()

        step_type, payload = self.plan[self.plan_idx]

        if step_type == "move":
            self._move_to_frame(payload, dt)
            return

        if step_type == "close":
            self.franka.gripper.set_joint_positions(
                self.franka.gripper.joint_closed_positions
            )
            self._advance()
            return

        if step_type == "open":
            self.franka.gripper.set_joint_positions(
                self.franka.gripper.joint_opened_positions
            )
            self._advance()
            return

        if step_type == "wait":
            self.wait_counter += 1
            if self.wait_counter >= payload:
                    if self.plan_idx > 0 and self.plan[self.plan_idx - 1][0] == "close":
                        current_task = self.tasks[self.task_idx]
                        set_wire_gravity_enabled(current_task["wire_root"], True)
                    self._advance()
            return


# ============================================================
# RUN
# ============================================================
import asyncio

# Keep references alive
WORLD_REF = None
DEMO_REF = None

async def setup_and_run():
    global WORLD_REF, DEMO_REF

    if World.instance():
        World.instance().clear_instance()

    world = World(stage_units_in_meters=1.0)

    # Initialize simulation context first
    await world.initialize_simulation_context_async()

    # Build controller/robot wrappers after world exists
    demo = CablePlugDemo(world)

    # Reset AFTER adding robot/controller so articulation handles initialize
    await world.reset_async()
    demo.reset()

    callback_name = "cable_plug_step"

    try:
        world.remove_physics_callback(callback_name)
    except Exception:
        pass

    world.add_physics_callback(callback_name, demo.on_physics_step)

    WORLD_REF = world
    DEMO_REF = demo

    print("Cable plug controller loaded.")
    print("Starting simulation...")

    await world.play_async()

asyncio.ensure_future(setup_and_run())

