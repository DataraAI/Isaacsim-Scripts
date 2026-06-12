from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import carb

from pxr import UsdGeom, Gf

import omni.usd
from omni.timeline import get_timeline_interface

# Core API
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path

# Robot + IK
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.controllers import DifferentialIKController


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATAHALL_USD = r"C:\Users\aayus\Downloads\Datacenter_Files\Assets\DigitalTwin\Assets\Datacenter\Facilities\Stages\Data_Hall\DataHall_Full_01.usd"

assets_root = get_assets_root_path()
FRANKA_USD = assets_root + "/Isaac/Robots/Franka/franka.usd"

ROBOT_PRIM_PATH = "/World/Franka"
TARGET_PRIM_PATH = "/World/TargetCube"

ROBOT_WORLD_POS = np.array([0.0, 0.0, 0.0])
TARGET_WORLD_POS = np.array([0.5, 0.0, 0.5])

ROBOT_SCALE = np.array([57.0, 57.0, 57.0])
TARGET_SCALE = np.array([1.0, 1.0, 1.0])

# -------------------------------------------------
# STAGE SETUP
# -------------------------------------------------
usd_context = omni.usd.get_context()
stage = usd_context.get_stage()

# Load environment
add_reference_to_stage(DATAHALL_USD, "/World/DataHall")

# Add Franka
add_reference_to_stage(FRANKA_USD, ROBOT_PRIM_PATH)

# Apply WORLD transform properly
franka_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
franka_xform = UsdGeom.Xformable(franka_prim)

franka_xform.ClearXformOpOrder()
franka_xform.AddTranslateOp().Set(Gf.Vec3d(*ROBOT_WORLD_POS))
franka_xform.AddScaleOp().Set(Gf.Vec3f(*ROBOT_SCALE))

# Create target cube
cube = UsdGeom.Cube.Define(stage, TARGET_PRIM_PATH)
cube_xform = UsdGeom.Xformable(cube.GetPrim())

cube_xform.ClearXformOpOrder()
cube_xform.AddTranslateOp().Set(Gf.Vec3d(*TARGET_WORLD_POS))
cube_xform.AddScaleOp().Set(Gf.Vec3f(*TARGET_SCALE))

cube.CreateDisplayColorAttr().Set([(1.0, 0.0, 0.0)])

# -------------------------------------------------
# WORLD + ROBOT
# -------------------------------------------------
world = World(stage_units_in_meters=1.0)
world.reset()

robot = SingleManipulator(
    prim_path=ROBOT_PRIM_PATH,
    name="franka",
    end_effector_prim_name="panda_hand"
)

world.scene.add(robot)

ik_controller = DifferentialIKController(
    name="ik_controller",
    robot_articulation=robot
)

# -------------------------------------------------
# TIMELINE CONTROL (START STOPPED)
# -------------------------------------------------
timeline = get_timeline_interface()
timeline.stop()

print("Simulation ready. Press PLAY in Isaac Sim.")

# -------------------------------------------------
# LOOP
# -------------------------------------------------
warning_printed = False

while simulation_app.is_running():

    # Step render always so UI updates
    world.step(render=True)

    if timeline.is_playing():

        # Get target WORLD pose
        cube_prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
        cube_xform = UsdGeom.Xformable(cube_prim)
        world_tf = cube_xform.ComputeLocalToWorldTransform(0)

        target_pos = world_tf.ExtractTranslation()

        # IK compute
        result = ik_controller.compute(
            target_position=np.array([target_pos[0], target_pos[1], target_pos[2]]),
            target_orientation=None
        )

        # Check convergence (API returns None or invalid if fails)
        if result is not None:
            robot.apply_action(result)
            warning_printed = False
        else:
            if not warning_printed:
                carb.log_warn("IK did not converge to target.")
                warning_printed = True

simulation_app.close()
