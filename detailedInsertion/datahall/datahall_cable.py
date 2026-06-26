from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
import typing
from pathlib import Path

import carb
import numpy as np
import omni.ui as ui
import omni.usd

from isaacsim.core.api import World
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices, rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    interface_config_loader,
)
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema

sys.path.append(str(Path(__file__).resolve().parent))
from collision_setup import apply_datahall_scale, enable_articulation_collisions, enable_static_collisions
from franka_motion_controller import FrankaMotionController
from port_frame import PortFrame


# =============================================================================
# 1. PATHS / PRIMS
# =============================================================================

NETWORK_CABLE_USD_PATH = "/home/aayush/isaacsim_assets/Network cable 001/model_Networkcable1_69323.usd"
NETWORK_CABLE_ROOT_PATH = "/World/NetworkCable"
TRACKED_PLUG_PRIM_PATH = f"{NETWORK_CABLE_ROOT_PATH}/E_crystal_head1_45"

FRANKA_PATH = "/World/Franka"
BLOCK_PATH = "/World/PickupBlock"
POST_LEFT_PATH = "/World/PickupPostLeft"
POST_RIGHT_PATH = "/World/PickupPostRight"

DATAHALL_USD = (
    "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin"
    "/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
)
DATAHALL_SCALE = 2.0
# Fixed work-table/robot base height for the AS4610 RJ45 test.
# The old QSFP/DataHall scripts used a much higher table. For this switch, the
# user-visible table should sit around 2.35 m so the robot approaches the RJ45
# row from a sane height instead of reaching down from above the port.
TABLE_HEIGHT = 2.35
TABLE_THICKNESS = 0.05
ROBOT_BASE_POS = np.array([0.55, -0.15, TABLE_HEIGHT], dtype=np.float64)

# AS4610 RJ45 12-pack component. This asset does NOT expose each Ethernet jack as
# its own prim, so the target port is inferred from this component bbox as a 2x6
# grid. Change TARGET_RJ45_ROW / TARGET_RJ45_COL to pick a different port.
DATAHALL_PORT_PRIM_PATH = (
    "/World/DataHall/Network_Switches/AS4610_01/AS4610_inst/AS4610_01/"
    "Switch/Net_12_Pack_no_LED_Component_02"
)
TARGET_RJ45_ROW = 0  # 0 = top row, 1 = bottom row
TARGET_RJ45_COL = 0  # 0 = most negative-Y port, 5 = most positive-Y port
AS4610_PORT_ROWS = 2
AS4610_PORT_COLS = 6
AS4610_Y_MARGIN_FRACTION = 0.090
AS4610_Z_MARGIN_FRACTION = 0.210
DATAHALL_PORT_LATERAL_OFFSET = np.array([0.0, 0.004, 0.0015], dtype=np.float64)  # +1.0mm right in Y, +0.5mm up in Z
# Positive Y is treated as "right" across the AS4610 RJ45 row. If the visual
# offset goes the wrong way, flip the Y value above to -0.0010 and leave Z alone.

# Robot/cable spawn placement. Z is no longer derived from the port height; the
# table and Franka base stay fixed at TABLE_HEIGHT=2.35 m. Y is still placed
# relative to the selected RJ45 port so the robot starts near the target lane.
ROBOT_BASE_Z_BELOW_TARGET_M = None  # unused; kept so old comments/debug are not misleading
ROBOT_BASE_Y_OFFSET_FROM_TARGET_M = 0.1675
ROBOT_BASE_X_M = 0.55

# Cable target construction from the port frame.
# The important correction: the controlled value is the plug XFORM CENTER, but
# collision/contact happens at the plug FRONT FACE. The previous version placed
# the plug center 50 mm in front of the port face, which means the physical nose
# of the connector was only ~31 mm in front. During insertion it then drove the
# nose ~18 mm into the port before the center reached the old final target.
#
# V6 defines pre-insert/final targets by the PLUG FRONT FACE, then converts those
# to the tracked plug xform-center target using the measured plug half-length.
DATAHALL_FINAL_PLUG_CENTER_AXIAL_DEPTH_M = 0.018  # legacy only; do not use for target math
DATAHALL_FINAL_PLUG_FRONT_AXIAL_DEPTH_M = 0.004   # first safe port test: nose only 4 mm past face
DATAHALL_PRE_INSERT_FRONT_GAP_M = 0.050           # nose is 50 mm in front of port face
DATAHALL_PORT_FACE_EXTRA_CLEARANCE_X_M = 0.000
PLUG_FRONT_TO_XFORM_X_OFFSET_M = 0.019            # overwritten from actual cable bbox after spawn
DATAHALL_PORT_FACE_X_M = float("nan")                # set from bbox max X in configure_datahall_cable_targets
# Legacy names kept only so older print/debug text does not confuse the file.
DATAHALL_PRE_INSERT_DISTANCE_FROM_PORT_FACE_X_M = DATAHALL_PRE_INSERT_FRONT_GAP_M
DATAHALL_PRE_INSERT_STANDOFF_M = DATAHALL_PRE_INSERT_FRONT_GAP_M
DATAHALL_REQUIRE_X_DOMINANT_PORT_AXIS = True
DATAHALL_MIN_ABS_X_AXIS = 0.85

# Before insertion is allowed to start, the measured plug pose must be locked on
# the port line and essentially horizontal. This prevents the robot from trying
# to insert while the cable is still diagonal/tilted from transit.
PREINSERT_AXIS_TO_PORT_TOL_DEG = 1.00
PREINSERT_HORIZONTAL_TOL_DEG = 1.00
PREINSERT_YZ_LINE_TOL = 0.00050

WORK_TABLE_PATH = "/World/WorkTable"

# Keep the Isaac UI clean. Only the live viewport stays visible.
VIEWPORT_ONLY_LAYOUT = False
VIEWPORT_KEEP_TITLES = ("Viewport", "Viewport 1")
VIEWPORT_HIDE_TOKENS = ("Sensors Output",)
VIEWPORT_LAYOUT_REAPPLY_FRAMES = 120

# V20 speed controls. The main time sink is excessive render/log overhead plus
# over-conservative servo/retreat waits. Keep pickup mechanically stable, but
# stop printing/bbox-checking every command during the demo.
FAST_DEMO_LOGGING = False
FAST_RENDER_EVERY_N_FRAMES = 2  # V22: after coarse pickup only; pickup/transit force render every frame

# Collision forgiveness for the target AS4610 RJ45 cage.
# The exact DataHall mesh colliders are too unforgiving for this demo: a tiny
# visual/intersection error at the RJ45 mouth can push the connector out or stall
# the X insert. Keep the switch visible, but make the selected 12-port RJ45
# component collision-free so insertion is judged by the measured plug pose
# servo, not by brittle mesh contact.
DISABLE_TARGET_RJ45_COMPONENT_COLLISION = True

# Broader invisible-collider fix. Disabling only the orange/white 12-port RJ45
# component is not enough when the DataHall asset has collision on parent/sibling
# switch meshes, rack face plates, or imported proxy geometry. This creates a
# visual-only collision tunnel around the selected insertion line. It keeps the
# scene visible but removes *all* switch/rack collision geometry that intersects
# the plug + gripper corridor.
DISABLE_AS4610_INSERT_COLLISION_TUNNEL = True
AS4610_INSERT_COLLISION_ROOT_PATH = "/World/DataHall/Network_Switches/AS4610_01"
INSERT_COLLISION_TUNNEL_X_BACK_PAD = 0.100   # extra room behind final plug center, into the switch
INSERT_COLLISION_TUNNEL_X_FRONT_PAD = 0.220  # room in front for fingers/gripper body
INSERT_COLLISION_TUNNEL_Y_PAD = 0.090        # wider than one RJ45 mouth so near-sibling colliders cannot block
INSERT_COLLISION_TUNNEL_Z_PAD = 0.075        # room above/below plug shell and latch

# Re-enable the same port/tunnel collisions after a successful insertion but
# before opening the hand. The insert needs the collision tunnel disabled, but
# release needs the port/rack collisions back on or the cable has nothing to sit
# in and can fall through the visual switch geometry.
REENABLE_PORT_COLLISION_BEFORE_RELEASE = True
POST_INSERT_COLLISION_SETTLE_FRAMES = 12
REENABLED_PORT_COLLISION_CONTACT_OFFSET = 0.0005
REENABLED_PORT_COLLISION_REST_OFFSET = -0.0005

# Re-enabling the imported switch collision is not enough if the cable is already
# inside a disabled/overlapping mesh corridor. Add a tiny invisible static cradle
# after successful insertion and before gripper release. This is the demo-safe
# version of an RJ45 socket retaining the connector: it supports the plug from
# below and lightly constrains Y without blocking the hand retreat along +X.
ENABLE_POST_INSERT_SOCKET_HOLD_PROXY = True
SOCKET_HOLD_PROXY_ROOT_PATH = "/World/PostInsertSocketHoldProxy"
SOCKET_HOLD_MATERIAL_PATH = "/World/Looks/SocketHoldPhysicsMaterial"
SOCKET_HOLD_STATIC_FRICTION = 20.0
SOCKET_HOLD_DYNAMIC_FRICTION = 20.0
SOCKET_HOLD_RESTITUTION = 0.0
SOCKET_HOLD_CONTACT_OFFSET = 0.0010
SOCKET_HOLD_REST_OFFSET = -0.0005
SOCKET_HOLD_SHELF_THICKNESS = 0.0040
SOCKET_HOLD_SIDE_RAIL_THICKNESS = 0.0030
SOCKET_HOLD_VERTICAL_OVERLAP = 0.0006
SOCKET_HOLD_CLEARANCE_Y = 0.0025
SOCKET_HOLD_X_PAD = 0.0140
SOCKET_HOLD_Y_PAD = 0.0060
SOCKET_HOLD_Z_PAD = 0.0060


# =============================================================================
# 2. SCENE SETTINGS
# =============================================================================

# -----------------------------------------------------------------------------
# PICKUP GEOMETRY — DO NOT "OPTIMIZE" THIS CASUALLY.
# -----------------------------------------------------------------------------
# These are copied from the standalone good pickup script in robot-local form.
# The DataHall robot can move in world space, but the cable/support geometry must
# stay at the exact same pose relative to the Franka base as the working file:
#
#   original robot base:      [0, 0, 0]
#   original block center:    [0.65, 0.00, 0.10]
#   original pickup target:   [0.642, 0.00, 0.1125]
#
# We convert these local pickup coordinates into world coordinates after the
# AS4610 target chooses ROBOT_BASE_POS. This preserves the working grasp while
# still letting the whole pickup island sit at TABLE_HEIGHT=2.35.
# Your imported cable USD is now manually scaled 2x in the asset file itself.
# Do NOT scale the cable again in this script. Only scale the pickup support
# geometry around the cable so the manually enlarged plug has a proportional
# cradle/slot.
PICKUP_SUPPORT_SCALE = 1.8
# Hard-set the cable/support pickup island Y in world space. This moves the
# support block, posts, cable spawn placement, and pickup grasp waypoint together.
# It does NOT scale the cable USD; the imported asset has already been manually doubled.
FIXED_PICKUP_WORLD_Y = -0.600

PICKUP_LOCAL_BLOCK_CENTER = np.array([0.65, 0.00, 0.10], dtype=np.float64)
BLOCK_CENTER = ROBOT_BASE_POS + PICKUP_LOCAL_BLOCK_CENTER
BLOCK_SIZE = np.array([0.04, 0.005, 0.025], dtype=np.float64) * PICKUP_SUPPORT_SCALE

# Keep the original pickup X/Y relationship, but raise the pickup Z to the new
# scaled block top. This preserves the diagonal pickup technique while matching
# the taller 2x support.
PICKUP_LOCAL_GRASP_POSITION = np.array(
    [0.642, 0.00, PICKUP_LOCAL_BLOCK_CENTER[2] + 0.5 * BLOCK_SIZE[2]],
    dtype=np.float64,
)

# Two small red posts make the cable slot visible. Scale only the support/post
# geometry; the cable asset itself is already scaled in the imported USD.
POST_SIZE = np.array([0.001, 0.001, 0.005], dtype=np.float64) * PICKUP_SUPPORT_SCALE
POST_GAP_Y = 0.008 * PICKUP_SUPPORT_SCALE
POST_X_OFFSET = -0.020 * PICKUP_SUPPORT_SCALE

# Cable plug rests this far above the top of the block. Scaled to match the
# manually enlarged cable asset.
PLUG_BLOCK_CLEARANCE = 0.004 * PICKUP_SUPPORT_SCALE
PICKUP_GRASP_POSITION = ROBOT_BASE_POS + PICKUP_LOCAL_GRASP_POSITION

PHYSICS_DT = 1.0 / 120.0
RENDERING_DT = 1.0 / 60.0

# Used by FrankaMotionController when you add cartesian waypoints.
TOOL_OFFSET = 0.111

TOP_DOWN_ORI = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
DIAGONAL_DOWN_ORI = np.array([0.5, 0.0, 0.86602540, 0.0], dtype=np.float64)
DIAGONAL_INSERT_ORI = np.array([0.0, -0.86602540, 0.0, 0.5], dtype=np.float64)
HORIZONTAL_ORI = np.array([0.70710678, 0, 0.70710678, 0.0], dtype=np.float64)

# Grip/contact tuning. These change physics, not the motion path.
# The original gripper delta was 0.02m per frame, which snaps shut in only a few
# frames and can squeeze/eject the connector before contact settles.
GRIPPER_ACTION_DELTA = 0.0025
GRIP_CLOSE_WAIT_FRAMES = 220  # V22: pickup is failing; give the fingers/contact solver time to actually settle

# High-friction physics material for the finger pads and connector plug.
# If the plug still slips, raise both friction values together. If the plug sticks
# unnaturally to everything, lower them together.
GRIP_STATIC_FRICTION = 30.0
GRIP_DYNAMIC_FRICTION = 30.0
GRIP_RESTITUTION = 0.0
GRIP_MATERIAL_PATH = "/World/Looks/HighGripPhysicsMaterial"

# Small contact offsets help tiny mesh contacts resolve before visible penetration.

GRIP_CONTACT_OFFSET = 0.006
GRIP_REST_OFFSET = 0.0

# Closed-loop plug pose servo. This is the cable version of the block insertion
# feedback loop: it reads /World/NetworkCable/E_crystal_head1_45 every frame and
# corrects the robot until the plug position AND orientation are inside tolerance.
# Position uses the tracked plug XFORM TRANSLATE because that is the value shown
# in the Isaac Sim Transform panel. Bbox center is still printed as secondary
# debug, but it is not the controlled target in this version.
ENABLE_PLUG_POSE_SERVO = True

# =============================================================================
# USER-SELECTED CABLE / PORT TARGET
# =============================================================================
# Change these two positions when you want a different insertion stroke.
# PRE_INSERT is the world-space TRANSLATE shown in the Transform panel for
# /World/NetworkCable/E_crystal_head1_45 before insertion starts.
# FINAL_INSERT is the seated/end pose after the slow X-only insertion stroke.
# The robot hand target is derived from these; do not hand-tune the gripper pose.
USER_SELECTED_CABLE_PRE_INSERT_POSITION = np.array([-0.50, 0.00, 0.325], dtype=np.float64)
USER_SELECTED_CABLE_FINAL_INSERT_POSITION = np.array([-0.55, 0.00, 0.325], dtype=np.float64)

# The plug's long dimension is local +X in the USD. This target is a 180-degree
# yaw around world Z, so local +X points along world -X and the connector stays
# horizontal for insertion.
USER_SELECTED_CABLE_TARGET_ORI_WXYZ = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

# Internal names used by the pre-insert alignment servo. Do not edit these;
# edit USER_SELECTED_* above.
PLUG_TARGET_POSITION = USER_SELECTED_CABLE_PRE_INSERT_POSITION.copy()
PLUG_FINAL_INSERT_POSITION = USER_SELECTED_CABLE_FINAL_INSERT_POSITION.copy()
PLUG_TARGET_ORI_WXYZ = USER_SELECTED_CABLE_TARGET_ORI_WXYZ.copy()
PLUG_INSERT_AXIS_WORLD = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

# Coarse routing is still hand/tool motion, but the last coarse waypoint is now
# derived from the cable target above. The final measured position is corrected
# by the plug feedback servo, not by trusting the hand waypoint.
COARSE_TRANSFER_LANE_Y = -0.50

# Do NOT add a huge fixed +125 mm height above every target.
# That was the slip bug for higher targets: z=0.325 produced a 0.450 m carry lane,
# then the next waypoint yanked the plug back down while the cable tail was loaded.
# Keep only a small clearance above the selected target, with the old 0.325 m
# floor preserved for low targets.
COARSE_TARGET_CLEARANCE_Z = 0.025
COARSE_TRANSFER_Z = max(0.325, float(PLUG_TARGET_POSITION[2]) + COARSE_TARGET_CLEARANCE_Z)

COARSE_TRANSFER_POSITION = np.array([0.0, COARSE_TRANSFER_LANE_Y, COARSE_TRANSFER_Z], dtype=np.float64)
COARSE_APPROACH_TARGET_POSITION = np.array(
    [float(PLUG_TARGET_POSITION[0]), float(PLUG_TARGET_POSITION[1]), COARSE_TRANSFER_Z],
    dtype=np.float64,
)
COARSE_NEAR_TARGET_POSITION = PLUG_TARGET_POSITION.copy()

# If the plug is nowhere near the target after coarse motion, do not let the servo
# pretend it can recover. That means the cable has already slipped out.
PRE_SERVO_MAX_POSITION_ERROR = 0.080
PRE_SERVO_MAX_TILT_DEG = 35.0
# Do not let the fine servo try to rescue a badly rotated plug. In the failed run,
# the plug entered servo about 33 deg off-axis; the servo then twisted the cable
# out of the gripper and the pose error exploded. These guards stop that early.
PRE_SERVO_MAX_AXIS_ANGLE_DEG = 12.0
PRE_SERVO_MAX_ORI_ERROR_DEG = 20.0

# Visual marker for the desired pre-insert plug target. Disabled for the clean run.
SHOW_USER_TARGET_MARKER = False
USER_TARGET_MARKER_PATH = "/World/UserSelectedCableTarget"
USER_TARGET_MARKER_SIZE = 0.008

# Visual final target/port block. Disabled for the clean run.
# No-overinsert protection comes from the strict X clamp below, not scene geometry.
SHOW_FINAL_TARGET_BLOCK = False
FINAL_TARGET_BLOCK_PATH = "/World/FinalInsertTargetBlock"
FINAL_TARGET_BLOCK_SIZE = np.array([0.012, 0.030, 0.030], dtype=np.float64)
FINAL_TARGET_BLOCK_COLLISION = False

# Final plug-position alignment tolerances.
# For now this is NOT insertion. The goal is to put the tracked plug transform
# translate at the user-selected port/alignment point with <= 0.5 mm total Y/Z
# offset, matching the Transform panel value.
PLUG_POSITION_TOL = 0.0010          # legacy total XYZ print limit
PLUG_X_TOL = 0.0010                 # 1.0 mm X tolerance for staging/alignment
PLUG_YZ_TOTAL_TOL = 0.00050         # 0.5 mm radial Y/Z tolerance — main metric
PLUG_ORIENTATION_TOL_DEG = 1.00     # full quaternion angular error
PLUG_HOLD_FRAMES = 1                # fast demo settle; still avoids one-frame false positives
PLUG_SERVO_MAX_FRAMES = 350
PLUG_SERVO_DEBUG_EVERY = 60

# Per-frame correction limits. These are intentionally small so the correction
# behaves like a servo, not a teleporting IK target.
PLUG_SERVO_POS_KP = 0.65           # legacy fallback value; adaptive gains below are used
# Fast/fine position servo inspired by the block insert: move aggressively while
# far from target, then slow down only near the 0.5 mm Y/Z gate.
PLUG_FAST_MODE_DISTANCE = 0.0100     # >6 mm error: fast catch-up
PLUG_MID_MODE_DISTANCE = 0.0030      # 2-6 mm error: medium catch-up
PLUG_FAST_POS_KP = 1.00
PLUG_MID_POS_KP = 0.85
PLUG_FINE_POS_KP = 0.95
PLUG_FAST_MAX_POS_STEP = 0.0080      # 6.0 mm/frame while far
PLUG_MID_MAX_POS_STEP = 0.0050       # 2.5 mm/frame while medium
PLUG_FINE_MAX_POS_STEP = 0.0030      # 2.0 mm/frame while settling
PLUG_SERVO_MAX_POS_STEP = PLUG_FAST_MAX_POS_STEP
PLUG_FAST_ORI_STEP_DEG = 2.50
PLUG_FINE_ORI_STEP_DEG = 1.25
PLUG_SERVO_MAX_ORI_STEP_DEG = PLUG_FAST_ORI_STEP_DEG
PLUG_SERVO_IK_POS_TOL = 0.0002
PLUG_SERVO_IK_ORI_TOL = 0.020       # radians-ish tolerance used by Lula IK
PLUG_LOCAL_DRIFT_WARN_POS = 0.0040  # if exceeded, plug is moving in fingers
PLUG_LOCAL_DRIFT_WARN_ORI_DEG = 5.0

# The uploaded baseline consistently plateaued around +1.2 mm Z error at the
# final pose. This Y/Z overdrive is the missing block-servo idea: if the measured
# plug sits low/right, command the hand slightly high/left until the measured
# plug itself is on the target line. This only runs near the target so it does
# not yank the cable during the long approach.
PLUG_YZ_OVERDRIVE_ENABLE_RADIUS = 0.012  # activate earlier; final bias was previously starving
PLUG_YZ_OVERDRIVE_X_ENABLE = 0.0100      # allow final Y/Z bias as soon as X is reasonably close
# Gentler than v3. The old 4 mm overdrive was accurate but could oscillate for
# hundreds of frames. This is closer to the block servo idea: enough bias to beat
# steady-state cable sag, not enough to throw the plug past the target line.
PLUG_YZ_OVERDRIVE_KP = 1.65
PLUG_YZ_OVERDRIVE_KI = 0.0025
PLUG_YZ_INTEGRAL_LEAK = 0.995
PLUG_YZ_INTEGRAL_LIMIT = 0.40
PLUG_YZ_MAX_OVERDRIVE = 0.0030      # max 3.0 mm Y/Z target bias; enough to beat cable sag quickly

# =============================================================================
# INSERT STROKE SETTINGS
# =============================================================================
ENABLE_PLUG_INSERT_SERVO = True
INSERT_TARGET_POSITION = PLUG_FINAL_INSERT_POSITION.copy()
INSERT_AXIS_WORLD = PLUG_INSERT_AXIS_WORLD / np.linalg.norm(PLUG_INSERT_AXIS_WORLD)

# Path requirement: during the X stroke, the measured plug Transform translate
# should stay close to the port line. Pre-insert still locks to 0.5 mm; the
# insert stroke gets a looser 1.0 mm gate so it does not stall before the plug
# actually reaches the port.
INSERT_YZ_TOTAL_TOL = 0.00100   # hard demo requirement: measured plug Y/Z path must never exceed 1.0 mm
INSERT_FINAL_X_TOL = 0.00050
INSERT_ORIENTATION_TOL_DEG = 1.00
INSERT_STABLE_HOLD_FRAMES = 1
INSERT_STROKE_MAX_FRAMES = 1200
INSERT_STROKE_DEBUG_EVERY = 40
INSERT_HARD_ABORT_YZ_TOL = 0.00105  # strict path guard: abort almost immediately if Y/Z tries to exceed 1 mm
INSERT_HARD_ABORT_ORI_TOL_DEG = 3.0 # abort if collision tilts the connector before final target

# Adaptive X insertion speed. V7 used 0.075 mm/frame and took ~675 frames
# for a 50 mm stroke even though Y/Z stayed far under the 0.5 mm limit.
# V8 moves fast while the plug is safely on the line, then slows only near the
# final X band or if Y/Z starts drifting toward the pause gate.
INSERT_X_FAST_STEP = 0.00100      # strict-path insert: slower X so Y/Z correction never spikes past 1 mm
INSERT_X_FINE_STEP = 0.00025       # strict-path insert: crawl when near final X or when Y/Z starts drifting
INSERT_X_FINE_DISTANCE = 0.006    # switch to fine mode earlier so the last centimeters do not kick Y/Z sideways
INSERT_X_YZ_SLOWDOWN_TOL = 0.00055 # slow X before Y/Z approaches the 1 mm path limit
INSERT_X_SLOW_STEP = INSERT_X_FAST_STEP  # legacy/debug alias
# Strict port safety: the commanded plug X is never allowed past the final target.
# For -X insertion, commanded_x will never be < final_x. For +X insertion, it
# will never be > final_x.
INSERT_STRICT_NO_PAST_TARGET_X = True
INSERT_X_MAX_PUSH_THROUGH = 0.0
INSERT_MAX_BACKSLIDE = 0.0040
INSERT_X_BACKTRACK_TOL = 0.00020

# Pause X advancement only when Y/Z is genuinely bad. V9 froze the X stroke for
# 760 frames because it paused at 0.49 mm and required recovery to 0.43 mm; the
# plug sat safely around 0.46 mm Y/Z error but never inserted. Keep moving unless
# Y/Z approaches the hard-abort band.
INSERT_STROKE_YZ_PAUSE_TOL = 0.00070
INSERT_STROKE_YZ_RESUME_TOL = 0.00040
INSERT_YZ_KP = 2.20
INSERT_YZ_KI = 0.0040
INSERT_YZ_INTEGRAL_LEAK = 0.997
INSERT_YZ_INTEGRAL_LIMIT = 0.75
INSERT_YZ_MAX_OVERDRIVE = 0.0100
INSERT_MAX_ORI_STEP_DEG = 0.70

# =============================================================================
# POST-INSERT RELEASE / RETREAT SETTINGS
# =============================================================================
# Runs only after the measured insert servo reports a successful final pose. The
# hand opens first, then retreats straight backward along world +X, which is the
# opposite direction of the -X insertion stroke. This deliberately does not move
# the cable target or change insertion depth.
ENABLE_POST_INSERT_RELEASE_RETREAT = True
POST_INSERT_OPEN_WAIT_FRAMES = 40
POST_INSERT_RETREAT_X_M = 0.120
POST_INSERT_RETREAT_LINEAR_STEP = 0.0120
POST_INSERT_RETREAT_POS_TOL = 0.0030

# =============================================================================
# TRANSIT SPEED SETTINGS
# =============================================================================
# These only affect the coarse carry to pre-insert. Pickup, grip, pre-insert
# servo, strict no-overinsert clamp, and final X insert logic are unchanged.
# The old values were deliberately conservative: 240/320/320 joint steps plus
# tiny linear steps. That made the transit phase crawl even though the measured
# plug stayed stable.
# The cable is now physically 2x larger in the imported USD, so the old fast
# transit is too violent: it yanks the long tail and the connector rotates out
# of the fingers. Keep the proven pickup, but make the carry deliberately slow
# and split translation from reorientation.
TRANSIT_LIFT_LINEAR_STEP = 0.0240      # V20: do not raise; V19 40mm/frame broke pickup      # controlled lift for the heavier/manual-2x cable
TRANSIT_REORIENT_JOINT_STEPS = 110     # V21: restore V18 working physical transit; V20 85 slipped the cable
TRANSIT_TRANSFER_JOINT_STEPS = 130     # V21: restore V18 working physical transit; V20 95 slipped the cable
TRANSIT_APPROACH_JOINT_STEPS = 120     # V21: restore V18 working physical transit; V20 85 slipped the cable
TRANSIT_DESCEND_LINEAR_STEP = 0.0040   # V21: restore V18 working descend; V20 6mm/frame was too aggressive

# Tail-clearance waypoints used when pickup is offset from the robot base.
# V5 dragged the held cable from x≈1.19 to x=0 while still diagonal-down, which
# created the ugly transit loop you saw. Reorient after a shorter clear-out move.
TAIL_CLEAR_X = 0.80
TAIL_CLEAR_Y = -0.58
MID_TRANSFER_X = 0.25


# =============================================================================
# 3. EDIT YOUR WAYPOINTS HERE
# =============================================================================

def queue_user_waypoints(controller: FrankaMotionController) -> None:
    """
    Queue the cable pickup and transit path.

    Pickup stays at y=-0.600 so the long manually-scaled cable tail avoids the
    Franka base. The robot base stays in the old port-relative lane. Because the
    pickup is now offset from the base, the transit must avoid two bad extremes:

      * reorienting at the far pickup station, which failed IK in V4
      * dragging all the way to x=0 while still diagonal-down, which made V5's
        transit look terrible and loaded the cable tail

    The compromise is a short tail-clearance move, then reorient at a mid
    waypoint, then carry toward the port in insertion orientation.
    """

    controller.clear_queue()

    tail_clear_position = np.array([TAIL_CLEAR_X, TAIL_CLEAR_Y, COARSE_TRANSFER_Z], dtype=np.float64)
    mid_transfer_position = np.array([MID_TRANSFER_X, TAIL_CLEAR_Y, COARSE_TRANSFER_Z], dtype=np.float64)

    controller.add_cartesian_waypoint(
        position=pickup_local_to_world(PICKUP_LOCAL_GRASP_POSITION),
        orientation=DIAGONAL_DOWN_ORI,
        max_frames=600,
        pos_tolerance=0.001,
        linear=True,
        linear_step=0.001,
        label="diagonal_pickup_down",
    )
    controller.add_gripper_command(action="close", wait_frames=GRIP_CLOSE_WAIT_FRAMES)

    controller.add_cartesian_waypoint(
        position=np.array([PICKUP_GRASP_POSITION[0], PICKUP_GRASP_POSITION[1], COARSE_TRANSFER_Z], dtype=np.float64),
        orientation=DIAGONAL_DOWN_ORI,
        max_frames=460,
        pos_tolerance=0.025,
        linear=True,
        linear_step=TRANSIT_LIFT_LINEAR_STEP,
        hold_gripper=True,
        label="diagonal_lift_after_grasp_medium",
    )

    controller.add_cartesian_waypoint(
        # Short move only: clear the long cable tail away from the pickup/support
        # before rotating. Do not drag the diagonal-down plug all the way to x=0.
        position=tail_clear_position,
        orientation=DIAGONAL_DOWN_ORI,
        max_frames=420,
        pos_tolerance=0.025,
        joint_interp=True,
        joint_steps=TRANSIT_TRANSFER_JOINT_STEPS,
        hold_gripper=True,
        label="tail_clear_before_reorient",
    )

    controller.add_cartesian_waypoint(
        # Reorient after the tail-clearance move. This avoids the failed V4
        # reorient at the far pickup station, but it also avoids V5's ugly full
        # diagonal transit across the table.
        position=tail_clear_position,
        orientation=DIAGONAL_INSERT_ORI,
        max_frames=420,
        pos_tolerance=0.025,
        joint_interp=True,
        joint_steps=TRANSIT_REORIENT_JOINT_STEPS,
        hold_gripper=True,
        label="mid_reorient_after_tail_clear",
    )

    controller.add_cartesian_waypoint(
        # Carry most of the remaining distance only after the plug is in the
        # insertion orientation.
        position=mid_transfer_position,
        orientation=DIAGONAL_INSERT_ORI,
        max_frames=420,
        pos_tolerance=0.025,
        joint_interp=True,
        joint_steps=TRANSIT_TRANSFER_JOINT_STEPS,
        hold_gripper=True,
        label="mid_transfer_insert_orientation",
    )

    controller.add_cartesian_waypoint(
        position=COARSE_APPROACH_TARGET_POSITION.copy(),
        orientation=DIAGONAL_INSERT_ORI,
        max_frames=420,
        pos_tolerance=0.016,
        joint_interp=True,
        joint_steps=TRANSIT_APPROACH_JOINT_STEPS,
        hold_gripper=True,
        label="approach_above_user_cable_target",
    )

    controller.add_cartesian_waypoint(
        position=COARSE_NEAR_TARGET_POSITION.copy(),
        orientation=DIAGONAL_INSERT_ORI,
        max_frames=300,
        pos_tolerance=0.006,
        linear=True,
        linear_step=TRANSIT_DESCEND_LINEAR_STEP,
        hold_gripper=True,
        label="coarse_descend_near_user_cable_target",
    )


# =============================================================================
# 4. SMALL USD HELPERS
# =============================================================================

def remove_prim_if_exists(prim_path: str) -> None:
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))


def get_bbox(prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    mn = np.array(box.GetMin(), dtype=np.float64)
    mx = np.array(box.GetMax(), dtype=np.float64)
    center = 0.5 * (mn + mx)
    dims = mx - mn
    return mn, mx, center, dims


# =============================================================================
# VIEWPORT / UI HELPERS
# =============================================================================

def _should_keep_viewport_window(title: str) -> bool:
    if any(token in title for token in VIEWPORT_HIDE_TOKENS):
        return False
    return title in VIEWPORT_KEEP_TITLES


def configure_viewport_only_layout() -> None:
    """Hide dock windows while keeping the live viewport window alive."""
    if not VIEWPORT_ONLY_LAYOUT:
        return

    try:
        for window in ui.Workspace.get_windows():
            title = getattr(window, "title", str(window))
            ui.Workspace.show_window(title, _should_keep_viewport_window(title))

        for viewport_title in VIEWPORT_KEEP_TITLES:
            viewport = ui.Workspace.get_window(viewport_title)
            if viewport is not None:
                ui.Workspace.show_window(viewport_title, True)
    except Exception as exc:
        carb.log_warn(f"Could not apply viewport-only layout: {exc}")


# =============================================================================
# DEBUG HELPERS
# =============================================================================

def fmt_vec(value: np.ndarray, decimals: int = 5) -> str:
    return np.array2string(
        np.round(np.asarray(value, dtype=np.float64), decimals),
        precision=decimals,
        suppress_small=False,
    )


def pickup_local_to_world(local_position: np.ndarray) -> np.ndarray:
    """Map the proven standalone pickup coordinates into the DataHall scene.

    X/Z stay translated from the Franka base like the working pickup. Y is
    intentionally pinned to FIXED_PICKUP_WORLD_Y because the cable/support spawn
    lane needs to sit at y=-0.600 regardless of the selected RJ45 port lane.
    """

    world_pos = np.asarray(ROBOT_BASE_POS, dtype=np.float64) + np.asarray(local_position, dtype=np.float64)
    world_pos[1] = float(FIXED_PICKUP_WORLD_Y + np.asarray(local_position, dtype=np.float64)[1])
    return world_pos


def get_prim_world_pose(prim_path: str):
    """Return world-space translation and orientation quaternion [w, x, y, z]."""

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None, None

    mat = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    pos = np.array(mat.ExtractTranslation(), dtype=np.float64)

    quat_gf = mat.ExtractRotationQuat()
    imag = quat_gf.GetImaginary()
    quat_wxyz = np.array(
        [quat_gf.GetReal(), imag[0], imag[1], imag[2]],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quat_wxyz))
    if norm > 1e-12:
        quat_wxyz /= norm
    return pos, quat_wxyz


def log_cable_pose(tag: str) -> None:
    """Print the cable/root pose plus the tracked plug pose/bbox.

    Root xform can stay mostly constant for deformable assets, so the tracked
    plug bbox center is the more useful number for pickup/insertion debugging.
    """

    print("-" * 88)
    print(f"[CABLE POSE DEBUG] {tag}")

    root_pos, root_quat = get_prim_world_pose(NETWORK_CABLE_ROOT_PATH)
    if root_pos is None:
        print(f"  cable_root missing: {NETWORK_CABLE_ROOT_PATH}")
    else:
        print(f"  root_path:       {NETWORK_CABLE_ROOT_PATH}")
        print(f"  root_pos:        {fmt_vec(root_pos)}")
        print(f"  root_ori_wxyz:   {fmt_vec(root_quat)}")

    plug_pos, plug_quat = get_prim_world_pose(TRACKED_PLUG_PRIM_PATH)
    if plug_pos is None:
        print(f"  tracked_plug missing: {TRACKED_PLUG_PRIM_PATH}")
    else:
        print(f"  plug_path:       {TRACKED_PLUG_PRIM_PATH}")
        print(f"  plug_xform_pos:  {fmt_vec(plug_pos)}")
        print(f"  plug_ori_wxyz:   {fmt_vec(plug_quat)}")

        plug_min, plug_max, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)
        print(f"  plug_bbox_min:   {fmt_vec(plug_min)}")
        print(f"  plug_bbox_max:   {fmt_vec(plug_max)}")
        print(f"  plug_bbox_ctr:   {fmt_vec(plug_center)}")
        print(f"  plug_dims_mm:    {fmt_vec(plug_dims * 1000.0, 3)}")

    print("-" * 88)


def get_current_command_info(controller: FrankaMotionController):
    """Read the active command from the controller for debug printing."""

    if controller.is_done():
        return None

    index = int(getattr(controller, "_current_command_index", -1))
    queue = getattr(controller, "_command_queue", [])
    if index < 0 or index >= len(queue):
        return None

    cmd = queue[index]
    cmd_type = str(cmd.get("type", "unknown"))
    label = str(cmd.get("label", ""))
    if not label:
        if cmd_type == "gripper":
            label = f"gripper_{cmd.get('action', 'command')}"
        else:
            label = f"command_{index}"

    return {
        "index": index,
        "type": cmd_type,
        "label": label,
        "cmd": cmd,
    }


def print_queued_commands(controller: FrankaMotionController) -> None:
    queue = getattr(controller, "_command_queue", [])
    print("=" * 88)
    print(f"[QUEUED CONTROLLER COMMANDS] count={len(queue)}")
    for i, cmd in enumerate(queue):
        cmd_type = str(cmd.get("type", "unknown"))
        label = str(cmd.get("label", "")) or f"gripper_{cmd.get('action', 'command')}"
        print(f"  [{i}] type={cmd_type} label={label}")
        if cmd_type == "cartesian":
            print(f"      target_position={fmt_vec(cmd.get('pos'))}")
            print(f"      target_ori_wxyz={fmt_vec(cmd.get('ori'))}")
            print(
                "      mode="
                f"linear={cmd.get('linear')} "
                f"joint_interp={cmd.get('joint_interp')} "
                f"hold_gripper={cmd.get('hold_gripper')} "
                f"target_is_hand={cmd.get('target_is_hand')}"
            )
        elif cmd_type == "gripper":
            print(f"      action={cmd.get('action')} wait_frames={cmd.get('max_frames')}")
    print("=" * 88)


def log_command_boundary(boundary: str, info) -> None:
    if info is None:
        return

    cmd = info["cmd"]
    print("=" * 88)
    print(
        f"[COMMAND {boundary}] index={info['index']} "
        f"type={info['type']} label={info['label']}"
    )
    if info["type"] == "cartesian":
        print(f"  commanded_position={fmt_vec(cmd.get('pos'))}")
        print(f"  commanded_ori_wxyz={fmt_vec(cmd.get('ori'))}")
        print(
            "  command_mode="
            f"linear={cmd.get('linear')} "
            f"joint_interp={cmd.get('joint_interp')} "
            f"hold_gripper={cmd.get('hold_gripper')} "
            f"target_is_hand={cmd.get('target_is_hand')}"
        )
    elif info["type"] == "gripper":
        print(f"  gripper_action={cmd.get('action')} wait_frames={cmd.get('max_frames')}")
    print("=" * 88)
    log_cable_pose(f"{boundary} command {info['index']} / {info['label']}")


def set_world_translate(prim_path: str, translation: np.ndarray) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    value = Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(value)
            return

    xform.AddTranslateOp().Set(value)


def make_static_box(prim_path: str, center: np.ndarray, size: np.ndarray, color=(0.2, 0.35, 1.0)) -> None:
    stage = omni.usd.get_context().get_stage()
    remove_prim_if_exists(prim_path)

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
    xform.AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


PORT_COLLISION_RESTORE_PATHS: typing.Set[str] = set()


def _remember_port_collision_path(prim: Usd.Prim) -> None:
    try:
        PORT_COLLISION_RESTORE_PATHS.add(str(prim.GetPath()))
    except Exception:
        pass


def disable_collision_recursive(root_path: str, label: str = "") -> typing.Tuple[int, int]:
    """Disable collision under a visual asset subtree.

    This is intentionally different from deleting the prims. The RJ45 cage stays
    visible, but its exact mesh collider stops blocking the connector when the
    visual alignment is off by a fraction of a millimeter.
    """

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[FORGIVING COLLISION WARNING] Missing prim: {root_path}")
        return 0, 0

    disabled_count = 0
    tagged_count = 0
    for prim in Usd.PrimRange(root):
        try:
            no_collision_attr = prim.GetAttribute("omni:no_collision")
            if not no_collision_attr or not no_collision_attr.IsValid():
                no_collision_attr = prim.CreateAttribute("omni:no_collision", Sdf.ValueTypeNames.Bool)
            no_collision_attr.Set(True)
            tagged_count += 1
            _remember_port_collision_path(prim)

            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                disabled_count += 1
                _remember_port_collision_path(prim)
        except Exception as exc:
            print(f"[FORGIVING COLLISION WARNING] Could not disable collision on {prim.GetPath()}: {exc}")

    print("=" * 88)
    print("[FORGIVING RJ45 COLLISION WINDOW]")
    print(f"  label:             {label or 'target'}")
    print(f"  root_path:         {root_path}")
    print("  visual_geometry:   kept visible")
    print("  physics_collision: disabled under this subtree")
    print(f"  tagged_no_collision_prims={tagged_count}")
    print(f"  disabled_collision_prims={disabled_count}")
    print("  reason: exact AS4610 RJ45 mesh collision was blocking insertion on tiny contact.")
    print("=" * 88)
    return disabled_count, tagged_count


def apply_target_rj45_forgiving_collision() -> None:
    if not DISABLE_TARGET_RJ45_COMPONENT_COLLISION:
        print("[FORGIVING RJ45 COLLISION WINDOW] disabled by config")
        return

    disable_collision_recursive(DATAHALL_PORT_PRIM_PATH, label="selected AS4610 12-port RJ45 component")


def _bbox_intersects(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> bool:
    return bool(
        a_min[0] <= b_max[0] and a_max[0] >= b_min[0]
        and a_min[1] <= b_max[1] and a_max[1] >= b_min[1]
        and a_min[2] <= b_max[2] and a_max[2] >= b_min[2]
    )


def _collision_enabled_attr_off(prim: Usd.Prim) -> bool:
    """Best-effort collision disable that works on exact mesh/proxy collision prims."""

    touched = False
    try:
        no_collision_attr = prim.GetAttribute("omni:no_collision")
        if not no_collision_attr or not no_collision_attr.IsValid():
            no_collision_attr = prim.CreateAttribute("omni:no_collision", Sdf.ValueTypeNames.Bool)
        no_collision_attr.Set(True)
        touched = True
    except Exception:
        pass

    try:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
            touched = True
    except Exception:
        pass

    try:
        if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            physx_collision = PhysxSchema.PhysxCollisionAPI(prim)
            # Not every Isaac/PhysX schema exposes the same attributes. These are
            # guarded so the script stays compatible across Isaac Sim builds.
            _set_schema_attr(physx_collision, "CreateContactOffsetAttr", 0.0002)
            _set_schema_attr(physx_collision, "CreateRestOffsetAttr", -0.0020)
            touched = True
    except Exception:
        pass

    if touched:
        _remember_port_collision_path(prim)
    return touched


def _collision_enabled_attr_on(prim: Usd.Prim) -> bool:
    """Best-effort restore for collision prims disabled by the insert tunnel."""

    touched = False
    try:
        no_collision_attr = prim.GetAttribute("omni:no_collision")
        if no_collision_attr and no_collision_attr.IsValid():
            no_collision_attr.Set(False)
            touched = True
    except Exception:
        pass

    try:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)
            touched = True
    except Exception:
        pass

    try:
        if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            physx_collision = PhysxSchema.PhysxCollisionAPI(prim)
            _set_schema_attr(physx_collision, "CreateContactOffsetAttr", REENABLED_PORT_COLLISION_CONTACT_OFFSET)
            _set_schema_attr(physx_collision, "CreateRestOffsetAttr", REENABLED_PORT_COLLISION_REST_OFFSET)
            touched = True
    except Exception:
        pass

    return touched


def restore_port_insert_collisions_before_release() -> typing.Tuple[int, int, int]:
    """Re-enable previously disabled target/tunnel collisions before hand release."""

    if not REENABLE_PORT_COLLISION_BEFORE_RELEASE:
        print("[PORT COLLISION RESTORE] disabled by config")
        return 0, 0, 0

    stage = omni.usd.get_context().get_stage()
    paths = sorted(PORT_COLLISION_RESTORE_PATHS)
    restored = 0
    missing = 0
    failed = 0

    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            missing += 1
            continue
        try:
            if _collision_enabled_attr_on(prim):
                restored += 1
        except Exception as exc:
            failed += 1
            print(f"[PORT COLLISION RESTORE WARNING] Could not restore {path}: {exc}")

    print("=" * 88)
    print("[PORT COLLISION RESTORED BEFORE RELEASE]")
    print("  timing:            after successful insert, before gripper open")
    print("  reason:            cable should not fall through visual-only RJ45/switch geometry after release")
    print(f"  tracked_paths:     {len(paths)}")
    print(f"  restored_prims:    {restored}")
    print(f"  missing_prims:     {missing}")
    print(f"  failed_prims:      {failed}")
    print(f"  settle_frames:     {POST_INSERT_COLLISION_SETTLE_FRAMES}")
    print("=" * 88)
    return restored, missing, failed



def create_socket_hold_physics_material() -> UsdShade.Material:
    """High-friction material for the invisible post-insert socket cradle."""

    stage = omni.usd.get_context().get_stage()
    material = UsdShade.Material.Define(stage, Sdf.Path(SOCKET_HOLD_MATERIAL_PATH))
    prim = material.GetPrim()

    usd_physics_mat = UsdPhysics.MaterialAPI.Apply(prim)
    _set_schema_attr(usd_physics_mat, "CreateStaticFrictionAttr", SOCKET_HOLD_STATIC_FRICTION)
    _set_schema_attr(usd_physics_mat, "CreateDynamicFrictionAttr", SOCKET_HOLD_DYNAMIC_FRICTION)
    _set_schema_attr(usd_physics_mat, "CreateRestitutionAttr", SOCKET_HOLD_RESTITUTION)

    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    _set_schema_attr(physx_mat, "CreateFrictionCombineModeAttr", "max")
    _set_schema_attr(physx_mat, "CreateRestitutionCombineModeAttr", "min")
    return material


def make_invisible_collision_box(prim_path: str, center: np.ndarray, size: np.ndarray, material: UsdShade.Material) -> None:
    """Create an invisible static collision cube that still participates in physics."""

    stage = omni.usd.get_context().get_stage()
    remove_prim_if_exists(prim_path)

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.0, 1.0, 0.35)])

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
    xform.AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))

    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    _set_schema_attr(physx_collision, "CreateContactOffsetAttr", SOCKET_HOLD_CONTACT_OFFSET)
    _set_schema_attr(physx_collision, "CreateRestOffsetAttr", SOCKET_HOLD_REST_OFFSET)
    _bind_material_to_prim(prim, material)

    # Keep the demo visually clean. Physics collision stays active even though
    # the visual cube is hidden.
    UsdGeom.Imageable(prim).MakeInvisible()


def spawn_post_insert_socket_hold_proxy() -> None:
    """Spawn an invisible static cradle under/around the inserted plug before release.

    The restored imported port mesh is not a reliable support once we inserted
    through a collision-free tunnel. This proxy simulates the RJ45 jack retaining
    the connector after the measured insertion succeeds.
    """

    if not ENABLE_POST_INSERT_SOCKET_HOLD_PROXY:
        print("[POST-INSERT SOCKET HOLD PROXY] disabled by config")
        return

    stage = omni.usd.get_context().get_stage()
    remove_prim_if_exists(SOCKET_HOLD_PROXY_ROOT_PATH)
    UsdGeom.Xform.Define(stage, Sdf.Path(SOCKET_HOLD_PROXY_ROOT_PATH))

    plug_min, plug_max, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)
    material = create_socket_hold_physics_material()

    x_len = float(plug_dims[0] + 2.0 * SOCKET_HOLD_X_PAD)
    y_width = float(plug_dims[1] + 2.0 * SOCKET_HOLD_Y_PAD)

    shelf_size = np.array([x_len, y_width, SOCKET_HOLD_SHELF_THICKNESS], dtype=np.float64)
    shelf_center = np.array([
        float(plug_center[0]),
        float(plug_center[1]),
        float(plug_min[2] - 0.5 * SOCKET_HOLD_SHELF_THICKNESS + SOCKET_HOLD_VERTICAL_OVERLAP),
    ], dtype=np.float64)

    rail_height = float(plug_dims[2] + 2.0 * SOCKET_HOLD_Z_PAD)
    rail_size = np.array([x_len, SOCKET_HOLD_SIDE_RAIL_THICKNESS, rail_height], dtype=np.float64)
    rail_y_offset = float(0.5 * plug_dims[1] + SOCKET_HOLD_CLEARANCE_Y + 0.5 * SOCKET_HOLD_SIDE_RAIL_THICKNESS)
    rail_z = float(plug_center[2])
    left_rail_center = np.array([plug_center[0], plug_center[1] + rail_y_offset, rail_z], dtype=np.float64)
    right_rail_center = np.array([plug_center[0], plug_center[1] - rail_y_offset, rail_z], dtype=np.float64)

    make_invisible_collision_box(
        f"{SOCKET_HOLD_PROXY_ROOT_PATH}/BottomShelf",
        shelf_center,
        shelf_size,
        material,
    )
    make_invisible_collision_box(
        f"{SOCKET_HOLD_PROXY_ROOT_PATH}/LeftSideRail",
        left_rail_center,
        rail_size,
        material,
    )
    make_invisible_collision_box(
        f"{SOCKET_HOLD_PROXY_ROOT_PATH}/RightSideRail",
        right_rail_center,
        rail_size,
        material,
    )

    print("=" * 88)
    print("[POST-INSERT SOCKET HOLD PROXY]")
    print("  timing:             after successful insert, before gripper open")
    print("  visual_geometry:    hidden")
    print("  physics_collision:  enabled static cradle")
    print(f"  root_path:          {SOCKET_HOLD_PROXY_ROOT_PATH}")
    print(f"  plug_bbox_min:      {fmt_vec(plug_min)}")
    print(f"  plug_bbox_max:      {fmt_vec(plug_max)}")
    print(f"  plug_dims_mm:       {fmt_vec(plug_dims * 1000.0, 3)}")
    print(f"  shelf_center:       {fmt_vec(shelf_center)}")
    print(f"  shelf_size_mm:      {fmt_vec(shelf_size * 1000.0, 3)}")
    print(f"  side_rail_size_mm:  {fmt_vec(rail_size * 1000.0, 3)}")
    print(f"  side_clearance_y_mm:{SOCKET_HOLD_CLEARANCE_Y * 1000.0:.3f}")
    print("  reason: restored imported mesh collision was still not retaining the released cable.")
    print("=" * 88)


def disable_collision_in_bbox(root_path: str, region_min: np.ndarray, region_max: np.ndarray, label: str = "") -> typing.Tuple[int, int, int]:
    """Disable visual-asset collisions only where the insert corridor passes.

    The failure mode now is not bad alignment; it is some invisible imported
    DataHall collider touching the plug/fingers in the last centimeters. This
    walks a broader AS4610 subtree and disables any collision-bearing prim whose
    world bbox intersects an expanded insertion tunnel. Visuals are left alone.
    """

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[INSERT COLLISION TUNNEL WARNING] Missing root prim: {root_path}")
        return 0, 0, 0

    scanned = 0
    disabled = 0
    bbox_failed = 0
    for prim in Usd.PrimRange(root):
        # Tag everything in the tunnel as no-collision. Only count disabling when
        # a collision API or collision-ish subtree is actually touched.
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
        is_boundable = prim.IsA(UsdGeom.Boundable) or prim.IsA(UsdGeom.Gprim)
        if not (has_collision or is_boundable):
            continue
        scanned += 1

        try:
            mn, mx, _, _ = get_bbox(str(prim.GetPath()))
        except Exception:
            bbox_failed += 1
            if has_collision and _collision_enabled_attr_off(prim):
                disabled += 1
            continue

        if not _bbox_intersects(mn, mx, region_min, region_max):
            continue

        if _collision_enabled_attr_off(prim):
            disabled += 1

    print("=" * 88)
    print("[AS4610 INSERT COLLISION TUNNEL]")
    print(f"  label:             {label or 'insert tunnel'}")
    print(f"  root_path:         {root_path}")
    print(f"  region_min:        {fmt_vec(region_min)}")
    print(f"  region_max:        {fmt_vec(region_max)}")
    print("  visual_geometry:   kept visible")
    print("  physics_collision: disabled for colliders intersecting this region")
    print(f"  scanned_prims:     {scanned}")
    print(f"  disabled_prims:    {disabled}")
    print(f"  bbox_failed_prims: {bbox_failed}")
    print("  reason: remove hidden parent/sibling/rack colliders that look like 'nothing' in the viewport.")
    print("=" * 88)
    return scanned, disabled, bbox_failed


def apply_as4610_insert_collision_tunnel() -> None:
    if not DISABLE_AS4610_INSERT_COLLISION_TUNNEL:
        print("[AS4610 INSERT COLLISION TUNNEL] disabled by config")
        return

    center_yz = 0.5 * (PLUG_TARGET_POSITION[1:3] + INSERT_TARGET_POSITION[1:3])
    x_min = min(float(PLUG_TARGET_POSITION[0]), float(INSERT_TARGET_POSITION[0])) - INSERT_COLLISION_TUNNEL_X_BACK_PAD
    x_max = max(float(PLUG_TARGET_POSITION[0]), float(INSERT_TARGET_POSITION[0])) + INSERT_COLLISION_TUNNEL_X_FRONT_PAD
    region_min = np.array([
        x_min,
        float(center_yz[0]) - INSERT_COLLISION_TUNNEL_Y_PAD,
        float(center_yz[1]) - INSERT_COLLISION_TUNNEL_Z_PAD,
    ], dtype=np.float64)
    region_max = np.array([
        x_max,
        float(center_yz[0]) + INSERT_COLLISION_TUNNEL_Y_PAD,
        float(center_yz[1]) + INSERT_COLLISION_TUNNEL_Z_PAD,
    ], dtype=np.float64)

    disable_collision_in_bbox(
        AS4610_INSERT_COLLISION_ROOT_PATH,
        region_min,
        region_max,
        label="selected AS4610 plug+gripper insertion corridor",
    )



def spawn_user_target_marker() -> None:
    """Spawn a small no-collision visual marker at the selected pre-insert target."""

    if not SHOW_USER_TARGET_MARKER:
        return

    stage = omni.usd.get_context().get_stage()
    remove_prim_if_exists(USER_TARGET_MARKER_PATH)

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(USER_TARGET_MARKER_PATH))
    cube.CreateSizeAttr(float(USER_TARGET_MARKER_SIZE))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.0, 1.0, 0.0)])

    xform = UsdGeom.Xformable(cube.GetPrim())
    target = PLUG_TARGET_POSITION
    xform.AddTranslateOp().Set(Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])))

    print("[USER CABLE PRE-INSERT TARGET MARKER]")
    print(f"  marker_path={USER_TARGET_MARKER_PATH}")
    print(f"  target_plug_transform_translate={fmt_vec(PLUG_TARGET_POSITION)}")
    print("  collision=disabled")


def make_visual_target_box(
    prim_path: str,
    center: np.ndarray,
    size: np.ndarray,
    color=(1.0, 0.55, 0.0),
    collision: bool = False,
) -> None:
    """Spawn a visible target/port block, optionally with collision.

    Collision is off by default because this block is centered on the final plug
    target. A colliding block there would push the connector away from the pose
    we are trying to measure.
    """

    stage = omni.usd.get_context().get_stage()
    remove_prim_if_exists(prim_path)

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
    xform.AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))

    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def spawn_final_insert_target_block() -> None:
    """Spawn a visible block at the final insert target."""

    if not SHOW_FINAL_TARGET_BLOCK:
        return

    make_visual_target_box(
        FINAL_TARGET_BLOCK_PATH,
        INSERT_TARGET_POSITION,
        FINAL_TARGET_BLOCK_SIZE,
        color=(1.0, 0.55, 0.0),
        collision=FINAL_TARGET_BLOCK_COLLISION,
    )

    print("[FINAL INSERT TARGET BLOCK]")
    print(f"  block_path={FINAL_TARGET_BLOCK_PATH}")
    print(f"  center_final_insert_transform={fmt_vec(INSERT_TARGET_POSITION)}")
    print(f"  size_mm={fmt_vec(FINAL_TARGET_BLOCK_SIZE * 1000.0, 2)}")
    print(f"  collision={'enabled' if FINAL_TARGET_BLOCK_COLLISION else 'disabled'}")


def _set_schema_attr(api, create_attr_name: str, value) -> bool:
    """Best-effort setter for generated USD/PhysX schema attributes."""

    create_attr = getattr(api, create_attr_name, None)
    if create_attr is None:
        return False

    try:
        attr = create_attr()
    except TypeError:
        attr = create_attr(value)

    attr.Set(value)
    return True


def create_high_grip_physics_material() -> UsdShade.Material:
    """Create one reusable physics material for plug/finger contact."""

    stage = omni.usd.get_context().get_stage()
    material = UsdShade.Material.Define(stage, Sdf.Path(GRIP_MATERIAL_PATH))
    prim = material.GetPrim()

    usd_physics_mat = UsdPhysics.MaterialAPI.Apply(prim)
    _set_schema_attr(usd_physics_mat, "CreateStaticFrictionAttr", GRIP_STATIC_FRICTION)
    _set_schema_attr(usd_physics_mat, "CreateDynamicFrictionAttr", GRIP_DYNAMIC_FRICTION)
    _set_schema_attr(usd_physics_mat, "CreateRestitutionAttr", GRIP_RESTITUTION)

    # Force high friction to win even if the other contacting surface has a lower
    # material. These attributes exist in PhysX-backed Isaac builds; the guards
    # keep the script usable if a schema version omits them.
    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    _set_schema_attr(physx_mat, "CreateFrictionCombineModeAttr", "max")
    _set_schema_attr(physx_mat, "CreateRestitutionCombineModeAttr", "min")

    return material


def _bind_material_to_prim(prim: Usd.Prim, material: UsdShade.Material) -> None:
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    try:
        binding_api.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    except TypeError:
        try:
            binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants)
        except TypeError:
            binding_api.Bind(material)


def _tune_collision_contact(prim: Usd.Prim) -> bool:
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        return False

    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    touched = False
    touched |= _set_schema_attr(physx_collision, "CreateContactOffsetAttr", GRIP_CONTACT_OFFSET)
    touched |= _set_schema_attr(physx_collision, "CreateRestOffsetAttr", GRIP_REST_OFFSET)
    return touched


def bind_high_grip_material_recursive(root_path: str, material: UsdShade.Material) -> typing.Tuple[int, int]:
    """Bind high-friction material to all geometry/collider prims below root_path."""

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[HIGH GRIP WARNING] Missing prim: {root_path}")
        return 0, 0

    bound_count = 0
    contact_tuned_count = 0

    for prim in Usd.PrimRange(root):
        is_geom = prim.IsA(UsdGeom.Gprim)
        is_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
        if not (is_geom or is_collider):
            continue

        _bind_material_to_prim(prim, material)
        bound_count += 1
        if _tune_collision_contact(prim):
            contact_tuned_count += 1

    # Binding the root gives descendants a fallback even if the referenced asset
    # has collision prims that are not regular Gprims.
    if bound_count == 0:
        _bind_material_to_prim(root, material)
        bound_count = 1

    return bound_count, contact_tuned_count


def apply_high_grip_setup() -> None:
    """Make the physical grasp grippier without attaching the cable to the robot."""

    material = create_high_grip_physics_material()

    targets = [
        # V22: bind the entire cable asset, not just the tracked plug prim.
        # The actual contact collider can live on a sibling/child mesh, so only
        # binding E_crystal_head1_45 can miss the surface the fingers touch.
        NETWORK_CABLE_ROOT_PATH,
        TRACKED_PLUG_PRIM_PATH,
        f"{FRANKA_PATH}/panda_leftfinger",
        f"{FRANKA_PATH}/panda_rightfinger",
    ]

    print("=" * 88)
    print("[APPLYING HIGH GRIP SETUP]")
    print("  No robot-to-cable attachment is being created.")
    print("  This only changes contact friction, restitution, contact offsets, and clamp speed.")
    print(f"  gripper_action_delta={GRIPPER_ACTION_DELTA:.4f}m/frame")
    print(f"  close_wait_frames={GRIP_CLOSE_WAIT_FRAMES}")
    print(f"  static_friction={GRIP_STATIC_FRICTION:.2f}")
    print(f"  dynamic_friction={GRIP_DYNAMIC_FRICTION:.2f}")
    print(f"  contact_offset={GRIP_CONTACT_OFFSET * 1000.0:.1f}mm")

    total_bound = 0
    total_contact_tuned = 0
    for path in targets:
        bound, contact_tuned = bind_high_grip_material_recursive(path, material)
        total_bound += bound
        total_contact_tuned += contact_tuned
        print(f"  target={path}")
        print(f"    bound_prims={bound}")
        print(f"    contact_tuned_prims={contact_tuned}")

    print(f"  total_bound_prims={total_bound}")
    print(f"  total_contact_tuned_prims={total_contact_tuned}")
    print("=" * 88)


def enable_gpu_dynamics() -> None:
    stage = omni.usd.get_context().get_stage()
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        scenes = [UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene")).GetPrim()]

    for prim in scenes:
        api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        api.CreateEnableGPUDynamicsAttr(True).Set(True)
        api.CreateBroadphaseTypeAttr("GPU").Set("GPU")
        api.CreateSolverTypeAttr("TGS").Set("TGS")


# =============================================================================
# 5. SPAWN ROBOT / BLOCK / CABLE
# =============================================================================

def spawn_robot(world: World) -> SingleManipulator:
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not find Isaac Sim assets folder")

    robot_prim = add_reference_to_stage(
        usd_path=assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        prim_path=FRANKA_PATH,
    )

    robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    physics_variant = robot_prim.GetVariantSet("Physics")
    physics_variants = list(physics_variant.GetVariantNames())
    if physics_variants:
        physics_variant.SetVariantSelection(
            next((v for v in physics_variants if v.lower() == "physx"), physics_variants[0])
        )
    # Keep the robot setup matched to the standalone good pickup file.
    # Do not enable extra articulation self-collisions here; that was not part
    # of the proven cable grasp and can change finger/contact behavior.

    gripper = ParallelGripper(
        end_effector_prim_path=f"{FRANKA_PATH}/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.001, 0.001]),
        action_deltas=np.array([GRIPPER_ACTION_DELTA, GRIPPER_ACTION_DELTA]),
    )

    return world.scene.add(
        SingleManipulator(
            prim_path=FRANKA_PATH,
            name="my_franka",
            end_effector_prim_path=f"{FRANKA_PATH}/panda_rightfinger",
            gripper=gripper,
            position=ROBOT_BASE_POS,
        )
    )


def spawn_block_and_posts() -> float:
    """Spawn the blue support block plus the two red slot posts."""

    remove_prim_if_exists(BLOCK_PATH)
    remove_prim_if_exists(POST_LEFT_PATH)
    remove_prim_if_exists(POST_RIGHT_PATH)

    block_top_z = BLOCK_CENTER[2] + 0.5 * BLOCK_SIZE[2]

    make_static_box(
        BLOCK_PATH,
        BLOCK_CENTER,
        BLOCK_SIZE,
        color=(0.2, 0.35, 1.0),
    )

    inner_y = 0.5 * POST_GAP_Y + 0.5 * POST_SIZE[1]
    post_z = block_top_z + 0.5 * POST_SIZE[2]
    post_x = BLOCK_CENTER[0] + POST_X_OFFSET

    left_post_center = np.array(
        [post_x, BLOCK_CENTER[1] + inner_y, post_z],
        dtype=np.float64,
    )
    right_post_center = np.array(
        [post_x, BLOCK_CENTER[1] - inner_y, post_z],
        dtype=np.float64,
    )

    make_static_box(
        POST_LEFT_PATH,
        left_post_center,
        POST_SIZE,
        color=(1.0, 0.0, 0.0),
    )
    make_static_box(
        POST_RIGHT_PATH,
        right_post_center,
        POST_SIZE,
        color=(1.0, 0.0, 0.0),
    )

    print("[SUPPORT SPAWNED]")
    print(f"  block_center={np.round(BLOCK_CENTER, 5)}")
    print(f"  fixed_pickup_world_y={FIXED_PICKUP_WORLD_Y:.5f}")
    print(f"  pickup_support_scale={PICKUP_SUPPORT_SCALE:.2f}")
    print(f"  block_size_mm={np.round(BLOCK_SIZE * 1000.0, 2)}")
    print(f"  block_top_z={block_top_z:.5f}")
    print(f"  left_post_center={np.round(left_post_center, 5)}")
    print(f"  right_post_center={np.round(right_post_center, 5)}")
    print(f"  post_size_mm={np.round(POST_SIZE * 1000.0, 2)}")
    print(f"  post_gap_y_mm={POST_GAP_Y * 1000.0:.2f}")
    print(f"  post_x_offset_mm={POST_X_OFFSET * 1000.0:.2f}")

    return block_top_z


def spawn_cable_on_block(block_top_z: float) -> None:
    remove_prim_if_exists(NETWORK_CABLE_ROOT_PATH)

    add_reference_to_stage(
        usd_path=NETWORK_CABLE_USD_PATH,
        prim_path=NETWORK_CABLE_ROOT_PATH,
    )

    stage = omni.usd.get_context().get_stage()

    plug_min, _, plug_center, _ = get_bbox(TRACKED_PLUG_PRIM_PATH)
    root_prim = stage.GetPrimAtPath(NETWORK_CABLE_ROOT_PATH)
    root_tf = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(root_prim)
    root_t = np.array(root_tf.ExtractTranslation(), dtype=np.float64)

    desired_plug_center_xy = BLOCK_CENTER[:2]
    desired_plug_min_z = block_top_z + PLUG_BLOCK_CLEARANCE

    delta = np.array(
        [
            desired_plug_center_xy[0] - plug_center[0],
            desired_plug_center_xy[1] - plug_center[1],
            desired_plug_min_z - plug_min[2],
        ],
        dtype=np.float64,
    )

    set_world_translate(NETWORK_CABLE_ROOT_PATH, root_t + delta)

    plug_min, plug_max, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)

    print("[CABLE SPAWNED]")
    print(f"  cable_root={NETWORK_CABLE_ROOT_PATH}")
    print(f"  tracked_plug={TRACKED_PLUG_PRIM_PATH}")
    print(f"  plug_center={np.round(plug_center, 5)}")
    print(f"  plug_dims_mm={np.round(plug_dims * 1000.0, 3)}")
    print(f"  plug_min_z={plug_min[2]:.5f}")
    print(f"  plug_max_z={plug_max[2]:.5f}")
    log_cable_pose("AFTER cable spawn/place")



def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Cannot normalize zero-length vector")
    return v / n


def orientation_for_cable_axis(axis_world: np.ndarray) -> np.ndarray:
    """Quaternion that points cable local +X along the port insert axis.

    The cable USD plug length axis is local +X. This keeps the local +Z as close
    as possible to world-up so the connector stays horizontal like the working
    non-DataHall cable insertion.
    """

    x_axis = _normalize_vec(axis_world)
    z_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(x_axis, z_hint))) > 0.95:
        z_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = z_hint - x_axis * float(np.dot(z_hint, x_axis))
    z_axis = _normalize_vec(z_axis)
    y_axis = _normalize_vec(np.cross(z_axis, x_axis))
    rot = np.column_stack([x_axis, y_axis, z_axis])
    quat = rot_matrices_to_quats(rot.reshape(1, 3, 3))[0]
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.linalg.norm(quat)


def measure_plug_front_to_xform_offset_for_minus_x_insert() -> float:
    """Return distance from tracked plug xform center to the leading -X nose.

    The port insertion direction is -X after the plug is reoriented. The cable
    xform translate is near the plug visual center, not the front tip. A 50 mm
    pre-insert gap must be measured from the front tip, otherwise the connector
    is already much closer to the port than the debug numbers claim.
    """

    plug_min, plug_max, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)
    # The plug is still roughly X-aligned on the pickup support. Its X bbox
    # length is the same dimension that becomes the -X insertion length.
    half_len = 0.5 * float(plug_dims[0])
    center_to_min = abs(float(plug_center[0] - plug_min[0]))
    center_to_max = abs(float(plug_max[0] - plug_center[0]))
    return float(max(half_len, center_to_min, center_to_max))


def _linspace_port_centers(lo: float, hi: float, count: int, margin_fraction: float) -> np.ndarray:
    """Return evenly spaced inferred port centers inside a component bbox axis."""

    lo = float(lo)
    hi = float(hi)
    span = hi - lo
    margin = span * float(margin_fraction)
    if count <= 1:
        return np.array([0.5 * (lo + hi)], dtype=np.float64)
    return np.linspace(lo + margin, hi - margin, count, dtype=np.float64)


def _selected_as4610_rj45_face_center() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Infer one RJ45 mouth center from the AS4610 12-pack component bbox.

    The AS4610 asset exposes the full orange 12-port RJ45 cage as one component,
    not 12 individual jack prims. The inspection script showed the component bbox
    is a clean 2 x 6 grid, so we target row/column centers inside that bbox.
    """

    if not (0 <= int(TARGET_RJ45_ROW) < AS4610_PORT_ROWS):
        raise ValueError(f"TARGET_RJ45_ROW must be 0..{AS4610_PORT_ROWS - 1}, got {TARGET_RJ45_ROW}")
    if not (0 <= int(TARGET_RJ45_COL) < AS4610_PORT_COLS):
        raise ValueError(f"TARGET_RJ45_COL must be 0..{AS4610_PORT_COLS - 1}, got {TARGET_RJ45_COL}")

    comp_min, comp_max, comp_center, comp_dims = get_bbox(DATAHALL_PORT_PRIM_PATH)

    # For the DataHall AS4610 front, robot approaches from +X and inserts along -X.
    port_face_x = float(comp_max[0]) + float(DATAHALL_PORT_FACE_EXTRA_CLEARANCE_X_M)

    y_centers = _linspace_port_centers(
        float(comp_min[1]),
        float(comp_max[1]),
        AS4610_PORT_COLS,
        AS4610_Y_MARGIN_FRACTION,
    )
    z_low_to_high = _linspace_port_centers(
        float(comp_min[2]),
        float(comp_max[2]),
        AS4610_PORT_ROWS,
        AS4610_Z_MARGIN_FRACTION,
    )
    # User-facing row 0 is the top row in the screenshot, so reverse the Z list.
    z_centers = z_low_to_high[::-1]

    face_center = np.array(
        [
            port_face_x,
            float(y_centers[int(TARGET_RJ45_COL)]),
            float(z_centers[int(TARGET_RJ45_ROW)]),
        ],
        dtype=np.float64,
    )
    # Small final visual correction requested after V10: shift the entire
    # pre-insert/final insert line 1.0 mm right and 0.5 mm up. X stays unchanged
    # so insertion depth is not affected.
    face_center = face_center + DATAHALL_PORT_LATERAL_OFFSET
    return face_center, comp_min, comp_max, comp_dims


def configure_datahall_cable_targets() -> None:
    """Target one inferred AS4610 RJ45 port from the 2x6 component bbox.

    The old QSFP flow used PortFrame because each QSFP connector had its own prim.
    The AS4610 RJ45 block does not. This function replaces PortFrame with a
    bbox-grid target: choose row/col, compute the port face, then use the same
    front-tip-referenced cable insertion logic from V6.
    """

    global USER_SELECTED_CABLE_PRE_INSERT_POSITION
    global USER_SELECTED_CABLE_FINAL_INSERT_POSITION
    global USER_SELECTED_CABLE_TARGET_ORI_WXYZ
    global PLUG_TARGET_POSITION, PLUG_FINAL_INSERT_POSITION, PLUG_TARGET_ORI_WXYZ
    global PLUG_INSERT_AXIS_WORLD, INSERT_TARGET_POSITION, INSERT_AXIS_WORLD
    global COARSE_TRANSFER_Z, COARSE_TRANSFER_POSITION, COARSE_APPROACH_TARGET_POSITION, COARSE_NEAR_TARGET_POSITION
    global PLUG_FRONT_TO_XFORM_X_OFFSET_M, DATAHALL_PORT_FACE_X_M
    global TABLE_HEIGHT, ROBOT_BASE_POS, BLOCK_CENTER, PICKUP_GRASP_POSITION

    face_center, comp_min, comp_max, comp_dims = _selected_as4610_rj45_face_center()
    port_face_x = float(face_center[0])
    DATAHALL_PORT_FACE_X_M = port_face_x

    try:
        PLUG_FRONT_TO_XFORM_X_OFFSET_M = measure_plug_front_to_xform_offset_for_minus_x_insert()
    except Exception as exc:
        print(f"[DATAHALL TARGET WARNING] Could not measure plug front offset from bbox: {exc}")
        print(f"[DATAHALL TARGET WARNING] Falling back to {PLUG_FRONT_TO_XFORM_X_OFFSET_M * 1000.0:.1f} mm")

    plug_front_offset = float(PLUG_FRONT_TO_XFORM_X_OFFSET_M)

    # For -X insertion, plug_front_x = plug_center_x - plug_front_offset.
    pre_center_x = port_face_x + DATAHALL_PRE_INSERT_FRONT_GAP_M + plug_front_offset
    final_center_x = port_face_x - DATAHALL_FINAL_PLUG_FRONT_AXIAL_DEPTH_M + plug_front_offset

    pre_insert_pos = np.array([pre_center_x, face_center[1], face_center[2]], dtype=np.float64)
    final_pos = np.array([final_center_x, face_center[1], face_center[2]], dtype=np.float64)

    target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    USER_SELECTED_CABLE_PRE_INSERT_POSITION = pre_insert_pos.copy()
    USER_SELECTED_CABLE_FINAL_INSERT_POSITION = final_pos.copy()
    USER_SELECTED_CABLE_TARGET_ORI_WXYZ = target_quat.copy()

    PLUG_TARGET_POSITION = pre_insert_pos.copy()
    PLUG_FINAL_INSERT_POSITION = final_pos.copy()
    PLUG_TARGET_ORI_WXYZ = target_quat.copy()
    PLUG_INSERT_AXIS_WORLD = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    INSERT_TARGET_POSITION = final_pos.copy()
    INSERT_AXIS_WORLD = PLUG_INSERT_AXIS_WORLD.copy()

    # Keep the work table and Franka base at the fixed AS4610 height requested by
    # the user. IMPORTANT: do NOT lock robot base Y to the pickup cable Y. The
    # cable/support pickup island is intentionally offset at y=-0.600 so the long
    # cable does not run through the robot base. The Franka base should stay in
    # the original port-relative lane that was visually good.
    TABLE_HEIGHT = 2.35
    ROBOT_BASE_POS = np.array(
        [ROBOT_BASE_X_M, float(face_center[1] + ROBOT_BASE_Y_OFFSET_FROM_TARGET_M), TABLE_HEIGHT],
        dtype=np.float64,
    )
    # Preserve the standalone pickup exactly relative to the shifted Franka base.
    # Do not derive this from the port, target height, or table collision geometry.
    BLOCK_CENTER = pickup_local_to_world(PICKUP_LOCAL_BLOCK_CENTER)
    PICKUP_GRASP_POSITION = pickup_local_to_world(PICKUP_LOCAL_GRASP_POSITION)

    # Preserve the standalone pickup lift/carry height relative to the Franka base.
    # In the good pickup file, local carry was 0.35 m for the 0.325 m target.
    # The previous AS4610 file used TABLE_HEIGHT+0.325, which lowered the lift by
    # 25 mm and changed the cable loading during grasp.
    COARSE_TRANSFER_Z = max(ROBOT_BASE_POS[2] + 0.350, float(PLUG_TARGET_POSITION[2]) + COARSE_TARGET_CLEARANCE_Z)
    COARSE_TRANSFER_POSITION = np.array([0.0, COARSE_TRANSFER_LANE_Y, COARSE_TRANSFER_Z], dtype=np.float64)
    COARSE_APPROACH_TARGET_POSITION = np.array(
        [float(PLUG_TARGET_POSITION[0]), float(PLUG_TARGET_POSITION[1]), COARSE_TRANSFER_Z],
        dtype=np.float64,
    )
    COARSE_NEAR_TARGET_POSITION = PLUG_TARGET_POSITION.copy()

    print("=" * 88)
    print("[DATAHALL AS4610 RJ45 TARGET - BBOX GRID + TIP-REFERENCED INSERT]")
    print(f"  component_prim:        {DATAHALL_PORT_PRIM_PATH}")
    print(f"  selected_port_id:      r{TARGET_RJ45_ROW}_c{TARGET_RJ45_COL}")
    print(f"  component_bbox_min:    {fmt_vec(comp_min)}")
    print(f"  component_bbox_max:    {fmt_vec(comp_max)}")
    print(f"  component_bbox_dims_mm:{fmt_vec(comp_dims * 1000.0, 3)}")
    print(f"  grid_rows_cols:        {AS4610_PORT_ROWS} x {AS4610_PORT_COLS}")
    print(f"  grid_margins_yz:       {AS4610_Y_MARGIN_FRACTION:.3f} / {AS4610_Z_MARGIN_FRACTION:.3f}")
    print(f"  port_face_center:      {fmt_vec(face_center)}")
    print(f"  applied_yz_offset_mm:  right_y={DATAHALL_PORT_LATERAL_OFFSET[1] * 1000.0:.3f}, up_z={DATAHALL_PORT_LATERAL_OFFSET[2] * 1000.0:.3f}")
    print(f"  port_face_x_bbox_max:  {port_face_x:.5f}")
    print("  servo_insert_axis:     [-1.  0.  0.]  # strict -X RJ45 insert")
    print(f"  plug_front_to_xform_mm:{PLUG_FRONT_TO_XFORM_X_OFFSET_M * 1000.0:.3f}")
    print(f"  pre_insert_front_gap_mm:{DATAHALL_PRE_INSERT_FRONT_GAP_M * 1000.0:.1f}")
    print(f"  final_front_depth_mm:  {DATAHALL_FINAL_PLUG_FRONT_AXIAL_DEPTH_M * 1000.0:.1f}")
    print(f"  pre_insert_plug_center:{fmt_vec(PLUG_TARGET_POSITION)}")
    print(f"  final_insert_plug_center:{fmt_vec(INSERT_TARGET_POSITION)}")
    print(f"  pre_insert_front_x:    {PLUG_TARGET_POSITION[0] - PLUG_FRONT_TO_XFORM_X_OFFSET_M:.5f}")
    print(f"  final_front_x:         {INSERT_TARGET_POSITION[0] - PLUG_FRONT_TO_XFORM_X_OFFSET_M:.5f}")
    print(f"  robot_base_pos:        {fmt_vec(ROBOT_BASE_POS)}  # original port-relative Y lane")
    print(f"  robot_to_pickup_y_delta_mm:{(FIXED_PICKUP_WORLD_Y - ROBOT_BASE_POS[1]) * 1000.0:.3f}")
    print(f"  robot_to_port_y_delta_mm:{(face_center[1] - ROBOT_BASE_POS[1]) * 1000.0:.3f}")
    print(f"  table_height:          {TABLE_HEIGHT:.5f}  # fixed AS4610 work-table height")
    print(f"  pickup_block_center:   {fmt_vec(BLOCK_CENTER)}")
    print(f"  pickup_grasp_position: {fmt_vec(PICKUP_GRASP_POSITION)}")
    print(f"  fixed_pickup_world_y:  {FIXED_PICKUP_WORLD_Y:.5f}")
    print(f"  pickup_support_scale:  {PICKUP_SUPPORT_SCALE:.2f}  # support only; cable USD already manually scaled")
    print(f"  pickup_local_block:    {fmt_vec(PICKUP_LOCAL_BLOCK_CENTER)}  # original working pickup island center")
    print(f"  pickup_local_grasp:    {fmt_vec(PICKUP_LOCAL_GRASP_POSITION)}  # same pickup, raised to scaled block top")
    print(f"  plug_target_ori_wxyz:  {fmt_vec(PLUG_TARGET_ORI_WXYZ)}")
    print(f"  coarse_hand_ori_wxyz:  {fmt_vec(DIAGONAL_INSERT_ORI)}")
    print(f"  coarse_transfer_z:     {COARSE_TRANSFER_Z:.5f}")
    print("  technique: land at 50 mm pre-insert front gap, then closed-loop reorient/lock the tracked plug horizontal before X insertion.")
    print("  note: AS4610 has no per-jack prim path; this is a row/column bbox-grid target.")
    print("=" * 88)


def build_work_table() -> None:
    """Spawn one simple table under the robot/cable pickup area."""

    center = np.array([0.30, float(ROBOT_BASE_POS[1] - 0.30), TABLE_HEIGHT - TABLE_THICKNESS * 0.5], dtype=np.float64)
    size = np.array([1.40, 1.10, TABLE_THICKNESS], dtype=np.float64)
    make_static_box(WORK_TABLE_PATH, center, size, color=(0.55, 0.35, 0.18))
    print("[WORK TABLE SPAWNED]")
    print(f"  table_center={fmt_vec(center)}")
    print(f"  table_size={fmt_vec(size)}")
    print(f"  table_top_z={TABLE_HEIGHT:.5f}")


# =============================================================================
# 6. CONTROLLER / IK SETUP
# =============================================================================

def build_controller(franka: SingleManipulator):
    kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")

    kinematics_solver = LulaKinematicsSolver(**kinematics_config)
    task_traj_gen = LulaTaskSpaceTrajectoryGenerator(**kinematics_config)
    art_kinematics = ArticulationKinematicsSolver(franka, kinematics_solver, "panda_hand")

    base_position, base_orientation = franka.get_world_pose()
    kinematics_solver.set_robot_base_pose(base_position, base_orientation)

    controller = FrankaMotionController(
        name="datahall_cable_support18x_yminus06_pathstrict_fast_v24",
        robot_articulation=franka,
        task_traj_gen=task_traj_gen,
        art_kinematics=art_kinematics,
        gripper=franka.gripper,
        tool_offset=TOOL_OFFSET,
        debug=False,
    )

    return controller, kinematics_solver, art_kinematics


def build_scene_and_controller():
    world = World(stage_units_in_meters=1.0)
    world.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)

    stage = omni.usd.get_context().get_stage()

    # light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    # light.CreateIntensityAttr(500.0)
    # light.CreateColorAttr((1.0, 1.0, 1.0))

    print("[SCENE] Loading DataHall...")
    add_reference_to_stage(usd_path=DATAHALL_USD, prim_path="/World/DataHall")
    apply_datahall_scale("/World/DataHall", DATAHALL_SCALE)
    switch_collider_count = enable_static_collisions("/World/DataHall/Network_Switches", "none")
    print(f"[SCENE] Static switch colliders enabled: {switch_collider_count}")
    apply_target_rj45_forgiving_collision()

    # assets_root = get_assets_root_path()
    # if assets_root is not None:
    #     add_reference_to_stage(
    #         usd_path=assets_root + "/Isaac/Environments/Grid/default_environment.usd",
    #         prim_path="/World/ground",
    #     )
    # else:
    #     world.scene.add_default_ground_plane()

    # Configure AS4610 row/column target before spawning the robot. The target
    # still comes from the RJ45 bbox-grid, but the table/robot/cable spawn height
    # is fixed at TABLE_HEIGHT=2.35 m as requested.
    configure_datahall_cable_targets()
    apply_as4610_insert_collision_tunnel()
    build_work_table()

    franka = spawn_robot(world)
    block_top_z = spawn_block_and_posts()
    spawn_cable_on_block(block_top_z)
    spawn_user_target_marker()
    spawn_final_insert_target_block()
    apply_high_grip_setup()
    enable_gpu_dynamics()

    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    world.reset()

    controller, kinematics_solver, art_kinematics = build_controller(franka)

    return world, franka, controller, kinematics_solver, art_kinematics



# =============================================================================
# 7. CLOSED-LOOP PLUG POSE SERVO
# =============================================================================

PHASE_COARSE_WAYPOINTS = 0
PHASE_PLUG_POSE_SERVO = 1
PHASE_PLUG_INSERT_SERVO = 2
PHASE_POST_INSERT_RELEASE_RETREAT = 3
PHASE_DONE = 4

plug_pose_servo: dict = {}
plug_pose_samples: typing.List[dict] = []
plug_insert_servo: dict = {}
plug_insert_samples: typing.List[dict] = []


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).flatten()
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_to_rot_wxyz(quat: np.ndarray) -> np.ndarray:
    return quats_to_rot_matrices(normalize_quat_wxyz(quat).reshape(1, 4))[0]


def rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    quat = rot_matrices_to_quats(np.asarray(rot, dtype=np.float64).reshape(1, 3, 3))
    if quat.ndim > 1:
        quat = quat[0]
    return normalize_quat_wxyz(quat)


def quat_angle_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    a = normalize_quat_wxyz(actual)
    b = normalize_quat_wxyz(target)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def quat_slerp_shortest(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = normalize_quat_wxyz(q0)
    q1 = normalize_quat_wxyz(q1)
    t = float(np.clip(fraction, 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quat_wxyz(q0 + t * (q1 - q0))
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return normalize_quat_wxyz(s0 * q0 + s1 * q1)


def rotation_angle_error_deg(actual_rot: np.ndarray, target_rot: np.ndarray) -> float:
    rel = np.asarray(target_rot, dtype=np.float64).T @ np.asarray(actual_rot, dtype=np.float64)
    trace_value = float(np.trace(rel))
    cos_angle = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def get_hand_pose_matrix() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hand_pos, hand_rot = art_kinematics.compute_end_effector_pose()
    hand_pos = np.asarray(hand_pos, dtype=np.float64).flatten()
    hand_rot = np.asarray(hand_rot, dtype=np.float64)
    if hand_rot.ndim == 3:
        hand_rot = hand_rot[0]
    hand_quat = rot_to_quat_wxyz(hand_rot)
    return hand_pos, hand_rot, hand_quat


def get_tracked_plug_pose_matrix() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return controlled plug position, rotation matrix, quaternion, and bbox center.

    Controlled position is the xform/world translate of E_crystal_head1_45 — the
    same value shown in the Isaac Sim Transform panel. The bbox center is returned
    only for debug so we can see how far visual geometry differs from the xform.
    """

    xform_pos, quat = get_prim_world_pose(TRACKED_PLUG_PRIM_PATH)
    if xform_pos is None or quat is None:
        raise RuntimeError(f"Could not read tracked plug pose: {TRACKED_PLUG_PRIM_PATH}")
    _, _, bbox_center, _ = get_bbox(TRACKED_PLUG_PRIM_PATH)
    quat = normalize_quat_wxyz(quat)
    rot = quat_to_rot_wxyz(quat)
    return np.asarray(xform_pos, dtype=np.float64), rot, quat, np.asarray(bbox_center, dtype=np.float64)


def plug_long_axis_world(plug_rot: np.ndarray) -> np.ndarray:
    axis = np.asarray(plug_rot, dtype=np.float64) @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 1e-12 else axis


def plug_axis_angle_to_insert_deg(plug_rot: np.ndarray) -> float:
    axis = plug_long_axis_world(plug_rot)
    target_axis = PLUG_INSERT_AXIS_WORLD / np.linalg.norm(PLUG_INSERT_AXIS_WORLD)
    dot = float(np.clip(np.dot(axis, target_axis), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def plug_tilt_out_of_horizontal_deg(plug_rot: np.ndarray) -> float:
    axis = plug_long_axis_world(plug_rot)
    return float(np.degrees(np.arcsin(np.clip(abs(axis[2]), 0.0, 1.0))))


def compute_plug_hand_offsets() -> typing.Tuple[np.ndarray, np.ndarray]:
    plug_pos, plug_rot, _, _ = get_tracked_plug_pose_matrix()
    hand_pos, hand_rot, _ = get_hand_pose_matrix()
    plug_offset_local = hand_rot.T @ (plug_pos - hand_pos)
    plug_rot_local = hand_rot.T @ plug_rot
    return plug_offset_local, plug_rot_local


def hand_pose_for_plug_pose(
    target_plug_pos: np.ndarray,
    target_plug_quat: np.ndarray,
    plug_offset_local: np.ndarray,
    plug_rot_local: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_plug_rot = quat_to_rot_wxyz(target_plug_quat)
    target_hand_rot = target_plug_rot @ np.asarray(plug_rot_local, dtype=np.float64).T
    target_hand_pos = np.asarray(target_plug_pos, dtype=np.float64) - target_hand_rot @ np.asarray(plug_offset_local, dtype=np.float64)
    target_hand_quat = rot_to_quat_wxyz(target_hand_rot)
    return target_hand_pos, target_hand_rot, target_hand_quat


def closed_gripper_hold_action(n_dof: int) -> ArticulationAction:
    return controller._with_closed_gripper(controller._hold_action(n_dof), n_dof)


def print_plug_pose_error(tag: str) -> typing.Tuple[float, float]:
    plug_pos, plug_rot, plug_quat, plug_bbox_center = get_tracked_plug_pose_matrix()
    target_quat = normalize_quat_wxyz(PLUG_TARGET_ORI_WXYZ)
    pos_err = plug_pos - PLUG_TARGET_POSITION
    pos_norm = float(np.linalg.norm(pos_err))
    yz_total = float(np.linalg.norm(pos_err[1:3]))
    ori_err = quat_angle_error_deg(plug_quat, target_quat)
    long_axis = plug_long_axis_world(plug_rot)
    axis_angle = plug_axis_angle_to_insert_deg(plug_rot)
    tilt = plug_tilt_out_of_horizontal_deg(plug_rot)

    print("=" * 88)
    print(f"[PLUG POSE CHECK] {tag}")
    bbox_minus_xform = plug_bbox_center - plug_pos
    print(f"  target_transform_pos:   {fmt_vec(PLUG_TARGET_POSITION)}")
    print(f"  actual_xform_translate: {fmt_vec(plug_pos)}  ← controlled value / Transform panel")
    print(f"  actual_bbox_center:     {fmt_vec(plug_bbox_center)}  secondary debug")
    print(f"  bbox_minus_xform_mm:    {fmt_vec(bbox_minus_xform * 1000.0, 3)}")
    print(f"  position_error_xyz_mm:  {fmt_vec(pos_err * 1000.0, 3)}")
    print(f"  position_error_norm_mm: {pos_norm * 1000.0:.3f}  legacy_limit={PLUG_POSITION_TOL * 1000.0:.3f}")
    print(f"  yz_total_error_mm:      {yz_total * 1000.0:.3f}  main_limit={PLUG_YZ_TOTAL_TOL * 1000.0:.3f}")
    print(f"  x_error_mm:             {pos_err[0] * 1000.0:.3f}  limit={PLUG_X_TOL * 1000.0:.3f}")
    print(f"  target_ori_wxyz:        {fmt_vec(target_quat)}")
    print(f"  actual_ori_wxyz:        {fmt_vec(plug_quat)}")
    print(f"  orientation_error_deg:  {ori_err:.3f}  limit={PLUG_ORIENTATION_TOL_DEG:.3f}")
    print(f"  plug_long_axis_world:   {fmt_vec(long_axis)}")
    print(f"  axis_angle_to_-X_deg:   {axis_angle:.3f}")
    print(f"  tilt_out_horizontal_deg:{tilt:.3f}")
    print("=" * 88)
    return pos_norm, ori_err


def print_plug_hand_offset_debug(tag: str) -> None:
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    hand_pos, hand_rot, hand_quat = get_hand_pose_matrix()
    offset_local = hand_rot.T @ (plug_pos - hand_pos)
    rot_local = hand_rot.T @ plug_rot

    ref_offset = plug_pose_servo.get("plug_offset_local")
    ref_rot = plug_pose_servo.get("plug_rot_local")
    offset_drift = float(np.linalg.norm(offset_local - ref_offset)) if ref_offset is not None else 0.0
    rot_drift = rotation_angle_error_deg(rot_local, ref_rot) if ref_rot is not None else 0.0

    print("-" * 88)
    print(f"[PLUG/HAND OFFSET DEBUG] {tag}")
    print(f"  hand_pos:              {fmt_vec(hand_pos)}")
    print(f"  hand_ori_wxyz:         {fmt_vec(hand_quat)}")
    _, _, bbox_center, _ = get_bbox(TRACKED_PLUG_PRIM_PATH)
    print(f"  plug_xform_translate:  {fmt_vec(plug_pos)}")
    print(f"  plug_bbox_center:      {fmt_vec(bbox_center)}")
    print(f"  plug_ori_wxyz:         {fmt_vec(plug_quat)}")
    print(f"  plug_offset_local:     {fmt_vec(offset_local)}")
    print(f"  plug_offset_drift_mm:  {offset_drift * 1000.0:.3f}")
    print(f"  plug_rot_local_drift:  {rot_drift:.3f} deg")
    if offset_drift > PLUG_LOCAL_DRIFT_WARN_POS or rot_drift > PLUG_LOCAL_DRIFT_WARN_ORI_DEG:
        print("  WARNING: plug is moving relative to the hand; this is grip/slip, not IK error.")
    print("-" * 88)


def init_plug_pose_servo() -> None:
    plug_pose_servo.clear()
    plug_pose_samples.clear()

    plug_offset_local, plug_rot_local = compute_plug_hand_offsets()
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    hand_pos, hand_rot, hand_quat = get_hand_pose_matrix()
    pos_err_norm, ori_err_deg = print_plug_pose_error("BEFORE closed-loop plug pose servo")

    target_hand_pos, _, target_hand_quat = hand_pose_for_plug_pose(
        PLUG_TARGET_POSITION,
        PLUG_TARGET_ORI_WXYZ,
        plug_offset_local,
        plug_rot_local,
    )

    plug_pose_servo.update({
        "frames": 0,
        "stable_frames": 0,
        "ik_fail_count": 0,
        "warned_ik": False,
        "plug_offset_local": plug_offset_local,
        "plug_rot_local": plug_rot_local,
        "initial_plug_pos": plug_pos.copy(),
        "initial_plug_quat": plug_quat.copy(),
        "initial_hand_pos": hand_pos.copy(),
        "initial_hand_quat": hand_quat.copy(),
        "target_hand_pos": target_hand_pos.copy(),
        "target_hand_quat": target_hand_quat.copy(),
        "max_position_error": pos_err_norm,
        "max_orientation_error_deg": ori_err_deg,
        "max_offset_drift": 0.0,
        "max_rot_local_drift_deg": 0.0,
        "yz_integral": np.zeros(2, dtype=np.float64),
        "last_yz_overdrive": np.zeros(2, dtype=np.float64),
        "last_command_plug_pos": plug_pos.copy(),
        "last_target_hand_pos": target_hand_pos.copy(),
        "last_servo_mode": "init",
        "last_pos_step_mm": 0.0,
    })

    print("=" * 88)
    print("[PLUG POSE SERVO INIT]")
    print("  control_object:          /World/NetworkCable/E_crystal_head1_45")
    print("  position_source:         tracked plug xform translate / Transform panel value")
    print("  orientation_source:      tracked plug xform quaternion")
    print(f"  plug_offset_local:       {fmt_vec(plug_offset_local)}")
    print(f"  target_transform_pos:    {fmt_vec(PLUG_TARGET_POSITION)}")
    print(f"  target_plug_ori_wxyz:    {fmt_vec(normalize_quat_wxyz(PLUG_TARGET_ORI_WXYZ))}")
    print(f"  target_hand_pos:         {fmt_vec(target_hand_pos)}")
    print(f"  target_hand_ori_wxyz:    {fmt_vec(target_hand_quat)}")
    print(f"  x_tolerance:             {PLUG_X_TOL * 1000.0:.3f} mm")
    print(f"  yz_total_tolerance:      {PLUG_YZ_TOTAL_TOL * 1000.0:.3f} mm  ← main port-position metric")
    print(f"  orientation_tolerance:   {PLUG_ORIENTATION_TOL_DEG:.3f} deg")
    print(f"  stable_hold_frames:      {PLUG_HOLD_FRAMES}")
    print(f"  fast/mid/fine pos step:  {PLUG_FAST_MAX_POS_STEP * 1000.0:.1f} / {PLUG_MID_MAX_POS_STEP * 1000.0:.1f} / {PLUG_FINE_MAX_POS_STEP * 1000.0:.1f} mm/frame")
    print(f"  fast/fine ori step:      {PLUG_FAST_ORI_STEP_DEG:.2f} / {PLUG_FINE_ORI_STEP_DEG:.2f} deg/frame")
    print(f"  yz_overdrive_kp/ki:      {PLUG_YZ_OVERDRIVE_KP:.3f} / {PLUG_YZ_OVERDRIVE_KI:.4f}")
    print(f"  yz_overdrive_limit:      {PLUG_YZ_MAX_OVERDRIVE * 1000.0:.3f} mm")
    print("=" * 88)
    print_plug_hand_offset_debug("SERVO INIT local plug-to-hand relationship")


def sample_plug_pose_servo_path(pos_err_norm: float, ori_err_deg: float) -> None:
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    hand_pos, hand_rot, _ = get_hand_pose_matrix()
    ref_offset = plug_pose_servo.get("plug_offset_local")
    ref_rot = plug_pose_servo.get("plug_rot_local")
    current_offset = hand_rot.T @ (plug_pos - hand_pos)
    current_rot_local = hand_rot.T @ plug_rot
    offset_drift = float(np.linalg.norm(current_offset - ref_offset)) if ref_offset is not None else 0.0
    rot_drift = rotation_angle_error_deg(current_rot_local, ref_rot) if ref_rot is not None else 0.0

    plug_pose_servo["max_position_error"] = max(float(plug_pose_servo.get("max_position_error", 0.0)), float(pos_err_norm))
    plug_pose_servo["max_orientation_error_deg"] = max(float(plug_pose_servo.get("max_orientation_error_deg", 0.0)), float(ori_err_deg))
    plug_pose_servo["max_offset_drift"] = max(float(plug_pose_servo.get("max_offset_drift", 0.0)), offset_drift)
    plug_pose_servo["max_rot_local_drift_deg"] = max(float(plug_pose_servo.get("max_rot_local_drift_deg", 0.0)), rot_drift)

    plug_pose_samples.append({
        "pos": plug_pos.copy(),
        "quat": plug_quat.copy(),
        "long_axis": plug_long_axis_world(plug_rot).copy(),
        "pos_err_norm": float(pos_err_norm),
        "ori_err_deg": float(ori_err_deg),
        "axis_angle_deg": plug_axis_angle_to_insert_deg(plug_rot),
        "tilt_deg": plug_tilt_out_of_horizontal_deg(plug_rot),
        "offset_drift": offset_drift,
        "rot_drift_deg": rot_drift,
    })


def update_plug_pose_servo_state() -> bool:
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    pos_err = PLUG_TARGET_POSITION - plug_pos
    pos_err_norm = float(np.linalg.norm(pos_err))
    x_err = float(pos_err[0])
    yz_err = np.asarray(pos_err[1:3], dtype=np.float64)
    yz_total = float(np.linalg.norm(yz_err))
    ori_err_deg = quat_angle_error_deg(plug_quat, PLUG_TARGET_ORI_WXYZ)
    sample_plug_pose_servo_path(pos_err_norm, ori_err_deg)

    plug_pose_servo["frames"] += 1
    inside_x = abs(x_err) <= PLUG_X_TOL
    inside_yz = yz_total <= PLUG_YZ_TOTAL_TOL
    inside_ori = ori_err_deg <= PLUG_ORIENTATION_TOL_DEG
    inside = inside_x and inside_yz and inside_ori
    if inside:
        plug_pose_servo["stable_frames"] += 1
    else:
        plug_pose_servo["stable_frames"] = 0

    if plug_pose_servo["frames"] == 1 or plug_pose_servo["frames"] % PLUG_SERVO_DEBUG_EVERY == 0 or inside:
        axis_angle = plug_axis_angle_to_insert_deg(plug_rot)
        tilt = plug_tilt_out_of_horizontal_deg(plug_rot)
        yz_overdrive = np.asarray(plug_pose_servo.get("last_yz_overdrive", np.zeros(2)), dtype=np.float64)
        command_plug_pos = np.asarray(plug_pose_servo.get("last_command_plug_pos", plug_pos), dtype=np.float64)
        servo_mode = str(plug_pose_servo.get("last_servo_mode", "?"))
        pos_step_mm = float(plug_pose_servo.get("last_pos_step_mm", 0.0))
        print(
            f"[PLUG PORT ALIGN SERVO] frame={plug_pose_servo['frames']} "
            f"pos_err={pos_err_norm * 1000.0:.3f}mm "
            f"x_err={x_err * 1000.0:.3f}mm "
            f"y_err={yz_err[0] * 1000.0:.3f}mm "
            f"z_err={yz_err[1] * 1000.0:.3f}mm "
            f"yz_total={yz_total * 1000.0:.3f}mm "
            f"ori_err={ori_err_deg:.3f}deg "
            f"axis_to_-X={axis_angle:.3f}deg "
            f"tilt={tilt:.3f}deg "
            f"mode={servo_mode} "
            f"pos_step={pos_step_mm:.3f}mm "
            f"yz_overdrive_mm={np.round(yz_overdrive * 1000.0, 3)} "
            f"command_plug={fmt_vec(command_plug_pos)} "
            f"inside_x={inside_x} inside_yz={inside_yz} inside_ori={inside_ori} "
            f"stable={plug_pose_servo['stable_frames']}/{PLUG_HOLD_FRAMES}"
        )

    if plug_pose_servo["stable_frames"] >= PLUG_HOLD_FRAMES:
        print("\n" + "=" * 88)
        print("[PLUG PORT ALIGN SERVO] target port/alignment pose held inside tolerance")
        print(f"  frames:               {plug_pose_servo['frames']}")
        print(f"  final_x_error_mm:     {x_err * 1000.0:.3f}")
        print(f"  final_y_error_mm:     {yz_err[0] * 1000.0:.3f}")
        print(f"  final_z_error_mm:     {yz_err[1] * 1000.0:.3f}")
        print(f"  final_yz_total_mm:    {yz_total * 1000.0:.3f}  limit={PLUG_YZ_TOTAL_TOL * 1000.0:.3f}")
        print(f"  final_ori_error_deg:  {ori_err_deg:.3f}")
        print("=" * 88)
        return True

    if plug_pose_servo["frames"] >= PLUG_SERVO_MAX_FRAMES:
        print("\n" + "=" * 88)
        print("[PLUG PORT ALIGN SERVO] FAILED: timed out before Y/Z port tolerance was met")
        print(f"  final_x_error_mm:     {x_err * 1000.0:.3f}")
        print(f"  final_y_error_mm:     {yz_err[0] * 1000.0:.3f}")
        print(f"  final_z_error_mm:     {yz_err[1] * 1000.0:.3f}")
        print(f"  final_yz_total_mm:    {yz_total * 1000.0:.3f}  limit={PLUG_YZ_TOTAL_TOL * 1000.0:.3f}")
        print(f"  final_ori_error_deg:  {ori_err_deg:.3f}")
        print(f"  stable_frames:        {plug_pose_servo['stable_frames']}/{PLUG_HOLD_FRAMES}")
        print("=" * 88)
        return True

    return False

def plug_pose_servo_action(joint_pos: np.ndarray) -> ArticulationAction:
    n_dof = int(joint_pos.shape[0])
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    target_quat = normalize_quat_wxyz(PLUG_TARGET_ORI_WXYZ)

    # Measured error from actual tracked plug xform translate to the user-selected
    # Transform-panel target.
    measured_err = PLUG_TARGET_POSITION - plug_pos

    pos_err_norm = float(np.linalg.norm(measured_err))
    yz_total = float(np.linalg.norm(measured_err[1:3]))

    # Fast/fine mode split, copied from the lesson of the block insert: get close
    # quickly first, then use a smaller correction only for the final sub-mm settle.
    if pos_err_norm > PLUG_FAST_MODE_DISTANCE:
        servo_mode = "fast"
        pos_kp = PLUG_FAST_POS_KP
        max_pos_step = PLUG_FAST_MAX_POS_STEP
    elif pos_err_norm > PLUG_MID_MODE_DISTANCE:
        servo_mode = "mid"
        pos_kp = PLUG_MID_POS_KP
        max_pos_step = PLUG_MID_MAX_POS_STEP
    else:
        servo_mode = "fine"
        pos_kp = PLUG_FINE_POS_KP
        max_pos_step = PLUG_FINE_MAX_POS_STEP

    # Y/Z overdrive only in fine/mid close range. v3 enabled a 4 mm overdrive too
    # early, so it often overshot and spent hundreds of frames unwinding.
    overdrive_allowed = (
        abs(float(measured_err[0])) <= PLUG_YZ_OVERDRIVE_X_ENABLE
        and yz_total <= PLUG_YZ_OVERDRIVE_ENABLE_RADIUS
    )
    if overdrive_allowed:
        yz_integral = np.asarray(plug_pose_servo.get("yz_integral", np.zeros(2)), dtype=np.float64)
        yz_integral = PLUG_YZ_INTEGRAL_LEAK * yz_integral + measured_err[1:3]
        yz_integral = np.clip(yz_integral, -PLUG_YZ_INTEGRAL_LIMIT, PLUG_YZ_INTEGRAL_LIMIT)
        plug_pose_servo["yz_integral"] = yz_integral
        yz_overdrive = PLUG_YZ_OVERDRIVE_KP * measured_err[1:3] + PLUG_YZ_OVERDRIVE_KI * yz_integral

        # Deadband kick: the previous fast version parked forever at ~0.56 mm
        # Y/Z error because the bias was barely smaller than the cable/contact
        # restoring force. When we are just outside the pass band, add a small
        # push in the measured-error direction so it crosses 0.5 mm quickly.
        if yz_total > PLUG_YZ_TOTAL_TOL:
            yz_dir = measured_err[1:3] / max(yz_total, 1e-9)
            kick_mag = min(0.0010, max(0.00020, yz_total - PLUG_YZ_TOTAL_TOL + 0.00030))
            yz_overdrive = yz_overdrive + yz_dir * kick_mag

        yz_overdrive = np.clip(yz_overdrive, -PLUG_YZ_MAX_OVERDRIVE, PLUG_YZ_MAX_OVERDRIVE)
    else:
        plug_pose_servo["yz_integral"] = np.zeros(2, dtype=np.float64)
        yz_overdrive = np.zeros(2, dtype=np.float64)

    servo_goal = PLUG_TARGET_POSITION.copy()
    servo_goal[1:3] += yz_overdrive
    plug_pose_servo["last_yz_overdrive"] = yz_overdrive.copy()
    plug_pose_servo["last_servo_mode"] = servo_mode

    err = servo_goal - plug_pos
    raw_step = pos_kp * err
    raw_step_norm = float(np.linalg.norm(raw_step))
    if raw_step_norm > max_pos_step:
        raw_step *= max_pos_step / raw_step_norm
    command_plug_pos = plug_pos + raw_step
    plug_pose_servo["last_command_plug_pos"] = command_plug_pos.copy()
    plug_pose_servo["last_pos_step_mm"] = float(np.linalg.norm(raw_step) * 1000.0)

    # Orientation also uses a fast/fine split. Big rotations are allowed to close
    # quickly, then we slow down near the target to avoid twisting the plug.
    ori_err_deg = quat_angle_error_deg(plug_quat, target_quat)
    max_ori_step = PLUG_FAST_ORI_STEP_DEG if ori_err_deg > 2.0 else PLUG_FINE_ORI_STEP_DEG
    if ori_err_deg <= 1e-9:
        command_plug_quat = target_quat
    else:
        frac = min(1.0, max_ori_step / ori_err_deg)
        command_plug_quat = quat_slerp_shortest(plug_quat, target_quat, frac)

    target_hand_pos, _, target_hand_quat = hand_pose_for_plug_pose(
        command_plug_pos,
        command_plug_quat,
        plug_pose_servo["plug_offset_local"],
        plug_pose_servo["plug_rot_local"],
    )
    plug_pose_servo["last_target_hand_pos"] = target_hand_pos.copy()

    action, success = art_kinematics.compute_inverse_kinematics(
        target_position=target_hand_pos,
        target_orientation=target_hand_quat,
        position_tolerance=PLUG_SERVO_IK_POS_TOL,
        orientation_tolerance=PLUG_SERVO_IK_ORI_TOL,
    )

    if not success:
        plug_pose_servo["ik_fail_count"] = int(plug_pose_servo.get("ik_fail_count", 0)) + 1
        if not plug_pose_servo.get("warned_ik", False) or plug_pose_servo["ik_fail_count"] % 120 == 0:
            print("[PLUG PORT ALIGN SERVO] IK failed; holding closed gripper")
            print(f"  measured_err_mm:      {fmt_vec(measured_err * 1000.0, 3)}")
            print(f"  yz_overdrive_mm:      {fmt_vec(yz_overdrive * 1000.0, 3)}")
            print(f"  command_plug_pos:     {fmt_vec(command_plug_pos)}")
            print(f"  command_plug_quat:    {fmt_vec(command_plug_quat)}")
            print(f"  target_hand_pos:      {fmt_vec(target_hand_pos)}")
            print(f"  target_hand_quat:     {fmt_vec(target_hand_quat)}")
            print(f"  ik_fail_count:        {plug_pose_servo['ik_fail_count']}")
            plug_pose_servo["warned_ik"] = True
        return closed_gripper_hold_action(n_dof)

    return controller._with_closed_gripper(action, n_dof)

def measure_plug_pose_servo_result() -> bool:
    pos_norm, ori_err = print_plug_pose_error("FINAL closed-loop plug pose result")
    print_plug_hand_offset_debug("FINAL local plug-to-hand relationship")

    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    final_err = PLUG_TARGET_POSITION - plug_pos
    final_x_abs = abs(float(final_err[0]))
    final_yz_total = float(np.linalg.norm(final_err[1:3]))
    final_axis_angle = plug_axis_angle_to_insert_deg(plug_rot)
    final_tilt = plug_tilt_out_of_horizontal_deg(plug_rot)
    horizontal_locked = final_tilt <= PREINSERT_HORIZONTAL_TOL_DEG
    axis_locked = final_axis_angle <= PREINSERT_AXIS_TO_PORT_TOL_DEG
    yz_line_locked = final_yz_total <= PREINSERT_YZ_LINE_TOL

    if len(plug_pose_samples) >= 2:
        positions = np.asarray([s["pos"] for s in plug_pose_samples], dtype=np.float64)
        pos_errs = np.asarray([s["pos_err_norm"] for s in plug_pose_samples], dtype=np.float64)
        ori_errs = np.asarray([s["ori_err_deg"] for s in plug_pose_samples], dtype=np.float64)
        axis_angles = np.asarray([s["axis_angle_deg"] for s in plug_pose_samples], dtype=np.float64)
        tilts = np.asarray([s["tilt_deg"] for s in plug_pose_samples], dtype=np.float64)
        offset_drifts = np.asarray([s["offset_drift"] for s in plug_pose_samples], dtype=np.float64)
        rot_drifts = np.asarray([s["rot_drift_deg"] for s in plug_pose_samples], dtype=np.float64)
        yz_errs = np.linalg.norm(PLUG_TARGET_POSITION[1:3].reshape(1, 2) - positions[:, 1:3], axis=1)

        dx = np.diff(positions[:, 0])
        max_positive_x_backtrack = float(np.max(np.maximum(0.0, dx))) if len(dx) else 0.0

        print("=" * 88)
        print("[PLUG TRANSFORM TARGET ALIGN PATH SUMMARY]")
        print(f"  samples:                    {len(plug_pose_samples)}")
        print(f"  start_plug_pos:             {fmt_vec(positions[0])}")
        print(f"  end_plug_pos:               {fmt_vec(positions[-1])}")
        print(f"  max_position_error_mm:      {float(np.max(pos_errs)) * 1000.0:.3f}")
        print(f"  final_position_error_mm:    {pos_norm * 1000.0:.3f}")
        print(f"  max_yz_total_error_mm:      {float(np.max(yz_errs)) * 1000.0:.3f}")
        print(f"  final_yz_total_error_mm:    {final_yz_total * 1000.0:.3f}  limit={PLUG_YZ_TOTAL_TOL * 1000.0:.3f}")
        print(f"  final_x_abs_error_mm:       {final_x_abs * 1000.0:.3f}  limit={PLUG_X_TOL * 1000.0:.3f}")
        print(f"  max_orientation_error_deg:  {float(np.max(ori_errs)):.3f}")
        print(f"  final_orientation_error_deg:{ori_err:.3f}")
        print(f"  max_axis_angle_to_-X_deg:   {float(np.max(axis_angles)):.3f}")
        print(f"  final_axis_angle_to_-X_deg: {float(axis_angles[-1]):.3f}")
        print(f"  max_tilt_horizontal_deg:    {float(np.max(tilts)):.3f}")
        print(f"  final_tilt_horizontal_deg:  {float(tilts[-1]):.3f}")
        print(f"  max_plug_offset_drift_mm:   {float(np.max(offset_drifts)) * 1000.0:.3f}")
        print(f"  max_plug_rot_drift_deg:     {float(np.max(rot_drifts)):.3f}")
        print(f"  max_positive_x_backtrack:   {max_positive_x_backtrack * 1000.0:.3f} mm")
        print(f"  ik_fail_count:              {int(plug_pose_servo.get('ik_fail_count', 0))}")
        print("=" * 88)

    print("=" * 88)
    print("[PRE-INSERT HORIZONTAL LOCK CHECK]")
    print(f"  required_pose:          plug front 50 mm in +X from bbox port face, same Y/Z as port line")
    print(f"  final_x_abs_error_mm:   {final_x_abs * 1000.0:.3f}  limit={PLUG_X_TOL * 1000.0:.3f}")
    print(f"  final_yz_total_mm:      {final_yz_total * 1000.0:.3f}  limit={PREINSERT_YZ_LINE_TOL * 1000.0:.3f}")
    print(f"  axis_angle_to_-X_deg:   {final_axis_angle:.3f}  limit={PREINSERT_AXIS_TO_PORT_TOL_DEG:.3f}")
    print(f"  tilt_out_horizontal_deg:{final_tilt:.3f}  limit={PREINSERT_HORIZONTAL_TOL_DEG:.3f}")
    print(f"  quat_error_deg:         {ori_err:.3f}  limit={PLUG_ORIENTATION_TOL_DEG:.3f}")
    print("  RESULT: ✓ ready to insert" if (horizontal_locked and axis_locked and yz_line_locked and ori_err <= PLUG_ORIENTATION_TOL_DEG and final_x_abs <= PLUG_X_TOL) else "  RESULT: ✗ not inserting; pre-insert pose is not locked")
    print("=" * 88)

    ok = (
        final_x_abs <= PLUG_X_TOL
        and yz_line_locked
        and ori_err <= PLUG_ORIENTATION_TOL_DEG
        and horizontal_locked
        and axis_locked
    )
    print("[PLUG TRANSFORM TARGET ALIGN RESULT] ✓ plug is locked horizontally on the 5 cm pre-insert port line" if ok else "[PLUG TRANSFORM TARGET ALIGN RESULT] ✗ plug is not locked for insertion")
    return ok




# =============================================================================
# 8. CLOSED-LOOP X INSERT SERVO
# =============================================================================

def _insert_direction() -> float:
    d = float(np.sign(float(INSERT_TARGET_POSITION[0]) - float(PLUG_TARGET_POSITION[0])))
    return d if abs(d) > 1e-9 else -1.0


def _insert_final_x_bounds() -> typing.Tuple[float, float]:
    """Return one-sided allowed final-X band.

    For -X insertion, the plug may stop before the target by INSERT_FINAL_X_TOL,
    but it may not go past the target. Example final_x=-0.55 gives
    [-0.55000, -0.54950].
    """

    final_x = float(INSERT_TARGET_POSITION[0])
    d = _insert_direction()
    if d < 0.0:
        return final_x, final_x + INSERT_FINAL_X_TOL
    return final_x - INSERT_FINAL_X_TOL, final_x


def _insert_x_in_final_band(actual_x: float) -> bool:
    low, high = _insert_final_x_bounds()
    x = float(actual_x)
    return low <= x <= high


def _insert_x_overshoot_distance(actual_x: float) -> float:
    d = _insert_direction()
    final_x = float(INSERT_TARGET_POSITION[0])
    x = float(actual_x)
    # Strict port rule: overshoot starts exactly at the final target plane, not
    # final +/- tolerance. For -X insertion, x < final_x is too far.
    if d < 0.0:
        return max(0.0, final_x - x)
    return max(0.0, x - final_x)


def _clip_insert_command_x(commanded_x: float) -> float:
    d = _insert_direction()
    final_x = float(INSERT_TARGET_POSITION[0])
    if INSERT_STRICT_NO_PAST_TARGET_X:
        if d < 0.0:
            return float(max(commanded_x, final_x))
        return float(min(commanded_x, final_x))

    push_limit = final_x + d * INSERT_X_MAX_PUSH_THROUGH
    if d < 0.0:
        return float(max(commanded_x, push_limit))
    return float(min(commanded_x, push_limit))


def port_face_x_for_debug() -> float:
    try:
        return float(DATAHALL_PORT_FACE_X_M)
    except Exception:
        return float("nan")


def init_plug_insert_servo() -> None:
    plug_insert_servo.clear()
    plug_insert_samples.clear()

    plug_offset_local, plug_rot_local = compute_plug_hand_offsets()
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    yz_err = plug_pos[1:3] - INSERT_TARGET_POSITION[1:3]
    yz_total = float(np.linalg.norm(yz_err))

    plug_insert_servo.update({
        "frames": 0,
        "stable_frames": 0,
        "ik_fail_count": 0,
        "warned_ik": False,
        "plug_offset_local": plug_offset_local,
        "plug_rot_local": plug_rot_local,
        "start_plug_pos": plug_pos.copy(),
        "start_plug_quat": plug_quat.copy(),
        "commanded_x": float(plug_pos[0]),
        "stroke_paused_for_yz": yz_total > INSERT_STROKE_YZ_PAUSE_TOL,
        "stroke_pause_count": 0,
        "yz_integral": np.zeros(2, dtype=np.float64),
        "last_yz_overdrive": np.zeros(2, dtype=np.float64),
        "last_command_plug_pos": plug_pos.copy(),
        "last_pos_step_mm": 0.0,
        "last_x_step_mm": 0.0,
        "last_insert_paused": False,
        "max_yz_total": yz_total,
        "max_x_overshoot": 0.0,
        "max_positive_x_backtrack": 0.0,
        "prev_x": float(plug_pos[0]),
    })

    print("=" * 88)
    print("[PLUG X INSERT SERVO INIT]")
    print("  control_object:          /World/NetworkCable/E_crystal_head1_45")
    print("  position_source:         tracked plug xform translate / Transform panel value")
    print(f"  pre_insert_transform:    {fmt_vec(PLUG_TARGET_POSITION)}")
    print(f"  final_insert_transform:  {fmt_vec(INSERT_TARGET_POSITION)}")
    print(f"  actual_start_transform:  {fmt_vec(plug_pos)}")
    print(f"  insert_axis_world:       {fmt_vec(INSERT_AXIS_WORLD)}")
    print(f"  fast/fine_x_step:        {INSERT_X_FAST_STEP * 1000.0:.3f} / {INSERT_X_FINE_STEP * 1000.0:.3f} mm/frame")
    print(f"  fine_x_distance:         {INSERT_X_FINE_DISTANCE * 1000.0:.3f} mm")
    print(f"  yz_slowdown_threshold:   {INSERT_X_YZ_SLOWDOWN_TOL * 1000.0:.3f} mm")
    x_low, x_high = _insert_final_x_bounds()
    print(f"  final_x_tolerance:       {INSERT_FINAL_X_TOL * 1000.0:.3f} mm, one-sided/no-past-target")
    print(f"  allowed_final_x_band:    [{x_low:.5f}, {x_high:.5f}]")
    print(f"  strict_no_past_target_x: {INSERT_STRICT_NO_PAST_TARGET_X}")
    print(f"  path_yz_tolerance:       {INSERT_YZ_TOTAL_TOL * 1000.0:.3f} mm")
    print(f"  hard_abort_yz_tolerance: {INSERT_HARD_ABORT_YZ_TOL * 1000.0:.3f} mm")
    print(f"  pause/resume YZ:         {INSERT_STROKE_YZ_PAUSE_TOL * 1000.0:.3f} / {INSERT_STROKE_YZ_RESUME_TOL * 1000.0:.3f} mm")
    print(f"  orientation_tolerance:   {INSERT_ORIENTATION_TOL_DEG:.3f} deg")
    print(f"  hard_abort_ori_tolerance:{INSERT_HARD_ABORT_ORI_TOL_DEG:.3f} deg")
    print("=" * 88)


def sample_plug_insert_path(pos_err_norm: float, ori_err_deg: float) -> None:
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    yz_err = plug_pos[1:3] - INSERT_TARGET_POSITION[1:3]
    yz_total = float(np.linalg.norm(yz_err))
    axis_angle = plug_axis_angle_to_insert_deg(plug_rot)
    tilt = plug_tilt_out_of_horizontal_deg(plug_rot)

    prev_x = float(plug_insert_servo.get("prev_x", plug_pos[0]))
    dx = float(plug_pos[0] - prev_x)
    d = _insert_direction()
    backtrack = max(0.0, -d * dx)
    plug_insert_servo["max_positive_x_backtrack"] = max(
        float(plug_insert_servo.get("max_positive_x_backtrack", 0.0)),
        backtrack,
    )
    plug_insert_servo["prev_x"] = float(plug_pos[0])
    plug_insert_servo["max_yz_total"] = max(float(plug_insert_servo.get("max_yz_total", 0.0)), yz_total)
    plug_insert_servo["max_x_overshoot"] = max(float(plug_insert_servo.get("max_x_overshoot", 0.0)), _insert_x_overshoot_distance(plug_pos[0]))

    plug_insert_samples.append({
        "pos": plug_pos.copy(),
        "quat": plug_quat.copy(),
        "yz_total": yz_total,
        "ori_err_deg": float(ori_err_deg),
        "axis_angle_deg": axis_angle,
        "tilt_deg": tilt,
        "pos_err_norm": float(pos_err_norm),
    })


def update_plug_insert_servo_state() -> bool:
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    final_err = INSERT_TARGET_POSITION - plug_pos
    pos_err_norm = float(np.linalg.norm(final_err))
    x_err = float(final_err[0])
    yz_err = np.asarray(final_err[1:3], dtype=np.float64)
    yz_total = float(np.linalg.norm(yz_err))
    ori_err_deg = quat_angle_error_deg(plug_quat, PLUG_TARGET_ORI_WXYZ)
    sample_plug_insert_path(pos_err_norm, ori_err_deg)

    plug_insert_servo["frames"] += 1
    overshoot = _insert_x_overshoot_distance(plug_pos[0])
    inside_x = _insert_x_in_final_band(plug_pos[0])
    inside_yz = yz_total <= INSERT_YZ_TOTAL_TOL
    inside_ori = ori_err_deg <= INSERT_ORIENTATION_TOL_DEG
    inside = inside_x and inside_yz and inside_ori

    if inside:
        plug_insert_servo["stable_frames"] += 1
    else:
        plug_insert_servo["stable_frames"] = 0

    if overshoot > 0.0:
        print("\n" + "=" * 88)
        print("[PLUG X INSERT SERVO] HARD STOP: X passed final safety band")
        print(f"  actual_transform:       {fmt_vec(plug_pos)}")
        print(f"  final_insert_transform: {fmt_vec(INSERT_TARGET_POSITION)}")
        print(f"  x_band_overshoot_mm:    {overshoot * 1000.0:.3f}")
        print("=" * 88)
        return True

    if yz_total > INSERT_HARD_ABORT_YZ_TOL or ori_err_deg > INSERT_HARD_ABORT_ORI_TOL_DEG:
        print("\n" + "=" * 88)
        print("[PLUG X INSERT SERVO] HARD STOP: plug left the port line / collision tilt detected")
        print(f"  actual_transform:       {fmt_vec(plug_pos)}")
        print(f"  final_insert_transform: {fmt_vec(INSERT_TARGET_POSITION)}")
        print(f"  final_x_error_mm:       {x_err * 1000.0:.3f}")
        print(f"  yz_total_mm:            {yz_total * 1000.0:.3f}  limit={INSERT_HARD_ABORT_YZ_TOL * 1000.0:.3f}")
        print(f"  ori_error_deg:          {ori_err_deg:.3f}  limit={INSERT_HARD_ABORT_ORI_TOL_DEG:.3f}")
        print(f"  front_x:                {plug_pos[0] - PLUG_FRONT_TO_XFORM_X_OFFSET_M:.5f}")
        print(f"  front_depth_mm:         {(port_face_x_for_debug() - (plug_pos[0] - PLUG_FRONT_TO_XFORM_X_OFFSET_M)) * 1000.0:.3f}")
        print("=" * 88)
        return True

    if plug_insert_servo["frames"] == 1 or plug_insert_servo["frames"] % INSERT_STROKE_DEBUG_EVERY == 0 or inside:
        yz_overdrive = np.asarray(plug_insert_servo.get("last_yz_overdrive", np.zeros(2)), dtype=np.float64)
        command_plug_pos = np.asarray(plug_insert_servo.get("last_command_plug_pos", plug_pos), dtype=np.float64)
        paused = bool(plug_insert_servo.get("last_insert_paused", False))
        print(
            f"[PLUG X INSERT SERVO] frame={plug_insert_servo['frames']} "
            f"x={plug_pos[0]:.5f} final_x={INSERT_TARGET_POSITION[0]:.5f} "
            f"x_err={x_err * 1000.0:.3f}mm "
            f"y_err={yz_err[0] * 1000.0:.3f}mm "
            f"z_err={yz_err[1] * 1000.0:.3f}mm "
            f"yz_total={yz_total * 1000.0:.3f}mm "
            f"ori_err={ori_err_deg:.3f}deg "
            f"axis_to_-X={plug_axis_angle_to_insert_deg(plug_rot):.3f}deg "
            f"tilt={plug_tilt_out_of_horizontal_deg(plug_rot):.3f}deg "
            f"paused={paused} "
            f"x_command={float(plug_insert_servo.get('commanded_x', plug_pos[0])):.5f} "
            f"front_x={plug_pos[0] - PLUG_FRONT_TO_XFORM_X_OFFSET_M:.5f} "
            f"front_depth={(port_face_x_for_debug() - (plug_pos[0] - PLUG_FRONT_TO_XFORM_X_OFFSET_M)) * 1000.0:.3f}mm "
            f"x_step={float(plug_insert_servo.get('last_x_step_mm', 0.0)):.3f}mm "
            f"pos_step={float(plug_insert_servo.get('last_pos_step_mm', 0.0)):.3f}mm "
            f"yz_overdrive_mm={np.round(yz_overdrive * 1000.0, 3)} "
            f"command_plug={fmt_vec(command_plug_pos)} "
            f"stable={plug_insert_servo['stable_frames']}/{INSERT_STABLE_HOLD_FRAMES}"
        )

    if plug_insert_servo["stable_frames"] >= INSERT_STABLE_HOLD_FRAMES:
        print("\n" + "=" * 88)
        print("[PLUG X INSERT SERVO] final insert pose held inside tolerance")
        print(f"  frames:               {plug_insert_servo['frames']}")
        print(f"  final_x_error_mm:     {x_err * 1000.0:.3f}")
        print(f"  final_y_error_mm:     {yz_err[0] * 1000.0:.3f}")
        print(f"  final_z_error_mm:     {yz_err[1] * 1000.0:.3f}")
        print(f"  final_yz_total_mm:    {yz_total * 1000.0:.3f}  limit={INSERT_YZ_TOTAL_TOL * 1000.0:.3f}")
        print(f"  final_ori_error_deg:  {ori_err_deg:.3f}")
        print("=" * 88)
        return True

    if plug_insert_servo["frames"] >= INSERT_STROKE_MAX_FRAMES:
        print("\n" + "=" * 88)
        print("[PLUG X INSERT SERVO] FAILED: timed out before final X target")
        print(f"  actual_transform:       {fmt_vec(plug_pos)}")
        print(f"  final_insert_transform: {fmt_vec(INSERT_TARGET_POSITION)}")
        print(f"  final_x_error_mm:       {x_err * 1000.0:.3f}")
        print(f"  final_yz_total_mm:      {yz_total * 1000.0:.3f}")
        print("=" * 88)
        return True

    return False


def plug_insert_servo_action(joint_pos: np.ndarray) -> ArticulationAction:
    n_dof = int(joint_pos.shape[0])
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    final_err = INSERT_TARGET_POSITION - plug_pos
    yz_err = np.asarray(final_err[1:3], dtype=np.float64)
    yz_total = float(np.linalg.norm(yz_err))

    paused = bool(plug_insert_servo.get("stroke_paused_for_yz", False))
    if paused and yz_total <= INSERT_STROKE_YZ_RESUME_TOL:
        paused = False
    elif (not paused) and yz_total >= INSERT_STROKE_YZ_PAUSE_TOL:
        paused = True
    plug_insert_servo["stroke_paused_for_yz"] = paused
    plug_insert_servo["last_insert_paused"] = paused
    if paused:
        plug_insert_servo["stroke_pause_count"] = int(plug_insert_servo.get("stroke_pause_count", 0)) + 1

    d = _insert_direction()
    commanded_x = float(plug_insert_servo.get("commanded_x", plug_pos[0]))
    x_remaining = max(0.0, d * (float(INSERT_TARGET_POSITION[0]) - float(plug_pos[0])))
    if paused:
        # Do not advance X while Y/Z is near the 0.5 mm path gate, but also do
        # not reset the commanded X back to the drifting actual plug X. V4 did
        # that and the connector slowly backed out forever after pausing.
        x_step = 0.0
    else:
        if x_remaining <= INSERT_X_FINE_DISTANCE or yz_total >= INSERT_X_YZ_SLOWDOWN_TOL:
            x_step = INSERT_X_FINE_STEP
        else:
            x_step = INSERT_X_FAST_STEP
        if not _insert_x_in_final_band(plug_pos[0]):
            # Advance toward the final X plane, but never command past it. This
            # is the port-insertion safety rule: no push-through allowance.
            next_commanded_x = commanded_x + d * x_step
            commanded_x = _clip_insert_command_x(next_commanded_x)
            x_step = abs(commanded_x - float(plug_insert_servo.get("commanded_x", plug_pos[0])))
        else:
            x_step = 0.0
    plug_insert_servo["commanded_x"] = commanded_x
    plug_insert_servo["last_x_step_mm"] = float(x_step * 1000.0)

    yz_integral = np.asarray(plug_insert_servo.get("yz_integral", np.zeros(2)), dtype=np.float64)
    yz_integral = INSERT_YZ_INTEGRAL_LEAK * yz_integral + yz_err
    yz_integral = np.clip(yz_integral, -INSERT_YZ_INTEGRAL_LIMIT, INSERT_YZ_INTEGRAL_LIMIT)
    plug_insert_servo["yz_integral"] = yz_integral
    yz_overdrive = INSERT_YZ_KP * yz_err + INSERT_YZ_KI * yz_integral
    yz_overdrive = np.clip(yz_overdrive, -INSERT_YZ_MAX_OVERDRIVE, INSERT_YZ_MAX_OVERDRIVE)

    command_plug_pos = np.array([
        commanded_x,
        float(INSERT_TARGET_POSITION[1] + yz_overdrive[0]),
        float(INSERT_TARGET_POSITION[2] + yz_overdrive[1]),
    ], dtype=np.float64)
    plug_insert_servo["last_yz_overdrive"] = yz_overdrive.copy()
    plug_insert_servo["last_command_plug_pos"] = command_plug_pos.copy()
    plug_insert_servo["last_pos_step_mm"] = float(np.linalg.norm(command_plug_pos - plug_pos) * 1000.0)

    target_quat = normalize_quat_wxyz(PLUG_TARGET_ORI_WXYZ)
    ori_err_deg = quat_angle_error_deg(plug_quat, target_quat)
    if ori_err_deg <= 1e-9:
        command_plug_quat = target_quat
    else:
        frac = min(1.0, INSERT_MAX_ORI_STEP_DEG / ori_err_deg)
        command_plug_quat = quat_slerp_shortest(plug_quat, target_quat, frac)

    target_hand_pos, _, target_hand_quat = hand_pose_for_plug_pose(
        command_plug_pos,
        command_plug_quat,
        plug_insert_servo["plug_offset_local"],
        plug_insert_servo["plug_rot_local"],
    )

    action, success = art_kinematics.compute_inverse_kinematics(
        target_position=target_hand_pos,
        target_orientation=target_hand_quat,
        position_tolerance=PLUG_SERVO_IK_POS_TOL,
        orientation_tolerance=PLUG_SERVO_IK_ORI_TOL,
    )

    if not success:
        plug_insert_servo["ik_fail_count"] = int(plug_insert_servo.get("ik_fail_count", 0)) + 1
        if not plug_insert_servo.get("warned_ik", False) or plug_insert_servo["ik_fail_count"] % 120 == 0:
            print("[PLUG X INSERT SERVO] IK failed; holding closed gripper")
            print(f"  actual_plug:        {fmt_vec(plug_pos)}")
            print(f"  command_plug_pos:   {fmt_vec(command_plug_pos)}")
            print(f"  target_hand_pos:    {fmt_vec(target_hand_pos)}")
            print(f"  target_hand_quat:   {fmt_vec(target_hand_quat)}")
            print(f"  ik_fail_count:      {plug_insert_servo['ik_fail_count']}")
            plug_insert_servo["warned_ik"] = True
        return closed_gripper_hold_action(n_dof)

    return controller._with_closed_gripper(action, n_dof)


def measure_plug_insert_result() -> bool:
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    final_err = plug_pos - INSERT_TARGET_POSITION
    final_x_abs = abs(float(final_err[0]))
    final_yz_total = float(np.linalg.norm(final_err[1:3]))
    final_ori = quat_angle_error_deg(plug_quat, PLUG_TARGET_ORI_WXYZ)

    if len(plug_insert_samples) >= 2:
        positions = np.asarray([s["pos"] for s in plug_insert_samples], dtype=np.float64)
        yz_totals = np.asarray([s["yz_total"] for s in plug_insert_samples], dtype=np.float64)
        ori_errs = np.asarray([s["ori_err_deg"] for s in plug_insert_samples], dtype=np.float64)
        axis_angles = np.asarray([s["axis_angle_deg"] for s in plug_insert_samples], dtype=np.float64)
        tilts = np.asarray([s["tilt_deg"] for s in plug_insert_samples], dtype=np.float64)
        dx = np.diff(positions[:, 0])
        d = _insert_direction()
        max_backtrack = float(np.max(np.maximum(0.0, -d * dx))) if len(dx) else 0.0
        max_x_overshoot = float(np.max([_insert_x_overshoot_distance(x) for x in positions[:, 0]]))
        final_x_in_band = _insert_x_in_final_band(positions[-1, 0])
        path_ok = float(np.max(yz_totals)) <= INSERT_YZ_TOTAL_TOL and max_backtrack <= INSERT_X_BACKTRACK_TOL and max_x_overshoot <= 1e-9 and final_x_in_band
        ori_ok = float(np.max(tilts)) <= 2.0 and float(np.max(axis_angles)) <= 2.0 and float(np.max(ori_errs)) <= INSERT_ORIENTATION_TOL_DEG

        print("=" * 88)
        print("[PLUG X INSERT PATH SUMMARY]")
        print(f"  samples:                    {len(positions)}")
        print(f"  start_plug_transform:       {fmt_vec(positions[0])}")
        print(f"  end_plug_transform:         {fmt_vec(positions[-1])}")
        print(f"  final_insert_target:        {fmt_vec(INSERT_TARGET_POSITION)}")
        print(f"  max_yz_total_offset:        {float(np.max(yz_totals)) * 1000.0:.3f} mm  limit={INSERT_YZ_TOTAL_TOL * 1000.0:.3f}")
        print(f"  final_yz_total_offset:      {final_yz_total * 1000.0:.3f} mm")
        print(f"  final_x_abs_error:          {final_x_abs * 1000.0:.3f} mm  limit={INSERT_FINAL_X_TOL * 1000.0:.3f}")
        print(f"  x_monotonic:                {max_backtrack <= INSERT_X_BACKTRACK_TOL}  max_backtrack={max_backtrack * 1000.0:.3f} mm")
        print(f"  max_x_band_overshoot:       {max_x_overshoot * 1000.0:.3f} mm")
        print(f"  final_x_in_band:            {final_x_in_band}")
        print(f"  max_orientation_error_deg:  {float(np.max(ori_errs)):.3f}")
        print(f"  final_orientation_error_deg:{final_ori:.3f}")
        print(f"  max_axis_angle_to_-X_deg:   {float(np.max(axis_angles)):.3f}")
        print(f"  max_tilt_horizontal_deg:    {float(np.max(tilts)):.3f}")
        print(f"  stroke_yz_pause_frames:     {int(plug_insert_servo.get('stroke_pause_count', 0))}")
        print(f"  ik_fail_count:              {int(plug_insert_servo.get('ik_fail_count', 0))}")
        print("  PATH RESULT: ✓ insert path stayed within Y/Z + X limits" if path_ok else "  PATH RESULT: ✗ insert path exceeded Y/Z or X limits")
        print("  ORIENTATION RESULT: ✓ plug stayed horizontal/aligned" if ori_ok else "  ORIENTATION RESULT: ✗ plug orientation drifted too much")
        print("=" * 88)
        ok = path_ok and ori_ok and final_x_abs <= INSERT_FINAL_X_TOL and final_yz_total <= INSERT_YZ_TOTAL_TOL and final_ori <= INSERT_ORIENTATION_TOL_DEG
    else:
        print("[PLUG X INSERT PATH SUMMARY] Not enough samples.")
        ok = False

    print("[PLUG X INSERT RESULT] ✓ inserted from pre-insert X to final X while holding Y/Z" if ok else "[PLUG X INSERT RESULT] ✗ insert failed path or endpoint checks")
    return ok

def pre_servo_grasp_sanity_ok(tag: str) -> bool:
    """Return False when the plug clearly slipped before the feedback servo starts."""

    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    pos_err = PLUG_TARGET_POSITION - plug_pos
    pos_err_norm = float(np.linalg.norm(pos_err))
    yz_total = float(np.linalg.norm(pos_err[1:3]))
    ori_err = quat_angle_error_deg(plug_quat, PLUG_TARGET_ORI_WXYZ)
    axis_angle = plug_axis_angle_to_insert_deg(plug_rot)
    tilt = plug_tilt_out_of_horizontal_deg(plug_rot)
    ok = (
        pos_err_norm <= PRE_SERVO_MAX_POSITION_ERROR
        and tilt <= PRE_SERVO_MAX_TILT_DEG
        and axis_angle <= PRE_SERVO_MAX_AXIS_ANGLE_DEG
        and ori_err <= PRE_SERVO_MAX_ORI_ERROR_DEG
    )

    print("=" * 88)
    print(f"[PRE-SERVO GRASP SANITY] {tag}")
    print(f"  pos_error_mm:       {pos_err_norm * 1000.0:.3f}  limit={PRE_SERVO_MAX_POSITION_ERROR * 1000.0:.1f}")
    print(f"  yz_total_mm:        {yz_total * 1000.0:.3f}")
    print(f"  orientation_deg:    {ori_err:.3f}  limit={PRE_SERVO_MAX_ORI_ERROR_DEG:.1f}")
    print(f"  axis_to_-X_deg:     {axis_angle:.3f}  limit={PRE_SERVO_MAX_AXIS_ANGLE_DEG:.1f}")
    print(f"  tilt_deg:           {tilt:.3f}  limit={PRE_SERVO_MAX_TILT_DEG:.1f}")
    print("  RESULT: ✓ safe to start fine servo" if ok else "  RESULT: ✗ plug already slipped/rotated; skipping servo before it can tweak out")
    print("=" * 88)
    return ok


def queue_post_insert_release_retreat() -> None:
    """Restore port collisions, open the gripper, then retreat along world +X.

    This runs only after measure_plug_insert_result() returns True. The cable is
    left in the port; the controller target is the hand itself, not the plug, so
    the retreat does not drag the measured connector target backward.
    """

    restore_port_insert_collisions_before_release()
    spawn_post_insert_socket_hold_proxy()
    controller.clear_queue()

    hand_pos, _, hand_quat = get_hand_pose_matrix()
    retreat_hand_pos = np.asarray(hand_pos, dtype=np.float64).copy()
    retreat_hand_pos[0] += float(POST_INSERT_RETREAT_X_M)

    if POST_INSERT_COLLISION_SETTLE_FRAMES > 0:
        controller.add_gripper_command(action="close", wait_frames=POST_INSERT_COLLISION_SETTLE_FRAMES)
    controller.add_gripper_command(action="open", wait_frames=POST_INSERT_OPEN_WAIT_FRAMES)
    controller.add_cartesian_waypoint(
        position=retreat_hand_pos,
        orientation=hand_quat,
        max_frames=520,
        pos_tolerance=POST_INSERT_RETREAT_POS_TOL,
        linear=True,
        linear_step=POST_INSERT_RETREAT_LINEAR_STEP,
        hold_gripper=False,
        target_is_hand=True,
        label="post_insert_linear_x_retreat",
    )

    print("=" * 88)
    print("[POST-INSERT COLLISION RESTORE + RELEASE + RETREAT QUEUED]")
    print("  result_required:       successful measured insert")
    print("  port_collision:        restored before gripper opens")
    print("  socket_hold_proxy:     spawned before gripper opens")
    print(f"  collision_settle:      {POST_INSERT_COLLISION_SETTLE_FRAMES} frames with gripper still closed")
    print(f"  gripper_action:        open for {POST_INSERT_OPEN_WAIT_FRAMES} frames")
    print(f"  start_hand_pos:        {fmt_vec(hand_pos)}")
    print(f"  retreat_hand_target:   {fmt_vec(retreat_hand_pos)}")
    print("  retreat_axis:          +X world, opposite of -X insertion")
    print(f"  retreat_distance_mm:   {POST_INSERT_RETREAT_X_M * 1000.0:.1f}")
    print(f"  retreat_linear_step_mm:{POST_INSERT_RETREAT_LINEAR_STEP * 1000.0:.1f}")
    print("  target_is_hand:        True  # do not drag the plug target backward")
    print("=" * 88)
    print_queued_commands(controller)


# =============================================================================
# 8. OPTIONAL PER-FRAME HOOK
# =============================================================================

def user_robot_step(
    world: World,
    franka: SingleManipulator,
    controller: FrankaMotionController,
    art_kinematics: ArticulationKinematicsSolver,
) -> None:
    """
    Optional hook that runs every frame while Play is active.

    You probably do not need this at first.
    Add printouts, live measurements, or custom checks here later.
    """
    pass


# =============================================================================
# 9. MAIN LOOP
# =============================================================================

world, franka, controller, kinematics_solver, art_kinematics = build_scene_and_controller()
configure_viewport_only_layout()
waypoints_queued = False
active_command_info = None
phase = PHASE_COARSE_WAYPOINTS
final_result_logged = False
viewport_layout_frames = 0
sim_frame_counter = 0

print("[READY - DATAHALL AS4610 RJ45 SUPPORT 1.8X Y -0.6 PATH-STRICT FAST SOCKET HOLD V24]")
print("  Spawned: Franka robot, pickup support/posts, and network cable.")
print("  Cable targets use the DataHall port bbox face plus the measured plug nose offset; the robot hand is only a carrier.")
print(f"  tracked plug: {TRACKED_PLUG_PRIM_PATH}")
print(f"  selected AS4610 RJ45 component: {DATAHALL_PORT_PRIM_PATH}")
print(f"  pre-insert plug xform translate: {fmt_vec(PLUG_TARGET_POSITION)}")
print(f"  selected plug target orientation: {fmt_vec(PLUG_TARGET_ORI_WXYZ)}")
print(f"  final insert plug xform translate: {fmt_vec(INSERT_TARGET_POSITION)}")
print("  visual target blocks: disabled")
print(f"  coarse transfer waypoint derived from target: {fmt_vec(COARSE_TRANSFER_POSITION)}")
print(f"  coarse approach-above-target waypoint: {fmt_vec(COARSE_APPROACH_TARGET_POSITION)}")
print(f"  coarse near-target waypoint derived from target: {fmt_vec(COARSE_NEAR_TARGET_POSITION)}")
print("  Pre-insert metric: plug FRONT FACE must be 50 mm in +X from the bbox-detected port face, same Y/Z as port line, horizontal <= 1.0 deg.")
print("  Insert only starts after that lock passes; then the plug FRONT FACE moves strict -X to the shallow safe depth without driving the connector center through the port face.")
print("  Cable USD is assumed already manually scaled; script only doubles the pickup support block/posts/clearance.")
print("  Pickup support/cable are at y=-0.600, while the Franka base stays in the original port-relative lane so the long cable avoids the robot base.")
print("  Transit is split but not dragged all the way across while diagonal: lift -> short tail-clearance move -> reorient at a mid waypoint -> transfer/approach in insert orientation.")
print("  Alignment servo restored from network_connector_diagonal.py. Insert stroke fixed: X no longer freezes at ~0.46-0.49 mm Y/Z error; it only pauses near the hard-abort band.")
print("  Forgiving collision mode: selected RJ45 component plus a broader AS4610 insert tunnel are visual-only so hidden parent/sibling/rack colliders cannot block insertion.")
print("  PATH-STRICT FAST V24: V22 grip recovery kept; insert slowed only enough to keep max Y/Z path error under 1.0 mm.")
print("  Viewport-only layout enabled: hiding dock windows except Viewport / Viewport 1.")
print(f"  PATH-STRICT FAST mid-reorient transit + old servo + strict insert joint steps: reorient={TRANSIT_REORIENT_JOINT_STEPS}, transfer={TRANSIT_TRANSFER_JOINT_STEPS}, approach={TRANSIT_APPROACH_JOINT_STEPS}")
print(f"  PATH-STRICT FAST physical-transit + old servo + strict insert linear steps: lift={TRANSIT_LIFT_LINEAR_STEP * 1000.0:.1f}mm/frame, descend={TRANSIT_DESCEND_LINEAR_STEP * 1000.0:.1f}mm/frame")
print(f"  PATH-STRICT FAST insert X steps: fast={INSERT_X_FAST_STEP * 1000.0:.1f}mm/frame, fine={INSERT_X_FINE_STEP * 1000.0:.1f}mm/frame, hold={INSERT_STABLE_HOLD_FRAMES} frames")
print(f"  PATH-STRICT Y/Z guard: slow_at={INSERT_X_YZ_SLOWDOWN_TOL * 1000.0:.2f}mm, pause_at={INSERT_STROKE_YZ_PAUSE_TOL * 1000.0:.2f}mm, resume_at={INSERT_STROKE_YZ_RESUME_TOL * 1000.0:.2f}mm, abort_at={INSERT_HARD_ABORT_YZ_TOL * 1000.0:.2f}mm")
print(f"  PATH-STRICT FAST release/retreat: settle={POST_INSERT_COLLISION_SETTLE_FRAMES} frames, open={POST_INSERT_OPEN_WAIT_FRAMES} frames, retreat_step={POST_INSERT_RETREAT_LINEAR_STEP * 1000.0:.1f}mm/frame")
print(f"  PATH-STRICT FAST logging: FAST_DEMO_LOGGING={FAST_DEMO_LOGGING}, render_every_n_frames={FAST_RENDER_EVERY_N_FRAMES}")
print(f"  PATH-STRICT pickup grip: static/dynamic friction={GRIP_STATIC_FRICTION:.1f}/{GRIP_DYNAMIC_FRICTION:.1f}, contact_offset={GRIP_CONTACT_OFFSET*1000.0:.1f}mm, close_wait={GRIP_CLOSE_WAIT_FRAMES} frames")
print("  V24 guardrail: grip/transit stay fast-safe; X insert pauses at 0.70 mm Y/Z and resumes at 0.40 mm so the 1.0 mm path limit is preserved.")
print(f"  fixed table/robot base height: {TABLE_HEIGHT:.3f} m")
print("  Insert technique: coarse land in front of port -> measured horizontal pose lock -> strict -X insert.")
print("  Post-insert: if final insert succeeds, port/tunnel collisions are restored, the gripper opens, then the hand retreats linearly +X away from the port.")
print("  Press Play.")

while simulation_app.is_running():
    sim_frame_counter += 1
    # V22: do NOT skip render during physical pickup/transit. In this scene the
    # cable contact/grasp became unreliable when V20/V21 reduced render cadence
    # during the coarse carry. Keep full render until the plug reaches the
    # pre-insert area; after that, reduce render cadence for speed.
    if phase == PHASE_COARSE_WAYPOINTS:
        render_this_frame = True
    else:
        render_this_frame = (FAST_RENDER_EVERY_N_FRAMES <= 1) or (sim_frame_counter % FAST_RENDER_EVERY_N_FRAMES == 0)
    world.step(render=render_this_frame)

    if not world.is_playing():
        if VIEWPORT_ONLY_LAYOUT:
            configure_viewport_only_layout()
        continue

    if VIEWPORT_ONLY_LAYOUT and viewport_layout_frames < VIEWPORT_LAYOUT_REAPPLY_FRAMES:
        configure_viewport_only_layout()
        viewport_layout_frames += 1

    if not waypoints_queued:
        queue_user_waypoints(controller)
        if FAST_DEMO_LOGGING:
            print_queued_commands(controller)
            print_plug_pose_error("AFTER queue_user_waypoints / BEFORE first command")
            log_cable_pose("AFTER queue_user_waypoints / BEFORE first command")
        else:
            print("[QUEUED CONTROLLER COMMANDS] fast-demo logging disabled; running pickup/transit.")
        waypoints_queued = True

    user_robot_step(world, franka, controller, art_kinematics)

    joint_pos = franka.get_joint_positions()
    if joint_pos is None:
        continue

    if phase == PHASE_PLUG_POSE_SERVO:
        if update_plug_pose_servo_state():
            print("\n[PHASE] CLOSED-LOOP PRE-INSERT ALIGNMENT complete")
            pre_insert_ok = measure_plug_pose_servo_result()
            if ENABLE_PLUG_INSERT_SERVO and pre_insert_ok:
                print("\n[PHASE] → CLOSED-LOOP X INSERT STROKE")
                init_plug_insert_servo()
                phase = PHASE_PLUG_INSERT_SERVO
            else:
                phase = PHASE_DONE
                print("\n[PHASE] DONE — pre-insert alignment failed or insertion disabled. Press Stop to reset/rerun.")
        else:
            franka.get_articulation_controller().apply_action(plug_pose_servo_action(joint_pos))
        continue

    if phase == PHASE_PLUG_INSERT_SERVO:
        if update_plug_insert_servo_state():
            print("\n[PHASE] CLOSED-LOOP X INSERT STROKE complete")
            insert_ok = measure_plug_insert_result()
            if insert_ok and ENABLE_POST_INSERT_RELEASE_RETREAT:
                print("\n[PHASE] → POST-INSERT RESTORE COLLISION + RELEASE + LINEAR X RETREAT")
                queue_post_insert_release_retreat()
                phase = PHASE_POST_INSERT_RELEASE_RETREAT
            else:
                phase = PHASE_DONE
                if insert_ok:
                    print("\n[PHASE] DONE — insert succeeded; release/retreat disabled. Press Stop to reset/rerun.")
                else:
                    print("\n[PHASE] DONE — insert result failed; keeping gripper closed for inspection. Press Stop to reset/rerun.")
        else:
            franka.get_articulation_controller().apply_action(plug_insert_servo_action(joint_pos))
        continue

    if phase == PHASE_POST_INSERT_RELEASE_RETREAT:
        if controller.is_done():
            print("\n" + "=" * 88)
            print("[POST-INSERT COLLISION RESTORE + RELEASE + RETREAT COMPLETE]")
            print("  Port/tunnel collisions were restored before release.")
            print("  Invisible socket hold proxy was active before release.")
            print("  Gripper opened.")
            print("  Hand retreated linearly along +X away from the port.")
            print("=" * 88)
            if FAST_DEMO_LOGGING:
                log_cable_pose("AFTER release + X retreat")
            phase = PHASE_DONE
            print("\n[PHASE] DONE — port collision restored, cable released after insertion, hand retreated. Press Stop to reset/rerun.")
        else:
            action = controller.forward(joint_pos)
            franka.get_articulation_controller().apply_action(action)
        continue

    if phase == PHASE_DONE:
        if not final_result_logged:
            if FAST_DEMO_LOGGING:
                log_cable_pose("FINAL HOLD / ALL PHASES DONE")
            final_result_logged = True
        continue

    # PHASE_COARSE_WAYPOINTS: execute your existing waypoint sequence with the
    # same before/after command debug you already had.
    if controller.is_done():
        if active_command_info is not None:
            if FAST_DEMO_LOGGING:
                log_command_boundary("AFTER", active_command_info)
            active_command_info = None

        print("\n[PHASE] COARSE WAYPOINTS complete")
        if FAST_DEMO_LOGGING:
            print_plug_pose_error("AFTER coarse waypoints / BEFORE closed-loop correction")
            log_cable_pose("AFTER coarse waypoints / BEFORE closed-loop correction")

        if ENABLE_PLUG_POSE_SERVO:
            if not pre_servo_grasp_sanity_ok("AFTER coarse waypoints"):
                print("[PHASE] DONE — coarse motion lost the plug. Fix carry path/grip before servo.")
                phase = PHASE_DONE
                continue

            print("[PHASE] → CLOSED-LOOP PRE-INSERT ALIGNMENT")
            controller.clear_queue()
            init_plug_pose_servo()
            phase = PHASE_PLUG_POSE_SERVO
        else:
            print("[PHASE] Plug pose servo disabled. Holding final coarse pose.")
            phase = PHASE_DONE
        continue

    current_info = get_current_command_info(controller)

    if current_info is not None:
        if active_command_info is None:
            active_command_info = current_info
            if FAST_DEMO_LOGGING:
                log_command_boundary("BEFORE", active_command_info)
        elif current_info["index"] != active_command_info["index"]:
            # The previous command completed between frames. Print its after-pose,
            # then immediately print the before-pose for the new command.
            log_command_boundary("AFTER", active_command_info)
            print_plug_pose_error(f"AFTER command {active_command_info['index']} / {active_command_info['label']}")
            active_command_info = current_info
            if FAST_DEMO_LOGGING:
                log_command_boundary("BEFORE", active_command_info)
            print_plug_pose_error(f"BEFORE command {active_command_info['index']} / {active_command_info['label']}")

    previous_info = active_command_info
    action = controller.forward(joint_pos)
    franka.get_articulation_controller().apply_action(action)

    # controller.forward() can complete a command immediately once tolerance /
    # wait-frame criteria are met. Detect that transition right away so every
    # command gets an AFTER pose print.
    next_info = get_current_command_info(controller)
    if previous_info is not None:
        if next_info is None or next_info["index"] != previous_info["index"]:
            if FAST_DEMO_LOGGING:
                log_command_boundary("AFTER", previous_info)
                print_plug_pose_error(f"AFTER command {previous_info['index']} / {previous_info['label']}")
            active_command_info = None

simulation_app.close()
