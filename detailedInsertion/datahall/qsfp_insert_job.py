"""Pick-and-insert job builder for QSFP modules using port frames."""

from __future__ import annotations

import numpy as np

from port_frame import PortFrame
from qsfp_module import QSFP_GRASP_OFFSET_TO_TOP_M, QSFP_LENGTH_M, pick_grasp_block_z

# Motion defaults (offsets match VERYGOOD/insert_at_prim_lula.py: HOVER 1.8, GRASP 1.515, PICK 1.5)
PICK_HOVER_OFFSET = 0.30
PICK_GRASP_OFFSET = 0.015
HOVER_CLEARANCE = PICK_HOVER_OFFSET
GRASP_CLEARANCE = PICK_GRASP_OFFSET
TRANSPORT_HOVER_CLEARANCE = 0.10
# Lateral align standoff (m) — stay well back from the port face while lining up.
ALIGN_STANDOFF = 0.18
# Legacy name used by insert_at_prim_lula logging.
APPROACH_STANDOFF = ALIGN_STANDOFF
# End standoff (m) for the slow axial move-in before the final insert crawl.
INSERT_CREEP_STANDOFF = 0.06
# Minimum leading-tip axial depth (m) along insert axis for a seated module.
# 0.0 = tip at or past the port face; small negative allows slight standoff.
MIN_SEATED_TIP_AXIAL_M = -0.015
SEAT_LATERAL_TOL_M = 0.008
SEAT_SETTLE_FRAMES = 60
# Fallback only when callers omit insert_tip_depth_m (tune depth in insert_at_prim_lula UI).
INSERT_TIP_DEPTH_M = 0.048
INSERT_FORCE_STOP_THRESHOLD = 18.0
TRANSIT_STANDOFF = 0.6

GRIPPER_CLOSE_FRAMES = 60
# One frame to command open; retreat starts immediately on the next frame.
GRIPPER_RELEASE_FRAMES = 30
TRANSIT_MAX_FRAMES = 300
ALIGN_MAX_FRAMES = 300
INSERT_APPROACH_MAX_FRAMES = 900
INSERT_STEP = 0.0015
INSERT_CREEP_STEP = 0.001
INSERT_CREEP_MAX_FRAMES = 1200
INSERT_MAX_FRAMES = 600
ALIGN_TOLERANCE = 0.0015
ALIGN_ORIENTATION_TOLERANCE = 0.05
INSERT_TOLERANCE = 0.0015
ALIGN_SETTLE_FRAMES = 2
INSERT_SETTLE_FRAMES = 1
RETREAT_STEP = 0.008
RETREAT_CLEAR_STEP = 0.05
RETREAT_MAX_FRAMES = 600
# Slow straight pull-back immediately after release (keeps fingers off the module).
RETREAT_STANDOFF = 0.2
# Full clear distance along the insert axis before the next pick job starts.
POST_RETREAT_CLEAR_STANDOFF = TRANSIT_STANDOFF

# 0-based command offsets within each queued job (for callbacks in the main script).
JOB_CMD_INSERT = 9
JOB_CMD_CLEAR = 12


def add_qsfp_insert_job(
    controller,
    port: PortFrame,
    pick_xy: np.ndarray,
    pick_z: float,
    insert_tip_depth_m: float = INSERT_TIP_DEPTH_M,
    pick_grasp_offset_to_top_m: float = QSFP_GRASP_OFFSET_TO_TOP_M,
    release_frames: int = GRIPPER_RELEASE_FRAMES,
    *,
    align_tolerance: float = ALIGN_TOLERANCE,
    align_settle_frames: int = ALIGN_SETTLE_FRAMES,
    align_max_frames: int = ALIGN_MAX_FRAMES,
    insert_step: float = INSERT_STEP,
    insert_max_frames: int = INSERT_MAX_FRAMES,
    insert_settle_frames: int = INSERT_SETTLE_FRAMES,
    align_standoff: float = ALIGN_STANDOFF,
    insert_creep_standoff: float = INSERT_CREEP_STANDOFF,
    insert_creep_step: float = INSERT_CREEP_STEP,
    insert_creep_max_frames: int = INSERT_CREEP_MAX_FRAMES,
) -> None:
    """Queue pick, transport, align, insert, release, and straight retreat for one port."""
    grasp_center_z = pick_grasp_block_z(
        pick_z, offset_to_top_m=pick_grasp_offset_to_top_m
    )
    pick_hover_z = grasp_center_z + HOVER_CLEARANCE
    pick_grasp_z = grasp_center_z + GRASP_CLEARANCE
    pick_lift_z = pick_hover_z
    transport_standoff = port.approach_position(TRANSIT_STANDOFF)
    align_goal = port.approach_position(align_standoff)
    creep_goal = port.approach_position(insert_creep_standoff)
    retreat_goal = port.approach_position(RETREAT_STANDOFF)
    clear_goal = port.approach_position(POST_RETREAT_CLEAR_STANDOFF)
    module_half_length = QSFP_LENGTH_M / 2.0
    insert_goal = port.center_goal_for_tip_depth(
        insert_tip_depth_m,
        module_half_length,
        module_orientation_wxyz=port.insert_rot,
    )
    axis = port.insert_axis
    origin = port.insert_origin
    down_ori = port.pick_down_rot
    insert_ori = port.insert_rot

    # 1. Hover over module in tray
    controller.add_cartesian_waypoint(
        position=np.array([pick_xy[0], pick_xy[1], pick_hover_z]),
        orientation=down_ori,
        pos_tolerance=0.05,
        max_frames=TRANSIT_MAX_FRAMES,
    )
    # 2. Grasp
    controller.add_cartesian_waypoint(
        position=np.array([pick_xy[0], pick_xy[1], pick_grasp_z]),
        orientation=down_ori,
        pos_tolerance=0.001,
        max_frames=TRANSIT_MAX_FRAMES,
    )
    controller.add_gripper_command(action="close", wait_frames=GRIPPER_CLOSE_FRAMES)
    # 3. Lift
    controller.add_cartesian_waypoint(
        position=np.array([pick_xy[0], pick_xy[1], pick_lift_z]),
        orientation=down_ori,
        pos_tolerance=0.05,
        max_frames=TRANSIT_MAX_FRAMES,
    )
    # 4. Rotate to the fixed horizontal insert orientation above the pickup area.
    controller.add_cartesian_waypoint(
        position=np.array([pick_xy[0], pick_xy[1], pick_lift_z]),
        orientation=insert_ori,
        pos_tolerance=0.05,
        max_frames=TRANSIT_MAX_FRAMES,
    )
    # 5. Move near the port while staying horizontal and facing the insert axis.
    controller.add_cartesian_waypoint(
        position=transport_standoff,
        orientation=insert_ori,
        pos_tolerance=0.05,
        max_frames=TRANSIT_MAX_FRAMES,
    )
    # 6. Move to the far align standoff (stay back from the port opening).
    controller.add_cartesian_waypoint(
        position=align_goal,
        orientation=insert_ori,
        pos_tolerance=0.01,
        max_frames=TRANSIT_MAX_FRAMES,
    )
    # 7. Fine lateral align while staying at the far standoff.
    controller.add_cartesian_waypoint(
        position=align_goal,
        orientation=insert_ori,
        pos_tolerance=align_tolerance,
        align_yz_only=True,
        track_block=True,
        insert_axis=axis,
        insert_origin=origin,
        max_frames=align_max_frames,
        settle_frames=align_settle_frames,
        orientation_tolerance=ALIGN_ORIENTATION_TOLERANCE,
    )
    # 8. Slow axial move-in from the align standoff toward the port mouth.
    controller.add_cartesian_waypoint(
        position=creep_goal,
        orientation=insert_ori,
        pos_tolerance=0.002,
        x_only_insert=True,
        cartesian_step=insert_creep_step,
        track_block=True,
        insert_axis=axis,
        insert_origin=origin,
        max_frames=insert_creep_max_frames,
        settle_frames=2,
    )
    # 9. Crawl until the leading module tip reaches insert_tip_depth_m.
    controller.add_cartesian_waypoint(
        position=insert_goal,
        orientation=insert_ori,
        pos_tolerance=INSERT_TOLERANCE,
        x_only_insert=True,
        cartesian_step=insert_step,
        track_block=True,
        insert_axis=axis,
        insert_origin=origin,
        insert_tip_depth_m=insert_tip_depth_m,
        module_half_length=module_half_length,
        compliant_insert=True,
        contact_force_threshold=INSERT_FORCE_STOP_THRESHOLD,
        stop_on_insert_blocked=True,
        max_frames=insert_max_frames,
        settle_frames=insert_settle_frames,
    )
    # 10. Release — command open, then retreat starts on the very next frame.
    controller.add_gripper_command(
        action="open",
        wait_frames=release_frames,
        full_open=True,
        freeze_arm=True,
    )
    # 11. Retreat straight back along the port axis from the release pose.
    controller.add_cartesian_waypoint(
        position=retreat_goal,
        orientation=insert_ori,
        pos_tolerance=0.005,
        x_only_insert=True,
        cartesian_step=RETREAT_STEP,
        insert_axis=axis,
        insert_origin=origin,
        post_contact_retreat=True,
        hold_current_orientation=True,
        max_frames=RETREAT_MAX_FRAMES,
        settle_frames=1,
    )
    # 12. Clear the rack so the next pick path misses the seated module.
    controller.add_cartesian_waypoint(
        position=clear_goal,
        orientation=insert_ori,
        pos_tolerance=0.02,
        x_only_insert=True,
        cartesian_step=RETREAT_CLEAR_STEP,
        insert_axis=axis,
        insert_origin=origin,
        post_contact_retreat=True,
        hold_current_orientation=True,
        keep_gripper_open=True,
        max_frames=RETREAT_MAX_FRAMES,
        settle_frames=1,
    )
