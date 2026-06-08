"""Pick-and-insert job builder for QSFP modules using port frames."""

from __future__ import annotations

import numpy as np

from port_frame import PortFrame
from qsfp_module import QSFP_LENGTH_M

# Motion defaults (offsets match VERYGOOD/insert_at_prim_lula.py: HOVER 1.8, GRASP 1.515, PICK 1.5)
PICK_HOVER_OFFSET = 0.30
PICK_GRASP_OFFSET = 0.015
HOVER_CLEARANCE = PICK_HOVER_OFFSET
GRASP_CLEARANCE = PICK_GRASP_OFFSET
TRANSPORT_HOVER_CLEARANCE = 0.10
APPROACH_STANDOFF = 0.10
# Minimum leading-tip axial depth (m) along insert axis for a seated module.
# 0.0 = tip at or past the port face; small negative allows slight standoff.
MIN_SEATED_TIP_AXIAL_M = -0.015
SEAT_LATERAL_TOL_M = 0.008
SEAT_SETTLE_FRAMES = 60
# Insert until block center reaches the port prim origin (no axial offset).
INSERT_STOP_DEPTH_M = 0.0
TRANSIT_STANDOFF = 0.25

GRIPPER_CLOSE_FRAMES = 60
# One frame to command open; retreat starts immediately on the next frame.
GRIPPER_RELEASE_FRAMES = 30
TRANSIT_MAX_FRAMES = 300
ALIGN_MAX_FRAMES = 300
INSERT_APPROACH_MAX_FRAMES = 900
INSERT_STEP = 0.002
INSERT_MAX_FRAMES = 300
ALIGN_TOLERANCE = 0.0015
ALIGN_ORIENTATION_TOLERANCE = 0.05
INSERT_TOLERANCE = 0.0015
ALIGN_SETTLE_FRAMES = 2
INSERT_SETTLE_FRAMES = 1
RETREAT_STEP = 0.008
RETREAT_CLEAR_STEP = 0.05
RETREAT_MAX_FRAMES = 300
# Slow straight pull-back immediately after release (keeps fingers off the module).
RETREAT_STANDOFF = 0.125
# Full clear distance along the insert axis before the next pick job starts.
POST_RETREAT_CLEAR_STANDOFF = TRANSIT_STANDOFF


def add_qsfp_insert_job(
    controller,
    port: PortFrame,
    pick_xy: np.ndarray,
    pick_z: float,
    release_frames: int = GRIPPER_RELEASE_FRAMES,
) -> None:
    """Queue pick, transport, align, insert, release, and straight retreat for one port."""
    pick_hover_z = pick_z + HOVER_CLEARANCE
    pick_grasp_z = pick_z + GRASP_CLEARANCE
    pick_lift_z = pick_hover_z
    transport_standoff = port.approach_position(TRANSIT_STANDOFF)
    approach_goal = port.approach_position(APPROACH_STANDOFF)
    retreat_goal = port.approach_position(RETREAT_STANDOFF)
    clear_goal = port.approach_position(POST_RETREAT_CLEAR_STANDOFF)
    insert_goal = port.point_along_axis(INSERT_STOP_DEPTH_M)

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
        verify_pick_lift=True,
        verify_pick_min_z=pick_lift_z - 0.08,
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
    # 6. Line up 10 cm away from the port before inserting.
    controller.add_cartesian_waypoint(
        position=approach_goal,
        orientation=insert_ori,
        pos_tolerance=0.01,
        max_frames=TRANSIT_MAX_FRAMES,
    )
    # 7. Fine lateral align while staying at the 10 cm approach standoff.
    controller.add_cartesian_waypoint(
        position=approach_goal,
        orientation=insert_ori,
        pos_tolerance=ALIGN_TOLERANCE,
        align_yz_only=True,
        track_block=True,
        insert_axis=axis,
        insert_origin=origin,
        max_frames=ALIGN_MAX_FRAMES,
        settle_frames=ALIGN_SETTLE_FRAMES,
        orientation_tolerance=ALIGN_ORIENTATION_TOLERANCE,
    )
    # 8. Insert to the port prim origin along the insert axis, then stop.
    controller.add_cartesian_waypoint(
        position=insert_goal,
        orientation=insert_ori,
        pos_tolerance=INSERT_TOLERANCE,
        x_only_insert=True,
        cartesian_step=INSERT_STEP,
        track_block=True,
        insert_axis=axis,
        insert_origin=origin,
        max_frames=INSERT_MAX_FRAMES,
        settle_frames=INSERT_SETTLE_FRAMES,
    )
    # 9. Release — command open, then retreat starts on the very next frame.
    controller.add_gripper_command(
        action="open",
        wait_frames=release_frames,
        full_open=True,
        freeze_arm=True,
    )
    # 10. Retreat 10 cm straight back along the port axis from the release pose.
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
    # 11. Clear the rack (30 cm back) so the next pick path misses the seated module.
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
