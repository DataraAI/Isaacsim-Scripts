"""Static port-sleeve colliders for QSFP insertion.

Network switch connectors live under USD instance proxies, so runtime mesh
collision authoring on `/World/DataHall/Network_Switches` does not create
colliders on the actual port geometry. These box walls form a simple cage
aligned to each PortFrame so released modules stay seated.

Colliders are spawned as Isaac `FixedCuboid` scene objects (not raw USD cubes)
so PhysX registers them before the first simulation reset.
"""

from __future__ import annotations

import carb
import numpy as np
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.objects import FixedCuboid
from pxr import Usd, UsdPhysics

from port_frame import PortFrame
from qsfp_module import QSFP_HEIGHT_M, QSFP_WIDTH_M

PORT_COLLIDER_ROOT = "/World/PortInsertColliders"
# Extra clearance around the module cross-section inside the sleeve (m).
PORT_CLEARANCE_M = 0.0015
# Sleeve depth along +insert_axis from the port opening (m).
PORT_SLEEVE_DEPTH_M = 0.060
# Extra open tunnel length along -insert_axis (toward the robot) (m).
PORT_MOUTH_BACK_M = 0.025
# Wall / back-plate thickness (m).
PORT_WALL_THICKNESS_M = 0.002
# Set True to draw bright green proxy boxes in the viewport.
PORT_COLLIDER_VISIBLE = False
# PhysX contact padding for the static sleeve walls (m).
PORT_COLLIDER_CONTACT_OFFSET_M = 0.0002

_PROXY_SIZE = 1.0
_PORT_COLLIDER_NAMES: list[str] = []
_PORT_SLEEVE_MATERIAL_PATH = "/World/Physics_Materials/port_sleeve"
_COLLISION_GROUP_ROOT = "/World/CollisionGroups"


def _collision_group_path(kind: str, port_index: int) -> str:
    return f"{_COLLISION_GROUP_ROOT}/{kind}_{port_index:02d}"


def _ensure_collision_group_root(stage: Usd.Stage) -> None:
    if not stage.GetPrimAtPath(_COLLISION_GROUP_ROOT).IsValid():
        stage.DefinePrim(_COLLISION_GROUP_ROOT, "Xform")


def _ensure_collision_group(stage: Usd.Stage, group_path: str) -> UsdPhysics.CollisionGroup:
    _ensure_collision_group_root(stage)
    prim = stage.GetPrimAtPath(group_path)
    if not prim.IsValid():
        return UsdPhysics.CollisionGroup.Define(stage, group_path)
    return UsdPhysics.CollisionGroup(prim)


def _add_prim_to_collision_group(
    stage: Usd.Stage, prim_path: str, group_path: str
) -> None:
    group = _ensure_collision_group(stage, group_path)
    collection = Usd.CollectionAPI.Get(group.GetPrim(), "colliders")
    if not collection:
        collection = Usd.CollectionAPI.Apply(group.GetPrim(), "colliders")
    collection.GetIncludesRel().AddTarget(prim_path)


def setup_per_port_collision_filtering(stage: Usd.Stage, num_ports: int) -> None:
    """Each port sleeve only collides with its own module, not other QSFP bodies."""
    if num_ports <= 0:
        return
    _ensure_collision_group_root(stage)

    module_group_paths = [
        _collision_group_path("qsfp_module", i) for i in range(num_ports)
    ]
    for port_index in range(num_ports):
        _ensure_collision_group(stage, module_group_paths[port_index])
        sleeve_group = _ensure_collision_group(
            stage, _collision_group_path("port_sleeve", port_index)
        )
        filtered = sleeve_group.GetFilteredGroupsRel()
        for other_index, module_group_path in enumerate(module_group_paths):
            if other_index == port_index:
                continue
            if module_group_path not in {str(t) for t in filtered.GetTargets()}:
                filtered.AddTarget(module_group_path)

    carb.log_info(
        f"Configured per-port collision filtering for {num_ports} QSFP ports"
    )


def register_module_collision_group(stage: Usd.Stage, port_index: int, prim_path: str) -> None:
    _add_prim_to_collision_group(
        stage, prim_path, _collision_group_path("qsfp_module", port_index)
    )


def _port_sleeve_physics_material() -> PhysicsMaterial:
    return PhysicsMaterial(
        prim_path=_PORT_SLEEVE_MATERIAL_PATH,
        static_friction=0.6,
        dynamic_friction=0.5,
        restitution=0.0,
    )


def _set_prim_collision_enabled(prim, enabled: bool) -> None:
    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr().Set(enabled)


def enable_port_insert_colliders_for_port(stage: Usd.Stage, port_index: int) -> None:
    """Turn on sleeve collisions for one port after the gripper releases the module."""
    base = f"{PORT_COLLIDER_ROOT}/port_{port_index:02d}"
    port_root = stage.GetPrimAtPath(base)
    if not port_root.IsValid():
        carb.log_warn(f"Port collider root not found: {base}")
        return
    for child in port_root.GetChildren():
        _set_prim_collision_enabled(child, True)
    carb.log_info(f"Enabled port sleeve collisions for port {port_index}")


def disable_all_port_insert_colliders(stage: Usd.Stage) -> None:
    """Disable every sleeve before a new run so inserts start unobstructed."""
    root = stage.GetPrimAtPath(PORT_COLLIDER_ROOT)
    if not root.IsValid():
        return
    for port_root in root.GetChildren():
        for child in port_root.GetChildren():
            _set_prim_collision_enabled(child, False)
    carb.log_info("Disabled all port sleeve collisions")


def _clear_port_colliders(world, stage: Usd.Stage) -> None:
    for name in list(_PORT_COLLIDER_NAMES):
        if world.scene.object_exists(name):
            world.scene.remove_object(name)
    _PORT_COLLIDER_NAMES.clear()

    root = stage.GetPrimAtPath(PORT_COLLIDER_ROOT)
    if not root.IsValid():
        return
    for child in list(root.GetChildren()):
        stage.RemovePrim(child.GetPath())


def _add_static_box(
    world,
    stage: Usd.Stage,
    port_index: int,
    name: str,
    prim_path: str,
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
    size_xyz: np.ndarray,
) -> None:
    color = np.array([0.15, 0.95, 0.25], dtype=np.float64)
    cuboid = world.scene.add(
        FixedCuboid(
            name=name,
            prim_path=prim_path,
            position=np.asarray(position, dtype=np.float64),
            orientation=np.asarray(orientation_wxyz, dtype=np.float64),
            scale=np.asarray(size_xyz, dtype=np.float64),
            size=_PROXY_SIZE,
            color=color,
            visible=PORT_COLLIDER_VISIBLE,
            physics_material=_port_sleeve_physics_material(),
        )
    )
    cuboid.set_contact_offset(PORT_COLLIDER_CONTACT_OFFSET_M)
    cuboid.set_rest_offset(0.0)
    # Sleeves stay non-colliding until release so the gripped module can be driven in.
    _set_prim_collision_enabled(cuboid.prim, False)
    _add_prim_to_collision_group(
        stage, prim_path, _collision_group_path("port_sleeve", port_index)
    )
    _PORT_COLLIDER_NAMES.append(name)


def build_port_insert_colliders(world, stage: Usd.Stage, ports: list[PortFrame]) -> int:
    """Create static sleeve colliders for each port frame."""
    if not ports:
        return 0

    if not stage.GetPrimAtPath(PORT_COLLIDER_ROOT).IsValid():
        stage.DefinePrim(PORT_COLLIDER_ROOT, "Xform")

    _clear_port_colliders(world, stage)
    setup_per_port_collision_filtering(stage, len(ports))

    count = 0
    inner_x = QSFP_WIDTH_M + 2.0 * PORT_CLEARANCE_M
    inner_y = QSFP_HEIGHT_M + 2.0 * PORT_CLEARANCE_M
    sleeve_depth = PORT_SLEEVE_DEPTH_M
    mouth_back = PORT_MOUTH_BACK_M
    total_depth = sleeve_depth + mouth_back
    wall = PORT_WALL_THICKNESS_M

    for index, port in enumerate(ports):
        x_axis, y_axis, z_axis = port.insert_frame_axes()
        origin = port.insert_origin
        tunnel_start = origin - z_axis * mouth_back
        tunnel_center = tunnel_start + z_axis * (total_depth * 0.5)
        ori = port.insert_rot
        base = f"{PORT_COLLIDER_ROOT}/port_{index:02d}"

        specs = [
            (
                "wall_neg_x",
                f"{base}/wall_neg_x",
                tunnel_center - x_axis * (inner_x * 0.5 + wall * 0.5),
                np.array([wall, inner_y + 2.0 * wall, total_depth], dtype=np.float64),
            ),
            (
                "wall_pos_x",
                f"{base}/wall_pos_x",
                tunnel_center + x_axis * (inner_x * 0.5 + wall * 0.5),
                np.array([wall, inner_y + 2.0 * wall, total_depth], dtype=np.float64),
            ),
            (
                "wall_neg_y",
                f"{base}/wall_neg_y",
                tunnel_center - y_axis * (inner_y * 0.5 + wall * 0.5),
                np.array([inner_x, wall, total_depth], dtype=np.float64),
            ),
            (
                "wall_pos_y",
                f"{base}/wall_pos_y",
                tunnel_center + y_axis * (inner_y * 0.5 + wall * 0.5),
                np.array([inner_x, wall, total_depth], dtype=np.float64),
            ),
            (
                "back_plate",
                f"{base}/back_plate",
                origin + z_axis * (sleeve_depth - wall * 0.5),
                np.array([inner_x + 2.0 * wall, inner_y + 2.0 * wall, wall], dtype=np.float64),
            ),
        ]

        for wall_name, path, position, size in specs:
            scene_name = f"port_collider_{index:02d}_{wall_name}"
            _add_static_box(world, stage, index, scene_name, path, position, ori, size)
            count += 1

        carb.log_info(
            f"Port collider {index}: origin={origin.tolist()} "
            f"inner={inner_x*1000:.1f}x{inner_y*1000:.1f}mm "
            f"depth={total_depth*1000:.1f}mm visible={PORT_COLLIDER_VISIBLE}"
        )

    carb.log_info(
        f"Port insert colliders: rebuilt {count} FixedCuboid boxes under {PORT_COLLIDER_ROOT}"
    )
    return count
