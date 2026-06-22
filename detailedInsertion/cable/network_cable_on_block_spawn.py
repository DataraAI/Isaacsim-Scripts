from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni.usd

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema


NETWORK_CABLE_USD_PATH = "/home/aayush/isaacsim_assets/Network cable 001/model_Networkcable1_69323.usd"
NETWORK_CABLE_ROOT_PATH = "/World/NetworkCable"
TRACKED_PLUG_PRIM_PATH = f"{NETWORK_CABLE_ROOT_PATH}/E_crystal_head1_45"

BLOCK_PATH = "/World/PickupBlock"

# Where the raised connector should sit.
BLOCK_CENTER = np.array([0.50, 0.00, 0.175], dtype=np.float64)

# Block dimensions in meters.
# Taller block = easier horizontal pickup clearance.
BLOCK_SIZE = np.array([0.025, 0.005, 0.025], dtype=np.float64)

# Small gap so the connector is not spawned interpenetrating the block.
PLUG_BLOCK_CLEARANCE = 0.004

GROUND_CLEARANCE = 0.002
SETTLE_FRAMES = 300


# ------------------------- USD helpers -------------------------

def get_bbox(prim_path):
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


def set_world_translate(prim_path, translation):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    value = Gf.Vec3d(
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    )

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(value)
            return

    xform.AddTranslateOp().Set(value)


def remove_prim_if_exists(prim_path):
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))


def enable_gpu_dynamics():
    stage = omni.usd.get_context().get_stage()
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        scenes = [UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene")).GetPrim()]

    for prim in scenes:
        api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        api.CreateEnableGPUDynamicsAttr(True).Set(True)
        api.CreateBroadphaseTypeAttr("GPU").Set("GPU")
        api.CreateSolverTypeAttr("TGS").Set("TGS")


def create_static_block():
    remove_prim_if_exists(BLOCK_PATH)

    stage = omni.usd.get_context().get_stage()

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(BLOCK_PATH))
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.35, 1.0)])

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(
        Gf.Vec3d(
            float(BLOCK_CENTER[0]),
            float(BLOCK_CENTER[1]),
            float(BLOCK_CENTER[2]),
        )
    )
    xform.AddScaleOp().Set(
        Gf.Vec3f(
            float(BLOCK_SIZE[0]),
            float(BLOCK_SIZE[1]),
            float(BLOCK_SIZE[2]),
        )
    )

    # Collision-only = static environment object.
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    block_top_z = BLOCK_CENTER[2] + 0.5 * BLOCK_SIZE[2]

    print("[BLOCK]")
    print(f"  center={np.round(BLOCK_CENTER, 5)}")
    print(f"  size_mm={np.round(BLOCK_SIZE * 1000.0, 2)}")
    print(f"  top_z={block_top_z:.5f}")

    return block_top_z


def load_cable():
    remove_prim_if_exists(NETWORK_CABLE_ROOT_PATH)
    add_reference_to_stage(
        usd_path=NETWORK_CABLE_USD_PATH,
        prim_path=NETWORK_CABLE_ROOT_PATH,
    )


def place_tracked_plug_on_block(block_top_z):
    """
    Move the whole cable root so E_crystal_head1_45 sits on top of the block.

    Important:
    - We do NOT move E_crystal_head1_45 directly.
    - We do NOT rotate the connector child.
    - We only translate /World/NetworkCable.
    """

    stage = omni.usd.get_context().get_stage()

    root_min, root_max, root_center, root_dims = get_bbox(NETWORK_CABLE_ROOT_PATH)
    plug_min, plug_max, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)

    root_prim = stage.GetPrimAtPath(NETWORK_CABLE_ROOT_PATH)
    root_pose = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(root_prim)
    root_t = np.array(root_pose.ExtractTranslation(), dtype=np.float64)

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

    root_min, root_max, root_center, root_dims = get_bbox(NETWORK_CABLE_ROOT_PATH)
    plug_min, plug_max, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)

    print("[CABLE PLACED]")
    print(f"  plug_center={np.round(plug_center, 5)}")
    print(f"  plug_min_z={plug_min[2]:.5f}")
    print(f"  plug_dims_mm={np.round(plug_dims * 1000.0, 3)}")
    print(f"  root_min={np.round(root_min, 5)}")
    print(f"  root_max={np.round(root_max, 5)}")

    if root_min[2] < -0.002:
        print("[WARN] Part of cable root bbox is below ground. The far end may be clipping initially.")
    else:
        print("[OK] Cable root bbox is above ground.")


# ------------------------- Scene setup -------------------------

world = World(stage_units_in_meters=1.0)
world.set_simulation_dt(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)

stage = omni.usd.get_context().get_stage()

light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
light.CreateIntensityAttr(500.0)
light.CreateColorAttr((1.0, 1.0, 1.0))

world.scene.add_default_ground_plane()

block_top_z = create_static_block()
load_cable()
place_tracked_plug_on_block(block_top_z)
enable_gpu_dynamics()

world.reset()

print("[READY]")
print("  One connector should start on top of the blue block.")
print("  Let it settle. If it falls/slides too much, increase BLOCK_SIZE or PLUG_BLOCK_CLEARANCE.")


settle_count = 0
printed_done = False

while simulation_app.is_running():
    world.step(render=True)

    if not world.is_playing():
        continue

    if printed_done:
        continue

    settle_count += 1

    if settle_count >= SETTLE_FRAMES:
        plug_min, plug_max, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)
        block_top_z = BLOCK_CENTER[2] + 0.5 * BLOCK_SIZE[2]

        print("[SETTLED]")
        print(f"  plug_center={np.round(plug_center, 5)}")
        print(f"  plug_min_z={plug_min[2]:.5f}")
        print(f"  block_top_z={block_top_z:.5f}")
        print(f"  height_above_block_mm={(plug_min[2] - block_top_z) * 1000.0:.2f}")

        printed_done = True

simulation_app.close()