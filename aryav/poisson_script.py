
import numpy as np
from plyfile import PlyData
import open3d as o3d

def gaussian_ply_to_mesh_poisson(ply_path, output_obj_path,
                                   opacity_floor=0.05,
                                   poisson_depth=9,
                                   density_trim_quantile=0.02):
    ply = PlyData.read(ply_path)
    verts = ply['vertex']

    x = np.array(verts['x'])
    y = np.array(verts['y'])
    z = np.array(verts['z'])
    positions = np.stack([x, y, z], axis=1)

    opacity_raw = np.array(verts['opacity'])
    opacities = 1 / (1 + np.exp(-opacity_raw))

    scale_0 = np.exp(np.array(verts['scale_0']))
    scale_1 = np.exp(np.array(verts['scale_1']))
    scale_2 = np.exp(np.array(verts['scale_2']))
    scales = np.stack([scale_0, scale_1, scale_2], axis=1).mean(axis=1)

    # Same filtering as before — drop ghost/low-opacity Gaussians
    mask = opacities > opacity_floor
    positions = positions[mask]
    opacities = opacities[mask]
    scales = scales[mask]

    print(f"Using {len(positions):,} Gaussians after opacity filtering")

    # Build an Open3D point cloud from the filtered Gaussian centers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    # Optional: drop the very largest (likely background/noise) Gaussians,
    # since huge splats often correspond to diffuse background, not solid surface
    if scales is not None:
        scale_cutoff = np.quantile(scales, 0.95)
        keep = scales < scale_cutoff
        pcd = pcd.select_by_index(np.where(keep)[0])
        print(f"Kept {len(pcd.points):,} points after trimming largest 5% of Gaussians by scale")

    # Poisson reconstruction needs normals — estimate them from local neighborhoods
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    print(f"Running Poisson reconstruction (depth={poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )

    # Poisson tends to extrapolate a "blobby" surface beyond where data actually
    # supports it — trim the lowest-density vertices to cut that off
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, density_trim_quantile)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    o3d.io.write_triangle_mesh(output_obj_path, mesh)
    print(f"Saved mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")
    print(f"Output: {output_obj_path}")

if __name__ == "__main__":
    gaussian_ply_to_mesh_poisson(
        "/home/aayush/Desktop/lyra_dynamic_demo_generated/60/gaussians_orig/gaussians_0.ply",
        "/home/aayush/Desktop/bmw_proxy_mesh_60_poisson.obj"
    )
