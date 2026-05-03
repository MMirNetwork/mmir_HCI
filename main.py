import open3d as o3d
import copy
import numpy as np
from PIL import Image


class RegResult:
    def __init__(self, fitness, inlier_rmse, transformation, method=""):
        self.fitness = fitness
        self.inlier_rmse = inlier_rmse
        self.transformation = transformation
        self.method = method

    def score(self):
        if self.inlier_rmse < 1e-9:
            return 0.0
        return self.fitness / (self.inlier_rmse + 1e-9)

    def __repr__(self):
        return f"[{self.method}] fitness={self.fitness:.4f}  rmse={self.inlier_rmse:.6f}  score={self.score():.2f}"


def normalize_orientation(pcd, up_axis="Y"):
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    R = np.eye(3)
    if up_axis == "Z":
        R = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)
    elif up_axis == "-Z":
        R = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=float)
    elif up_axis == "X":
        R = np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)
    elif up_axis == "-Y":
        R = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ center
    pcd_out = copy.deepcopy(pcd)
    pcd_out.transform(T)
    return pcd_out, T


def remove_outliers(pcd, nb=20, std=2.0):
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    return cl


def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = remove_outliers(pcd_down)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30))
    pcd_down.orient_normals_consistent_tangent_plane(30)
    return pcd_down


def rotation_candidates(source, target):
    src_c = np.asarray(source.get_center())
    tgt_c = np.asarray(target.get_center())
    candidates = []
    for deg in [0, 90, 180, 270]:
        a = np.radians(deg)
        Ry = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]], dtype=float)
        t = tgt_c - Ry @ src_c
        T = np.eye(4)
        T[:3, :3] = Ry
        T[:3, 3] = t
        candidates.append((T, f"Yaw{deg}"))
    return candidates


def refine_icp(source, target, init_transform, voxel_size):
    T = init_transform
    for scale, iters in [(8, 40), (4, 60), (2, 80), (1, 100)]:
        try:
            res = o3d.pipelines.registration.registration_icp(
                source, target, voxel_size * scale, T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iters))
            T = res.transformation
        except Exception:
            pass
    ev = o3d.pipelines.registration.evaluate_registration(source, target, voxel_size, T)
    return RegResult(ev.fitness, ev.inlier_rmse, T, "ICP")


def find_best_registration(source_down, target_down, voxel_size):
    threshold = voxel_size * 5
    print("Generating candidates...")
    raw = []
    for T, name in rotation_candidates(source_down, target_down):
        ev = o3d.pipelines.registration.evaluate_registration(source_down, target_down, threshold, T)
        raw.append(RegResult(ev.fitness, ev.inlier_rmse, T, name))
    raw.sort(key=lambda r: r.score(), reverse=True)
    for r in raw:
        print(f"  {r}")

    print("Refining with ICP...")
    refined = []
    for cand in raw:
        r = refine_icp(source_down, target_down, cand.transformation, voxel_size)
        r.method = cand.method + "+ICP"
        refined.append(r)
        print(f"  {r}")

    refined.sort(key=lambda r: r.score(), reverse=True)
    print(f"Best: {refined[0]}")
    return refined[0]


def bake_vertex_colors(mesh, tex_path):
    tex = np.array(Image.open(tex_path))
    th, tw = tex.shape[:2]

    n_verts = len(mesh.vertices)
    color_acc = np.zeros((n_verts, 3), dtype=np.float64)
    count_acc = np.zeros(n_verts, dtype=np.float64)

    triangles = np.asarray(mesh.triangles)
    triangle_uvs = np.asarray(mesh.triangle_uvs)

    for i, tri in enumerate(triangles):
        uvs = triangle_uvs[i * 3 : i * 3 + 3]
        uv_c = uvs.mean(axis=0)
        px_x = int(np.clip(uv_c[0],       0.0, 0.999) * (tw - 1))
        px_y = int(np.clip(1.0 - uv_c[1], 0.0, 0.999) * (th - 1))
        fc = tex[px_y, px_x, :3] / 255.0
        for vi in tri:
            color_acc[vi] += fc
            count_acc[vi] += 1

    count_acc = np.maximum(count_acc, 1)
    vertex_colors = (color_acc / count_acc[:, None]).astype(np.float32)
    print(f"Baked {n_verts} vertex colors  (mean brightness {vertex_colors.mean():.3f})")
    return vertex_colors


def transfer_texture(mesh_scan, mesh_digital, scan_vertex_colors, dist_threshold=0.03):
    scan_pcd = o3d.geometry.PointCloud()
    scan_pcd.points = mesh_scan.vertices
    scan_pcd.colors = o3d.utility.Vector3dVector(scan_vertex_colors)

    tree = o3d.geometry.KDTreeFlann(scan_pcd)
    dig_verts = np.asarray(mesh_digital.vertices)

    colors  = []
    ignored = 0
    for pt in dig_verts:
        k, idx, dist2 = tree.search_knn_vector_3d(pt, 1)
        dist = np.sqrt(dist2[0])
        if dist > dist_threshold:
            colors.append([0.5, 0.5, 0.5])
            ignored += 1
        else:
            colors.append(scan_vertex_colors[idx[0]])

    mesh_digital.vertex_colors = o3d.utility.Vector3dVector(colors)
    print(f"Texture transfer done  (ignored {ignored}/{len(dig_verts)} > {dist_threshold}m)")


def show(window_data):
    visualizers = []
    for geometries, name in window_data:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=name, width=700, height=500)
        for g in (geometries if isinstance(geometries, list) else [geometries]):
            vis.add_geometry(g)
        visualizers.append(vis)
    running = True
    while running:
        for vis in visualizers:
            if not vis.poll_events():
                running = False
                break
            vis.update_renderer()
    for vis in visualizers:
        vis.destroy_window()


if __name__ == "__main__":
    VOXEL       = 0.005
    SCAN_OBJ    = "Models/coffee_machine-scanned_result-iphone/coffee_machine-scanned_result-iphone.obj"
    TEXTURE     = "texture.jpg"
    DIGITAL_OBJ = "Models/coffee_machine-digital-with_joints.obj"

    print("Loading point clouds...")
    pcd_scan    = o3d.io.read_point_cloud("cropped_1.ply")
    pcd_digital = o3d.io.read_point_cloud("cropped_2.ply")
    print(f"  scan={len(pcd_scan.points):,}  digital={len(pcd_digital.points):,}")

    pcd_digital.scale(0.001, center=np.zeros(3))
    pcd_scan    = remove_outliers(pcd_scan)
    pcd_digital = remove_outliers(pcd_digital)

    print("Normalizing axes...")
    pcd_scan_n,    T_scan_n    = normalize_orientation(pcd_scan,    up_axis="Y")
    pcd_digital_n, T_digital_n = normalize_orientation(pcd_digital, up_axis="-Z")

    print(f"Preprocessing  voxel={VOXEL}...")
    scan_down    = preprocess(pcd_scan_n,    VOXEL)
    digital_down = preprocess(pcd_digital_n, VOXEL)
    print(f"  scan={len(scan_down.points):,}  digital={len(digital_down.points):,}")

    best = find_best_registration(scan_down, digital_down, VOXEL)

    T_align = best.transformation @ T_scan_n

    print("Loading meshes...")
    mesh_scan    = o3d.io.read_triangle_mesh(SCAN_OBJ,    enable_post_processing=True)
    mesh_digital = o3d.io.read_triangle_mesh(DIGITAL_OBJ, enable_post_processing=True)
    mesh_digital.scale(0.001, center=np.zeros(3))

    print("Baking texture into scan vertex colors...")
    scan_vertex_colors = bake_vertex_colors(mesh_scan, TEXTURE)

    mesh_scan_aligned    = copy.deepcopy(mesh_scan);    mesh_scan_aligned.transform(T_align)
    mesh_digital_aligned = copy.deepcopy(mesh_digital); mesh_digital_aligned.transform(T_digital_n)

    mesh_scan_aligned.vertex_colors = o3d.utility.Vector3dVector(scan_vertex_colors)

    transfer_texture(mesh_scan_aligned, mesh_digital_aligned, scan_vertex_colors)

    mesh_digital_final = copy.deepcopy(mesh_digital)
    mesh_digital_final.vertex_colors = mesh_digital_aligned.vertex_colors

    C_SCAN    = [1.0, 0.7, 0.0]
    C_DIGITAL = [0.0, 0.6, 0.9]

    pc_s = copy.deepcopy(scan_down); pc_s.paint_uniform_color(C_SCAN)
    pc_d = copy.deepcopy(digital_down); pc_d.paint_uniform_color(C_DIGITAL)

    pc_s_aligned = copy.deepcopy(scan_down)
    pc_s_aligned.paint_uniform_color(C_SCAN)
    pc_s_aligned.transform(best.transformation)
    pc_d_ref = copy.deepcopy(digital_down)
    pc_d_ref.paint_uniform_color(C_DIGITAL)

    show([
        ([pc_s, pc_d],                  "Before"),
        ([pc_s_aligned, pc_d_ref],      f"Aligned  {best.method}"),
        ([mesh_scan_aligned, pc_d_ref], "Scan mesh + Digital PCD"),
        ([mesh_digital_final],          "Result"),
    ])