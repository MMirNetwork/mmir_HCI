import os, zipfile, tempfile, glob, json
import open3d as o3d
import copy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import xatlas

def load_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    elif ext in (".glb", ".gltf"):
        import trimesh
        raw = trimesh.load(path)
        tm = trimesh.util.concatenate(list(raw.geometry.values())) if hasattr(raw, "geometry") else raw
        m = o3d.geometry.TriangleMesh()
        m.vertices  = o3d.utility.Vector3dVector(np.array(tm.vertices,  dtype=np.float64))
        m.triangles = o3d.utility.Vector3iVector(np.array(tm.faces,     dtype=np.int32))
        if hasattr(tm.visual, "uv") and tm.visual.uv is not None:
            idx = np.array(tm.faces, dtype=np.int32)
            m.triangle_uvs = o3d.utility.Vector2dVector(np.array(tm.visual.uv, dtype=np.float64)[idx.ravel()])
            mat = tm.visual.material
            if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                m.textures = [o3d.geometry.Image(np.array(mat.baseColorTexture.convert("RGB"), dtype=np.uint8))]
                m.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(tm.faces), dtype=np.int32))
        m.compute_vertex_normals()
        return m
    elif ext == ".usdz":
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(path)
        verts_all, faces_all, uvs_all, off = [], [], [], 0
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Mesh": continue
            um = UsdGeom.Mesh(prim)
            pts  = np.array(um.GetPointsAttr().Get(), dtype=np.float64)
            fidx = np.array(um.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            verts_all.append(pts); faces_all.append(fidx.reshape(-1, 3) + off); off += len(pts)
            st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            if st and st.IsDefined():
                uv_vals = np.array(st.Get(), dtype=np.float64)
                uv_idx  = st.GetIndices()
                uvs_all.append((uv_vals[np.array(uv_idx, dtype=np.int32)] if uv_idx is not None and len(uv_idx) else uv_vals).reshape(-1, 2))
        m = o3d.geometry.TriangleMesh()
        all_v = np.concatenate(verts_all); all_f = np.concatenate(faces_all)
        m.vertices  = o3d.utility.Vector3dVector(all_v)
        m.triangles = o3d.utility.Vector3iVector(all_f)
        if uvs_all:
            all_uv = np.concatenate(uvs_all)
            if len(all_uv) == len(all_f) * 3:
                m.triangle_uvs = o3d.utility.Vector2dVector(all_uv)
                with zipfile.ZipFile(path) as z:
                    cands = [n for n in z.namelist() if n.lower().endswith(("_tex0.png", "_tex0.jpg"))] \
                         or [n for n in z.namelist() if n.lower().endswith(".png")]
                    if cands:
                        with tempfile.TemporaryDirectory() as tmp:
                            z.extract(cands[0], tmp)
                            img = Image.open(os.path.join(tmp, cands[0])).convert("RGB")
                            m.textures = [o3d.geometry.Image(np.array(img, dtype=np.uint8))]
                            m.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(all_f), dtype=np.int32))
        m.compute_vertex_normals()
        return m
    else:
        raise ValueError(f"Unsupported format: {ext}")


def create_debug_texture(width=2048, height=2048, grid_size=20):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    cell_w = width / grid_size
    cell_h = height / grid_size

    try:
        font = ImageFont.truetype("arial.ttf", int(min(cell_w, cell_h) * 0.3))
    except Exception:
        font = ImageFont.load_default()

    for i in range(grid_size):
        for j in range(grid_size):
            x0 = j * cell_w
            y0 = i * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            r = int(255 * (j / grid_size))
            g = int(255 * (i / grid_size))
            b = 150
            bg_color = (r, g, b)
            draw.rectangle([x0, y0, x1, y1], fill=bg_color, outline="black")
            text = f"{i},{j}"
            draw.text((x0 + cell_w*0.1, y0 + cell_h*0.1), text, fill="white", font=font)
            draw.text((x0 + cell_w*0.1 + 2, y0 + cell_h*0.1 + 2), text, fill="black", font=font)
    return img


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


def compute_scale_metric(pcd, method="median_radius"):
    pts = np.asarray(pcd.points)
    if method == "aabb_diag":
        bb = pcd.get_axis_aligned_bounding_box()
        return float(np.linalg.norm(bb.get_extent()))
    elif method == "median_radius":
        c = pts.mean(axis=0)
        d = np.linalg.norm(pts - c, axis=1)
        return float(np.median(d))
    else:
        raise ValueError(method)


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


def parameterize_mesh_xatlas(mesh):
    print("Parameterizing digital mesh with xatlas (creating optimal UV map)...")
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, triangles)
    atlas.generate()

    vmapping, new_triangles, new_uvs = atlas[0]
    new_vertices = vertices[vmapping]

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)

    uv_flat = new_uvs[new_triangles].reshape(-1, 2)
    new_mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_flat)
    new_mesh.compute_vertex_normals()

    if mesh.has_vertex_colors():
        old_colors = np.asarray(mesh.vertex_colors)
        new_colors = old_colors[vmapping]
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)

    return new_mesh


def bake_digital_texture(mesh_digital, mesh_scan, tex_path, out_size=4096, max_dist=0.05):
    print(f"Baking high-res texture ({out_size}x{out_size}) for digital model...")
    tex_img = (tex_path if isinstance(tex_path, Image.Image) else Image.open(tex_path)).convert("RGB")
    tex_w, tex_h = tex_img.size
    tex_arr = np.array(tex_img).astype(np.float32) / 255.0

    digital_uvs = np.asarray(mesh_digital.triangle_uvs)
    if len(digital_uvs) == 0:
        print("WARNING: Digital model has no UVs.")
        return None

    digital_tris  = np.asarray(mesh_digital.triangles)
    digital_verts = np.asarray(mesh_digital.vertices)
    T = len(digital_tris)

    out_tex  = np.zeros((out_size, out_size, 3), dtype=np.float32)
    out_mask = np.zeros((out_size, out_size),    dtype=bool)

    scan_t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_scan)
    scan_scene  = o3d.t.geometry.RaycastingScene()
    scan_scene.add_triangles(scan_t_mesh)
    scan_tris = np.asarray(mesh_scan.triangles)
    scan_uvs  = np.asarray(mesh_scan.triangle_uvs)

    pix_xs, pix_ys, pos_list = [], [], []

    for ti in range(T):
        uv0 = digital_uvs[ti*3 + 0]
        uv1 = digital_uvs[ti*3 + 1]
        uv2 = digital_uvs[ti*3 + 2]

        p0 = np.array([uv0[0]*(out_size-1), (1.0-uv0[1])*(out_size-1)])
        p1 = np.array([uv1[0]*(out_size-1), (1.0-uv1[1])*(out_size-1)])
        p2 = np.array([uv2[0]*(out_size-1), (1.0-uv2[1])*(out_size-1)])

        min_x = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
        max_x = min(int(np.ceil (max(p0[0], p1[0], p2[0]))), out_size-1)
        min_y = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
        max_y = min(int(np.ceil (max(p0[1], p1[1], p2[1]))), out_size-1)
        if max_x < min_x or max_y < min_y:
            continue

        xs, ys = np.meshgrid(np.arange(min_x, max_x+1),
                             np.arange(min_y, max_y+1))
        xs = xs.ravel().astype(np.float32)
        ys = ys.ravel().astype(np.float32)

        denom = (p1[1]-p2[1])*(p0[0]-p2[0]) + (p2[0]-p1[0])*(p0[1]-p2[1])
        if abs(denom) < 1e-12:
            continue
        l0 = ((p1[1]-p2[1])*(xs-p2[0]) + (p2[0]-p1[0])*(ys-p2[1])) / denom
        l1 = ((p2[1]-p0[1])*(xs-p2[0]) + (p0[0]-p2[0])*(ys-p2[1])) / denom
        l2 = 1.0 - l0 - l1

        eps = -1e-4
        inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
        if not np.any(inside):
            continue
        l0i, l1i, l2i = l0[inside], l1[inside], l2[inside]
        xi, yi = xs[inside].astype(np.int32), ys[inside].astype(np.int32)

        v0 = digital_verts[digital_tris[ti, 0]]
        v1 = digital_verts[digital_tris[ti, 1]]
        v2 = digital_verts[digital_tris[ti, 2]]
        pos = (l0i[:,None]*v0) + (l1i[:,None]*v1) + (l2i[:,None]*v2)

        pix_xs.append(xi)
        pix_ys.append(yi)
        pos_list.append(pos.astype(np.float32))

    if not pos_list:
        print("WARNING: no UV triangles rasterized.")
        return None

    pix_xs = np.concatenate(pix_xs)
    pix_ys = np.concatenate(pix_ys)
    positions = np.concatenate(pos_list, axis=0)
    print(f"  Rasterized {len(positions):,} pixels across {T:,} triangles")

    qp = o3d.core.Tensor(positions, dtype=o3d.core.Dtype.Float32)
    ans = scan_scene.compute_closest_points(qp)
    s_tri  = ans['primitive_ids'].numpy()
    s_bary = ans['primitive_uvs'].numpy()
    s_pts  = ans['points'].numpy()
    dists  = np.linalg.norm(positions - s_pts, axis=1)

    su = s_bary[:, 0]
    sv = s_bary[:, 1]
    sw = 1.0 - su - sv

    suv0 = scan_uvs[s_tri*3 + 0]
    suv1 = scan_uvs[s_tri*3 + 1]
    suv2 = scan_uvs[s_tri*3 + 2]
    final_uvs = sw[:,None]*suv0 + su[:,None]*suv1 + sv[:,None]*suv2

    px = np.clip(final_uvs[:,0], 0.0, 0.999) * (tex_w - 1)
    py = np.clip(1.0 - final_uvs[:,1], 0.0, 0.999) * (tex_h - 1)
    colors = tex_arr[py.astype(np.int32), px.astype(np.int32)]

    valid = dists < max_dist
    colors[~valid] = [0.5, 0.5, 0.5]

    out_tex[pix_ys, pix_xs] = colors
    out_mask[pix_ys, pix_xs] = True

    try:
        import cv2
        kernel = np.ones((3,3), np.uint8)
        dilated_mask = cv2.dilate(out_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
        inv = (~out_mask).astype(np.uint8) * 255
        bgr = (out_tex[..., ::-1] * 255).astype(np.uint8)
        filled = cv2.inpaint(bgr, inv, 3, cv2.INPAINT_TELEA)
        out_tex = filled[..., ::-1].astype(np.float32) / 255.0
    except Exception:
        pass

    return Image.fromarray((np.clip(out_tex, 0, 1)*255).astype(np.uint8))


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


def _d2_hist(pcd, n_samples=4000, n_pairs=200_000, n_bins=64, max_ratio=4.0):
    pts = np.asarray(pcd.points)
    if len(pts) > n_samples:
        idx = np.random.default_rng(0).choice(len(pts), n_samples, replace=False); pts = pts[idx]
    rng = np.random.default_rng(1)
    i = rng.integers(0, len(pts), n_pairs); j = rng.integers(0, len(pts), n_pairs)
    mask = i != j; i, j = i[mask], j[mask]
    d = np.linalg.norm(pts[i] - pts[j], axis=1); med = np.median(d)
    if med < 1e-12: return np.zeros(n_bins)
    hist, _ = np.histogram(d / med, bins=n_bins, range=(0.0, max_ratio), density=True)
    s = hist.sum()
    return hist / s if s > 0 else hist


def compute_volume(pcd):
    try:
        hull, _ = pcd.compute_convex_hull()
        return float(hull.get_volume())
    except:
        return 0.0


def compute_edge_score(pcd):
    try:
        _, idx = pcd.compute_convex_hull()
        return float(len(idx)) / len(pcd.points)
    except:
        return 0.0


def compute_scan_descriptors(pcd):
    pts = np.asarray(pcd.points)
    ext = np.sort(np.asarray(pcd.get_axis_aligned_bounding_box().get_extent(), float))[::-1]
    aabb_n = (ext / ext.max()).tolist() if ext.max() > 1e-12 else [1.0, 0.0, 0.0]
    centered = pts - pts.mean(axis=0)
    w = np.sort(np.linalg.eigvalsh(np.cov(centered.T)))[::-1]; w = np.clip(w, 1e-18, None)
    try:
        hull, _ = pcd.compute_convex_hull(); hull.compute_vertex_normals()
        v = hull.get_volume(); a = hull.get_surface_area()
        hull_c = float(v / (a ** 1.5)) if a > 1e-12 else 0.0
    except Exception:
        hull_c = 0.0
    return {
        "size_median_radius":      compute_scale_metric(pcd, "median_radius"),
        "aabb_extent_sorted_norm": np.array(aabb_n),
        "pca_eigenvalue_ratios":   np.array([float(w[1]/w[0]), float(w[2]/w[0])]),
        "convex_hull_compactness": hull_c,
        "d2_histogram":            _d2_hist(pcd),
        "volume":                  compute_volume(pcd),
        "edge_score":              compute_edge_score(pcd)
    }


def descriptor_distance(scan_desc, idx_data):
    if idx_data.get("variant") == "vol_edge":
        vol_dist = abs(scan_desc["volume"] - float(idx_data["volume"]))
        edge_dist = abs(scan_desc["edge_score"] - float(idx_data["edge_score"]))
        return float(vol_dist + edge_dist * 100.0)

    w = {"d2": 3.0, "pca": 1.0, "aabb": 1.0, "hull": 0.5, "size": 0.3}
    a = scan_desc["d2_histogram"]; b = np.asarray(idx_data["d2_histogram"])
    d2_dist = 0.5 * np.sum(((a - b) ** 2) / (a + b + 1e-12))
    pca_dist  = np.linalg.norm(scan_desc["pca_eigenvalue_ratios"]   - np.asarray(idx_data["pca_eigenvalue_ratios"]))
    aabb_dist = np.linalg.norm(scan_desc["aabb_extent_sorted_norm"] - np.asarray(idx_data["aabb_extent_sorted_norm"]))
    hull_dist = abs(scan_desc["convex_hull_compactness"] - float(idx_data["convex_hull_compactness"]))
    size_dist = abs(np.log(max(scan_desc["size_median_radius"], 1e-9) / max(float(idx_data["size_median_radius"]), 1e-9)))
    return float(w["d2"]*d2_dist + w["pca"]*pca_dist + w["aabb"]*aabb_dist + w["hull"]*hull_dist + w["size"]*size_dist)


def get_top_candidates(pcd_scan, index_folder, top_k=5):
    scan_desc = compute_scan_descriptors(pcd_scan)
    entries = []
    for f in sorted(glob.glob(os.path.join(index_folder, "*.index.json"))):
        with open(f, "r") as fp: entries.append(json.load(fp))
    if not entries:
        raise SystemExit(f"Keine Indexdateien in {index_folder} – bitte erst index_digital_models.py ausführen.")
    ranked = sorted(entries, key=lambda d: descriptor_distance(scan_desc, d))
    print("\nTop candidates (pre-ranking):")
    out = []
    for i, d in enumerate(ranked[:top_k]):
        name = os.path.splitext(d["source_file"])[0]
        ply_path = os.path.join(index_folder, d["source_file"])
        print(f"  #{i+1}  {d['source_file']}   dist={descriptor_distance(scan_desc, d):.3f}")
        out.append((name, ply_path))
    return out


if __name__ == "__main__":
    VOXEL       = 0.005
    SCAN_OBJ    = "./scanned and digital models/Scanned-House_Key.glb"
    TEXTURE     = "texture.jpg"
    INDEX_FOLDER = "./scanned and digital models/Digital_Models_Cropped"

    BAKE_DEBUG_GRID_TO_DIGITAL = False
    SHOW_UV_GRID_ON_DIGITAL = False

    if BAKE_DEBUG_GRID_TO_DIGITAL or SHOW_UV_GRID_ON_DIGITAL:
        print("Creating debug texture with grid and numbers...")
        debug_img = create_debug_texture(2048, 2048, 20)
        debug_img.save("debug_grid.jpg")
        if BAKE_DEBUG_GRID_TO_DIGITAL:
            TEXTURE = "debug_grid.jpg"
            print("Using debug_grid.jpg as source texture for baking.")

    print("Loading point clouds...")
    pcd_scan    = o3d.io.read_point_cloud("cropped_1.ply")
    pcd_scan    = remove_outliers(pcd_scan)
    print(f"  scan={len(pcd_scan.points):,}")

    print("Pre-ranking digital models from index...")
    candidates = get_top_candidates(pcd_scan, INDEX_FOLDER, top_k=5)

    winner = None  
    for cand_name, cand_ply in candidates:
        print(f"\n========== Candidate: {cand_name} ==========")
        pcd_digital = o3d.io.read_point_cloud(cand_ply)
        print(f"  digital={len(pcd_digital.points):,}")

        pcd_digital.scale(0.001, center=np.zeros(3))
        pcd_digital_c = remove_outliers(pcd_digital)

        print("Auto-scaling digital cloud to match scan size...")
        size_scan    = compute_scale_metric(pcd_scan,    "median_radius")
        size_digital = compute_scale_metric(pcd_digital_c, "median_radius")
        auto_scale   = size_scan / max(size_digital, 1e-12)
        print(f"  size_scan={size_scan:.4f}  size_digital={size_digital:.4f}  factor={auto_scale:.4f}")
        pcd_digital_c.scale(auto_scale, center=np.zeros(3))

        print("Normalizing axes...")
        pcd_scan_n,    T_scan_n    = normalize_orientation(pcd_scan,    up_axis="Y")
        pcd_digital_n, T_digital_n = normalize_orientation(pcd_digital_c, up_axis="-Z")

        print(f"Preprocessing  voxel={VOXEL}...")
        scan_down    = preprocess(pcd_scan_n,    VOXEL)
        digital_down = preprocess(pcd_digital_n, VOXEL)
        print(f"  scan={len(scan_down.points):,}  digital={len(digital_down.points):,}")

        best = find_best_registration(scan_down, digital_down, VOXEL)
        sc = best.score()
        print(f"  -> score {cand_name}: {sc:.2f}")
        if winner is None or sc > winner[0]:
            winner = (sc, best, cand_name, cand_ply, auto_scale,
                      T_scan_n, T_digital_n, scan_down, digital_down, pcd_digital_c)

    if winner is None:
        raise SystemExit("Kein Kandidat lieferte eine Registrierung.")

    (_, best, _dig_base, _dig_ply, auto_scale,
     T_scan_n, T_digital_n, scan_down, digital_down, pcd_digital) = winner
    print(f"\n>>> Auto-selected: {_dig_base}  {best}\n")

    DIGITAL_OBJ = os.path.join(os.path.dirname(_dig_ply), _dig_base + ".obj")
    if not os.path.exists(DIGITAL_OBJ):
        alt = os.path.join(os.path.dirname(INDEX_FOLDER.rstrip("/\\")), _dig_base + ".obj")
        if os.path.exists(alt): DIGITAL_OBJ = alt

    T_align = best.transformation @ T_scan_n

    print("Loading meshes...")
    mesh_scan    = load_mesh(SCAN_OBJ)
    mesh_digital = load_mesh(DIGITAL_OBJ)
    mesh_digital.scale(0.001, center=np.zeros(3))
    mesh_digital.scale(auto_scale, center=np.zeros(3))  

    if mesh_scan.has_textures():
        TEXTURE = Image.fromarray(np.asarray(mesh_scan.textures[0]))

    mesh_scan_aligned    = copy.deepcopy(mesh_scan);    mesh_scan_aligned.transform(T_align)
    mesh_digital_aligned = copy.deepcopy(mesh_digital); mesh_digital_aligned.transform(T_digital_n)

    print("Generating unique UV map for digital model to avoid overlap using xatlas...")
    mesh_digital_aligned = parameterize_mesh_xatlas(mesh_digital_aligned)

    if SHOW_UV_GRID_ON_DIGITAL:
        print("Saving digital_uv_debug.obj with raw xatlas UVs...")
        mesh_debug = copy.deepcopy(mesh_digital_aligned)
        T_digital_n_inv = np.linalg.inv(T_digital_n)
        mesh_debug.transform(T_digital_n_inv)
        mesh_debug.scale(1000.0, center=np.zeros(3))
        mesh_debug.textures = [o3d.geometry.Image(np.asarray(debug_img))]
        mesh_debug.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(mesh_debug.triangles), dtype=np.int32))
        o3d.io.write_triangle_mesh("digital_uv_debug.obj", mesh_debug, write_triangle_uvs=True)
        try:
            with open("digital_uv_debug.mtl", "r") as f:
                mtl_c = f.read()
            with open("digital_uv_debug.mtl", "w") as f:
                f.write(mtl_c.replace(".png", ".jpg"))
        except: pass

    print("Baking texture to digital model UVs...")
    digital_tex_img = bake_digital_texture(mesh_digital_aligned, mesh_scan_aligned, TEXTURE)

    if digital_tex_img is not None:
        base_name = os.path.splitext(os.path.basename(DIGITAL_OBJ))[0]
        out_tex_name = f"{base_name}_texture.jpg"

        digital_tex_img.save(out_tex_name)
        print(f"Saved {out_tex_name}")

        mesh_digital_aligned.textures = [o3d.geometry.Image(np.asarray(digital_tex_img))]
        mesh_digital_aligned.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(mesh_digital_aligned.triangles), dtype=np.int32))

        T_digital_n_inv = np.linalg.inv(T_digital_n)

        T_digital_n_inv = np.linalg.inv(T_digital_n).copy()
        T_digital_n_inv[:3, 3] *= 1000.0
        aabb = mesh_digital_aligned.get_axis_aligned_bounding_box()
        center = aabb.get_center()
        min_y  = aabb.min_bound[1]

        T_place = np.eye(4)
        T_place[0, 3] = -center[0]
        T_place[2, 3] = -center[2]
        T_place[1, 3] = -min_y
        mesh_digital_aligned.transform(T_place)

        out_obj_path = f"{base_name}_textured.obj"
        o3d.io.write_triangle_mesh(out_obj_path, mesh_digital_aligned, write_triangle_uvs=True)
        print(f"Saved {out_obj_path} (restored to original digital model orientation)")

        out_mtl_path = f"{base_name}_textured.mtl"
        auto_tex = f"{base_name}_textured_0.png"
        try:
            with open(out_mtl_path, "r") as f:
                mtl_content = f.read()
            mtl_content = mtl_content.replace(auto_tex, out_tex_name)
            with open(out_mtl_path, "w") as f:
                f.write(mtl_content)
            import os
            if os.path.exists(auto_tex):
                os.remove(auto_tex)
            print(f"Updated MTL to reference {out_tex_name}")
        except Exception as e:
            print(f"Could not update MTL file: {e}")

    C_SCAN    = [1.0, 0.7, 0.0]
    C_DIGITAL = [0.0, 0.6, 0.9]

    pc_s = copy.deepcopy(scan_down); pc_s.paint_uniform_color(C_SCAN)
    pc_d = copy.deepcopy(digital_down); pc_d.paint_uniform_color(C_DIGITAL)

    pc_s_aligned = copy.deepcopy(scan_down)
    pc_s_aligned.paint_uniform_color(C_SCAN)
    pc_s_aligned.transform(best.transformation)
    pc_d_ref = copy.deepcopy(digital_down)
    pc_d_ref.paint_uniform_color(C_DIGITAL)

    if digital_tex_img is not None:
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.albedo_img = o3d.geometry.Image(np.asarray(digital_tex_img))

    show([
        ([pc_s, pc_d],                  "Before"),
        ([pc_s_aligned, pc_d_ref],      f"Aligned  {best.method}"),
        ([mesh_digital_aligned],        "Result Textured Model"),
    ])