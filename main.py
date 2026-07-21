import os, zipfile, tempfile, glob, json
import open3d as o3d
import copy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# xatlas wird fuer Weg A NICHT mehr benoetigt (Original-UVs bleiben erhalten).


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


def _safe_name(name):
    keep = "-_."
    return "".join(c if (c.isalnum() or c in keep) else "_" for c in name)


# ============================================================================
#  GEAENDERT: OBJ-Loader, der die o/g-Gruppen UND die Original-UVs erhaelt
# ============================================================================
def load_obj_grouped(path):
    """
    Liest ein OBJ und gibt eine Liste von Teilen zurueck. Jedes Teil:
       {name, vertices (np Nx3), triangles (np Mx3, lokale Indizes),
        triangle_uvs (np (M*3)x2) oder None}

    Die Original-UVs werden erhalten: Der UV-Index aus 'f v/vt/vn' wird
    parallel zum Vertex-Index gelesen, sodass jede Ecke jeder Flaeche ihre
    korrekte UV behaelt -- unabhaengig davon, ob Vertex-Index == UV-Index ist.
    """
    all_v  = []   # globale Vertex-Positionen
    all_vt = []   # globale UV-Koordinaten
    groups = []   # Liste (name, faces) mit faces = Liste von (vtri, uvtri)
    cur_name = "model"
    cur_faces = []

    def flush():
        nonlocal cur_faces
        if cur_faces:
            groups.append((cur_name, cur_faces))
            cur_faces = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            t = tok[0]
            if t == "v":
                all_v.append([float(tok[1]), float(tok[2]), float(tok[3])])
            elif t == "vt":
                # nur U,V verwenden (eine evtl. dritte Komponente W ignorieren)
                all_vt.append([float(tok[1]), float(tok[2])])
            elif t in ("o", "g"):
                flush()
                cur_name = tok[1] if len(tok) > 1 else "unnamed"
            elif t == "f":
                v_idx, uv_idx = [], []
                for p in tok[1:]:
                    parts = p.split("/")
                    vi = int(parts[0])
                    vi = len(all_v) + vi if vi < 0 else vi - 1
                    v_idx.append(vi)
                    # UV-Index (zweites Feld), falls vorhanden und nicht leer
                    if len(parts) >= 2 and parts[1] != "":
                        ti = int(parts[1])
                        ti = len(all_vt) + ti if ti < 0 else ti - 1
                        uv_idx.append(ti)
                    else:
                        uv_idx.append(None)
                # Polygon -> Dreiecke (Fan-Triangulation), UV synchron mitfuehren
                for k in range(1, len(v_idx) - 1):
                    vtri  = (v_idx[0],  v_idx[k],  v_idx[k + 1])
                    uvtri = (uv_idx[0], uv_idx[k], uv_idx[k + 1])
                    cur_faces.append((vtri, uvtri))

    flush()

    all_v  = np.asarray(all_v,  dtype=np.float64)
    all_vt = np.asarray(all_vt, dtype=np.float64) if all_vt else np.zeros((0, 2))

    if not groups:
        groups = [("model", [])]

    parts = []
    for name, faces in groups:
        if not faces:
            continue

        vtris  = np.asarray([fc[0] for fc in faces], dtype=np.int64)  # (M,3)
        uvtris = [fc[1] for fc in faces]                              # Liste (M) von (3,)

        # --- Vertices lokal remappen ---
        used = np.unique(vtris.ravel())
        remap = {g: l for l, g in enumerate(used)}
        loc_v = all_v[used]
        loc_f = np.vectorize(remap.get)(vtris).astype(np.int32)       # (M,3)

        # --- UVs synchron in Open3D-triangle_uvs-Format (M*3, 2) ---
        part_has_uv = (len(all_vt) > 0) and all(
            all(u is not None for u in tri) for tri in uvtris
        )
        if part_has_uv:
            uv_index_flat = np.asarray(
                [u for tri in uvtris for u in tri], dtype=np.int64
            )                                                        # (M*3,)
            tri_uvs = all_vt[uv_index_flat]                          # (M*3, 2)
        else:
            tri_uvs = None
            if len(all_vt) > 0:
                print(f"  WARN: Teil '{name}' hat unvollstaendige UVs "
                      f"-> wird ohne Original-UV behandelt.")

        parts.append({
            "name":         _safe_name(name),
            "vertices":     loc_v,
            "triangles":    loc_f,
            "triangle_uvs": tri_uvs,
        })

    n_uv_parts = sum(1 for p in parts if p["triangle_uvs"] is not None)
    print(f"OBJ geladen: {len(parts)} Teile (mit Faces), {len(all_v):,} Vertices, "
          f"{len(all_vt):,} UVs  |  Teile mit Original-UV: {n_uv_parts}/{len(parts)}")
    return parts



def part_to_mesh(part):
    """Baut ein Open3D-Mesh aus einem Teil und HAENGT die Original-UVs an,
    falls vorhanden."""
    m = o3d.geometry.TriangleMesh()
    m.vertices  = o3d.utility.Vector3dVector(part["vertices"])
    m.triangles = o3d.utility.Vector3iVector(part["triangles"])
    if part.get("triangle_uvs") is not None:
        m.triangle_uvs = o3d.utility.Vector2dVector(part["triangle_uvs"])
    m.compute_vertex_normals()
    return m


def adaptive_tex_size(n_tris, min_size=256, max_size=2048, ref_tris=1200):
    """Texturgroesse (Zweierpotenz) abhaengig von der Face-Anzahl."""
    if n_tris <= 0:
        return min_size
    frac = (n_tris / ref_tris) ** 0.5
    target = max_size * min(1.0, max(0.0, frac))
    size = min_size
    while size < target and size < max_size:
        size *= 2
    return int(max(min_size, min(size, max_size)))


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
            x0 = j * cell_w; y0 = i * cell_h; x1 = x0 + cell_w; y1 = y0 + cell_h
            bg_color = (int(255*(j/grid_size)), int(255*(i/grid_size)), 150)
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


# ============================================================================
#  bake nimmt vorgebaute scan_scene/scan_uvs entgegen (1x bauen)
#  (unveraendert -- arbeitet rein auf den triangle_uvs des digitalen Meshes,
#   egal ob diese von xatlas oder aus dem Original kommen)
# ============================================================================
def bake_digital_texture(mesh_digital, scan_scene, scan_uvs,
                         tex_arr, tex_w, tex_h, out_size=2048, max_dist=0.05):
    digital_uvs = np.asarray(mesh_digital.triangle_uvs)
    if len(digital_uvs) == 0:
        return None
    digital_tris  = np.asarray(mesh_digital.triangles)
    digital_verts = np.asarray(mesh_digital.vertices)
    
    # Normals berechnen fuer echtes Raycasting!
    mesh_digital.compute_triangle_normals()
    digital_tri_normals = np.asarray(mesh_digital.triangle_normals)
    
    T = len(digital_tris)
    out_tex  = np.zeros((out_size, out_size, 3), dtype=np.float32)
    out_mask = np.zeros((out_size, out_size),    dtype=bool)
    pix_xs, pix_ys, pos_list, norm_list = [], [], [], []

    for ti in range(T):
        uv0 = digital_uvs[ti*3 + 0]; uv1 = digital_uvs[ti*3 + 1]; uv2 = digital_uvs[ti*3 + 2]
        p0 = np.array([uv0[0]*(out_size-1), (1.0-uv0[1])*(out_size-1)])
        p1 = np.array([uv1[0]*(out_size-1), (1.0-uv1[1])*(out_size-1)])
        p2 = np.array([uv2[0]*(out_size-1), (1.0-uv2[1])*(out_size-1)])
        min_x = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
        max_x = min(int(np.ceil (max(p0[0], p1[0], p2[0]))), out_size-1)
        min_y = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
        max_y = min(int(np.ceil (max(p0[1], p1[1], p2[1]))), out_size-1)
        if max_x < min_x or max_y < min_y:
            continue
        xs, ys = np.meshgrid(np.arange(min_x, max_x+1), np.arange(min_y, max_y+1))
        xs = xs.ravel().astype(np.float32); ys = ys.ravel().astype(np.float32)
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
        
        v0 = digital_verts[digital_tris[ti, 0]]; v1 = digital_verts[digital_tris[ti, 1]]; v2 = digital_verts[digital_tris[ti, 2]]
        pos = (l0i[:,None]*v0) + (l1i[:,None]*v1) + (l2i[:,None]*v2)
        
        # Face Normal
        n = digital_tri_normals[ti]
        norm = np.tile(n, (len(l0i), 1))
        
        pix_xs.append(xi); pix_ys.append(yi)
        pos_list.append(pos.astype(np.float32))
        norm_list.append(norm.astype(np.float32))

    if not pos_list:
        return None
    pix_xs = np.concatenate(pix_xs); pix_ys = np.concatenate(pix_ys)
    positions = np.concatenate(pos_list, axis=0)
    normals = np.concatenate(norm_list, axis=0)
    
    # --- DIAGNOSE / SCHUTZ ---
    bad = ~np.isfinite(positions).all(axis=1) | ~np.isfinite(normals).all(axis=1)
    if bad.any():
        print(f"      WARN: {bad.sum():,} ungueltige Positionen (NaN/inf) -> verworfen")
        positions = positions[~bad]
        normals = normals[~bad]
        pix_xs = pix_xs[~bad]; pix_ys = pix_ys[~bad]
    if len(positions) == 0:
        print("      (keine gueltigen Positionen -> Teil uebersprungen)")
        return None
    print(f"      {len(positions):,} Pixel -> Normal-based Raycast")
    # --- ENDE ---

    # Wir schiessen echte Strahlen entlang der Normalen!
    # Startpunkt: Etwas ueber der Flaeche (max_dist in Richtung der Normalen)
    # Richtung: Entgegen der Normalen (-1.0 * normal)
    origins = positions + normals * max_dist
    directions = -normals
    rays = np.concatenate([origins, directions], axis=1).astype(np.float32)

    qp = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scan_scene.cast_rays(qp)
    
    t_hit = ans['t_hit'].numpy()
    s_tri  = ans['primitive_ids'].numpy()
    s_bary = ans['primitive_uvs'].numpy()
    
    # Der Ray startet bei +max_dist und fliegt mit Speed 1. 
    # Wenn er bei t_hit aufschlaegt, ist der Abstand zum originalen Punkt: abs(t_hit - max_dist)
    dists = np.abs(t_hit - max_dist)
    valid = np.isfinite(t_hit) & (dists < max_dist)
    
    # Schutz vor IndexError bei Strahlen, die nichts getroffen haben (ID=4294967295)
    s_tri = np.clip(s_tri, 0, (len(scan_uvs)//3) - 1)

    su = s_bary[:, 0]; sv = s_bary[:, 1]; sw = 1.0 - su - sv
    suv0 = scan_uvs[s_tri*3 + 0]; suv1 = scan_uvs[s_tri*3 + 1]; suv2 = scan_uvs[s_tri*3 + 2]
    final_uvs = sw[:,None]*suv0 + su[:,None]*suv1 + sv[:,None]*suv2
    
    px = np.clip(final_uvs[:,0], 0.0, 0.999) * (tex_w - 1)
    py = np.clip(1.0 - final_uvs[:,1], 0.0, 0.999) * (tex_h - 1)
    colors = tex_arr[py.astype(np.int32), px.astype(np.int32)]
    
    colors[~valid] = [0.5, 0.5, 0.5]
    out_tex[pix_ys, pix_xs] = colors
    out_mask[pix_ys, pix_xs] = True

    try:
        import cv2
        inv = (~out_mask).astype(np.uint8) * 255
        bgr = (out_tex[..., ::-1] * 255).astype(np.uint8)
        filled = cv2.inpaint(bgr, inv, 3, cv2.INPAINT_TELEA)
        out_tex = filled[..., ::-1].astype(np.float32) / 255.0
    except Exception:
        pass

    return Image.fromarray((np.clip(out_tex, 0, 1)*255).astype(np.uint8))


# ============================================================================
#  Multi-Part-OBJ schreiben.
#  GEAENDERT: schreibt die (Original-)UVs des Meshes; robust gegen Teile
#  ohne UV, sodass der vt-Offset nie auseinanderlaeuft.
# ============================================================================
def write_multipart_obj(out_obj_path, baked_parts, tex_subdir="textures", make_zip=True):
    base = os.path.splitext(out_obj_path)[0]
    out_mtl_path = base + ".mtl"
    mtl_basename = os.path.basename(out_mtl_path)
    out_dir = os.path.dirname(out_obj_path) or "."
    tex_dir = os.path.join(out_dir, tex_subdir)
    os.makedirs(tex_dir, exist_ok=True)

    obj_lines = [f"mtllib {mtl_basename}"]
    mtl_lines = []
    v_off = 0
    vt_off = 0
    written = []

    for part in baked_parts:
        name = part["name"]; mesh = part["mesh"]; tex = part["tex"]
        verts = np.asarray(mesh.vertices)
        tris  = np.asarray(mesh.triangles)
        tuvs  = np.asarray(mesh.triangle_uvs)
        mat_name = f"mat_{name}"

        # --- ABSICHERUNG: UV-Konsistenz pruefen (sonst Offset-Drift!) ---
        write_uv = (len(tuvs) == len(tris) * 3) and (tex is not None)
        if (tex is not None) and (len(tuvs) != len(tris) * 3):
            print(f"  WARN: {name} hat {len(tuvs)} UVs statt {len(tris)*3} "
                  f"-> Teil wird OHNE Textur geschrieben.")

        if write_uv:
            tex_name = f"{name}_texture.jpg"
            tex.convert("RGB").save(os.path.join(tex_dir, tex_name), quality=95)
            written.append(os.path.join(tex_dir, tex_name))
            mtl_lines += [f"newmtl {mat_name}", "Ka 1.0 1.0 1.0", "Kd 1.0 1.0 1.0",
                          "Ks 0.0 0.0 0.0", "d 1.0", "illum 2",
                          f"map_Kd {tex_subdir}/{tex_name}", ""]
        else:
            mtl_lines += [f"newmtl {mat_name}", "Kd 0.8 0.8 0.8", "illum 1", ""]

        obj_lines.append(f"o {name}")
        obj_lines.append(f"g {name}")
        for v in verts:
            obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        if write_uv:
            for uv in tuvs:
                obj_lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
        obj_lines.append(f"usemtl {mat_name}")

        if write_uv:
            for i, tri in enumerate(tris):
                a, b, c   = tri[0]+1+v_off, tri[1]+1+v_off, tri[2]+1+v_off
                ta, tb, tc = i*3+1+vt_off, i*3+2+vt_off, i*3+3+vt_off
                obj_lines.append(f"f {a}/{ta} {b}/{tb} {c}/{tc}")
            vt_off += len(tuvs)
        else:
            for tri in tris:
                a, b, c = tri[0]+1+v_off, tri[1]+1+v_off, tri[2]+1+v_off
                obj_lines.append(f"f {a} {b} {c}")
            # vt_off bleibt unveraendert -> kein Drift

        v_off += len(verts)

    with open(out_obj_path, "w") as f:
        f.write("\n".join(obj_lines) + "\n")
    with open(out_mtl_path, "w") as f:
        f.write("\n".join(mtl_lines) + "\n")
    print(f"\nGeschrieben: {out_obj_path}")
    print(f"Geschrieben: {out_mtl_path}")
    print(f"{len(written)} Texturen in: {tex_dir}")

    if make_zip and written:
        zip_path = base + "_textures.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for tp in written:
                zf.write(tp, os.path.join(tex_subdir, os.path.basename(tp)))
        print(f"ZIP mit {len(written)} Texturen: {zip_path}")


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
        raise SystemExit(f"Keine Indexdateien in {index_folder} - bitte erst index_digital_models.py ausfuehren.")
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
    VOXEL_RATIO  = 0.04
    SCAN_OBJ     = "./scanned and digital models/Scanned-Coffee_Maker.glb"
    TEXTURE      = "texture.jpg"
    INDEX_FOLDER = "./scanned and digital models/Digital_Models_Cropped"

    # ---- Per-Teil-Textur-Einstellungen ----
    TEX_MIN    = 256          # kleinste Teil-Textur (Zweierpotenz)
    TEX_MAX    = 512         # groesste Teil-Textur (Zweierpotenz)
    TEX_SUBDIR = "textures"   # Unterordner fuer die JPGs
    MAKE_ZIP   = True         # alle JPGs zusaetzlich als ZIP

    # Weg A: Original-UVs verwenden (xatlas aus). Auf True setzen, falls ein
    # Modell mal KEINE brauchbaren UVs hat -> dann faellt es auf xatlas zurueck.
    USE_ORIGINAL_UVS = True

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
    pcd_scan = o3d.io.read_point_cloud("cropped_1.ply")
    pcd_scan = remove_outliers(pcd_scan)
    print(f"  scan={len(pcd_scan.points):,}")
    
    # Dynamische Voxel-Groesse berechnen (verhindert Out-Of-Memory bei grossen Skalierungen)
    size_scan_global = compute_scale_metric(pcd_scan, "median_radius")
    VOXEL = max(size_scan_global * VOXEL_RATIO, 1e-4)
    MAX_DIST = max(size_scan_global * 0.4, 1e-4)
    print(f"  scan_size={size_scan_global:.4f} -> dynamic voxel={VOXEL:.4f}, max_dist={MAX_DIST:.4f}")

    print("Pre-ranking digital models from index...")
    candidates = get_top_candidates(pcd_scan, INDEX_FOLDER, top_k=3)

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

    # ---- GLOBALE Transformation (einmal fuer das ganze Modell) ----
    T_align = best.transformation @ T_scan_n

    print("Loading scan mesh + texture...")
    mesh_scan = load_mesh(SCAN_OBJ)
    if mesh_scan.has_textures():
        TEXTURE_IMG = Image.fromarray(np.asarray(mesh_scan.textures[0])).convert("RGB")
    else:
        TEXTURE_IMG = (TEXTURE if isinstance(TEXTURE, Image.Image) else Image.open(TEXTURE)).convert("RGB")
    tex_w, tex_h = TEXTURE_IMG.size
    tex_arr = np.array(TEXTURE_IMG).astype(np.float32) / 255.0

    # Scan ins aligned Frame bringen (EINMAL) und Raycasting-Scene EINMAL bauen
    mesh_scan_aligned = copy.deepcopy(mesh_scan); mesh_scan_aligned.transform(T_align)
    scan_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_scan_aligned)
    scan_scene = o3d.t.geometry.RaycastingScene(); scan_scene.add_triangles(scan_t)
    scan_uvs = np.asarray(mesh_scan_aligned.triangle_uvs)

    # ---- Digitales OBJ MIT o/g-Gruppen UND Original-UVs laden ----
    print("Loading digital OBJ with parts (Original-UVs erhalten)...")
    parts = load_obj_grouped(DIGITAL_OBJ)

    # gleiche globale Skalierung wie beim Punktwolken-Matching
    global_scale = 0.001 * auto_scale

    # Alle Teile EINMAL in aligned Space bringen
    # WICHTIG: triangle_uvs sind invariant unter Skalierung/Transform der
    # Geometrie -> sie bleiben bei .scale()/.transform() korrekt erhalten.
    aligned_meshes = []
    for part in parts:
        m = part_to_mesh(part)                 # haengt Original-UVs an (falls vorhanden)
        m.scale(global_scale, center=np.zeros(3))
        m.transform(T_digital_n)
        aligned_meshes.append(m)

    # Platzierungs-Transform EINMAL global aus Gesamt-AABB
    all_pts = np.concatenate([np.asarray(m.vertices) for m in aligned_meshes])
    center  = all_pts.mean(axis=0)
    g_min   = all_pts.min(axis=0)
    T_place = np.eye(4)
    T_place[0, 3] = -center[0]
    T_place[2, 3] = -center[2]
    T_place[1, 3] = -g_min[1]

    # ---- Pro Teil: (Original-UV ODER xatlas-Fallback) + Bake ----
    baked_parts = []
    final_view_meshes = []
    for part, m_aligned in zip(parts, aligned_meshes):
        name = part["name"]
        n_tris = len(part["triangles"])
        tex_size = adaptive_tex_size(n_tris, min_size=TEX_MIN, max_size=TEX_MAX)

        has_orig_uv = (np.asarray(m_aligned.triangle_uvs).shape[0] == n_tris * 3)

        if USE_ORIGINAL_UVS and has_orig_uv:
            print(f"\n--- Teil: {name}  (tris={n_tris:,}, tex={tex_size}x{tex_size}, "
                  f"Original-UV) ---")
            m_uv = m_aligned                    # UVs bleiben original, KEIN xatlas
        else:
            # Fallback: nur wenn ein Teil keine brauchbaren Original-UVs hat
            print(f"\n--- Teil: {name}  (tris={n_tris:,}, tex={tex_size}x{tex_size}, "
                  f"xatlas-Fallback) ---")
            import xatlas
            verts = np.asarray(m_aligned.vertices)
            tris  = np.asarray(m_aligned.triangles)
            atlas = xatlas.Atlas(); atlas.add_mesh(verts, tris); atlas.generate()
            vmapping, new_tris, new_uvs = atlas[0]
            m_uv = o3d.geometry.TriangleMesh()
            m_uv.vertices  = o3d.utility.Vector3dVector(verts[vmapping])
            m_uv.triangles = o3d.utility.Vector3iVector(new_tris)
            m_uv.triangle_uvs = o3d.utility.Vector2dVector(new_uvs[new_tris].reshape(-1, 2))
            m_uv.compute_vertex_normals()

        tex = bake_digital_texture(m_uv, scan_scene, scan_uvs,
                                   tex_arr, tex_w, tex_h, out_size=tex_size, max_dist=MAX_DIST)
        if tex is None:
            print(f"   WARN: kein Bake fuer {name} (keine gemeinsame Flaeche?)")

        m_out = copy.deepcopy(m_uv)
        m_out.transform(T_place)
        baked_parts.append({"name": name, "mesh": m_out, "tex": tex})

        # fuer die Vorschau (eine Textur pro Teil)
        if tex is not None:
            mv = copy.deepcopy(m_out)
            mv.textures = [o3d.geometry.Image(np.asarray(tex))]
            mv.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(mv.triangles), dtype=np.int32))
            final_view_meshes.append(mv)

    # ---- Ein OBJ + eine MTL + JPGs pro Teil + ZIP ----
    base_name = os.path.splitext(os.path.basename(DIGITAL_OBJ))[0]
    out_obj_path = f"{base_name}_textured.obj"
    write_multipart_obj(out_obj_path, baked_parts,
                        tex_subdir=TEX_SUBDIR, make_zip=MAKE_ZIP)

    # ---- Visualisierung ----
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
        (final_view_meshes,             "Result Textured Model (per part)"),
    ])
