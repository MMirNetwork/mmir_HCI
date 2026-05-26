import os, zipfile, tempfile
import numpy as np
import open3d as o3d
from PIL import Image, ImageFilter


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
            m.triangle_uvs = o3d.utility.Vector2dVector(
                np.array(tm.visual.uv, dtype=np.float64)[idx.ravel()])
            mat = tm.visual.material
            if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                m.textures = [o3d.geometry.Image(
                    np.array(mat.baseColorTexture.convert("RGB"), dtype=np.uint8))]
                m.triangle_material_ids = o3d.utility.IntVector(
                    np.zeros(len(tm.faces), dtype=np.int32))
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
                uvs_all.append((uv_vals[np.array(uv_idx, dtype=np.int32)]
                                if uv_idx is not None and len(uv_idx) else uv_vals).reshape(-1, 2))
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
                            m.triangle_material_ids = o3d.utility.IntVector(
                                np.zeros(len(all_f), dtype=np.int32))
        m.compute_vertex_normals()
        return m
    else:
        raise ValueError(f"Unsupported format: {ext}")



def find_ground_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=2000):
    work = pcd
    best = None
    for _ in range(3):
        plane_model, inliers = work.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)
        vert_score = max(abs(normal[1]), abs(normal[2]))
        if vert_score > 0.85 or best is None:
            best = (plane_model, inliers, work)
            if vert_score > 0.85:
                break
        work = work.select_by_index(inliers, invert=True)
    return best



def bake_ground_texture(ground_pcd, plane_model, mesh_scan, out_path,
                        resolution=2048, max_dist=0.05):
    """
    1) Bounding-Rechteck der Boden-Inlier in der Ebene bestimmen.
    2) Pro Pixel die 3D-Weltposition auf der Ebene berechnen.
    3) Closest-Point gegen das Scan-Mesh => baryzentrische Koords.
    4) Farbe aus Mesh-Textur via UV sampeln.
    5) Löcher mit cv2.inpaint schließen.
    """
    if not mesh_scan.has_textures() or not mesh_scan.has_triangle_uvs():
        print("  ⚠ Scan-Mesh hat keine Textur/UVs – fallback auf Vertex-Color-Projektion.")
        return _fallback_vertex_color_texture(ground_pcd, plane_model, out_path, resolution)

    n = np.array(plane_model[:3], dtype=np.float64)
    n /= np.linalg.norm(n)
    d = float(plane_model[3])
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper); u /= np.linalg.norm(u)
    v = np.cross(n, u)

    pts = np.asarray(ground_pcd.points)
    uv  = np.column_stack([pts @ u, pts @ v])
    uv_min = uv.min(axis=0)
    uv_max = uv.max(axis=0)
    size_world = uv_max - uv_min      

    scale = (resolution - 1) / float(size_world.max())
    W = int(np.ceil(size_world[0] * scale)) + 1
    H = int(np.ceil(size_world[1] * scale)) + 1
    print(f"  → Boden-Größe ~ {size_world[0]:.3f} x {size_world[1]:.3f} m  ->  {W}x{H} px")


    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    uu = uv_min[0] + xs.ravel() / scale
    vv = uv_min[1] + ys.ravel() / scale

    p0 = -d * n
    world_pts = (p0[None, :]
                 + uu[:, None] * u[None, :]
                 + vv[:, None] * v[None, :]).astype(np.float32)


    print("  → Raycasting (closest-points) gegen Scan-Mesh …")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_scan))
    ans = scene.compute_closest_points(o3d.core.Tensor(world_pts, dtype=o3d.core.Dtype.Float32))
    s_tri  = ans['primitive_ids'].numpy()
    s_bary = ans['primitive_uvs'].numpy()
    s_pts  = ans['points'].numpy()
    dists  = np.linalg.norm(world_pts - s_pts, axis=1)


    scan_uvs = np.asarray(mesh_scan.triangle_uvs)     
    su, sv = s_bary[:, 0], s_bary[:, 1]
    sw     = 1.0 - su - sv
    uv0 = scan_uvs[s_tri * 3 + 0]
    uv1 = scan_uvs[s_tri * 3 + 1]
    uv2 = scan_uvs[s_tri * 3 + 2]
    final_uvs = sw[:, None] * uv0 + su[:, None] * uv1 + sv[:, None] * uv2

    tex_img = np.asarray(mesh_scan.textures[0])
    th, tw  = tex_img.shape[:2]
    px = np.clip(final_uvs[:, 0],       0.0, 0.999) * (tw - 1)
    py = np.clip(1.0 - final_uvs[:, 1], 0.0, 0.999) * (th - 1)
    colors = tex_img[py.astype(np.int32), px.astype(np.int32)].astype(np.float32) / 255.0


    valid = dists < max_dist
    print(f"  → {valid.sum():,} / {len(valid):,} Pixel mit gültigem Mesh-Treffer "
          f"({100*valid.mean():.1f}%)")

    img  = colors.reshape(H, W, 3)
    mask = valid.reshape(H, W)


    try:
        import cv2
        bgr = (img[..., ::-1] * 255).astype(np.uint8)
        inv = (~mask).astype(np.uint8) * 255
        filled = cv2.inpaint(bgr, inv, 3, cv2.INPAINT_TELEA)
        img = filled[..., ::-1].astype(np.float32) / 255.0
    except Exception as e:
        print(f"  ⚠ cv2.inpaint nicht verfügbar ({e}) – Median-Fallback.")
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        pil = pil.filter(ImageFilter.MedianFilter(5))
        img = np.array(pil, dtype=np.float32) / 255.0

    out_img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out_img).save(out_path, "JPEG", quality=92)
    print(f"  → Boden-Textur gespeichert: {out_path} ({W}x{H})")


    obj_path = os.path.splitext(out_path)[0] + ".obj"
    _write_ground_plane_obj(obj_path, p0, u, v, uv_min, uv_max,
                            os.path.basename(out_path))
    print(f"  → Boden-Plane gespeichert: {obj_path}")

    return out_path


def _write_ground_plane_obj(obj_path, p0, u, v, uv_min, uv_max, texture_filename):
    """Schreibt ein 2-Dreieck-Rechteck mit UVs 0..1 + passende .mtl."""
    corners_uv = np.array([
        [uv_min[0], uv_min[1]],
        [uv_max[0], uv_min[1]],
        [uv_max[0], uv_max[1]],
        [uv_min[0], uv_max[1]],
    ])
    verts = p0[None, :] + corners_uv[:, 0:1] * u[None, :] + corners_uv[:, 1:2] * v[None, :]

    mtl_path  = os.path.splitext(obj_path)[0] + ".mtl"
    mtl_name  = os.path.basename(mtl_path)
    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n")
        f.write("usemtl ground\n")
        f.write("f 1/1 2/2 3/3\n")
        f.write("f 1/1 3/3 4/4\n")
    with open(mtl_path, "w") as f:
        f.write("newmtl ground\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\n")
        f.write(f"map_Kd {texture_filename}\n")


def _fallback_vertex_color_texture(ground_pcd, plane_model, out_path, resolution):
    """Alter Pfad falls Mesh keine Textur hat."""
    pts    = np.asarray(ground_pcd.points)
    colors = np.asarray(ground_pcd.colors)
    if len(colors) == 0:
        colors = np.full_like(pts, 0.5)
    n = np.array(plane_model[:3]); n /= np.linalg.norm(n)
    helper = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(n, helper); u /= np.linalg.norm(u); v = np.cross(n, u)
    uv = np.column_stack([pts @ u, pts @ v]); uv -= uv.min(axis=0)
    scale = (resolution - 1) / float(uv.max()); uv = (uv * scale).astype(np.int32)
    W = uv[:, 0].max() + 1; H = uv[:, 1].max() + 1
    img = np.zeros((H, W, 3), dtype=np.float32); cnt = np.zeros((H, W), dtype=np.int32)
    for (x, y), c in zip(uv, colors):
        img[y, x] += c; cnt[y, x] += 1
    mask = cnt > 0; img[mask] /= cnt[mask, None]
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3))
    pil.save(out_path, "JPEG", quality=92)
    return out_path



def keep_main_object(pcd, eps=None, min_points=50):
    pts = np.asarray(pcd.points)
    if eps is None:
        diag = np.linalg.norm(pts.max(0) - pts.min(0))
        eps  = max(diag * 0.005, 1e-4)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.max() < 0:
        print("  ⚠ Kein Cluster gefunden – Punktwolke bleibt unverändert.")
        return pcd
    center = pts.mean(axis=0)
    sizes  = np.bincount(labels[labels >= 0])
    topk   = np.argsort(sizes)[::-1][:3]
    best_lbl, best_d = None, np.inf
    for lbl in topk:
        c = pts[labels == lbl].mean(axis=0)
        d = np.linalg.norm(c[:2] - center[:2])
        if d < best_d:
            best_d, best_lbl = d, lbl
    print(f"  → Hauptcluster: Label {best_lbl}, {sizes[best_lbl]} Punkte "
          f"(von {len(sizes)} Clustern insgesamt)")
    return pcd.select_by_index(np.where(labels == best_lbl)[0])


def main():
    dateiname = input("Bitte Dateipfad eingeben: ").strip().strip('"')
    base, _   = os.path.splitext(dateiname)

    print("• Lade Mesh …")
    mesh_scan = load_mesh(dateiname)
    print(f"  Mesh hat Textur: {mesh_scan.has_textures()}  "
          f"UVs: {mesh_scan.has_triangle_uvs()}")

    print("• Sample 1 Mio. Punkte …")
    pcd = mesh_scan.sample_points_uniformly(number_of_points=1_000_000)

    print("• Vorschau Original (schließen zum Fortfahren) …")
    o3d.visualization.draw_geometries([pcd], window_name="Original-Scan (vor Crop)")

    diag = np.linalg.norm(np.asarray(pcd.points).max(0) - np.asarray(pcd.points).min(0))
    dist_th = diag * 0.005
    print(f"• RANSAC Boden-Suche (dist_threshold={dist_th:.4f}) …")
    plane_model, inliers, work = find_ground_plane(pcd, distance_threshold=dist_th)
    a, b, c, d = plane_model
    print(f"  → Ebene: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0   "
          f"({len(inliers)} Inlier)")

    ground_pcd  = work.select_by_index(inliers)
    without_grd = work.select_by_index(inliers, invert=True)

    tex_path = base + "_ground_texture.jpg"
    print("• Backe Boden-Textur via Raycasting …")
    bake_ground_texture(ground_pcd, plane_model, mesh_scan, tex_path,
                        resolution=2048, max_dist=max(dist_th * 4, 0.02))

    print("• Clustering: Hauptobjekt suchen …")
    main_obj = keep_main_object(without_grd)

    out_pcd = base + "_object.ply"
    o3d.io.write_point_cloud(out_pcd, main_obj)
    print(f"  → Punktwolke gespeichert: {out_pcd}")

    print("• Vorschau (schließen zum Fortfahren) …")
    ground_pcd.paint_uniform_color([0.7, 0.5, 0.3])
    o3d.visualization.draw_geometries(
        [main_obj, ground_pcd],
        window_name="Auto-Crop Ergebnis (Boden=braun, Objekt=Original)"
    )

    print("• Manuelles Nachschneiden (optional) …")
    o3d.visualization.draw_geometries_with_editing(
        [main_obj], window_name="Cropping Tool"
    )


if __name__ == "__main__":
    main()
