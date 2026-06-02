import os
import zipfile
import tempfile
import numpy as np
import open3d as o3d
from PIL import Image


def load_mesh(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".obj":
        m = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
        m.compute_vertex_normals()
        m.compute_triangle_normals()
        return m

    elif ext in (".glb", ".gltf"):
        import trimesh
        raw = trimesh.load(path)
        tm = trimesh.util.concatenate(list(raw.geometry.values())) if hasattr(raw, "geometry") else raw

        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(np.array(tm.vertices, dtype=np.float64))
        m.triangles = o3d.utility.Vector3iVector(np.array(tm.faces, dtype=np.int32))

        if hasattr(tm.visual, "uv") and tm.visual.uv is not None:
            idx = np.array(tm.faces, dtype=np.int32)
            m.triangle_uvs = o3d.utility.Vector2dVector(
                np.array(tm.visual.uv, dtype=np.float64)[idx.ravel()]
            )

            mat = tm.visual.material
            if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                tex = np.array(mat.baseColorTexture.convert("RGB"), dtype=np.uint8)
                m.textures = [o3d.geometry.Image(tex)]
                m.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(tm.faces), dtype=np.int32))

        m.compute_vertex_normals()
        m.compute_triangle_normals()
        return m

    elif ext == ".usdz":
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(path)
        verts_all, faces_all, uvs_all = [], [], []
        off = 0

        for prim in stage.Traverse():
            if prim.GetTypeName() != "Mesh":
                continue

            um = UsdGeom.Mesh(prim)
            pts = np.array(um.GetPointsAttr().Get(), dtype=np.float64)
            fidx = np.array(um.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

            verts_all.append(pts)
            faces_all.append(fidx.reshape(-1, 3) + off)
            off += len(pts)

            st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            if st and st.IsDefined():
                uv_vals = np.array(st.Get(), dtype=np.float64)
                uv_idx = st.GetIndices()
                if uv_idx is not None and len(uv_idx):
                    uvs = uv_vals[np.array(uv_idx, dtype=np.int32)]
                else:
                    uvs = uv_vals
                uvs_all.append(uvs.reshape(-1, 2))

        if not verts_all or not faces_all:
            raise RuntimeError("Keine Mesh-Daten in USDZ gefunden.")

        all_v = np.concatenate(verts_all)
        all_f = np.concatenate(faces_all)

        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(all_v)
        m.triangles = o3d.utility.Vector3iVector(all_f)

        if uvs_all:
            all_uv = np.concatenate(uvs_all)
            if len(all_uv) == len(all_f) * 3:
                m.triangle_uvs = o3d.utility.Vector2dVector(all_uv)

                with zipfile.ZipFile(path) as z:
                    cands = [n for n in z.namelist() if n.lower().endswith(("_tex0.png", "_tex0.jpg"))]
                    if not cands:
                        cands = [n for n in z.namelist() if n.lower().endswith(".png") or n.lower().endswith(".jpg")]

                    if cands:
                        with tempfile.TemporaryDirectory() as tmp:
                            z.extract(cands[0], tmp)
                            img = Image.open(os.path.join(tmp, cands[0])).convert("RGB")
                            m.textures = [o3d.geometry.Image(np.array(img, dtype=np.uint8))]
                            m.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(all_f), dtype=np.int32))

        m.compute_vertex_normals()
        m.compute_triangle_normals()
        return m

    else:
        raise ValueError(f"Unsupported format: {ext}")


def find_ground_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=600):
    work = pcd
    best = None

    for _ in range(3):
        plane_model, inliers = work.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        normal = np.array(plane_model[:3], dtype=np.float64)
        normal /= (np.linalg.norm(normal) + 1e-12)

        vert_score = max(abs(normal[1]), abs(normal[2]))
        if vert_score > 0.85 or best is None:
            best = (plane_model, inliers, work)
            if vert_score > 0.85:
                break

        work = work.select_by_index(inliers, invert=True)

    return best


def _write_ground_plane_obj(obj_path, p0, u, v, uv_min, uv_max, texture_filename):
    corners_uv = np.array([
        [uv_min[0], uv_min[1]],
        [uv_max[0], uv_min[1]],
        [uv_max[0], uv_max[1]],
        [uv_min[0], uv_max[1]],
    ], dtype=np.float64)

    verts = p0[None, :] + corners_uv[:, 0:1] * u[None, :] + corners_uv[:, 1:2] * v[None, :]

    mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
    mtl_name = os.path.basename(mtl_path)

    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n")
        f.write("usemtl ground\n")
        f.write("f 1/1 2/2 3/3\n")
        f.write("f 1/1 3/3 4/4\n")

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("newmtl ground\n")
        f.write("Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\n")
        f.write(f"map_Kd {texture_filename}\n")


def keep_main_object(pcd, eps=None, min_points=40):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd

    if eps is None:
        diag = np.linalg.norm(pts.max(0) - pts.min(0))
        eps = max(diag * 0.008, 1e-4)

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.max() < 0:
        print("  ⚠ Kein Cluster gefunden – Punktwolke bleibt unverändert.")
        return pcd

    center = pts.mean(axis=0)
    sizes = np.bincount(labels[labels >= 0])
    topk = np.argsort(sizes)[::-1][:3]

    best_lbl, best_d = None, np.inf
    for lbl in topk:
        c = pts[labels == lbl].mean(axis=0)
        d = np.linalg.norm(c[:2] - center[:2])
        if d < best_d:
            best_d, best_lbl = d, lbl

    print(f"  → Hauptcluster: Label {best_lbl}, {sizes[best_lbl]} Punkte")
    return pcd.select_by_index(np.where(labels == best_lbl)[0])


def _plane_basis_from_model(plane_model):
    n = np.array(plane_model[:3], dtype=np.float64)
    n /= (np.linalg.norm(n) + 1e-12)
    d = float(plane_model[3])

    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n, helper)
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    p0 = -d * n

    return n, d, p0, u, v


def _rasterize_object_mask_from_points(main_obj_pcd, plane_model, uv_min, scale, W, H,
                                       expand_px=16, safety_px=12):
    import cv2

    pts = np.asarray(main_obj_pcd.points)
    _, _, _, u, v = _plane_basis_from_model(plane_model)

    uu = pts @ u
    vv = pts @ v

    xs = np.round((uu - uv_min[0]) * scale).astype(np.int32)
    ys = np.round((vv - uv_min[1]) * scale).astype(np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    mask[ys[valid], xs[valid]] = 255

    k1 = max(3, expand_px)
    if k1 % 2 == 0:
        k1 += 1
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))

    mask = cv2.dilate(mask, kernel1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    if safety_px > 0:
        k2 = max(3, safety_px)
        if k2 % 2 == 0:
            k2 += 1
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
        mask = cv2.dilate(mask, kernel2, iterations=1)

    return mask


def _extract_roi(img, repair, pad=64):
    ys, xs = np.where(repair > 0)
    if len(ys) == 0:
        return img.copy(), repair.copy(), (0, img.shape[0], 0, img.shape[1])

    y0 = max(0, ys.min() - pad)
    y1 = min(img.shape[0], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(img.shape[1], xs.max() + pad + 1)

    return img[y0:y1, x0:x1].copy(), repair[y0:y1, x0:x1].copy(), (y0, y1, x0, x1)


def _fast_local_fill_telea(img, hole_mask, radius=3):
    import cv2
    mask = (hole_mask > 0).astype(np.uint8) * 255
    return cv2.inpaint(img.astype(np.uint8), mask, radius, cv2.INPAINT_TELEA)


def _soften_filled_region_strong(img, hole_mask,
                                 inner_expand_px=6,
                                 outer_expand_px=18,
                                 blur_ksize=21,
                                 bilateral_d=9,
                                 bilateral_sigma_color=35,
                                 bilateral_sigma_space=35,
                                 alpha_strength=0.85):
    import cv2
    import numpy as np

    base_mask = (hole_mask > 0).astype(np.uint8) * 255

    k1 = inner_expand_px * 2 + 1
    k2 = outer_expand_px * 2 + 1
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))

    inner = cv2.dilate(base_mask, kernel1, iterations=1)
    outer = cv2.dilate(base_mask, kernel2, iterations=1)

    transition = outer.copy()

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    gauss = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    smooth = cv2.bilateralFilter(
        gauss,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )

    alpha = cv2.GaussianBlur(transition.astype(np.float32) / 255.0, (31, 31), 0)
    alpha = np.clip(alpha * alpha_strength, 0.0, 1.0)[..., None]

    out = img.astype(np.float32) * (1.0 - alpha) + smooth.astype(np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, inner, outer, (alpha[..., 0] * 255).astype(np.uint8)


def _save_debug_image(path, arr):
    Image.fromarray(arr).save(path)


def _make_overlay(base_img, mask, color=(255, 0, 0), alpha=0.35):
    out = base_img.astype(np.float32).copy()
    m = (mask > 0).astype(np.float32)[..., None]
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    out = out * (1.0 - m * alpha) + color_arr * (m * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def bake_ground_texture_remove_object_fast(
    ground_pcd,
    plane_model,
    mesh_scan,
    main_obj_pcd,
    out_path,
    resolution=600,
    max_dist=0.01,
    object_mask_expand_px=16,
    object_mask_safety_px=12,
    roi_pad_px=64,
    inpaint_radius=3,
    soften_inner_expand_px=6,
    soften_outer_expand_px=18,
    soften_blur_ksize=21,
    soften_bilateral_d=9,
    soften_bilateral_sigma_color=35,
    soften_bilateral_sigma_space=35,
    soften_alpha_strength=0.85,
    save_debug=True,
):
    import cv2

    if not mesh_scan.has_textures() or not mesh_scan.has_triangle_uvs():
        raise RuntimeError("Scan-Mesh hat keine Textur/UVs. Für diesen Bake werden UV + Textur benötigt.")

    n, d, p0, u, v = _plane_basis_from_model(plane_model)

    pts = np.asarray(ground_pcd.points)
    uv = np.column_stack([pts @ u, pts @ v])
    uv_min = uv.min(axis=0)
    uv_max = uv.max(axis=0)
    size_world = uv_max - uv_min

    scale = (resolution - 1) / float(max(size_world.max(), 1e-9))
    W = int(np.ceil(size_world[0] * scale)) + 1
    H = int(np.ceil(size_world[1] * scale)) + 1
    print(f"  → Boden-Größe ~ {size_world[0]:.3f} x {size_world[1]:.3f} m -> {W}x{H} px")

    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    uu = uv_min[0] + xs.ravel() / scale
    vv = uv_min[1] + ys.ravel() / scale
    world_pts = (p0[None, :] + uu[:, None] * u[None, :] + vv[:, None] * v[None, :]).astype(np.float32)

    print("  → Raycasting Boden-Textur …")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_scan))
    ans = scene.compute_closest_points(o3d.core.Tensor(world_pts, dtype=o3d.core.Dtype.Float32))

    s_tri = ans["primitive_ids"].numpy()
    s_bary = ans["primitive_uvs"].numpy()
    s_pts = ans["points"].numpy()

    dists = np.linalg.norm(world_pts - s_pts, axis=1)
    hit_valid = dists < max_dist

    scan_uvs = np.asarray(mesh_scan.triangle_uvs)
    su, sv = s_bary[:, 0], s_bary[:, 1]
    sw = 1.0 - su - sv

    uv0 = scan_uvs[s_tri * 3 + 0]
    uv1 = scan_uvs[s_tri * 3 + 1]
    uv2 = scan_uvs[s_tri * 3 + 2]
    final_uvs = sw[:, None] * uv0 + su[:, None] * uv1 + sv[:, None] * uv2

    tex_img = np.asarray(mesh_scan.textures[0])
    th, tw = tex_img.shape[:2]

    tx = np.clip(final_uvs[:, 0], 0.0, 0.999) * (tw - 1)
    ty = np.clip(1.0 - final_uvs[:, 1], 0.0, 0.999) * (th - 1)
    colors = tex_img[ty.astype(np.int32), tx.astype(np.int32)].astype(np.uint8)

    img = colors.reshape(H, W, 3)

    print("  → Erzeuge präzise Lochmaske aus projiziertem Hauptobjekt …")
    repair = _rasterize_object_mask_from_points(
        main_obj_pcd=main_obj_pcd,
        plane_model=plane_model,
        uv_min=uv_min,
        scale=scale,
        W=W,
        H=H,
        expand_px=object_mask_expand_px,
        safety_px=object_mask_safety_px,
    )

    base_dbg = os.path.splitext(out_path)[0]

    if save_debug:
        _save_debug_image(base_dbg + "_repair_mask.png", repair)
        _save_debug_image(base_dbg + "_full_overlay.png", _make_overlay(img, repair, color=(255, 0, 0), alpha=0.35))

    if repair.max() == 0:
        print("  → Reparaturmaske leer, speichere nur gebackene Textur.")
    else:
        if np.any(~hit_valid):
            invalid = (~hit_valid).reshape(H, W).astype(np.uint8) * 255
            invalid_outside_hole = cv2.bitwise_and(invalid, cv2.bitwise_not(repair))
            if invalid_outside_hole.max() > 0:
                print("  → Fülle kleine Raycast-Lücken außerhalb des Lochs …")
                img = cv2.inpaint(img, invalid_outside_hole, 2, cv2.INPAINT_TELEA)
                if save_debug:
                    _save_debug_image(base_dbg + "_invalid_outside_hole.png", invalid_outside_hole)

        roi_img, roi_mask, roi_box = _extract_roi(img, repair, pad=roi_pad_px)
        y0, y1, x0, x1 = roi_box

        if save_debug:
            _save_debug_image(base_dbg + "_roi_before_fill.png", roi_img)
            _save_debug_image(base_dbg + "_roi_mask.png", roi_mask)
            _save_debug_image(base_dbg + "_roi_overlay.png", _make_overlay(roi_img, roi_mask, color=(255, 0, 0), alpha=0.40))

        print("  → Fülle Loch schnell per lokalem Telea-Inpaint nur in ROI …")
        filled_roi = _fast_local_fill_telea(roi_img, roi_mask, radius=inpaint_radius)

        if save_debug:
            _save_debug_image(base_dbg + "_roi_after_fill.png", filled_roi)

        print("  → Starkes Softening auf Übergangsring anwenden …")
        softened_roi, inner_mask, outer_mask, feather_alpha = _soften_filled_region_strong(
            filled_roi,
            roi_mask,
            inner_expand_px=soften_inner_expand_px,
            outer_expand_px=soften_outer_expand_px,
            blur_ksize=soften_blur_ksize,
            bilateral_d=soften_bilateral_d,
            bilateral_sigma_color=soften_bilateral_sigma_color,
            bilateral_sigma_space=soften_bilateral_sigma_space,
            alpha_strength=soften_alpha_strength,
        )

        if save_debug:
            _save_debug_image(base_dbg + "_roi_soften_inner_mask.png", inner_mask)
            _save_debug_image(base_dbg + "_roi_soften_outer_mask.png", outer_mask)
            _save_debug_image(base_dbg + "_roi_feather_alpha.png", feather_alpha)
            _save_debug_image(base_dbg + "_roi_after_soften.png", softened_roi)
            _save_debug_image(base_dbg + "_roi_debug_overlay.png", _make_overlay(softened_roi, outer_mask, color=(0, 255, 0), alpha=0.30))

        img[y0:y1, x0:x1] = softened_roi

    Image.fromarray(img.astype(np.uint8)).save(out_path, "JPEG", quality=95)
    print(f"  → Boden-Textur gespeichert: {out_path} ({W}x{H})")

    obj_path = os.path.splitext(out_path)[0] + ".obj"
    _write_ground_plane_obj(obj_path, p0, u, v, uv_min, uv_max, os.path.basename(out_path))
    print(f"  → Boden-Plane gespeichert: {obj_path}")

    return out_path


def main():
    dateiname = input("Bitte Dateipfad eingeben: ").strip().strip('"')
    base, _ = os.path.splitext(dateiname)

    print("• Lade Mesh …")
    mesh_scan = load_mesh(dateiname)
    print(f"  Mesh hat Textur: {mesh_scan.has_textures()}  UVs: {mesh_scan.has_triangle_uvs()}")

    print("• Sample Punkte für schnelle Ebenensuche …")
    pcd_fast = mesh_scan.sample_points_uniformly(number_of_points=200_000)

    diag = np.linalg.norm(np.asarray(pcd_fast.points).max(0) - np.asarray(pcd_fast.points).min(0))
    dist_th = diag * 0.005
    print(f"• RANSAC Boden-Suche (dist_threshold={dist_th:.4f}) …")

    plane_model, inliers, work = find_ground_plane(
        pcd_fast,
        distance_threshold=dist_th,
        ransac_n=3,
        num_iterations=600,
    )

    a, b, c, d = plane_model
    print(f"  → Ebene: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0   ({len(inliers)} Inlier)")

    ground_pcd = work.select_by_index(inliers)
    without_grd = work.select_by_index(inliers, invert=True)

    print("• Clustering: Hauptobjekt suchen …")
    main_obj = keep_main_object(without_grd)

    out_pcd = base + "_object.ply"
    o3d.io.write_point_cloud(out_pcd, main_obj)
    print(f"  → Punktwolke gespeichert: {out_pcd}")

    tex_path = base + "_ground_texture.jpg"
    print("• Backe Boden-Textur + entferne Objekt schnell …")
    bake_ground_texture_remove_object_fast(
        ground_pcd=ground_pcd,
        plane_model=plane_model,
        mesh_scan=mesh_scan,
        main_obj_pcd=main_obj,
        out_path=tex_path,
        resolution=600,
        max_dist=0.01,
        object_mask_expand_px=16,
        object_mask_safety_px=12,
        roi_pad_px=24,
        inpaint_radius=3,
        soften_inner_expand_px=6,
        soften_outer_expand_px=10,
        soften_blur_ksize=31,
        soften_bilateral_d=9,
        soften_bilateral_sigma_color=35,
        soften_bilateral_sigma_space=35,
        soften_alpha_strength=0.85,
        save_debug=True,
    )

    print("• Vorschau (schließen zum Fortfahren) …")
    ground_pcd.paint_uniform_color([0.7, 0.5, 0.3])
    o3d.visualization.draw_geometries(
        [main_obj, ground_pcd],
        window_name="Auto-Crop Ergebnis (Boden=braun, Objekt=Original)"
    )

    print("• Manuelles Nachschneiden (optional) …")
    o3d.visualization.draw_geometries_with_editing([main_obj], window_name="Cropping Tool")


if __name__ == "__main__":
    main()