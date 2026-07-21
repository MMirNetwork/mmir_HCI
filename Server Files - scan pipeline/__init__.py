import copy
import os

import numpy as np
import open3d as o3d
from PIL import Image

from src.scan.config import PipelineConfig
from src.scan.descriptors import DescriptorExtractor
from src.scan.io_utils import MeshIO
from src.scan.registration import CloudRegistrator
from src.scan.scan_processing import ScanProcessor
from src.scan.texture_baking import TextureBaker


def save_alignment_png(
    scan_down,
    digital_down,
    transformation,
    out_path="alignment_debug.png",
    width=1280,
    height=960,
):
    """Render the aligned point clouds to an offscreen PNG."""
    src = copy.deepcopy(scan_down)
    tgt = copy.deepcopy(digital_down)

    src.transform(transformation)
    src.paint_uniform_color([1.0, 0.7, 0.0])
    tgt.paint_uniform_color([0.0, 0.6, 0.9])

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        vis.add_geometry(src)
        vis.add_geometry(tgt)

        opt = vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.array([0.1, 0.1, 0.1])

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(out_path, do_render=True)
        vis.destroy_window()
        print(f"[DEBUG] Saved alignment image: {out_path}")
    except Exception as e:
        # Export point clouds when offscreen rendering is unavailable.
        print(
            f"[DEBUG] Offscreen rendering failed ({e}); "
            "exporting PLY files instead."
        )
        base = os.path.splitext(out_path)[0]
        o3d.io.write_point_cloud(base + "_scan.ply", src)
        o3d.io.write_point_cloud(base + "_digital.ply", tgt)
        print(f"[DEBUG] Exported: {base}_scan.ply / {base}_digital.ply")


def save_alignment_ply(
    scan_down,
    digital_down,
    transformation,
    out_path="alignment_debug.ply",
):
    """Save both aligned point clouds in a single PLY file."""
    src = copy.deepcopy(scan_down)
    tgt = copy.deepcopy(digital_down)

    src.transform(transformation)
    src.paint_uniform_color([1.0, 0.7, 0.0])
    tgt.paint_uniform_color([0.0, 0.6, 0.9])

    combined = src + tgt
    o3d.io.write_point_cloud(out_path, combined)
    print(
        f"[DEBUG] Saved alignment point cloud: {out_path} "
        "(orange=scan, blue=digital)"
    )


def show_alignment_interactive(
    scan_down,
    digital_down,
    transformation,
    window_name="Alignment Debug (scan=orange, digital=blue)",
):
    """Display both aligned point clouds in an interactive window."""
    src = copy.deepcopy(scan_down)
    tgt = copy.deepcopy(digital_down)

    src.transform(transformation)
    src.paint_uniform_color([1.0, 0.7, 0.0])
    tgt.paint_uniform_color([0.0, 0.6, 0.9])

    # Add a coordinate frame for orientation.
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1,
        origin=[0, 0, 0],
    )

    o3d.visualization.draw_geometries(
        [src, tgt, axes],
        window_name=window_name,
        width=1280,
        height=960,
        point_show_normal=False,
    )


def process_scan(config: PipelineConfig):
    """Register an incoming scan against the digital model library."""
    if not config.scan_obj_path:
        raise ValueError("A scan file path is required.")
    if not os.path.isfile(config.scan_obj_path):
        raise FileNotFoundError(f"Scan file not found: {config.scan_obj_path}")
    if not config.index_folder:
        raise ValueError("A digital model index folder is required.")
    if not os.path.isdir(config.index_folder):
        raise FileNotFoundError(
            f"Digital model index folder not found: {config.index_folder}"
        )
    if config.digital_unit_scale <= 0:
        raise ValueError("digital_unit_scale must be greater than zero.")

    os.makedirs(config.output_folder, exist_ok=True)

    # Prepare the incoming scan.
    mesh_scan = MeshIO.load_mesh(config.scan_obj_path)

    pcd_scan_raw = mesh_scan.sample_points_uniformly(number_of_points=100000)
    pcd_scan_raw = ScanProcessor.remove_outliers(pcd_scan_raw)

    ground_plane_model = None
    ground_pcd = None
    main_obj_pcd = None

    cropped_path = getattr(config, "cropped_scan_path", "") or ""
    if cropped_path and os.path.exists(cropped_path):
        # Use the configured pre-cropped point cloud when available.
        print(f"Loading point cloud (fixed cropped scan): {cropped_path}")
        pcd_scan = o3d.io.read_point_cloud(cropped_path)
        pcd_scan = ScanProcessor.remove_outliers(pcd_scan)
    else:
        if config.remove_scan_ground:
            # Remove the ground and retain the primary object cluster.
            print(
                "Auto-cropping incoming scan "
                "(RANSAC ground removal + DBSCAN main object)..."
            )

            diag = np.linalg.norm(
                np.asarray(pcd_scan_raw.get_max_bound())
                - np.asarray(pcd_scan_raw.get_min_bound())
            )
            dist_thr = max(diag * config.ground_dist_threshold_ratio, 1e-4)
            found = ScanProcessor.find_ground_plane(
                pcd_scan_raw,
                distance_threshold=dist_thr,
            )

            if found is not None:
                ground_plane_model, inliers, work = found
                ground_pcd = work.select_by_index(inliers)
                pcd_no_ground = work.select_by_index(inliers, invert=True)
                print(f"  Removed {len(inliers):,} ground points.")

                # Clear residual points near the detected ground plane.
                plane_clearance = max(dist_thr * 2.5, 1e-4)
                pcd_no_ground = ScanProcessor.remove_points_near_plane(
                    pcd_no_ground,
                    ground_plane_model,
                    plane_clearance,
                )
            else:
                pcd_no_ground = pcd_scan_raw
        else:
            print(
                "Ground removal disabled; skipping RANSAC plane detection."
            )
            pcd_no_ground = pcd_scan_raw

        # DBSCAN remains useful in both modes to discard isolated scan noise.
        pcd_scan = ScanProcessor.keep_main_object(
            pcd_no_ground,
            radial_trim_target_ratio=config.radial_trim_target_ratio,
        )
        pcd_scan = ScanProcessor.remove_outliers(pcd_scan)
        main_obj_pcd = pcd_scan
        print(f"  Cropped scan points: {len(pcd_scan.points):,}")

    tex_path = None

    # Register the scan against candidate models.
    size_scan_global = ScanProcessor.compute_scale_metric(
        pcd_scan,
        "median_radius",
    )
    voxel_size = max(size_scan_global * config.voxel_ratio, 1e-4)
    max_dist = max(
        voxel_size * config.texture_projection_distance_voxels,
        1e-4,
    )
    print(
        f"  scan_size={size_scan_global:.4f} -> "
        f"dynamic voxel={voxel_size:.4f}, max_dist={max_dist:.4f}"
    )

    candidates = DescriptorExtractor.get_top_candidates(
        pcd_scan,
        config.index_folder,
        top_k=config.top_k_candidates,
    )

    winner = None
    for cand_name, cand_path in candidates:
        # Sample each OBJ candidate into an in-memory point cloud.
        cand_obj = cand_path
        if not cand_obj.lower().endswith(".obj"):
            cand_obj = os.path.splitext(cand_obj)[0] + ".obj"
        if not os.path.exists(cand_obj):
            print(
                f"  WARN: No OBJ found for candidate '{cand_name}' "
                f"({cand_obj}); skipping it."
            )
            continue

        print(f"\n========== Candidate: {cand_name} ==========")
        cand_mesh = MeshIO.load_mesh(cand_obj)
        pcd_digital = cand_mesh.sample_points_uniformly(number_of_points=50000)
        if config.remove_detached_digital_components:
            pcd_digital = ScanProcessor.remove_detached_components(pcd_digital)
        if len(pcd_digital.points) == 0:
            print(f"  WARN: '{cand_obj}' produced no points; skipping it.")
            continue

        pcd_digital.scale(config.digital_unit_scale, center=np.zeros(3))
        pcd_digital_c = ScanProcessor.remove_outliers(pcd_digital)

        size_scan = ScanProcessor.compute_scale_metric(pcd_scan, "median_radius")
        size_digital = ScanProcessor.compute_scale_metric(
            pcd_digital_c,
            "median_radius",
        )
        auto_scale = size_scan / max(size_digital, 1e-12)
        print(
            f"  size_scan={size_scan:.4f}  "
            f"size_digital={size_digital:.4f}  factor={auto_scale:.4f}"
        )
        pcd_digital_c.scale(auto_scale, center=np.zeros(3))
        size_digital_final = ScanProcessor.compute_scale_metric(
            pcd_digital_c,
            "median_radius",
        )

        pcd_scan_n, T_scan_n = ScanProcessor.normalize_orientation(
            pcd_scan,
            up_axis=config.scan_up_axis,
        )
        pcd_digital_n, T_digital_n = ScanProcessor.normalize_orientation(
            pcd_digital_c,
            up_axis=config.digital_up_axis,
        )

        scan_down = ScanProcessor.preprocess_point_cloud(
            pcd_scan_n,
            voxel_size,
        )
        digital_down = ScanProcessor.preprocess_point_cloud(
            pcd_digital_n,
            voxel_size,
        )

        best = CloudRegistrator.find_best_registration(
            scan_down,
            digital_down,
            voxel_size,
            similarity_scale_bounds=(
                config.similarity_scale_min,
                config.similarity_scale_max,
            ),
        )
        selection_key = CloudRegistrator._selection_key(best)
        print(
            f"  -> result {cand_name}: "
            f"symmetric_fitness={best.symmetric_fitness():.4f}, "
            f"rmse={best.inlier_rmse:.6f}"
        )
        if winner is None or selection_key > winner[0]:
            winner = (
                selection_key,
                best,
                cand_name,
                cand_obj,
                auto_scale,
                size_digital_final,
                T_scan_n,
                T_digital_n,
                scan_down,
                digital_down,
            )

    if winner is None:
        raise RuntimeError("No candidate produced a valid registration.")

    (
        selection_key,
        best,
        dig_base,
        cand_obj,
        auto_scale,
        size_digital_final,
        T_scan_n,
        T_digital_n,
        scan_down,
        digital_down,
    ) = winner
    print(f"\n>>> Auto-selected: {dig_base}  {best}\n")
    print(
        f"[SCALE] ICP scale (scan->digital) = {best.icp_scale:.4f}"
    )

    # Validate scale only on the actual overlap.  The full scan AABB can be
    # much larger than the digital model when the scan still contains a hand,
    # support, ground remnants, or other disconnected geometry.  Using that
    # box as a scale estimate produced misleading ratios (and must never drive
    # a scale correction).
    _sd = copy.deepcopy(scan_down)
    _sd.transform(best.transformation)
    _scan_pts = np.asarray(_sd.points)
    _digital_pts = np.asarray(digital_down.points)
    _inlier_threshold = max(voxel_size, 1e-6)

    _digital_tree = o3d.geometry.KDTreeFlann(digital_down)
    _scan_inlier_mask = np.zeros(len(_scan_pts), dtype=bool)
    for _i, _p in enumerate(_scan_pts):
        _k, _idx, _dist2 = _digital_tree.search_knn_vector_3d(_p, 1)
        if _k and _dist2[0] <= _inlier_threshold * _inlier_threshold:
            _scan_inlier_mask[_i] = True

    _scan_tree = o3d.geometry.KDTreeFlann(_sd)
    _digital_inlier_mask = np.zeros(len(_digital_pts), dtype=bool)
    for _i, _p in enumerate(_digital_pts):
        _k, _idx, _dist2 = _scan_tree.search_knn_vector_3d(_p, 1)
        if _k and _dist2[0] <= _inlier_threshold * _inlier_threshold:
            _digital_inlier_mask[_i] = True

    def _robust_overlap_diag(points, mask):
        selected = points[mask]
        if len(selected) < 10:
            return float("nan")
        lo = np.percentile(selected, 2.5, axis=0)
        hi = np.percentile(selected, 97.5, axis=0)
        return float(np.linalg.norm(hi - lo))

    _scan_overlap_diag = _robust_overlap_diag(_scan_pts, _scan_inlier_mask)
    _digital_overlap_diag = _robust_overlap_diag(
        _digital_pts,
        _digital_inlier_mask,
    )
    _overlap_ratio = _scan_overlap_diag / max(_digital_overlap_diag, 1e-12)
    print(
        f"[SCALE CHECK] overlap_scan_diag={_scan_overlap_diag:.4f}  "
        f"overlap_digital_diag={_digital_overlap_diag:.4f}  "
        f"ratio={_overlap_ratio:.4f} (target ~1.0; ICP inliers only)"
    )
    print(
        f"[SCALE CHECK] inliers scan={int(_scan_inlier_mask.sum()):,}/"
        f"{len(_scan_pts):,}, digital={int(_digital_inlier_mask.sum()):,}/"
        f"{len(_digital_pts):,}, threshold={_inlier_threshold:.4f}"
    )

    if config.output_mode != "textures_only":
        save_alignment_ply(
            scan_down,
            digital_down,
            best.transformation,
            out_path=os.path.join(config.output_folder, "alignment_debug.ply"),
        )

    # Use the matched OBJ for texture baking.
    digital_obj_path = cand_obj
    if not digital_obj_path.lower().endswith(".obj"):
        digital_obj_path = os.path.splitext(digital_obj_path)[0] + ".obj"
    if not os.path.exists(digital_obj_path):
        raise RuntimeError(
            f"Could not find a .obj file for texture baking for model "
            f"'{dig_base}'.\n"
            f"Expected: {digital_obj_path}"
        )
    print(f"Using digital OBJ for baking: {digital_obj_path}")

    # Prepare a single, explicit digital-to-scan similarity transform.
    # Registration estimates scan_normalized -> digital_normalized:
    #     d = (s * R) @ scan_normalized + t
    # Baking keeps the scan mesh fixed, so the digital mesh needs the exact
    # inverse transform.  Applying the inverse matrix directly avoids splitting
    # scale/rotation/translation and prevents the scale from being inverted or
    # applied twice.
    T_icp = np.asarray(best.transformation, dtype=np.float64)
    s = float(best.icp_scale)
    if not np.isfinite(s) or s <= 0.0:
        raise RuntimeError(f"Invalid ICP similarity scale: {s}")

    T_digital_to_scan = (
        np.linalg.inv(T_scan_n)
        @ np.linalg.inv(T_icp)
        @ T_digital_n
    )
    print(
        f"[DEBUG] Applying inverse similarity transform digital->scan "
        f"(scale={1.0 / s:.4f}; ICP scan->digital scale={s:.4f})"
    )

    global_scale = config.digital_unit_scale * auto_scale
    print(
        f"[DEBUG] global_scale={global_scale:.6f} "
        f"(unit_scale={config.digital_unit_scale:.6f}, "
        f"auto_scale={auto_scale:.4f})"
    )

    if mesh_scan.has_textures():
        texture_img = Image.fromarray(
            np.asarray(mesh_scan.textures[0])
        ).convert("RGB")
    elif config.texture_path:
        if not os.path.isfile(config.texture_path):
            raise FileNotFoundError(
                f"Fallback texture not found: {config.texture_path}"
            )
        texture_img = Image.open(config.texture_path).convert("RGB")
    else:
        raise RuntimeError(
            "The scan has no embedded texture and no fallback texture was provided."
        )

    tex_w, tex_h = texture_img.size
    tex_arr = np.array(texture_img).astype(np.float32) / 255.0

    # Compute a robust fallback color for unseen surfaces.
    fallback_sample = tex_arr.reshape(-1, 3)
    if len(fallback_sample) > 200000:
        step = max(1, len(fallback_sample) // 200000)
        fallback_sample = fallback_sample[::step]
    fallback_color = np.median(fallback_sample, axis=0)
    print(
        "[BAKE] Robust fallback color for unseen regions: "
        f"RGB={np.round(fallback_color * 255).astype(int).tolist()}"
    )

    # Keep the textured scan in its original coordinate system. Move the
    # digital model into that system with the inverse registration transform.
    mesh_scan_aligned = copy.deepcopy(mesh_scan)

    parts = MeshIO.load_obj_grouped(digital_obj_path)

    # Apply unit/automatic scale once, then the complete inverse similarity
    # transform once. T_digital_to_scan already contains normalization,
    # rotation, translation, and the inverse ICP scale.
    aligned_meshes = []
    for part in parts:
        m = MeshIO.part_to_mesh(part)
        m.scale(global_scale, center=np.zeros(3))
        m.transform(T_digital_to_scan)
        aligned_meshes.append(m)

    print(
        f"[DEBUG] Scan Box (Aligned): Center={mesh_scan_aligned.get_center()}, "
        "Extent="
        f"{mesh_scan_aligned.get_axis_aligned_bounding_box().get_extent()}"
    )
    digital_pts = (
        np.concatenate([np.asarray(m.vertices) for m in aligned_meshes])
        if aligned_meshes
        else np.zeros((1, 3))
    )
    print(
        f"[DEBUG] Digital Box: Center={digital_pts.mean(axis=0)}, "
        f"Extent={digital_pts.max(axis=0) - digital_pts.min(axis=0)}"
    )
    print(f"[DEBUG] auto_scale={auto_scale:.4f}, global_scale={global_scale:.4f}")

    # Keep only scan triangles near the matched digital model.
    scan_vertices = np.asarray(mesh_scan_aligned.vertices)
    scan_triangles = np.asarray(mesh_scan_aligned.triangles)
    scan_uvs_all = np.asarray(mesh_scan_aligned.triangle_uvs)
    digital_min = digital_pts.min(axis=0)
    digital_max = digital_pts.max(axis=0)
    crop_margin = max(max_dist * 2.0, 1e-6)
    triangle_centers = scan_vertices[scan_triangles].mean(axis=1)
    keep_triangles = np.all(
        (triangle_centers >= digital_min - crop_margin)
        & (triangle_centers <= digital_max + crop_margin),
        axis=1,
    )
    kept_ids = np.flatnonzero(keep_triangles)
    if len(kept_ids) > 0:
        scan_bake_mesh = o3d.geometry.TriangleMesh()
        scan_bake_mesh.vertices = o3d.utility.Vector3dVector(scan_vertices)
        scan_bake_mesh.triangles = o3d.utility.Vector3iVector(
            scan_triangles[kept_ids]
        )
        if len(scan_uvs_all) == len(scan_triangles) * 3:
            scan_bake_mesh.triangle_uvs = o3d.utility.Vector2dVector(
                scan_uvs_all.reshape(-1, 3, 2)[kept_ids].reshape(-1, 2)
            )
        scan_uvs = np.asarray(scan_bake_mesh.triangle_uvs)
        print(
            f"[BAKE] Kept {len(kept_ids):,}/{len(scan_triangles):,} "
            "scan triangles near the digital model."
        )
    else:
        scan_bake_mesh = mesh_scan_aligned
        scan_uvs = scan_uvs_all
        print("[BAKE] WARN: Scan crop was empty; using the complete scan mesh.")

    if len(scan_uvs) != len(scan_bake_mesh.triangles) * 3:
        raise RuntimeError("The cropped scan mesh has no valid triangle UVs.")

    scan_t = o3d.t.geometry.TriangleMesh.from_legacy(scan_bake_mesh)
    scan_scene = o3d.t.geometry.RaycastingScene()
    scan_scene.add_triangles(scan_t)

    all_pts = np.concatenate([np.asarray(m.vertices) for m in aligned_meshes])
    center = all_pts.mean(axis=0)
    g_min = all_pts.min(axis=0)
    T_place = np.eye(4)
    T_place[0, 3] = -center[0]
    T_place[2, 3] = -center[2]
    T_place[1, 3] = -g_min[1]

    baked_parts = []
    for part, m_aligned in zip(parts, aligned_meshes):
        name = part["name"]
        n_tris = len(part["triangles"])
        tex_size = TextureBaker.adaptive_tex_size(
            n_tris,
            config.tex_min_size,
            config.tex_max_size,
        )

        has_orig_uv = np.asarray(m_aligned.triangle_uvs).shape[0] == n_tris * 3
        if config.use_original_uvs and has_orig_uv:
            m_uv = m_aligned
        else:
            import xatlas

            verts = np.asarray(m_aligned.vertices)
            tris = np.asarray(m_aligned.triangles)
            atlas = xatlas.Atlas()
            atlas.add_mesh(verts, tris)
            atlas.generate()
            vmapping, new_tris, new_uvs = atlas[0]
            m_uv = o3d.geometry.TriangleMesh()
            m_uv.vertices = o3d.utility.Vector3dVector(verts[vmapping])
            m_uv.triangles = o3d.utility.Vector3iVector(new_tris)
            m_uv.triangle_uvs = o3d.utility.Vector2dVector(
                new_uvs[new_tris].reshape(-1, 2)
            )
            m_uv.compute_vertex_normals()

        tex = TextureBaker.bake_digital_texture(
            m_uv,
            scan_scene,
            scan_uvs,
            tex_arr,
            tex_w,
            tex_h,
            tex_size,
            max_dist,
            fallback_color=fallback_color,
            part_name=name,
        )
        m_out = copy.deepcopy(m_uv)
        m_out.transform(T_place)
        baked_parts.append({"name": name, "mesh": m_out, "tex": tex})

    # Bake the ground texture from the original scan.
    if (
        config.remove_scan_ground
        and ground_plane_model is not None
        and ground_pcd is not None
        and len(ground_pcd.points) > 0
        and config.output_mode != "textures_only"
    ):
        if mesh_scan.has_textures() and mesh_scan.has_triangle_uvs():
            try:
                plane_basis = ScanProcessor.plane_basis_from_model(
                    ground_plane_model
                )
                ground_out = os.path.join(
                    config.output_folder,
                    f"{dig_base}_ground.jpg",
                )
                obj_pcd = main_obj_pcd if main_obj_pcd is not None else pcd_scan
                tex_path = TextureBaker.bake_ground_texture_remove_object_fast(
                    ground_pcd,
                    plane_basis,
                    mesh_scan,
                    obj_pcd,
                    ground_out,
                )
                print(f"[GROUND] Baked ground texture: {tex_path}")
            except Exception as e:
                print(f"[GROUND] WARN: Ground baking failed ({e}); skipping it.")
                tex_path = None
        else:
            print("[GROUND] Scan has no texture or UVs; skipping ground baking.")
    else:
        if not config.remove_scan_ground:
            print("[GROUND] Ground removal disabled; skipping ground baking.")
        else:
            print(
                "[GROUND] No ground plane was found or a fixed crop is active; "
                "skipping ground baking."
            )

    return baked_parts, tex_path, dig_base
