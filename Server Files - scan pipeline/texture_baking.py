import os

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont


class TextureBaker:
    """Handle raycasting, texture baking, and image inpainting."""

    @staticmethod
    def adaptive_tex_size(
        n_tris,
        min_size=256,
        max_size=2048,
        ref_tris=1200,
    ):
        """Calculate a power-of-two texture size from the face count."""
        if n_tris <= 0:
            return min_size
        frac = (n_tris / ref_tris) ** 0.5
        target = max_size * min(1.0, max(0.0, frac))
        size = min_size
        while size < target and size < max_size:
            size *= 2
        return int(max(min_size, min(size, max_size)))

    @staticmethod
    def create_debug_texture(width=2048, height=2048, grid_size=20):
        """Generate a debug texture grid."""
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        cell_w = width / grid_size
        cell_h = height / grid_size
        try:
            font = ImageFont.truetype(
                "arial.ttf",
                int(min(cell_w, cell_h) * 0.3),
            )
        except Exception:
            font = ImageFont.load_default()

        for i in range(grid_size):
            for j in range(grid_size):
                x0 = j * cell_w
                y0 = i * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                bg_color = (
                    int(255 * (j / grid_size)),
                    int(255 * (i / grid_size)),
                    150,
                )
                draw.rectangle(
                    [x0, y0, x1, y1],
                    fill=bg_color,
                    outline="black",
                )
                text = f"{i},{j}"
                draw.text(
                    (x0 + cell_w * 0.1, y0 + cell_h * 0.1),
                    text,
                    fill="white",
                    font=font,
                )
                draw.text(
                    (x0 + cell_w * 0.1 + 2, y0 + cell_h * 0.1 + 2),
                    text,
                    fill="black",
                    font=font,
                )

        return img

    @staticmethod
    def bake_digital_texture(
        mesh_digital,
        scan_scene,
        scan_uvs,
        tex_arr,
        tex_w,
        tex_h,
        out_size=2048,
        max_dist=0.05,
        fallback_color=None,
        part_name="part",
    ):
        """Bake a scan texture onto the UV space of a digital mesh."""
        digital_uvs = np.asarray(mesh_digital.triangle_uvs)
        if len(digital_uvs) == 0:
            return None

        # Compute a robust fallback color for unseen regions.
        if fallback_color is None:
            sample = np.asarray(tex_arr, dtype=np.float32).reshape(-1, 3)
            if len(sample) > 200000:
                step = max(1, len(sample) // 200000)
                sample = sample[::step]
            fallback_color = np.median(sample, axis=0)
        fallback_color = np.clip(
            np.asarray(fallback_color, dtype=np.float32).reshape(3),
            0.0,
            1.0,
        )

        digital_tris = np.asarray(mesh_digital.triangles)
        digital_verts = np.asarray(mesh_digital.vertices)

        mesh_digital.compute_triangle_normals()
        digital_tri_normals = np.asarray(mesh_digital.triangle_normals)

        T = len(digital_tris)
        out_tex = np.zeros((out_size, out_size, 3), dtype=np.float32)
        out_mask = np.zeros((out_size, out_size), dtype=bool)
        hit_mask = np.zeros((out_size, out_size), dtype=bool)

        pix_xs, pix_ys, pos_list, norm_list = [], [], [], []

        for ti in range(T):
            uv0 = digital_uvs[ti * 3 + 0]
            uv1 = digital_uvs[ti * 3 + 1]
            uv2 = digital_uvs[ti * 3 + 2]

            p0 = np.array(
                [uv0[0] * (out_size - 1), (1.0 - uv0[1]) * (out_size - 1)]
            )
            p1 = np.array(
                [uv1[0] * (out_size - 1), (1.0 - uv1[1]) * (out_size - 1)]
            )
            p2 = np.array(
                [uv2[0] * (out_size - 1), (1.0 - uv2[1]) * (out_size - 1)]
            )

            min_x = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
            max_x = min(
                int(np.ceil(max(p0[0], p1[0], p2[0]))),
                out_size - 1,
            )
            min_y = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
            max_y = min(
                int(np.ceil(max(p0[1], p1[1], p2[1]))),
                out_size - 1,
            )

            if max_x < min_x or max_y < min_y:
                continue

            xs, ys = np.meshgrid(
                np.arange(min_x, max_x + 1),
                np.arange(min_y, max_y + 1),
            )
            xs = xs.ravel().astype(np.float32)
            ys = ys.ravel().astype(np.float32)

            denom = (
                (p1[1] - p2[1]) * (p0[0] - p2[0])
                + (p2[0] - p1[0]) * (p0[1] - p2[1])
            )
            if abs(denom) < 1e-12:
                continue

            l0 = (
                (p1[1] - p2[1]) * (xs - p2[0])
                + (p2[0] - p1[0]) * (ys - p2[1])
            ) / denom
            l1 = (
                (p2[1] - p0[1]) * (xs - p2[0])
                + (p0[0] - p2[0]) * (ys - p2[1])
            ) / denom
            l2 = 1.0 - l0 - l1

            eps = -1e-4
            inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
            if not np.any(inside):
                continue

            l0i = l0[inside]
            l1i = l1[inside]
            l2i = l2[inside]
            xi = xs[inside].astype(np.int32)
            yi = ys[inside].astype(np.int32)

            v0 = digital_verts[digital_tris[ti, 0]]
            v1 = digital_verts[digital_tris[ti, 1]]
            v2 = digital_verts[digital_tris[ti, 2]]
            pos = (
                l0i[:, None] * v0
                + l1i[:, None] * v1
                + l2i[:, None] * v2
            )

            n = digital_tri_normals[ti]
            norm = np.tile(n, (len(l0i), 1))

            pix_xs.append(xi)
            pix_ys.append(yi)
            pos_list.append(pos.astype(np.float32))
            norm_list.append(norm.astype(np.float32))

        if not pos_list:
            return None

        pix_xs = np.concatenate(pix_xs)
        pix_ys = np.concatenate(pix_ys)
        positions = np.concatenate(pos_list, axis=0)
        normals = np.concatenate(norm_list, axis=0)

        bad = (
            ~np.isfinite(positions).all(axis=1)
            | ~np.isfinite(normals).all(axis=1)
        )
        if bad.any():
            positions = positions[~bad]
            normals = normals[~bad]
            pix_xs = pix_xs[~bad]
            pix_ys = pix_ys[~bad]

        if len(positions) == 0:
            return None

        # Cast rays from both sides of each digital surface.
        forward_rays = np.concatenate(
            [positions + normals * max_dist, -normals],
            axis=1,
        ).astype(np.float32)
        reverse_rays = np.concatenate(
            [positions - normals * max_dist, normals],
            axis=1,
        ).astype(np.float32)

        forward_ans = scan_scene.cast_rays(
            o3d.core.Tensor(
                forward_rays,
                dtype=o3d.core.Dtype.Float32,
            )
        )
        reverse_ans = scan_scene.cast_rays(
            o3d.core.Tensor(
                reverse_rays,
                dtype=o3d.core.Dtype.Float32,
            )
        )

        forward_t = forward_ans["t_hit"].numpy()
        reverse_t = reverse_ans["t_hit"].numpy()
        forward_dist = np.abs(forward_t - max_dist)
        reverse_dist = np.abs(reverse_t - max_dist)
        forward_valid = np.isfinite(forward_t) & (forward_dist < max_dist)
        reverse_valid = np.isfinite(reverse_t) & (reverse_dist < max_dist)

        # Choose the ray hit nearest to the digital surface.
        use_reverse = reverse_valid & (
            ~forward_valid | (reverse_dist < forward_dist)
        )
        ray_valid = forward_valid | reverse_valid
        ray_dist = np.where(use_reverse, reverse_dist, forward_dist)
        s_tri = np.where(
            use_reverse,
            reverse_ans["primitive_ids"].numpy(),
            forward_ans["primitive_ids"].numpy(),
        )
        s_bary = np.where(
            use_reverse[:, None],
            reverse_ans["primitive_uvs"].numpy(),
            forward_ans["primitive_uvs"].numpy(),
        )

        # Use closest points only to fill genuine ray misses. Never replace a
        # valid directional ray with a lateral nearest-surface lookup; doing so
        # smears colors across sharp edges. Keep the original generous distance
        # for coverage, but only as a fallback when neither ray hit.
        query = o3d.core.Tensor(
            positions.astype(np.float32),
            dtype=o3d.core.Dtype.Float32,
        )
        closest = scan_scene.compute_closest_points(query)
        closest_points = closest["points"].numpy()
        closest_dist = np.linalg.norm(closest_points - positions, axis=1)
        closest_tri = closest["primitive_ids"].numpy()
        closest_bary = closest["primitive_uvs"].numpy()
        closest_valid = np.isfinite(closest_dist) & (
            closest_dist < max_dist * 2.0
        )
        use_closest = (~ray_valid) & closest_valid

        s_tri[use_closest] = closest_tri[use_closest]
        s_bary[use_closest] = closest_bary[use_closest]
        valid = ray_valid | use_closest
        closest_valid_count = int(np.sum(use_closest))

        s_tri = np.clip(s_tri, 0, (len(scan_uvs) // 3) - 1)

        su = s_bary[:, 0]
        sv = s_bary[:, 1]
        sw = 1.0 - su - sv

        suv0 = scan_uvs[s_tri * 3 + 0]
        suv1 = scan_uvs[s_tri * 3 + 1]
        suv2 = scan_uvs[s_tri * 3 + 2]
        final_uvs = (
            sw[:, None] * suv0
            + su[:, None] * suv1
            + sv[:, None] * suv2
        )

        px = np.clip(final_uvs[:, 0], 0.0, 0.999) * (tex_w - 1)
        py = np.clip(1.0 - final_uvs[:, 1], 0.0, 0.999) * (tex_h - 1)
        colors = tex_arr[py.astype(np.int32), px.astype(np.int32)]

        # Store only valid scan hits before repairing the UV islands.
        out_tex[pix_ys[valid], pix_xs[valid]] = colors[valid]
        out_mask[pix_ys, pix_xs] = True
        hit_mask[pix_ys[valid], pix_xs[valid]] = True

        ray_hit_ratio = np.sum(ray_valid) / max(len(ray_valid), 1) * 100
        hit_ratio = np.sum(valid) / max(len(valid), 1) * 100
        print(
            f"      [DEBUG Bake] Part: {part_name}, tris: {T}, "
            f"tex_size: {out_size}x{out_size}, "
            f"Rasterized pixels: {len(valid)}, "
            f"Ray hits: {ray_hit_ratio:.1f}%, "
            f"Final hits: {hit_ratio:.1f}% "
            f"(closest selected: {closest_valid_count}), "
            f"max_dist: {max_dist:.4f}"
        )

        try:
            import cv2

            bgr = (out_tex[..., ::-1] * 255).astype(np.uint8)

            # Inpaint parts with enough genuine scan hits.
            if hit_ratio >= 10.0 and np.any(hit_mask):
                missing_inside = out_mask & (~hit_mask)
                repair = missing_inside | (~out_mask)
                filled = cv2.inpaint(
                    bgr,
                    repair.astype(np.uint8) * 255,
                    5,
                    cv2.INPAINT_TELEA,
                )
                out_tex = filled[..., ::-1].astype(np.float32) / 255.0
            else:
                # Fill sparsely observed parts with a robust color.
                if np.any(hit_mask):
                    part_fallback = np.median(out_tex[hit_mask], axis=0)
                else:
                    part_fallback = fallback_color
                out_tex[out_mask & (~hit_mask)] = part_fallback
                out_tex[~out_mask] = part_fallback
                print(
                    "      [FALLBACK] Too few hits; using robust fill color "
                    f"RGB={np.round(part_fallback * 255).astype(int).tolist()}"
                )
        except Exception:
            # Apply the same fallback when OpenCV is unavailable.
            part_fallback = (
                np.median(out_tex[hit_mask], axis=0)
                if np.any(hit_mask)
                else fallback_color
            )
            out_tex[out_mask & (~hit_mask)] = part_fallback
            out_tex[~out_mask] = part_fallback

        return Image.fromarray(
            (np.clip(out_tex, 0, 1) * 255).astype(np.uint8)
        )

    @staticmethod
    def _write_ground_plane_obj(
        obj_path,
        p0,
        u,
        v,
        uv_min,
        uv_max,
        texture_filename,
    ):
        corners_uv = np.array(
            [
                [uv_min[0], uv_min[1]],
                [uv_max[0], uv_min[1]],
                [uv_max[0], uv_max[1]],
                [uv_min[0], uv_max[1]],
            ],
            dtype=np.float64,
        )

        verts = (
            p0[None, :]
            + corners_uv[:, 0:1] * u[None, :]
            + corners_uv[:, 1:2] * v[None, :]
        )

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

    @staticmethod
    def _rasterize_object_mask_from_points(
        main_obj_pcd,
        plane_basis,
        uv_min,
        scale,
        W,
        H,
        expand_px=16,
        safety_px=12,
    ):
        import cv2

        pts = np.asarray(main_obj_pcd.points)
        _, _, _, u, v = plane_basis

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
        kernel1 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k1, k1),
        )

        mask = cv2.dilate(mask, kernel1, iterations=1)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel1,
            iterations=2,
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8),
            connectivity=8,
        )
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest, 255, 0).astype(np.uint8)

        if safety_px > 0:
            k2 = max(3, safety_px)
            if k2 % 2 == 0:
                k2 += 1
            kernel2 = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (k2, k2),
            )
            mask = cv2.dilate(mask, kernel2, iterations=1)

        return mask

    @staticmethod
    def bake_ground_texture_remove_object_fast(
        ground_pcd,
        plane_basis,
        mesh_scan,
        main_obj_pcd,
        out_path,
        resolution=600,
        max_dist=0.01,
        object_mask_expand_px=16,
        object_mask_safety_px=12,
        inpaint_radius=3,
        soften_alpha_strength=0.85,
    ):
        """Bake the floor texture and fill the removed object's footprint."""
        import cv2

        if not mesh_scan.has_textures() or not mesh_scan.has_triangle_uvs():
            raise RuntimeError("Scan mesh has no textures or UVs.")

        n, d, p0, u, v = plane_basis
        pts = np.asarray(ground_pcd.points)
        uv = np.column_stack([pts @ u, pts @ v])
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        size_world = uv_max - uv_min

        scale = (resolution - 1) / float(max(size_world.max(), 1e-9))
        W = int(np.ceil(size_world[0] * scale)) + 1
        H = int(np.ceil(size_world[1] * scale)) + 1

        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        uu = uv_min[0] + xs.ravel() / scale
        vv = uv_min[1] + ys.ravel() / scale
        world_pts = (
            p0[None, :]
            + uu[:, None] * u[None, :]
            + vv[:, None] * v[None, :]
        ).astype(np.float32)

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(mesh_scan)
        )
        ans = scene.compute_closest_points(
            o3d.core.Tensor(world_pts, dtype=o3d.core.Dtype.Float32)
        )

        s_tri = ans["primitive_ids"].numpy()
        s_bary = ans["primitive_uvs"].numpy()
        s_pts = ans["points"].numpy()

        dists = np.linalg.norm(world_pts - s_pts, axis=1)
        hit_valid = dists < max_dist

        scan_uvs = np.asarray(mesh_scan.triangle_uvs)
        su = s_bary[:, 0]
        sv = s_bary[:, 1]
        sw = 1.0 - su - sv

        uv0 = scan_uvs[s_tri * 3 + 0]
        uv1 = scan_uvs[s_tri * 3 + 1]
        uv2 = scan_uvs[s_tri * 3 + 2]
        final_uvs = (
            sw[:, None] * uv0
            + su[:, None] * uv1
            + sv[:, None] * uv2
        )

        tex_img = np.asarray(mesh_scan.textures[0])
        th, tw = tex_img.shape[:2]

        tx = np.clip(final_uvs[:, 0], 0.0, 0.999) * (tw - 1)
        ty = np.clip(1.0 - final_uvs[:, 1], 0.0, 0.999) * (th - 1)
        colors = tex_img[
            ty.astype(np.int32),
            tx.astype(np.int32),
        ].astype(np.uint8)
        img = colors.reshape(H, W, 3)

        repair = TextureBaker._rasterize_object_mask_from_points(
            main_obj_pcd,
            plane_basis,
            uv_min,
            scale,
            W,
            H,
            expand_px=object_mask_expand_px,
            safety_px=object_mask_safety_px,
        )

        if repair.max() > 0:
            if np.any(~hit_valid):
                invalid = (~hit_valid).reshape(H, W).astype(np.uint8) * 255
                invalid_outside_hole = cv2.bitwise_and(
                    invalid,
                    cv2.bitwise_not(repair),
                )
                if invalid_outside_hole.max() > 0:
                    img = cv2.inpaint(
                        img,
                        invalid_outside_hole,
                        2,
                        cv2.INPAINT_TELEA,
                    )

            ys_idx, xs_idx = np.where(repair > 0)
            y0 = max(0, ys_idx.min() - 64)
            y1 = min(H, ys_idx.max() + 65)
            x0 = max(0, xs_idx.min() - 64)
            x1 = min(W, xs_idx.max() + 65)

            roi_img = img[y0:y1, x0:x1].copy()
            roi_mask = repair[y0:y1, x0:x1]

            # Fill the masked region locally.
            mask = (roi_mask > 0).astype(np.uint8) * 255
            filled_roi = cv2.inpaint(
                roi_img,
                mask,
                inpaint_radius,
                cv2.INPAINT_TELEA,
            )

            # Smooth the transition around the repaired region.
            k1, k2 = 13, 37
            kernel1 = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (k1, k1),
            )
            kernel2 = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (k2, k2),
            )

            transition = cv2.dilate(mask, kernel2, iterations=1)

            gauss = cv2.GaussianBlur(filled_roi, (21, 21), 0)
            smooth = cv2.bilateralFilter(gauss, 9, 35, 35)

            alpha = cv2.GaussianBlur(
                transition.astype(np.float32) / 255.0,
                (31, 31),
                0,
            )
            alpha = np.clip(
                alpha * soften_alpha_strength,
                0.0,
                1.0,
            )[..., None]

            out = (
                filled_roi.astype(np.float32) * (1.0 - alpha)
                + smooth.astype(np.float32) * alpha
            )
            img[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)

        Image.fromarray(img.astype(np.uint8)).save(
            out_path,
            "JPEG",
            quality=95,
        )

        obj_path = os.path.splitext(out_path)[0] + ".obj"
        TextureBaker._write_ground_plane_obj(
            obj_path,
            p0,
            u,
            v,
            uv_min,
            uv_max,
            os.path.basename(out_path),
        )

        return out_path
