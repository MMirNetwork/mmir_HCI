import copy

import numpy as np
import open3d as o3d


class ScanProcessor:
    @staticmethod
    def find_ground_plane(
        pcd,
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=600,
    ):
        """Find the ground plane in a point cloud using RANSAC."""
        work = pcd
        best = None

        for _ in range(3):
            plane_model, inliers = work.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
            )

            normal = np.array(plane_model[:3], dtype=np.float64)
            normal /= np.linalg.norm(normal) + 1e-12

            vert_score = max(abs(normal[1]), abs(normal[2]))
            if vert_score > 0.85 or best is None:
                best = (plane_model, inliers, work)
                if vert_score > 0.85:
                    break

            work = work.select_by_index(inliers, invert=True)

        return best

    @staticmethod
    def keep_main_object(
        pcd,
        eps=None,
        min_points=40,
        radial_trim_target_ratio=None,
    ):
        """Extract the main object as a robust merged cluster set."""
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return pcd

        diag = np.linalg.norm(pts.max(0) - pts.min(0))
        if eps is None:
            # Use a wider radius to keep the object connected.
            eps = max(diag * 0.02, 1e-4)

        labels = np.array(
            pcd.cluster_dbscan(
                eps=eps,
                min_points=min_points,
                print_progress=False,
            )
        )
        if labels.max() < 0:
            print("  WARN: No cluster found, keeping original point cloud.")
            return pcd

        sizes = np.bincount(labels[labels >= 0])
        core_lbl = int(np.argmax(sizes))
        core_c = pts[labels == core_lbl].mean(axis=0)

        # Merge clusters near the largest cluster.
        merge_radius = diag * 0.35
        keep = {core_lbl}
        for lbl in range(len(sizes)):
            if lbl == core_lbl or sizes[lbl] == 0:
                continue
            c = pts[labels == lbl].mean(axis=0)
            if np.linalg.norm(c - core_c) <= merge_radius:
                keep.add(lbl)

        idx = np.where(np.isin(labels, list(keep)))[0]
        print(
            f"  DBSCAN: {len(sizes)} clusters, kept {len(keep)} "
            f"({len(idx):,} of {len(pts):,} points)"
        )
        main = pcd.select_by_index(idx)

        # Apply optional dataset-specific radial trimming.
        if radial_trim_target_ratio is not None:
            main = ScanProcessor.trim_radial_outliers(
                main,
                target_ratio=radial_trim_target_ratio,
            )
        return main

    @staticmethod
    def trim_radial_outliers(
        pcd,
        target_ratio,
        max_frac_removed=0.15,
        step_pct=0.5,
        min_pct=80.0,
    ):
        """Trim radial outliers until the target scale ratio is reached."""
        pts = np.asarray(pcd.points)
        n0 = len(pts)
        if n0 < 100:
            return pcd

        def ratio_of(mask):
            p = pts[mask]
            c = p.mean(axis=0)
            d = np.linalg.norm(p - c, axis=1)
            med = float(np.median(d))
            if med <= 1e-9:
                return 0.0, med
            aabb = float(np.linalg.norm(p.max(0) - p.min(0)))
            return aabb / med, med

        mask = np.ones(n0, dtype=bool)
        r0, _ = ratio_of(mask)
        if r0 <= target_ratio:
            print(
                f"  RADIAL TRIM: ratio={r0:.2f} already <= "
                f"{target_ratio} (no trim)"
            )
            return pcd

        pct = 100.0
        while pct > min_pct:
            c = pts[mask].mean(axis=0)
            d_all = np.linalg.norm(pts - c, axis=1)
            thr = np.percentile(d_all[mask], pct)
            cand = mask & (d_all <= thr)
            removed_frac = 1.0 - cand.sum() / n0
            if removed_frac > max_frac_removed:
                break
            mask = cand
            r, _ = ratio_of(mask)
            if r <= target_ratio:
                break
            pct -= step_pct

        r_final, med_final = ratio_of(mask)
        kept = int(mask.sum())
        print(
            f"  RADIAL TRIM: ratio {r0:.2f} -> {r_final:.2f} "
            f"(target {target_ratio}), kept {kept:,}/{n0:,} "
            f"({100.0 * kept / n0:.1f}%), median_radius={med_final:.4f}"
        )
        return pcd.select_by_index(np.where(mask)[0])

    @staticmethod
    def plane_basis_from_model(plane_model):
        """Compute basis vectors and the normal from a plane model."""
        n = np.array(plane_model[:3], dtype=np.float64)
        n /= np.linalg.norm(n) + 1e-12
        d = float(plane_model[3])

        helper = (
            np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(n[0]) < 0.9
            else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        )
        u = np.cross(n, helper)
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(n, u)
        p0 = -d * n

        return n, d, p0, u, v

    @staticmethod
    def normalize_orientation(pcd, up_axis="Y"):
        """Align a point cloud with the specified up axis."""
        pts = np.asarray(pcd.points)
        center = pts.mean(axis=0)
        R = np.eye(3)
        if up_axis == "Z":
            R = np.array(
                [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                dtype=float,
            )
        elif up_axis == "-Z":
            R = np.array(
                [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                dtype=float,
            )
        elif up_axis == "X":
            R = np.array(
                [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                dtype=float,
            )
        elif up_axis == "-Y":
            R = np.array(
                [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                dtype=float,
            )

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ center

        pcd_out = copy.deepcopy(pcd)
        pcd_out.transform(T)
        return pcd_out, T

    @staticmethod
    def remove_outliers(pcd, nb=20, std=2.0):
        """Remove statistical outliers from a point cloud."""
        cl, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb,
            std_ratio=std,
        )
        return cl

    @staticmethod
    def remove_detached_components(
        pcd,
        eps=None,
        min_ratio=0.15,
        verbose=True,
    ):
        """Remove a spatially detached side object from a point cloud."""
        pts = np.asarray(pcd.points)
        if len(pts) < 50:
            return pcd

        # Find a clear axis-aligned gap separating a compact side object.
        total_e = pts.max(0) - pts.min(0)
        total_vol = float(np.prod(np.maximum(total_e, 1e-6)))
        best = None
        for axis in range(3):
            v = np.sort(pts[:, axis])
            axis_range = v[-1] - v[0]
            if axis_range <= 0:
                continue
            gaps = np.diff(v)
            gi = int(np.argmax(gaps))
            gap_w = float(gaps[gi])
            if gap_w < axis_range * 0.05:
                continue
            split = (v[gi] + v[gi + 1]) / 2.0
            n_low = int((pts[:, axis] < split).sum())
            n_high = len(pts) - n_low
            if n_low == 0 or n_high == 0:
                continue
            keep_low = n_low >= n_high
            minor = (
                pts[pts[:, axis] >= split]
                if keep_low
                else pts[pts[:, axis] < split]
            )
            if len(minor) < len(pts) * 0.02:
                continue
            e = minor.max(0) - minor.min(0)
            minor_volfrac = (
                float(np.prod(np.maximum(e, 1e-6))) / total_vol
            )
            score = (minor_volfrac, -gap_w)
            if best is None or score < best[0]:
                best = (score, axis, split, keep_low, len(minor))

        if best is None:
            return pcd

        _, axis, split, keep_low, minor_n = best
        keep_mask = (
            pts[:, axis] < split
            if keep_low
            else pts[:, axis] >= split
        )
        kept = int(keep_mask.sum())
        if verbose and kept < len(pts):
            print(
                "  [Digital] Removed detached component: "
                f"kept {kept} of {len(pts)} points "
                f"(gap split on axis {'XYZ'[axis]} @ {split:.1f}, "
                f"removed={minor_n} pts)."
            )
        return pcd.select_by_index(np.where(keep_mask)[0])

    @staticmethod
    def remove_points_near_plane(pcd, plane_model, distance):
        """Remove points within a specified distance of a plane."""
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return pcd
        a, b, c, d = [float(x) for x in plane_model]
        n = np.array([a, b, c], dtype=np.float64)
        nn = np.linalg.norm(n) + 1e-12
        signed = (pts @ n + d) / nn
        keep_mask = np.abs(signed) > distance
        kept = int(keep_mask.sum())
        if kept < len(pts) * 0.3:
            return pcd
        if kept < len(pts):
            print(
                "  [Scan] Removed ground remnants near plane: "
                f"kept {kept} of {len(pts)} points "
                f"(|dist| > {distance:.4f})."
            )
        return pcd.select_by_index(np.where(keep_mask)[0])

    @staticmethod
    def compute_scale_metric(pcd, method="robust_extent"):
        """Compute the selected point-cloud scale metric."""
        pts = np.asarray(pcd.points)
        if method == "robust_extent":
            lo = np.percentile(pts, 2.5, axis=0)
            hi = np.percentile(pts, 97.5, axis=0)
            return float(np.linalg.norm(hi - lo))
        elif method == "aabb_diag":
            bb = pcd.get_axis_aligned_bounding_box()
            return float(np.linalg.norm(bb.get_extent()))
        elif method == "median_radius":
            c = pts.mean(axis=0)
            d = np.linalg.norm(pts - c, axis=1)
            return float(np.median(d))
        else:
            raise ValueError(method)

    @staticmethod
    def preprocess_point_cloud(pcd, voxel_size):
        """Downsample a point cloud and estimate its normals."""
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down = ScanProcessor.remove_outliers(pcd_down)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 4,
                max_nn=30,
            )
        )
        pcd_down.orient_normals_consistent_tangent_plane(30)
        return pcd_down
