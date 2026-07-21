import glob
import json
import os

import numpy as np
import open3d as o3d

from src.scan.scan_processing import ScanProcessor


class DescriptorExtractor:
    """Compute point-cloud descriptors and rank matching digital models."""

    @staticmethod
    def _d2_hist(
        pcd,
        n_samples=4000,
        n_pairs=200_000,
        n_bins=64,
        max_ratio=4.0,
    ):
        pts = np.asarray(pcd.points)
        if len(pts) > n_samples:
            idx = np.random.default_rng(0).choice(
                len(pts),
                n_samples,
                replace=False,
            )
            pts = pts[idx]

        rng = np.random.default_rng(1)
        i = rng.integers(0, len(pts), n_pairs)
        j = rng.integers(0, len(pts), n_pairs)
        mask = i != j
        i, j = i[mask], j[mask]

        d = np.linalg.norm(pts[i] - pts[j], axis=1)
        med = np.median(d)
        if med < 1e-12:
            return np.zeros(n_bins)

        hist, _ = np.histogram(
            d / med,
            bins=n_bins,
            range=(0.0, max_ratio),
            density=True,
        )
        s = hist.sum()
        return hist / s if s > 0 else hist

    @staticmethod
    def compute_volume(pcd):
        try:
            hull, _ = pcd.compute_convex_hull()
            if not hull.is_watertight():
                return 0.0
            return float(hull.get_volume())
        except Exception:
            return 0.0

    @staticmethod
    def compute_edge_score(pcd):
        try:
            _, idx = pcd.compute_convex_hull()
            return float(len(idx)) / len(pcd.points)
        except Exception:
            return 0.0

    @staticmethod
    def compute_scan_descriptors(pcd):
        """Compute shape descriptors for a point cloud."""
        pts = np.asarray(pcd.points)
        ext = np.sort(
            np.asarray(
                pcd.get_axis_aligned_bounding_box().get_extent(),
                float,
            )
        )[::-1]
        aabb_n = (
            (ext / ext.max()).tolist()
            if ext.max() > 1e-12
            else [1.0, 0.0, 0.0]
        )

        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        w = np.sort(np.linalg.eigvalsh(cov))[::-1]
        w = np.clip(w, 1e-18, None)

        try:
            hull, _ = pcd.compute_convex_hull()
            hull.compute_vertex_normals()
            a = hull.get_surface_area()
            v = hull.get_volume() if hull.is_watertight() else 0.0
            hull_c = float(v / (a**1.5)) if a > 1e-12 else 0.0
        except Exception:
            hull_c = 0.0

        return {
            "size_median_radius": ScanProcessor.compute_scale_metric(
                pcd,
                "median_radius",
            ),
            "aabb_extent_sorted_norm": np.array(aabb_n),
            "pca_eigenvalue_ratios": np.array(
                [float(w[1] / w[0]), float(w[2] / w[0])]
            ),
            "convex_hull_compactness": hull_c,
            "d2_histogram": DescriptorExtractor._d2_hist(pcd),
            "volume": DescriptorExtractor.compute_volume(pcd),
            "edge_score": DescriptorExtractor.compute_edge_score(pcd),
        }

    @staticmethod
    def descriptor_distance(scan_desc, idx_data):
        """Compute the weighted distance between two descriptor sets."""
        if idx_data.get("variant") == "vol_edge":
            vol_dist = abs(scan_desc["volume"] - float(idx_data["volume"]))
            edge_dist = abs(
                scan_desc["edge_score"] - float(idx_data["edge_score"])
            )
            return float(vol_dist + edge_dist * 100.0)

        w = {
            "d2": 3.0,
            "pca": 1.0,
            "aabb": 1.0,
            "hull": 0.5,
        }
        a = scan_desc["d2_histogram"]
        b = np.asarray(idx_data["d2_histogram"])

        d2_dist = 0.5 * np.sum(((a - b) ** 2) / (a + b + 1e-12))
        pca_dist = np.linalg.norm(
            scan_desc["pca_eigenvalue_ratios"]
            - np.asarray(idx_data["pca_eigenvalue_ratios"])
        )
        aabb_dist = np.linalg.norm(
            scan_desc["aabb_extent_sorted_norm"]
            - np.asarray(idx_data["aabb_extent_sorted_norm"])
        )
        hull_dist = abs(
            scan_desc["convex_hull_compactness"]
            - float(idx_data["convex_hull_compactness"])
        )
        return float(
            w["d2"] * d2_dist
            + w["pca"] * pca_dist
            + w["aabb"] * aabb_dist
            + w["hull"] * hull_dist
        )

    @staticmethod
    def get_top_candidates(pcd_scan, index_folder, top_k=5):
        """Return the best matching models from the central index."""
        scan_desc = DescriptorExtractor.compute_scan_descriptors(pcd_scan)

        index_path = os.path.join(index_folder, "index.json")
        if not os.path.exists(index_path):
            raise SystemExit(
                f"Index database not found at {index_path}. "
                "Please wait for auto-indexer."
            )

        with open(index_path, "r") as fp:
            db = json.load(fp)

        entries = list(db.values())
        if not entries:
            raise SystemExit(f"No entries in index database {index_path}.")

        ranked = sorted(
            entries,
            key=lambda d: DescriptorExtractor.descriptor_distance(scan_desc, d),
        )

        print("\nTop candidates (pre-ranking):")
        out = []
        for i, d in enumerate(ranked[:top_k]):
            name = os.path.splitext(d["source_file"])[0]

            # Normalize legacy PLY entries to their OBJ counterparts.
            src = d["source_file"]
            if not src.lower().endswith(".obj"):
                src = os.path.splitext(src)[0] + ".obj"

            obj_path = os.path.join(index_folder, src)
            dist = DescriptorExtractor.descriptor_distance(scan_desc, d)
            print(f"  #{i + 1}  {d['source_file']}   dist={dist:.3f}")
            out.append((name, obj_path))

        return out
