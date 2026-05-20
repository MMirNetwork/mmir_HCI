import os, json, argparse, glob
import numpy as np
import open3d as o3d


def load_point_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        return o3d.io.read_point_cloud(path)
  
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) == 0:
        raise ValueError(f"Kein Mesh / keine Punkte in {path}")
    return mesh.sample_points_uniformly(number_of_points=50000)


def compute_size_median_radius(pcd):
    pts = np.asarray(pcd.points)
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    return float(np.median(d))


def compute_aabb_extent_sorted_norm(pcd):
    ext = pcd.get_axis_aligned_bounding_box().get_extent()
    ext = np.sort(np.asarray(ext, dtype=float))[::-1] 
    m = ext.max()
    if m < 1e-12:
        return [1.0, 0.0, 0.0]
    return (ext / m).tolist()


def compute_pca_eigenvalue_ratios(pcd):
    pts = np.asarray(pcd.points)
    pts = pts - pts.mean(axis=0)
    cov = np.cov(pts.T)
    w = np.linalg.eigvalsh(cov)
    w = np.sort(w)[::-1]
    w = np.clip(w, 1e-18, None)
    return [float(w[1] / w[0]), float(w[2] / w[0])]


def compute_convex_hull_compactness(pcd):
    try:
        hull, _ = pcd.compute_convex_hull()
        hull.compute_vertex_normals()
        v = hull.get_volume()
        a = hull.get_surface_area()
        if a < 1e-12:
            return 0.0
        return float(v / (a ** 1.5))
    except Exception:
        return 0.0


def compute_d2_histogram(pcd, n_samples=4000, n_pairs=200_000, n_bins=64, max_ratio=4.0):
    pts = np.asarray(pcd.points)
    if len(pts) > n_samples:
        idx = np.random.default_rng(0).choice(len(pts), n_samples, replace=False)
        pts = pts[idx]
    rng = np.random.default_rng(1)
    i = rng.integers(0, len(pts), n_pairs)
    j = rng.integers(0, len(pts), n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    d = np.linalg.norm(pts[i] - pts[j], axis=1)
    med = np.median(d)
    if med < 1e-12:
        return [0.0] * n_bins
    d_norm = d / med
    hist, _ = np.histogram(d_norm, bins=n_bins, range=(0.0, max_ratio), density=True)
    s = hist.sum()
    if s > 0:
        hist = hist / s  
    return hist.tolist()


def compute_fpfh_sample(pcd, n_points=512):
    size = compute_size_median_radius(pcd)
    if size < 1e-12:
        return None, None, None
    p = o3d.geometry.PointCloud(pcd)
    p.scale(1.0 / size, center=np.zeros(3))

    voxel = 0.02 
    p_down = p.voxel_down_sample(voxel)
    p_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        p_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 10, max_nn=100),
    )
    pts = np.asarray(p_down.points)
    feats = np.asarray(fpfh.data).T 

    if len(pts) > n_points:
        idx = np.random.default_rng(42).choice(len(pts), n_points, replace=False)
        pts = pts[idx]
        feats = feats[idx]

    return pts.tolist(), feats.tolist(), voxel


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


def index_file(path, force=False, variant="full"):
    out_path = os.path.splitext(path)[0] + ".index.json"
    if os.path.exists(out_path) and not force:
        print(f"  [skip] {os.path.basename(out_path)} existiert (use --force)")
        return

    print(f"  indexing {os.path.basename(path)} ...")
    pcd = load_point_cloud(path)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if variant == "full":
        data = {
            "source_file": os.path.basename(path),
            "variant": "full",
            "num_points": int(len(pcd.points)),
            "size_median_radius": compute_size_median_radius(pcd),
            "aabb_extent_sorted_norm": compute_aabb_extent_sorted_norm(pcd),
            "pca_eigenvalue_ratios": compute_pca_eigenvalue_ratios(pcd),
            "convex_hull_compactness": compute_convex_hull_compactness(pcd),
            "d2_histogram": compute_d2_histogram(pcd),
        }
        pts, feats, voxel = compute_fpfh_sample(pcd)
        if pts is not None:
            data["fpfh_sample_points"] = pts
            data["fpfh_sample"] = feats
            data["normalized_voxel"] = voxel
    else:
        data = {
            "source_file": os.path.basename(path),
            "variant": "vol_edge",
            "volume": compute_volume(pcd),
            "edge_score": compute_edge_score(pcd)
        }

    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"    -> {os.path.basename(out_path)}  ({len(pcd.points):,} pts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Ordner mit digitalen Modellen (.ply / .obj)")
    ap.add_argument("--ext", default=".ply", help="Dateiendung der Quelldateien (default .ply)")
    ap.add_argument("--force", action="store_true", help="Vorhandene Indizes überschreiben")
    ap.add_argument("--variant", choices=["full", "vol_edge"], default="full")
    args = ap.parse_args()

    pattern = os.path.join(args.folder, f"*{args.ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Keine {args.ext}-Dateien in {args.folder}")
        return

    print(f"Indexiere {len(files)} Dateien aus {args.folder} mit Variante: {args.variant}")
    for p in files:
        try:
            index_file(p, force=args.force, variant=args.variant)
        except Exception as e:
            print(f"  [FEHLER] {p}: {e}")
    print("Fertig.")


if __name__ == "__main__":
    main()