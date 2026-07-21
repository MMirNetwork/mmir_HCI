import copy

import numpy as np
import open3d as o3d


class RegResult:
    """Store the result of a point-cloud registration."""

    def __init__(
        self,
        fitness,
        inlier_rmse,
        transformation,
        method="",
        icp_scale=1.0,
        reverse_fitness=None,
    ):
        self.fitness = fitness
        self.reverse_fitness = (
            fitness if reverse_fitness is None else reverse_fitness
        )
        self.inlier_rmse = inlier_rmse
        self.transformation = transformation
        self.method = method
        self.icp_scale = icp_scale  # Additional scale resolved by scaling ICP.

    def symmetric_fitness(self):
        """Compute the harmonic mean of forward and reverse overlap."""
        total = self.fitness + self.reverse_fitness
        if total < 1e-12:
            return 0.0
        return 2.0 * self.fitness * self.reverse_fitness / total

    def score(self):
        """Compute a diagnostic score from symmetric overlap and RMSE."""
        if self.inlier_rmse < 1e-9:
            return 0.0
        return self.symmetric_fitness() / (self.inlier_rmse + 1e-9)

    def __repr__(self):
        return (
            f"[{self.method}] forward={self.fitness:.4f}  "
            f"reverse={self.reverse_fitness:.4f}  "
            f"symmetric={self.symmetric_fitness():.4f}  "
            f"rmse={self.inlier_rmse:.6f}"
        )


class CloudRegistrator:
    """Handle ICP registration of point clouds."""

    @staticmethod
    def rotation_candidates(source, target):
        """Generate all 24 axis-aligned 3D orientations."""
        src_c = np.asarray(source.get_center())
        tgt_c = np.asarray(target.get_center())

        candidates = []
        axes = np.eye(3)
        index = 0
        for x_axis in axes:
            for x_sign in (-1.0, 1.0):
                x = x_axis * x_sign
                for y_axis in axes:
                    if abs(np.dot(x_axis, y_axis)) > 0.5:
                        continue
                    for y_sign in (-1.0, 1.0):
                        y = y_axis * y_sign
                        z = np.cross(x, y)
                        R = np.column_stack((x, y, z))
                        if np.linalg.det(R) < 0.0:
                            continue

                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = tgt_c - R @ src_c
                        candidates.append((T, f"Orientation{index:02d}"))
                        index += 1

        return candidates

    @staticmethod
    def refine_icp(
        source,
        target,
        init_transform,
        voxel_size,
        similarity_scale_bounds=(0.8, 1.25),
    ):
        """Refine an alignment using rigid and bounded similarity ICP."""
        T = init_transform

        # Refine progressively from coarse to fine scales.
        for scale, iters in [(8, 40), (4, 60), (2, 80), (1, 100)]:
            try:
                res = o3d.pipelines.registration.registration_icp(
                    source,
                    target,
                    voxel_size * scale,
                    T,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=iters,
                    ),
                )
                T = res.transformation
            except Exception:
                pass

        rigid_ev = o3d.pipelines.registration.evaluate_registration(
            source,
            target,
            voxel_size,
            T,
        )
        rigid_reverse_ev = o3d.pipelines.registration.evaluate_registration(
            target,
            source,
            voxel_size,
            np.linalg.inv(T),
        )
        rigid_result = RegResult(
            rigid_ev.fitness,
            rigid_ev.inlier_rmse,
            T,
            "ICP",
            icp_scale=1.0,
            reverse_fitness=rigid_reverse_ev.fitness,
        )

        # Resolve small residual scale errors after rigid alignment.
        try:
            scaled = o3d.pipelines.registration.registration_icp(
                source,
                target,
                voxel_size * 2.0,
                T,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    with_scaling=True
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-7,
                    relative_rmse=1e-7,
                    max_iteration=80,
                ),
            )
            scaled_T = scaled.transformation
            icp_scale = float(
                np.cbrt(abs(np.linalg.det(scaled_T[:3, :3])))
            )
            scale_min, scale_max = similarity_scale_bounds

            if scale_min <= icp_scale <= scale_max:
                scaled_ev = o3d.pipelines.registration.evaluate_registration(
                    source,
                    target,
                    voxel_size,
                    scaled_T,
                )
                scaled_reverse_ev = (
                    o3d.pipelines.registration.evaluate_registration(
                        target,
                        source,
                        voxel_size,
                        np.linalg.inv(scaled_T),
                    )
                )
                scaled_result = RegResult(
                    scaled_ev.fitness,
                    scaled_ev.inlier_rmse,
                    scaled_T,
                    "SimilarityICP",
                    icp_scale=icp_scale,
                    reverse_fitness=scaled_reverse_ev.fitness,
                )

                rigid_key = CloudRegistrator._selection_key(rigid_result)
                scaled_key = CloudRegistrator._selection_key(scaled_result)
                if scaled_key > rigid_key:
                    return scaled_result
        except Exception:
            pass

        return rigid_result

    @staticmethod
    def _selection_key(r):
        """Prioritize overlap and use RMSE as the tie-breaker."""
        # Group fitness values before comparing RMSE.
        fitness_bucket = round(r.symmetric_fitness(), 2)
        return (fitness_bucket, -r.inlier_rmse)

    @staticmethod
    def find_best_registration(
        source_down,
        target_down,
        voxel_size,
        similarity_scale_bounds=(0.8, 1.25),
    ):
        """Find the best ICP registration among rotation candidates."""
        threshold = voxel_size * 5
        print("  Generating rotation candidates...")

        raw = []
        for T, name in CloudRegistrator.rotation_candidates(
            source_down,
            target_down,
        ):
            ev = o3d.pipelines.registration.evaluate_registration(
                source_down,
                target_down,
                threshold,
                T,
            )
            raw.append(RegResult(ev.fitness, ev.inlier_rmse, T, name))

        raw.sort(key=CloudRegistrator._selection_key, reverse=True)
        for r in raw:
            print(f"    {r}")

        print("  Refining the strongest candidates with ICP...")
        refined = []
        for cand in raw[:8]:
            r = CloudRegistrator.refine_icp(
                source_down,
                target_down,
                cand.transformation,
                voxel_size,
                similarity_scale_bounds=similarity_scale_bounds,
            )
            r.method = cand.method + "+" + r.method
            refined.append(r)
            print(f"    {r}")

        # Prefer geometric overlap and use RMSE only as a tie-breaker.
        refined.sort(key=CloudRegistrator._selection_key, reverse=True)
        best = refined[0]
        print(f"  Best registration: {best}")

        return best
