import glob
import json
import os

import numpy as np

from src.scan.descriptors import DescriptorExtractor
from src.scan.io_utils import MeshIO


class DatabaseIndexer:
    @staticmethod
    def _load_or_create_index(index_path):
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _save_index(index_path, data):
        with open(index_path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def index_folder(folder_path, force=False, digital_unit_scale=1.0):
        if digital_unit_scale <= 0:
            raise ValueError("digital_unit_scale must be greater than zero.")

        index_path = os.path.join(folder_path, "index.json")
        db = DatabaseIndexer._load_or_create_index(index_path)

        # Index OBJ files using in-memory point-cloud samples.
        obj_files = set(glob.glob(os.path.join(folder_path, "*.obj")))
        files_to_process = sorted(obj_files)

        updated = False
        for filepath in files_to_process:
            filename = os.path.basename(filepath)

            indexed_scale = db.get(filename, {}).get("digital_unit_scale")
            if (
                not force
                and filename in db
                and indexed_scale is not None
                and np.isclose(float(indexed_scale), digital_unit_scale)
            ):
                continue

            print(f"Indexing new model: {filename}")
            try:
                # Sample the OBJ mesh without persisting a point cloud.
                mesh = MeshIO.load_mesh(filepath)
                pcd = mesh.sample_points_uniformly(number_of_points=50000)

                # Convert the model into the configured working unit.
                pcd.scale(digital_unit_scale, center=np.zeros(3))

                pcd, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=20,
                    std_ratio=2.0,
                )

                # Compute model descriptors.
                desc = DescriptorExtractor.compute_scan_descriptors(pcd)

                # Convert arrays for JSON serialization.
                desc_json = {}
                for k, v in desc.items():
                    if isinstance(v, np.ndarray):
                        desc_json[k] = v.tolist()
                    else:
                        desc_json[k] = v

                desc_json["source_file"] = filename
                desc_json["digital_unit_scale"] = digital_unit_scale

                db[filename] = desc_json
                updated = True
                print(f"  -> Success: {filename}")
            except Exception as e:
                print(f"  [ERROR] Failed to index {filename}: {e}")

        if updated:
            DatabaseIndexer._save_index(index_path, db)
            print(f"Updated index saved to {index_path}")
