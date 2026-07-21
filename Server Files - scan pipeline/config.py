from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """Configure the scanning and registration pipeline."""

    # Required inputs
    scan_obj_path: str
    index_folder: str

    # File paths
    texture_path: Optional[str] = None
    output_folder: str = "output"
    output_mode: str = "textures_only"  # Options: "full" or "textures_only".

    # Scan processing
    manual_crop: bool = False
    cropped_scan_path: Optional[str] = None
    # Disable this for scans that already contain only the object. When False,
    # no RANSAC ground detection/removal or ground-texture baking is performed.
    remove_scan_ground: bool = True
    ground_dist_threshold_ratio: float = 0.005
    radial_trim_target_ratio: Optional[float] = None

    # Registration
    voxel_ratio: float = 0.04
    top_k_candidates: int = 3
    digital_unit_scale: float = 1.0
    scan_up_axis: str = "Y"
    digital_up_axis: str = "Y"
    remove_detached_digital_components: bool = False
    similarity_scale_min: float = 0.8
    similarity_scale_max: float = 1.25

    # Texture settings
    texture_projection_distance_voxels: float = 2.0
    tex_min_size: int = 256
    tex_max_size: int = 512
    tex_subdir: str = "textures"
    make_zip: bool = True
    use_original_uvs: bool = True

    # Debug and visualization
    bake_debug_grid: bool = False
    show_uv_grid: bool = False
    show_visualizations: bool = True
