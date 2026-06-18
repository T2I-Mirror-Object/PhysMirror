from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class SceneConfig:
    """Holds all physical parameters for the scene."""
    # Object placement
    gap: float = 0.1
    min_angle: float = -np.pi/4
    max_angle: float = np.pi/4

    # Rotation
    object_base_rotation: float = 0.0  # Initial Y-axis rotation in degrees, applied before randomness
    include_object_random_rotation: bool = False

    # Object scaling
    object_scale: float = 1.0
    
    # Mirror dimensions
    mirror_thickness: float = 0.1
    mirror_gap_side: float = 2.0
    mirror_gap_top: float = 2.0
    mirror_gap_ahead: float = 3.0
    mirror_height: Optional[float] = None  # When set, fixes the mirror height and auto-scales objects to fit
    
    # Room dimensions
    room_width: float = 20.0
    room_depth: float = 20.0
    wall_height: float = 10.0
    wall_thickness: float = 0.5
    paint_floor: bool = False
    paint_walls: bool = False
    paint_mirror_wall: bool = False

    camera_method: str = "random_side"  # Options: 'random_side', 'fixed_front'
    camera_dist_multiplier: float = 1.2  # Changed from hardcoded 1.5. Lower = Closer.
    camera_elevation: float = 10.0       # Look slightly down (degrees). 0 is perfectly horizontal.
    camera_look_at_height: float = 0.8   # Lift the camera target (e.g., 0.8m off the floor)
    camera_azim_min: float = 20.0 
    camera_azim_max: float = 25.0
    camera_azim: float = 22.5
    inference_mode: bool = True
    
    renderer_type: str = "depth"        # Options: 'depth', 'segmentation'
    render_size: int = 512

    include_floor: bool = True
    include_walls: bool = True
    include_mirror_surface: bool = False  # The glass itself
    include_mirror_frame: bool = True
    include_mirror_wall: bool = False