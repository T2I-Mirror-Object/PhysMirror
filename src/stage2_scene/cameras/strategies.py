import torch
import random
import math
from typing import Tuple
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.structures import Meshes

from .base import BaseCameraStrategy

class RandomSideStrategy(BaseCameraStrategy):
    """
    Implements the user's logic:
    - Distance: 1.5x the mirror gap
    - Azimuth: Randomly chooses Left (-25 to -20) or Right (+20 to +25)
    - Elevation: Default 0.0 (Eye level)
    """
    
    def calculate_pose(self, scene_meshes: Meshes) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Calculate Distance
        # "camera_distance = mirror_gap_ahead * 1.5"
        dist = self.cfg.mirror_gap_ahead * self.cfg.camera_dist_multiplier
        
        # 2. Calculate Azimuth
        if self.cfg.inference_mode:
            # The Random Logic
            base_azimuth = 0.0
            min_angle = self.cfg.camera_azim_min
            max_angle = self.cfg.camera_azim_max
            
            # 50% chance for left range, 50% for right range
            if random.random() < 0.5:
                # Left side: [-25, -20]
                azim = random.uniform(base_azimuth - max_angle, base_azimuth - min_angle)
            else:
                # Right side: [+20, +25]
                azim = random.uniform(base_azimuth + min_angle, base_azimuth + max_angle)
        else:
            azim = self.cfg.camera_azim
            
        # 3. Elevation (Assuming 0.0 unless specified in config)
        elev = self.cfg.camera_elevation

        # 4. Target Position (Tripod Height)
        at = torch.tensor(
            [[0.0, self.cfg.camera_look_at_height, 0.0]], 
            device=self.device
        )
        
        print(f"[Camera] Selected Pose: Dist={dist:.2f}, Elev={elev:.2f}, Azim={azim:.2f}, At={at}")

        # 4. Convert Spherical -> Cartesian (R, T)
        # This PyTorch3D helper function does the hard math for you.
        R, T = look_at_view_transform(
            dist=dist, 
            elev=elev, 
            azim=azim,
            at=at,
            device=self.device
        )
        
        return R, T


class FixedFrontalStrategy(BaseCameraStrategy):
    """
    A simple strategy for debugging. 
    Always looks straight at the scene from the front.
    """
    def calculate_pose(self, scene_meshes: Meshes) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.cfg.mirror_gap_ahead * self.cfg.camera_dist_multiplier
        elev = self.cfg.camera_elevation
        azim = 0.0 # Perfectly centered
        
        R, T = look_at_view_transform(
            dist=dist, 
            elev=elev, 
            azim=azim, 
            device=self.device
        )
        return R, T