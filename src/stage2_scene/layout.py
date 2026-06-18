import torch
import random
import numpy as np
from typing import List, Tuple
from pytorch3d.structures import Meshes

from .config import SceneConfig
from .utils import MeshUtils

class LayoutEngine:
    def __init__(self, config: SceneConfig, device: str = "cuda"):
        self.cfg = config
        self.device = torch.device(device)

    def arrange(self, meshes: List[Meshes]) -> List[Meshes]:
        """
        Unified logic:
        1. Ground everything.
        2. Rotate everything.
        3. Arrange (works for 1 or N).
        """
        # 1. Scale objects
        if self.cfg.object_scale != 1.0:
            meshes = [MeshUtils.scale(m, self.cfg.object_scale) for m in meshes]

        # 2. Place on floor (Y=0)
        meshes = [self._ground_object(m) for m in meshes]

        # 2b. Auto-scale to match fixed mirror_height (if set)
        if self.cfg.mirror_height is not None:
            _, group_h, _ = MeshUtils.get_group_dims(meshes)
            available_h = self.cfg.mirror_height - self.cfg.mirror_gap_top
            if group_h > 0 and available_h > 0:
                auto_scale = available_h / group_h
                meshes = [MeshUtils.scale(m, auto_scale) for m in meshes]
                meshes = [self._ground_object(m) for m in meshes]
                print(f"[Layout] mirror_height={self.cfg.mirror_height}: auto-scaled objects by {auto_scale:.3f}x (natural_h={group_h:.3f} -> target_h={available_h:.3f})")

        # 3. Base rotation (fixed initial angle)
        if self.cfg.object_base_rotation != 0.0:
            base_rad = np.radians(self.cfg.object_base_rotation)
            meshes = [MeshUtils.rotate_y(m, base_rad, self.device) for m in meshes]

        # 4. Random Rotation (added on top of base rotation)
        if self.cfg.include_object_random_rotation:
            meshes = self._random_rotate(meshes)

        # 5. Horizontal Arrangement & Centering
        # We change logic here: ALWAYS run the arrange logic.
        meshes = self._arrange_horizontally(meshes)
        
        return meshes

    def _ground_object(self, mesh: Meshes) -> Meshes:
        """Shifts object so its bottom touches y=0."""
        bounds = MeshUtils.get_bounds(mesh)
        min_y = bounds[0, 1]
        translation = torch.tensor([0.0, -min_y, 0.0], device=self.device)
        return MeshUtils.translate(mesh, translation)

    def _random_rotate(self, meshes: List[Meshes]) -> List[Meshes]:
        """Applies random Y-rotation to a list of meshes."""
        processed = []
        for mesh in meshes:
            angle = random.uniform(self.cfg.min_angle, self.cfg.max_angle)
            processed.append(MeshUtils.rotate_y(mesh, angle, self.device))
        return processed

    def _arrange_horizontally(self, meshes: List[Meshes]) -> List[Meshes]:
        """
        This function now handles N=1 safely.
        """
        # If only 1 object, this list starts with it.
        positioned = [meshes[0]]
        prev_mesh = meshes[0]

        # If len is 1, this loop simply DOES NOT RUN (0 iterations).
        # Logic remains safe!
        for mesh in meshes[1:]:
            prev_bounds = MeshUtils.get_bounds(prev_mesh)
            curr_bounds = MeshUtils.get_bounds(mesh)
        
            shift_x = prev_bounds[1, 0] - curr_bounds[0, 0] + self.cfg.gap
            translation = torch.tensor([shift_x, 0, 0], device=self.device)
        
            new_mesh = MeshUtils.translate(mesh, translation)
            positioned.append(new_mesh)
            prev_mesh = new_mesh
        
        # Finally, CENTER the whole group.
        # If N=1, it centers that single object perfectly at X=0.
        return self._center_group(positioned)

    def _center_group(self, meshes: List[Meshes]) -> List[Meshes]:
        """Shifts the whole group so the center is at X=0."""
        first_min_x = MeshUtils.get_bounds(meshes[0])[0, 0]
        last_max_x = MeshUtils.get_bounds(meshes[-1])[1, 0]
        
        center_x = (first_min_x + last_max_x) / 2.0
        translation = torch.tensor([-center_x, 0, 0], device=self.device)
        
        return [MeshUtils.translate(m, translation) for m in meshes]

    def calculate_mirror_dims(self, meshes: List[Meshes]):
        """
        Calculates how big the mirror should be to fit all objects.
        Returns: (frame_width, frame_height)
        """
        # Get the size of the group of objects
        group_w, group_h, _ = MeshUtils.get_group_dims(meshes)
        
        # Add the gaps from config
        frame_width = group_w + (self.cfg.mirror_gap_side * 2)
        frame_height = group_h + (self.cfg.mirror_gap_top) # Assuming bottom is floor
        
        # Ensure minimum size (optional)
        frame_width = max(frame_width, 2.0)
        frame_height = max(frame_height, 2.0)
        
        return frame_width, frame_height

    def position_mirror(self, frame: Meshes, z_pos: float) -> Meshes:
        """
        Moves the created mirror props to the correct location (behind objects).
        """
        bounds = MeshUtils.get_bounds(frame)
        min_y = bounds[0, 1] # Assumes bounds is [[min_x, min_y, min_z], [max_x...]]
        
        # We need to move up by exactly the amount it is underground
        y_shift = -min_y

        # 2. Apply Transformation
        # We apply the same Y-shift to both Frame and Surface so they stay attached.
        translation = torch.tensor([0.0, y_shift, z_pos], device=self.device)
        
        frame = MeshUtils.translate(frame, translation)
        
        return frame