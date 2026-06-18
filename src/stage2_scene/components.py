import torch
import trimesh
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from typing import List

class SceneComponents:
    def __init__(self, device="cuda"):
        self.device = device

    def create_mirror_frame(self, outer_width, outer_height, thickness, depth=0.1) -> Meshes:
        """
        Creates a hollow frame using Trimesh subtraction.
        Much easier than manually placing 4 bars!
        """
        # 1. Create Outer Box
        outer = trimesh.creation.box(extents=[outer_width, outer_height, depth])
        
        # 2. Create Inner Box (Hole)
        inner_w = outer_width - (2 * thickness)
        inner_h = outer_height - (2 * thickness)
        # Make inner depth slightly larger to ensure clean cut
        inner = trimesh.creation.box(extents=[inner_w, inner_h, depth * 1.2])
        
        # 3. Boolean Subtraction: Outer - Inner
        frame_geom = outer.difference(inner)
        
        # 4. Convert to Pytorch3D
        return self._to_pytorch3d(frame_geom, color=[0.4, 0.4, 0.4]) # Dark grey frame

    def create_floor(self, width, depth, y_pos=0.0) -> Meshes:
        """Creates a simple floor plane."""
        floor_geom = trimesh.creation.box(extents=[width, 0.1, depth])
        floor_geom.apply_translation([0, y_pos - 0.05, 0]) # Shift down so top is at y_pos
        return self._to_pytorch3d(floor_geom, color=[0.8, 0.8, 0.8])

    def create_mirror_wall(self, wall_width, wall_height, hole_width, hole_height, depth=0.1) -> Meshes:
        """
        Creates a wall with a rectangular hole where the mirror is.
        The hole matches the mirror frame's inner opening so reflections remain visible.

        Both wall and mirror frame are created centered at Y=0, then position_mirror
        grounds them (shifts bottom to Y=0). So in local space the hole must be at
        the same Y as where the mirror frame center sits relative to the wall center.
        Wall center is at Y=0, frame center is at Y=0, but frame is shorter.
        After grounding, the frame bottom aligns with wall bottom. So in local space
        the hole center needs to be offset downward by: (wall_height - hole_height) / 2
        """
        # 1. Full wall
        wall = trimesh.creation.box(extents=[wall_width, wall_height, depth])

        # 2. Hole matching the mirror's inner area, shifted to bottom of wall
        hole = trimesh.creation.box(extents=[hole_width, hole_height, depth * 1.2])
        y_offset = -(wall_height - hole_height) / 2.0
        hole.apply_translation([0, y_offset, 0])

        # 3. Boolean subtraction
        wall_with_hole = wall.difference(hole)

        return self._to_pytorch3d(wall_with_hole, color=[0.75, 0.75, 0.75])

    def create_mirror_surface(self, width, height, thickness=0.01) -> Meshes:
        """The reflective glass part."""
        # Create a thin box
        glass = trimesh.creation.box(extents=[width, height, thickness])
        # Color it light blue-ish to distinguish (optional, texture overwrites this)
        return self._to_pytorch3d(glass, color=[0.9, 0.9, 1.0])

    def create_room_walls(self, width, depth, height, thickness=0.5) -> List[Meshes]:
        """
        Creates 4 walls automatically.
        """
        walls_tri = []
        
        # 1. Back Wall (-Z)
        back = trimesh.creation.box(extents=[width, height, thickness])
        back.apply_translation([0, height/2, -depth/2 - thickness/2])
        walls_tri.append(back)
        
        # 2. Front Wall (+Z)
        front = trimesh.creation.box(extents=[width, height, thickness])
        front.apply_translation([0, height/2, depth/2 + thickness/2])
        walls_tri.append(front)
        
        # 3. Left Wall (-X)
        left = trimesh.creation.box(extents=[thickness, height, depth])
        left.apply_translation([-width/2 - thickness/2, height/2, 0])
        walls_tri.append(left)

        # 4. Right Wall (+X)
        right = trimesh.creation.box(extents=[thickness, height, depth])
        right.apply_translation([width/2 + thickness/2, height/2, 0])
        walls_tri.append(right)

        # Convert all to Pytorch3D
        return [self._to_pytorch3d(w, color=[0.7, 0.7, 0.7]) for w in walls_tri]

    def _to_pytorch3d(self, tri_mesh, color) -> Meshes:
        """Helper to convert Trimesh -> Pytorch3D"""
        verts = torch.tensor(tri_mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(tri_mesh.faces, dtype=torch.int64, device=self.device)
        
        # Add simple solid color texture
        # (Needed because PyTorch3D renderer expects texture)
        verts_rgb = torch.tensor(color, dtype=torch.float32, device=self.device)
        verts_rgb = verts_rgb[None].repeat(verts.shape[0], 1)
        textures = TexturesVertex(verts_features=[verts_rgb])
        
        return Meshes(verts=[verts], faces=[faces], textures=textures)