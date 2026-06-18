from typing import List, Dict, Any
import torch
from pytorch3d.structures import Meshes, join_meshes_as_scene

# Import our new modular pieces
from .config import SceneConfig
from .loader import MeshLoader
from .layout import LayoutEngine
from .components import SceneComponents
from .physics import ScenePhysics
from .utils import MeshUtils

from .cameras import get_camera_strategy

class SceneBuilder:
    def __init__(self, config: SceneConfig, device: str = "cuda"):
        self.cfg = config
        self.device = device
        
        # Initialize Sub-Modules
        self.loader = MeshLoader(device)
        self.layout = LayoutEngine(config, device)
        self.components = SceneComponents(device)
        self.physics = ScenePhysics(device)

    def build(self, object_paths: List[str]) -> Dict[str, List[Meshes]]:
        """
        The main pipeline: Load -> Arrange -> Build Environment -> Reflect.
        """
        print(f"[Stage 2] Building scene with {len(object_paths)} objects...")
        
        # 1. LOAD & ARRANGE
        objects = self.loader.load_meshes(object_paths)
        objects = self.layout.arrange(objects) # Handles grounding, rotation, gaps
        
        # 2. CALCULATE MIRROR SIZE
        # Dynamic sizing based on the arranged objects
        m_width, m_height = self.layout.calculate_mirror_dims(objects)
        
        # 3. Create Environment (Conditional)
        floor = []
        if self.cfg.include_floor:
            floor = [self.components.create_floor(self.cfg.room_width, self.cfg.room_depth)]
            
        walls = []
        if self.cfg.include_walls:
            walls = self.components.create_room_walls(self.cfg.room_width, self.cfg.room_depth, self.cfg.wall_height)

        # 4. Create Mirror Components (Conditional)
        mirror_frame = []

        _frame = self.components.create_mirror_frame(m_width, m_height, self.cfg.mirror_thickness)

        # Position them
        _frame = self.layout.position_mirror(_frame, z_pos=-self.cfg.mirror_gap_ahead)

        if self.cfg.include_mirror_frame:
            mirror_frame = [_frame]

        # 4b. Create Mirror Wall (wall surrounding the mirror with a hole for the opening)
        mirror_wall = []
        if self.cfg.include_mirror_wall:
            # Hole size = mirror frame inner dimensions (frame outer minus frame thickness on each side)
            hole_w = m_width - (2 * self.cfg.mirror_thickness)
            hole_h = m_height - (2 * self.cfg.mirror_thickness)
            _wall = self.components.create_mirror_wall(
                wall_width=self.cfg.room_width,
                wall_height=self.cfg.wall_height,
                hole_width=hole_w,
                hole_height=hole_h,
            )
            _wall = self.layout.position_mirror(_wall, z_pos=-self.cfg.mirror_gap_ahead)
            mirror_wall = [_wall]

        # 5. CREATE REFLECTIONS (Physics)
        reflections = self.physics.create_reflections(objects, mirror_z_pos=-self.cfg.mirror_gap_ahead)

        # 6. RETURN DICTIONARY (Compatible with your old code)
        return {
            "objects": objects,
            "mirror_frame": mirror_frame,
            "mirror_wall": mirror_wall,
            "reflections": reflections,
            "floor": floor,
            "walls": walls
        }
        
    def get_complete_scene(self, scene_dict: Dict) -> Meshes:
        """Helper to merge everything into one huge mesh for rendering."""
        
        # 1. Flatten the dictionary
        raw_list = (
            scene_dict["objects"] +
            scene_dict["mirror_frame"] +
            scene_dict.get("mirror_wall", []) +
            scene_dict["reflections"] +
            scene_dict["floor"] +
            scene_dict["walls"]
        )

        # 2. SANITIZE LIST (The Fix)
        clean_list = []
        for i, m in enumerate(raw_list):
            # Check if texture is missing
            if m.textures is None:
                print(f"[Auto-Fix] Mesh {i} had no texture. Painting it white.")
                # Paint it default white
                m_fixed = MeshUtils.paint_mesh(m, color=[0.9, 0.9, 0.9], device=self.device)
                clean_list.append(m_fixed)
            else:
                clean_list.append(m)

        # 3. Join safe meshes
        return join_meshes_as_scene(clean_list)
