from typing import List, Dict, Any, Callable, Optional
import torch
from .base import BaseScenePipeline
from ..renderers.segmentation import SegmentationRenderer
from ..scene_utils.colors import ColorPalette
from ..scene_utils.metadata import Seg2AnyFormatter
from ..utils import MeshUtils

class SceneSegmentationPipeline(BaseScenePipeline):
    def __init__(self, config, device="cuda"):
        super().__init__(config, device)
        
        self.cfg.include_mirror_surface = False
        self.renderer = SegmentationRenderer(config.render_size, device)
        self.palette = ColorPalette()
        self.formatter = Seg2AnyFormatter()

    def _paint_and_track(self, 
                         mesh_list: List, 
                         prompt_getter: Callable[[int], str], 
                         json_prompts: List[str], 
                         json_colors: List[List[int]], 
                         start_idx: int) -> int:
        """
        Reusable helper to:
        1. Assign a unique color to each mesh in the list.
        2. Paint the mesh in-place.
        3. Append the prompt and color to the JSON tracking lists.
        4. Return the next available color index.
        """
        current_idx = start_idx
        
        for i, mesh in enumerate(mesh_list):
            # 1. Get Color
            color_255 = self.palette.get_color(current_idx)
            color_norm = self.palette.get_normalized_color(current_idx)
            current_idx += 1
            
            # 2. Paint Mesh (Modifies the list in-place)
            mesh_list[i] = MeshUtils.paint_mesh(
                mesh, color=color_norm, device=self.device
            )
            
            # 3. Add to JSON
            prompt = prompt_getter(i)
            json_prompts.append(prompt)
            json_colors.append(color_255)
            
        return current_idx

    def run(self, 
            object_paths: List[str], 
            object_prompts: List[str], 
            global_caption: str, 
            camera_params: Optional[Dict[str, torch.Tensor]] = None,
            **kwargs) -> Dict[str, Any]:
        
        print("[SegPipeline] Starting execution...")
        
        # 1. Build Geometry (Kept separate so we can paint before combining)
        scene_dict = self.builder.build(object_paths)
        
        # We need lists to collect data for JSON and Rendering
        json_prompts = []
        json_colors = []
        color_idx = 0 

        # 2. Semantic Painting
        color_idx = self._paint_and_track(
            scene_dict['objects'],
            lambda i: object_prompts[i],
            json_prompts, json_colors, color_idx
        )

        color_idx = self._paint_and_track(
            scene_dict['reflections'],
            lambda i: f"Reflection of {object_prompts[i]}",
            json_prompts, json_colors, color_idx
        )

        if self.cfg.include_mirror_frame:
            color_idx = self._paint_and_track(
                scene_dict['mirror_frame'],
                lambda i: "A decorative mirror frame",
                json_prompts, json_colors, color_idx
            )

        if self.cfg.include_floor:
            if getattr(self.cfg, 'paint_floor', False):
                color_idx = self._paint_and_track(
                    scene_dict['floor'],
                    lambda i: "A floor",
                    json_prompts, json_colors, color_idx
                )
            else:
                scene_dict['floor'] = [
                    MeshUtils.paint_mesh(m, [0.0, 0.0, 0.0], self.device) for m in scene_dict['floor']
                ]

        if self.cfg.include_mirror_wall:
            if getattr(self.cfg, 'paint_mirror_wall', False):
                color_idx = self._paint_and_track(
                    scene_dict['mirror_wall'],
                    lambda i: "A wall behind the mirror",
                    json_prompts, json_colors, color_idx
                )
            else:
                scene_dict['mirror_wall'] = [
                    MeshUtils.paint_mesh(m, [0.0, 0.0, 0.0], self.device) for m in scene_dict['mirror_wall']
                ]

        if self.cfg.include_walls:
            if getattr(self.cfg, 'paint_walls', False):
                color_idx = self._paint_and_track(
                    scene_dict['walls'],
                    lambda i: "Walls",
                    json_prompts, json_colors, color_idx
                )
            else:
                scene_dict['walls'] = [
                    MeshUtils.paint_mesh(m, [0.0, 0.0, 0.0], self.device) for m in scene_dict['walls']
                ]

        # 3. Merge Scene AFTER Painting
        full_scene = self.builder.get_complete_scene(scene_dict)
        
        # 4. Position Camera (Apply forced params if they exist)
        if camera_params is not None and "R" in camera_params and "T" in camera_params:
            R = camera_params["R"]
            T = camera_params["T"]
            print("[SegPipeline] Using provided camera parameters.")
        else:
            R, T = self.cam_strategy.calculate_pose(full_scene)
            print("[SegPipeline] Calculated new camera parameters.")

        # 5. Render Segmentation Map
        seg_map = self.renderer.render(full_scene, R, T) # Shape [1, 3, H, W]

        # 6. Generate JSON
        json_data = self.formatter.format(
            caption=global_caption,
            seed=42, 
            object_prompts=json_prompts,
            object_colors=json_colors
        )

        return {
            "segmentation_map": seg_map,
            "json_data": json_data,
            "camera_params": {"R": R, "T": T},
            "scene_dict": scene_dict
        }