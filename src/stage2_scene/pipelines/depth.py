from typing import List, Dict, Any, Optional
import torch
from .base import BaseScenePipeline
from ..renderers.depth import DepthRenderer

class SceneDepthPipeline(BaseScenePipeline):
    def __init__(self, config, device="cuda"):
        super().__init__(config, device)
        self.renderer = DepthRenderer(
            image_size=config.render_size,
            device=device
        )

    def run(
        self, 
        object_paths: List[str], 
        camera_params: Optional[Dict[str, torch.Tensor]] = None,
        prebuilt_scene_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        
        print("[DepthPipeline] Starting execution...")
        
        # 1. Geometry & Camera (Instant if prebuilt_scene_dict and camera_params are provided)
        full_scene, R, T, final_dict = self._build_and_position(
            object_paths, 
            camera_params=camera_params, 
            prebuilt_scene_dict=prebuilt_scene_dict
        )
        
        # 2. Render Depth
        depth_map = self.renderer.render(full_scene, R, T)
        
        return {
            "depth_map": depth_map,
            "camera_params": {"R": R, "T": T},
            "scene_dict": final_dict
        }