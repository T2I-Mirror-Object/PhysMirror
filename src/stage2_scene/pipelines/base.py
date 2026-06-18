from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import torch

from ..config import SceneConfig
from ..builder import SceneBuilder
from ..cameras import get_camera_strategy

class BaseScenePipeline(ABC):
    def __init__(self, config: SceneConfig, device: str = "cuda"):
        self.cfg = config
        self.device = device
        
        self.builder = SceneBuilder(config, device)
        StrategyClass = get_camera_strategy(config.camera_method)
        self.cam_strategy = StrategyClass(config, device)

    def _build_and_position(
        self, 
        object_paths: List[str], 
        camera_params: Optional[Dict[str, torch.Tensor]] = None,
        prebuilt_scene_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, Dict]:
        
        # Step A: Build Geometry (or use provided)
        if prebuilt_scene_dict is not None:
            scene_dict = prebuilt_scene_dict
            print("[BaseScenePipeline] Using prebuilt scene dictionary.")
        else:
            scene_dict = self.builder.build(object_paths)
            print("[BaseScenePipeline] Built new scene geometry.")
            
        full_scene = self.builder.get_complete_scene(scene_dict)

        # Step B: Determine Pose
        if camera_params is not None and "R" in camera_params and "T" in camera_params:
            R = camera_params["R"]
            T = camera_params["T"]
            print("[BaseScenePipeline] Using provided camera parameters.")
        else:
            R, T = self.cam_strategy.calculate_pose(full_scene)
            print("[BaseScenePipeline] Calculated new camera parameters.")
        
        return full_scene, R, T, scene_dict

    @abstractmethod
    def run(self, object_paths: List[str], **kwargs) -> Dict[str, Any]:
        pass