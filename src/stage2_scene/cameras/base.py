from abc import ABC, abstractmethod
from typing import Tuple, Dict
import torch
from pytorch3d.structures import Meshes

# Import your config class for type hinting
from ..config import SceneConfig

class BaseCameraStrategy(ABC):
    def __init__(self, config: SceneConfig, device: str = "cuda"):
        self.cfg = config
        self.device = device

    @abstractmethod
    def calculate_pose(self, scene_meshes: Meshes) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determines where the camera should stand based on the scene.

        Args:
            scene_meshes: The entire scene (used if you want to center on objects)

        Returns:
            R: Rotation matrix (1, 3, 3)
            T: Translation vector (1, 3)
        """
        pass