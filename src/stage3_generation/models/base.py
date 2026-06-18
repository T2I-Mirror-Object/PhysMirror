from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from PIL import Image
import torch

class BaseT2IModel(ABC):
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = None # The actual model pipeline loads here

    @abstractmethod
    def load_model(self, model_id: str, **kwargs):
        """
        Load weights. 
        model_id could be a HuggingFace path or local checkpoint path.
        """
        pass

    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        condition_image: Union[Image.Image, torch.Tensor], 
        negative_prompt: str = "",
        seed: int = 42,
        **kwargs
    ) -> Image.Image:
        """
        The core generation logic.
        Args:
            condition_image: The Depth Map or Segmentation Map.
        Returns:
            The final PIL Image.
        """
        pass