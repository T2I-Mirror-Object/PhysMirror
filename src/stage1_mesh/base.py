from abc import ABC, abstractmethod
import os

class BaseTextTo3D(ABC):
    def __init__(self, device: str = "cuda", output_dir: str = "outputs/meshes"):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def generate(self, prompt: str, object_name: str) -> str:
        """
        Generates a 3D mesh from text.
        
        Args:
            prompt (str): The full description or the specific object name.
            object_name (str): A short name for the file (e.g., 'chair').
            
        Returns:
            str: The absolute path to the saved .obj/.ply file.
        """
        pass