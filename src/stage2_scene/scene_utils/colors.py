import torch
import random

class ColorPalette:
    """Generates distinct RGB colors for segmentation."""
    def __init__(self):
        # Define some standard distinctive colors (Seg2Any likely prefers these standard sets)
        self.distinct_colors = [
            [128, 0, 0],   # Red-ish
            [0, 128, 0],   # Green-ish
            [128, 128, 0], # Olive
            [0, 0, 128],   # Blue-ish
            [128, 0, 128], # Purple
            [0, 128, 128], # Teal
        ]
    
    def get_color(self, index: int) -> list:
        """Returns [R, G, B] in 0-255 range."""
        if index < len(self.distinct_colors):
            return self.distinct_colors[index]
        
        # Fallback for many objects: Random distinct color
        return [random.randint(50, 200) for _ in range(3)]
        
    def get_normalized_color(self, index: int) -> list:
        """Returns [R, G, B] in 0.0-1.0 range for PyTorch3D."""
        c = self.get_color(index)
        return [x / 255.0 for x in c]