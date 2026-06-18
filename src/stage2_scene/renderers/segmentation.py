import torch
from pytorch3d.renderer import (
    MeshRasterizer, SoftPhongShader, 
    AmbientLights, TexturesVertex
)
from .base import BaseRenderer

class SegmentationRenderer(BaseRenderer):
    def __init__(self, image_size=512, device="cuda"):
        super().__init__(image_size, device)
        
        # 1. Use Ambient Light Only (Intensity 1.0)
        # This ensures the color we see is exactly the color we painted.
        self.lights = AmbientLights(device=device)

    def render(self, scene_mesh, R, T, fov=60.0) -> torch.Tensor:
        """
        Renders the segmentation map (RGB).
        """
        # 1. Standard Camera/Rasterizer setup (inherited)
        cameras = self._create_cameras(R, T, fov)
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=self.raster_settings
        )

        # 2. Shader: Use SoftPhong but with ambient lights only
        # This effectively renders "Unlit" colors
        shader = SoftPhongShader(
            device=self.device, 
            cameras=cameras,
            lights=self.lights
        )

        # 3. Render
        fragments = rasterizer(scene_mesh)
        images = shader(fragments, scene_mesh)
        
        # Output is (1, H, W, 4) -> (RGBA)
        # We need (1, 3, H, W) usually, or just (1, H, W, 3)
        seg_map = images[..., :3] # Drop Alpha
        
        return seg_map.permute(0, 3, 1, 2) # (1, C, H, W)