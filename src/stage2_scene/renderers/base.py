import torch
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer, 
    FoVPerspectiveCameras
)

class BaseRenderer:
    def __init__(self, image_size: int = 512, device: str = "cuda"):
        self.device = torch.device(device)
        self.image_size = image_size
        
        # SHARED: Standard rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1,
        )

    def _create_cameras(self, R, T, fov):
        """Helper to create camera object from pose."""
        return FoVPerspectiveCameras(
            device=self.device, 
            R=R, 
            T=T, 
            fov=fov
        )

    def _get_fragments(self, scene_mesh, R, T, fov):
        """
        Shared Pipeline: 
        1. Create Camera 
        2. Create Rasterizer 
        3. Rasterize -> Return Fragments (geometry info)
        """
        cameras = self._create_cameras(R, T, fov)
        
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=self.raster_settings
        )
        
        return rasterizer(scene_mesh)