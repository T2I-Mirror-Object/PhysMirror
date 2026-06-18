import torch
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer, 
    FoVPerspectiveCameras
)
from .base import BaseRenderer

class DepthRenderer(BaseRenderer):
    def render(self, scene_mesh, R, T, fov=60.0, normalize=True, invert=True) -> torch.Tensor:
        """
        Renders the depth map.
        Returns: Tensor of shape (1, H, W) on GPU.
        """
        # 1. Get Geometry (Reuses Base Class Logic)
        fragments = self._get_fragments(scene_mesh, R, T, fov)
        
        # 2. Extract Z-Buffer
        # Shape: (N, H, W, faces_per_pixel) -> take first face
        depth_map = fragments.zbuf[0, ..., 0] 
        
        # 3. Post-Processing (Normalization logic)
        if normalize:
            valid_mask = depth_map > 0 # -1 usually indicates background
            
            if valid_mask.sum() > 0:
                min_d = depth_map[valid_mask].min()
                max_d = depth_map[valid_mask].max()
                
                denom = max_d - min_d
                if denom < 1e-6:
                    denom = 1.0
                
                # Normalize valid pixels to [0, 1]
                depth_map[valid_mask] = (depth_map[valid_mask] - min_d) / denom
                
                # Set background
                # If Invert=True (White=Near), Background should be Black (Far/0.0)
                # If Invert=False (Black=Near), Background should be White (Far/1.0)
                depth_map[~valid_mask] = 1.0 if not invert else 0.0

        if invert:
            # Standard Depth: Dark=Near, Light=Far
            # Inverted: Light=Near, Dark=Far
            depth_map = 1.0 - depth_map
            
            # Re-clamp background to 0 (Black) if it became negative or weird
            if normalize:
                depth_map[~valid_mask] = 0.0

        return depth_map.unsqueeze(0)

    def get_raw_depth_and_cameras(self, scene_mesh, R, T, fov=60.0):
        """
        Returns the raw Z-buffer and the Camera object.
        """
        # 1. Create Cameras (Using your BaseRenderer helper)
        cameras = self._create_cameras(R, T, fov)
        
        # 2. Create Rasterizer locally
        # We must create a new rasterizer linked to these specific cameras
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=self.raster_settings
        )
        
        # 3. Rasterize
        fragments = rasterizer(scene_mesh)
        
        # 4. Extract Raw Depth (No normalization)
        depth_map = fragments.zbuf[0, ..., 0]
        
        return depth_map, cameras