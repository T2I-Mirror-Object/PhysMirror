import torch
import numpy as np
from typing import List, Any
import cv2

from .base import BaseScenePipeline
from ..renderers.depth import DepthRenderer
from ..renderers.segmentation import SegmentationRenderer
from ..scene_utils.correspondence import CorrespondenceUtils
from ..utils import MeshUtils

class SceneCorrespondencePipeline(BaseScenePipeline):
    def __init__(self, config, device="cuda"):
        super().__init__(config, device)
        
        # 1. Depth Renderer (For occlusion checking)
        self.renderer = DepthRenderer(config.render_size, device)

        # 2. RGB Renderer (For visual background)
        self.rgb_renderer = SegmentationRenderer(config.render_size, device)

    def run(self, object_paths: List[str], num_pairs: int = 50, draw_intersections: bool = True) -> np.ndarray:
        print(f"[Correspondence] Processing scene with {len(object_paths)} objects...")
        
        # 1. Build Scene
        # We need the full scene mesh to act as "occluders" (blocking lines of sight)
        scene_dict = self.builder.build(object_paths)
        # Objects -> Light Gray
        scene_dict["objects"] = [
            MeshUtils.paint_mesh(m, [0.7, 0.7, 0.7], self.device) 
            for m in scene_dict["objects"]
        ]
        
        # Reflections -> Darker Gray (Visual cue it's a reflection)
        scene_dict["reflections"] = [
            MeshUtils.paint_mesh(m, [0.3, 0.3, 0.3], self.device) 
            for m in scene_dict["reflections"]
        ]
        
        # Mirror Frame -> Gold/Orange
        scene_dict["mirror_frame"] = [
            MeshUtils.paint_mesh(m, [0.8, 0.5, 0.2], self.device)
            for m in scene_dict["mirror_frame"]
        ]

        # Mirror Wall -> Light Gray
        scene_dict["mirror_wall"] = [
            MeshUtils.paint_mesh(m, [0.75, 0.75, 0.75], self.device)
            for m in scene_dict.get("mirror_wall", [])
        ]
        full_mesh = self.builder.get_complete_scene(scene_dict)
        
        # 2. Get Raw Depth & Cameras
        # We use the NEW method we added to DepthRenderer
        R, T = self.cam_strategy.calculate_pose(full_mesh)
        # 3. Render Background Image (RGB)
        rgb_tensor = self.rgb_renderer.render(full_mesh, R, T) # (1, 3, H, W)
        
        # Convert Tensor to Numpy Image (H, W, 3)
        canvas = rgb_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

        # Scale to 0-255 uint8
        if canvas.max() <= 1.0:
            canvas = (canvas * 255).astype(np.uint8)
            
        # Convert RGB to BGR (for OpenCV)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        # 4. Render Depth (for Logic)
        depth_map, cameras = self.renderer.get_raw_depth_and_cameras(full_mesh, R, T)
        
        # 5. Find Correspondence Pairs
        all_valid_pairs = []
        mirror_z = -self.cfg.mirror_gap_ahead
        
        # Iterate only through the Real Objects
        for mesh in scene_dict["objects"]:
            # A. Sample
            pts_obj, pts_ref = CorrespondenceUtils.sample_point_pairs(
                mesh, mirror_z, num_pairs
            )
            
            # B. Filter Visible
            pairs = CorrespondenceUtils.get_visible_pairs(
                pts_obj, pts_ref, depth_map, cameras, 
                self.cfg.render_size, tolerance=0.05
            )
            all_valid_pairs.extend(pairs)
            
        print(f"[Correspondence] Found {len(all_valid_pairs)} visible pairs.")

        # 6. Draw Correspondence Lines on the RGB Canvas
        final_image = CorrespondenceUtils.draw_correspondences(canvas, all_valid_pairs)

        # 7. Optionally draw intersection / vanishing points
        if draw_intersections:
            print("\n[Correspondence] Calculating Vanishing Points...")
            raw_intersections = CorrespondenceUtils.calculate_intersections(all_valid_pairs)
            if raw_intersections:
                H, W = self.cfg.render_size, self.cfg.render_size
                visible_intersections = [
                    (int(px), int(py)) for (px, py) in raw_intersections
                    if 0 <= int(px) < W and 0 <= int(py) < H
                ]
                if visible_intersections:
                    print(f" > Found {len(visible_intersections)} intersection points inside the image.")
                    for i, (ix, iy) in enumerate(visible_intersections[:20]):
                        print(f"   Point {i+1}: ({ix}, {iy})")
                    if len(visible_intersections) > 20:
                        print(f"   ... and {len(visible_intersections) - 20} more.")
                    final_image = CorrespondenceUtils.draw_intersections(final_image, visible_intersections)
                else:
                    print(" > No intersections found within image boundaries.")

                avg_x = sum(p[0] for p in raw_intersections) / len(raw_intersections)
                avg_y = sum(p[1] for p in raw_intersections) / len(raw_intersections)
                print(f" > Average Vanishing Point: ({avg_x:.2f}, {avg_y:.2f})")
            else:
                print(" > No intersections found (lines might be parallel or too far apart).")
        
        return final_image