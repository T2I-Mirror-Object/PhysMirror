import torch
import numpy as np
import cv2
import math
from typing import List, Tuple, Optional
from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVPerspectiveCameras

class CorrespondenceUtils:
    
    @staticmethod
    def sample_point_pairs(mesh: Meshes, mirror_z: float, num_pairs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1. Samples N random vertices from the mesh.
        2. Calculates their mathematical reflection across the mirror plane.
        Returns: (points_obj, points_ref) both shape (N, 3)
        """
        verts = mesh.verts_packed()
        if len(verts) == 0:
            return torch.empty((0,3)), torch.empty((0,3))
            
        # Random sampling
        indices = torch.randperm(len(verts))[:num_pairs]
        points_obj = verts[indices]
        
        # Calculate Reflection: (x, y, z) -> (x, y, 2*Z_mirror - z)
        # Assumes mirror is a plane perpendicular to Z-axis
        points_ref = points_obj.clone()
        points_ref[:, 2] = (2 * mirror_z) - points_obj[:, 2]
        
        return points_obj, points_ref

    @staticmethod
    def get_visible_pairs(
        points_obj: torch.Tensor,
        points_ref: torch.Tensor,
        depth_map: torch.Tensor,
        cameras: FoVPerspectiveCameras,
        image_size: int,
        tolerance: float = 0.05
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Projects points to screen and verifies visibility against the depth map.
        Returns a list of visible ((u1,v1), (u2,v2)) tuples.
        """
        valid_pairs = []
        H, W = image_size, image_size
        
        # 1. Project to Screen Space (Pixels)
        # Returns (N, 3) where z is the depth
        screen_obj = cameras.transform_points_screen(points_obj, image_size=(H, W))
        screen_ref = cameras.transform_points_screen(points_ref, image_size=(H, W))
        
        # Move to CPU for loop
        screen_obj = screen_obj.cpu().numpy()
        screen_ref = screen_ref.cpu().numpy()
        depth_map_np = depth_map.cpu().numpy()
        
        for i in range(len(screen_obj)):
            # Check Object Visibility
            vis_obj, uv_obj = CorrespondenceUtils._is_visible(screen_obj[i], depth_map_np, H, W, tolerance)
            
            # Check Reflection Visibility
            vis_ref, uv_ref = CorrespondenceUtils._is_visible(screen_ref[i], depth_map_np, H, W, tolerance)
            
            if vis_obj and vis_ref:
                valid_pairs.append((uv_obj, uv_ref))
                
        return valid_pairs

    @staticmethod
    def _is_visible(screen_pt, depth_map, H, W, tolerance):
        u, v, z = screen_pt[0], screen_pt[1], screen_pt[2]
        
        # Pixel coordinates
        u_int, v_int = int(u), int(v)
        
        # 1. Bounds Check
        if u_int < 0 or u_int >= W or v_int < 0 or v_int >= H:
            return False, None
            
        # 2. Depth Check
        # Get depth from Z-buffer
        buffer_z = depth_map[v_int, u_int]
        
        # If buffer is -1, it's background (far plane)
        if buffer_z < 0:
            return False, None

        # Point is visible if it's NOT significantly deeper than the buffer
        # PyTorch3D z-buffer logic implies larger Z = farther away
        if z > (buffer_z + tolerance):
            return False, None # Occluded
            
        return True, (u_int, v_int)

    @staticmethod
    def draw_correspondences(image: np.ndarray, pairs: List[Tuple]) -> np.ndarray:
        """Draws lines and dots on the image."""
        canvas = image.copy()
        for start, end in pairs:
            # Line: Cyan
            cv2.line(canvas, start, end, (255, 255, 0), 1)
            # Dots: Red (Object), Green (Reflection)
            cv2.circle(canvas, start, 3, (0, 0, 255), -1) 
            cv2.circle(canvas, end, 3, (0, 255, 0), -1)
        return canvas

    
    @staticmethod
    def draw_intersections(image: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
        """Draws intersection points in White."""
        canvas = image.copy()
        for pt in points:
            # Intersection: White, slightly larger
            cv2.circle(canvas, pt, 2, (255, 255, 255), -1)
        return canvas

    @staticmethod
    def calculate_intersections(pairs: List[Tuple], angle_threshold_deg: float = 5.0) -> List[Tuple[float, float]]:
        """
        Calculates intersection for unique pairs, SKIPPING pairs that are nearly parallel.
        angle_threshold_deg: Minimum angle difference required to trust the intersection.
        """
        intersections = []
        num_lines = len(pairs)
        
        # Pre-calculate direction vectors to save time
        # vectors[i] = (dx, dy) normalized
        vectors = []
        for (start, end) in pairs:
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length < 1e-6:
                vectors.append((0, 0)) # degenerate line
            else:
                vectors.append((dx / length, dy / length))

        # Compare pairs
        for i in range(num_lines):
            for j in range(i + 1, num_lines):
                vec1 = vectors[i]
                vec2 = vectors[j]
                
                # Check Angle using Dot Product
                # dot = x1*x2 + y1*y2
                dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
                
                # Clamp for safety
                dot = max(-1.0, min(1.0, dot))
                
                # Angle in degrees
                angle = math.degrees(math.acos(abs(dot))) # abs() because 180 deg is also parallel
                
                # If lines are too similar (e.g. < 5 degrees apart), skip
                if angle < angle_threshold_deg:
                    continue

                # Calculate Intersection
                pt = CorrespondenceUtils._get_line_intersection(pairs[i], pairs[j])
                if pt is not None:
                    intersections.append(pt)
                    
        return intersections

    @staticmethod
    def _get_line_intersection(line1, line2) -> Optional[Tuple[float, float]]:
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        # Parallel lines check (redundant given the angle check, but safe to keep)
        if abs(denom) < 1e-6:
            return None
            
        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
        return (px, py)

    @staticmethod
    def calculate_convergence_score(intersections: List[Tuple[float, float]], image_size: int) -> float:
        """
        Calculates a 'Physics Correctness' score (0.0 to 1.0).
        
        Logic:
        1. Find the Median Center of the intersections (Robust to outliers).
        2. Calculate the distance of every point to this center.
        3. Calculate the average error of the remaining points.
        4. Normalize to a 0-1 score using an exponential decay.
        """
        if not intersections:
            return 0.0
            
        points = np.array(intersections) # Shape (N, 2)
        
        # 1. Robust Centroid (Median)
        # We use median instead of mean because mean is destroyed by one bad point like (315, 319)
        centroid = np.median(points, axis=0)
        
        # 2. Calculate Euclidean Distances to Centroid
        # dists shape: (N,)
        dists = np.linalg.norm(points - centroid, axis=1)
        
        # 3. Raw Error (Average Pixel Distance)
        # This tells us: "On average, lines miss the perfect point by X pixels"
        avg_pixel_error = np.mean(dists)
        
        # 4. Normalize to [0, 1] Score
        # We normalize by image size so the score is resolution-independent.
        # We use exp(-alpha * error) to map 0->1 and inf->0
        normalized_error = avg_pixel_error / image_size
        
        # Alpha controls strictness. 
        # If alpha=50:
        #   - 1 pixel error (on 512px) -> Score ~0.99 (Great)
        #   - 10 pixel error           -> Score ~0.90 (Good)
        #   - 50 pixel error           -> Score ~0.60 (Mediocre)
        #   - 100 pixel error          -> Score ~0.37 (Bad)
        score = np.exp(-50.0 * normalized_error)
        
        return score, avg_pixel_error