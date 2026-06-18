import sys
import os
import torch
import cv2
import argparse
import numpy as np
import trimesh

# -----------------------------------------------------------------------------
# 1. SETUP & IMPORTS
# -----------------------------------------------------------------------------
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage2_scene.config import SceneConfig
from src.stage2_scene.pipelines.correspondence import SceneCorrespondencePipeline

# -----------------------------------------------------------------------------
# 2. HELPER: Create Dummy Object
# -----------------------------------------------------------------------------
def create_dummy_object(output_dir: str):
    """Creates a single Cube to test correspondence."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Object: A Cube (slightly offset so it's not perfectly centered)
    path = os.path.join(output_dir, "obj_cube.obj")
    
    if not os.path.exists(path):
        # Create a box with side length 0.8
        mesh = trimesh.creation.box(extents=[0.8, 0.8, 0.8])
        # Apply a small translation/rotation so it looks interesting
        mesh.apply_translation([0.0, 0.4, 0.0]) # Lift it up slightly
        mesh.export(path)
        
    # Return as a list, because the pipeline expects a list of paths
    return [path]

# -----------------------------------------------------------------------------
# 3. MAIN DEBUG LOGIC
# -----------------------------------------------------------------------------
def debug_correspondence(args):
    print("="*60)
    print("DEBUG: Stage 2 Correspondence Visualization (Single Object)")
    print("="*60)
    
    output_dir = "outputs/debug_correspondence"
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Prepare Dummy Data
    print(f"[Setup] Creating dummy object in {output_dir}...")
    object_paths = create_dummy_object(output_dir)
    
    # 2. Configure Scene
    cfg = SceneConfig(
        render_size=512,
        gap=0.5,
        mirror_gap_ahead=3.0, 
        camera_method="random_side",
        
        # Camera tweaks for better visibility
        camera_dist_multiplier=1.2,
        camera_look_at_height=0.5,
        camera_elevation=15.0
    )

    # 3. Initialize Pipeline
    print(f"[Pipeline] Initializing SceneCorrespondencePipeline on {device}...")
    try:
        pipeline = SceneCorrespondencePipeline(cfg, device=device)
    except Exception as e:
        print(f"[Error] Failed to init pipeline: {e}")
        return

    # 4. Run Pipeline
    print(f"[Run] Processing object with {args.num_pairs} point pairs...")
    try:
        result_img = pipeline.run(
            object_paths=object_paths, 
            num_pairs=args.num_pairs
        )
        
        # 5. Save Output
        save_path = os.path.join(output_dir, "correspondence_result_single.png")
        cv2.imwrite(save_path, result_img)
        
        print(f"\n[Success] Visualization saved to: {save_path}")
        print("-" * 50)
        print("Visual Check Guide:")
        print("1. Look for a 'Cloud' of RED dots (The Cube).")
        print("2. Look for a 'Cloud' of GREEN dots (The Reflection).")
        print("3. Verify that CYAN lines connect them straight through the mirror.")
        print("-" * 50)

    except Exception as e:
        print(f"[Error] Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Correspondence Pipeline")
    
    parser.add_argument(
        "--num_pairs", 
        type=int, 
        default=50, 
        help="Number of random point pairs to sample."
    )
    
    args = parser.parse_args()
    debug_correspondence(args)