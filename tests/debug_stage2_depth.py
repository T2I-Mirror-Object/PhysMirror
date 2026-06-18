import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import trimesh

# -----------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP
# -----------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage2_scene.config import SceneConfig
from src.stage2_scene.builder import SceneBuilder
from src.stage2_scene.pipelines.depth import SceneDepthPipeline

# -----------------------------------------------------------------------------
# 2. HELPER: Create Dummy Data
# -----------------------------------------------------------------------------
def create_dummy_object(path: str):
    """Creates a simple asymmetric L-shape so we can check orientation."""
    if os.path.exists(path):
        return
    
    # Create two boxes and merge them to make an L-shape
    # This helps verify if the reflection is actually flipped correctly
    box1 = trimesh.creation.box(extents=[0.5, 2.0, 0.5])
    box2 = trimesh.creation.box(extents=[1.0, 0.5, 0.5])
    box2.apply_translation([0.5, -0.75, 0]) # Move to bottom right
    
    mesh = trimesh.util.concatenate([box1, box2])
    mesh.export(path)
    print(f"[Setup] Created dummy object at {path}")

# -----------------------------------------------------------------------------
# 3. MAIN DEBUG LOGIC
# -----------------------------------------------------------------------------
def debug_render_depth():
    print("="*60)
    print("DEBUGGING STAGE 2: Scene Composition & Depth Rendering")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "outputs/debug_render"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Setup Dummy Data
    dummy_obj_path = os.path.join(output_dir, "test_shape.obj")
    create_dummy_object(dummy_obj_path)

    # 2. Configure the Scene
    cfg = SceneConfig(
        # Physics Params
        gap=0.2,
        mirror_gap_ahead=3.0,
        mirror_thickness=0.1,
        mirror_height=3.0,  # Fix mirror height; objects auto-scale to fill it. Set to None to use dynamic sizing.

        # Strategy Params
        camera_method="random_side", # Change to 'fixed_front' to debug alignment
        renderer_type="depth",
        render_size=512
    )
    
    # 3. Initialize Pipeline
    try:
        pipeline = SceneDepthPipeline(cfg, device=device)
    except Exception as e:
        print(f"[Error] Failed to init pipeline: {e}")
        return
    
    # 4. Run the Pipeline
    results = pipeline.run([dummy_obj_path, dummy_obj_path])
    
    depth_map = results["depth_map"]   # Shape: (1, 512, 512)
    cam_params = results["camera_params"]
    
    print(f"\n[Result] Depth Map Shape: {depth_map.shape}")
    print(f"[Result] Camera Rotation (R):\n{cam_params['R']}")
    print(f"[Result] Camera Translation (T):\n{cam_params['T']}")

    # 5. Visualize and Save
    # Remove batch dim -> (H, W)
    depth_img = depth_map.squeeze().cpu().numpy()
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: The Continuous Grayscale Map
    plt.subplot(1, 2, 1)
    plt.title("T2I Depth Map (0=Far, 1=Near)")
    
    # Use 'gray' colormap to mimic the actual input to diffusion models
    # vmin=0, vmax=1 ensures we lock the range
    plt.imshow(depth_img, cmap='gray', vmin=0.0, vmax=1.0) 
    plt.axis('off') 
    
    # Plot 2: Histogram (To verify continuous distribution)
    plt.subplot(1, 2, 2)
    plt.title("Depth Value Distribution")
    # We want to see a spread of values, not just 0 and 1
    plt.hist(depth_img.flatten(), bins=50, range=(0, 1), color='black')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    
    save_path = os.path.join(output_dir, "debug_depth.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n[Success] Depth map saved to: {save_path}")
    print("Check the image! You should see:")
    print(" 1. Two L-shaped objects in the foreground.")
    print(" 2. A mirror frame behind them.")
    print(" 3. Two reflected L-shapes 'inside' the mirror.")

if __name__ == "__main__":
    debug_render_depth()