import sys
import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage2_scene.config import SceneConfig
from src.stage2_scene.pipelines.segmentation import SceneSegmentationPipeline

def create_dummy_objects(output_dir: str):
    """Creates a Cube and a Sphere to test multiple object segmentation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Object 1: A Cube
    path1 = os.path.join(output_dir, "obj1_cube.obj")
    if not os.path.exists(path1):
        trimesh.creation.box(extents=[1, 1, 1]).export(path1)
        
    # Object 2: A Sphere (offset slightly so they don't overlap)
    path2 = os.path.join(output_dir, "obj2_sphere.obj")
    if not os.path.exists(path2):
        # Create sphere and move it
        mesh = trimesh.creation.icosphere(radius=0.6)
        # We don't need to move it much because the LayoutEngine handles arrangement,
        # but let's shift it just in case logic fails.
        mesh.apply_translation([1.5, 0, 0]) 
        mesh.export(path2)
        
    return path1, path2

def debug_segmentation():
    print("="*60)
    print("DEBUGGING STAGE 2: Segmentation Rendering & JSON Generation")
    print("="*60)
    
    output_dir = "outputs/debug_segmentation"
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Prepare Dummy Data
    obj1_path, obj2_path = create_dummy_objects(output_dir)
    object_paths = [obj1_path, obj2_path]
    
    # Define Inputs specifically for Seg2Any
    object_prompts = [
        "A red wooden cube", 
        "A blue metallic sphere"
    ]
    global_caption = "A simple test scene containing a cube and a sphere."

    # 2. Configure Pipeline
    cfg = SceneConfig(
        camera_method="random_side", # Angle view to see separation
        render_size=512,
        # Layout params
        gap=0.5,
        mirror_gap_ahead=3.0
    )

    # 3. Initialize Pipeline
    # This will init SegmentationRenderer, ColorPalette, and Seg2AnyFormatter
    try:
        pipeline = SceneSegmentationPipeline(cfg, device=device)
    except Exception as e:
        print(f"[Error] Failed to init pipeline: {e}")
        return

    # 4. Run Pipeline
    print(f"[Debug] Processing {len(object_paths)} objects...")
    results = pipeline.run(
        object_paths=object_paths,
        object_prompts=object_prompts,
        global_caption=global_caption
    )
    
    seg_map_tensor = results["segmentation_map"] # Shape [1, 3, H, W]
    json_data = results["json_data"]

    # 5. Save & Verify JSON
    json_path = os.path.join(output_dir, "segmentation_meta.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n[Result] JSON Metadata saved to: {json_path}")
    print("-" * 40)
    print(json.dumps(json_data, indent=2))
    print("-" * 40)

    # 6. Save & Verify Image
    # Convert [1, 3, H, W] -> [H, W, 3] numpy
    seg_img = seg_map_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 1, 1)
    plt.title("Segmentation Map (Flat Colors)")
    plt.imshow(seg_img)
    plt.axis('off')
    
    img_save_path = os.path.join(output_dir, "segmentation_map.png")
    plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0)
    print(f"[Result] Segmentation Image saved to: {img_save_path}")

    # 7. Verification Logic
    print("\n[Verification Check]")
    print("1. Are the objects visible against a Black background?")
    print("2. Does Object 1 (Cube) have the color listed in JSON entry 0?")
    print(f"   -> JSON says: {json_data['segments_info'][0]['color']}")
    print("3. Does Object 2 (Sphere) have the color listed in JSON entry 1?")
    print(f"   -> JSON says: {json_data['segments_info'][1]['color']}")
    print("4. Are the Mirror/Floor/Walls black (ignored)?")

if __name__ == "__main__":
    debug_segmentation()