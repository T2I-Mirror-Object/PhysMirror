"""
Standalone Stage 2 Segmentation inference script.

Reads prompts from a text file, matches them to pre-generated 3D meshes in a 
flat directory, constructs the 3D scene using the updated config, 
and renders both the semantic segmentation map (PNG) and metadata (JSON) for Seg2Any.

Usage:
    python scripts/generate_seg_maps_standalone.py \
        --prompts_file data/prompts.txt \
        --meshes_dir output/meshes \
        --output_dir output/segmentation_maps \
        --camera_file data/cameras.txt
"""

import sys
import os
import json
import argparse
import traceback
import torch
import numpy as np
from PIL import Image

# Ensure the src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage1_mesh.objects_extractors import get_objects_extractor
from src.stage2_scene.config import SceneConfig
from src.stage2_scene.pipelines.segmentation import SceneSegmentationPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cameras(path: str) -> list:
    cameras = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [float(x.strip()) for x in line.split(",")]
            cameras.append({
                "camera_dist_multiplier": parts[0],
                "camera_elevation": parts[1],
                "camera_look_at_height": parts[2],
                "camera_azim": parts[3]
            })
    return cameras

def load_prompts(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def manifest_path(output_dir: str) -> str:
    return os.path.join(output_dir, "manifest.json")

def load_manifest(output_dir: str) -> dict:
    path = manifest_path(output_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_manifest(output_dir: str, manifest: dict) -> None:
    with open(manifest_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def run(args):
    print("=" * 70)
    print("Standalone 3D-to-Segmentation Rendering (Stage 2)")
    print(f"Prompts:     {args.prompts_file}")
    print(f"Camera File: {args.camera_file}")
    print(f"Meshes Dir:  {args.meshes_dir}")
    print(f"Output Dir:  {args.output_dir}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = load_manifest(args.output_dir)
    prompts = load_prompts(args.prompts_file)
    cameras = load_cameras(args.camera_file)

    if len(prompts) != len(cameras):
        print(f"[ERROR] Number of prompts ({len(prompts)}) does not match number of cameras ({len(cameras)}).")
        sys.exit(1)

    # 1. Setup Extractor (Needed to reconstruct filenames & pass object prompts)
    extractor_type = args.extractor
    print(f"Loading object extractor ({extractor_type}) to map prompts to filenames...")
    extractor = get_objects_extractor(extractor_type)

    # 2. Setup Scene Pipeline using the updated configuration logic
    print("Initializing SceneSegmentationPipeline...")
    scene_cfg = SceneConfig(
        gap=args.gap,
        object_scale=args.object_scale,
        object_base_rotation=args.object_base_rotation,
        include_object_random_rotation=args.random_rotation,
        mirror_gap_ahead=args.mirror_gap_ahead,
        mirror_gap_side=args.mirror_gap_side,
        mirror_gap_top=args.mirror_gap_top,
        mirror_thickness=args.mirror_thickness,
        mirror_height=args.mirror_height,
        camera_method=args.camera_method,
        camera_dist_multiplier=1.0,  # Placeholder, set per prompt
        camera_elevation=0.0,        # Placeholder, set per prompt
        camera_look_at_height=0.0,   # Placeholder, set per prompt
        camera_azim=0.0,             # Placeholder, set per prompt
        inference_mode=False,
        render_size=args.render_size,
        include_floor=not args.no_floor,
        include_walls=not args.no_walls,
        # Forcing parameters crucial for Seg2Any as seen in your full pipeline script
        renderer_type="segmentation",
        include_mirror_surface=False,  # Let Flux hallucinate the reflection
        include_mirror_frame=not args.no_mirror_frame,
        include_mirror_wall=not args.no_mirror_wall
    )
    
    pipeline = SceneSegmentationPipeline(scene_cfg, device=device)

    # 3. Process Prompts
    for idx, (prompt, cam_cfg) in enumerate(zip(prompts, cameras)):
        if not prompt:
            continue

        # Update pipeline configuration for this prompt's camera settings
        pipeline.cfg.camera_dist_multiplier = cam_cfg["camera_dist_multiplier"]
        pipeline.cfg.camera_elevation = cam_cfg["camera_elevation"]
        pipeline.cfg.camera_look_at_height = cam_cfg["camera_look_at_height"]
        pipeline.cfg.camera_azim = cam_cfg["camera_azim"]

        key = str(idx)
        # Resumability check
        seg_path = manifest.get(key, {}).get("segmentation_map_path", "")
        if seg_path and os.path.exists(seg_path):
            print(f"\n[{idx + 1}/{len(prompts)}] Skipping: Segmentation map already exists for '{prompt[:50]}...'")
            continue

        print(f"\n[{idx + 1}/{len(prompts)}] Rendering: '{prompt}'")

        try:
            # Reconstruct expected mesh filenames
            object_descriptions = extractor.extract(prompt)
            mesh_paths = []
            
            for obj_prompt in object_descriptions:
                safe_name = "".join(c for c in obj_prompt if c.isalnum() or c in " -_").strip()
                safe_name = safe_name.replace(" ", "_")[:64]
                if not safe_name: safe_name = "unknown_object"
                
                glb_path = os.path.join(args.meshes_dir, f"{safe_name}.glb")
                obj_path = os.path.join(args.meshes_dir, f"{safe_name}.obj")
                
                if os.path.exists(glb_path):
                    mesh_paths.append(glb_path)
                elif os.path.exists(obj_path):
                    mesh_paths.append(obj_path)
                else:
                    # Append one of them so the existing missing file check can catch it
                    mesh_paths.append(glb_path)

            # Validate that all required meshes exist
            missing = [p for p in mesh_paths if not os.path.exists(p)]
            if missing:
                print(f"  [WARNING] Missing meshes for this prompt. Skipping.")
                for m in missing:
                    print(f"    -> Cannot find: {m}")
                continue

            # Render Scene
            # Note: Segmentation pipeline requires object_prompts and global_caption
            results = pipeline.run(
                object_paths=mesh_paths,
                object_prompts=object_descriptions,
                global_caption=prompt
            )
            
            seg_map_tensor = results["segmentation_map"]  # [1, 3, H, W]
            json_data = results["json_data"]              # Dict metadata

            # Define output paths
            base_filename = f"seg_{idx + 1:03d}"
            save_img_path = os.path.join(args.output_dir, f"{base_filename}.png")
            save_json_path = os.path.join(args.output_dir, f"{base_filename}.json")

            # Save the JSON metadata
            with open(save_json_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            # Save the Segmentation Map as PNG
            seg_img_np = seg_map_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            if seg_img_np.max() <= 1.0: 
                seg_img_np = (seg_img_np * 255).astype(np.uint8)
            Image.fromarray(seg_img_np).save(save_img_path)

            # Update Manifest
            manifest[key] = {
                "prompt": prompt,
                "segmentation_map_path": save_img_path,
                "metadata_path": save_json_path
            }
            save_manifest(args.output_dir, manifest)

            print(f"  -> Saved seg map: {save_img_path}")
            print(f"  -> Saved metadata: {save_json_path}")

        except Exception as e:
            print(f"  [ERROR] Rendering failed for index {idx}.")
            traceback.print_exc()

        # Keep VRAM clean between renders
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Done. Segmentation maps and JSON metadata generated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render segmentation maps and metadata from a folder of pre-generated 3D meshes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompts_file", "-p", type=str, required=True,
                        help="Path to the text file containing prompts.")
    parser.add_argument("--meshes_dir", "-m", type=str, required=True,
                        help="Directory where the generated .glb or .obj meshes are stored.")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Directory to save the resulting PNGs and JSONs.")
    parser.add_argument("--camera_file", type=str, required=True,
                        help="Text file containing camera parameters per prompt (dist, elevation, look at height, azim).")
    
    # Scene configuration arguments with defaults
    parser.add_argument("--gap", type=float, default=0.5)
    parser.add_argument("--object_scale", type=float, default=1.5)
    parser.add_argument("--object_base_rotation", type=float, default=180.0)
    parser.add_argument("--random_rotation", action="store_true", default=False)
    
    parser.add_argument("--mirror_gap_ahead", type=float, default=1.7)
    parser.add_argument("--mirror_gap_side", type=float, default=2.0)
    parser.add_argument("--mirror_gap_top", type=float, default=2.0)
    parser.add_argument("--mirror_thickness", type=float, default=0.1)
    parser.add_argument("--mirror_height", type=float, default=None)
    
    parser.add_argument("--camera_method", type=str, default="random_side")
    parser.add_argument("--render_size", type=int, default=1024)
    
    parser.add_argument("--no_floor", action="store_true", help="Disable floor rendering (defaults to include).")
    parser.add_argument("--no_walls", action="store_true", help="Disable walls rendering (defaults to include).")
    parser.add_argument("--no_mirror_frame", action="store_true", help="Disable mirror frame rendering (defaults to include).")
    parser.add_argument("--no_mirror_wall", action="store_true", help="Disable mirror wall rendering (defaults to include).")
    
    parser.add_argument("--extractor", type=str, default="simple2", help="Object extractor type to use.")
    
    args = parser.parse_args()
    run(args)