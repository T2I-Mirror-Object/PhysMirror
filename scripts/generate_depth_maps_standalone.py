"""
Standalone Stage 2 inference script.

Reads prompts from a text file, determines which meshes belong to each prompt,
loads those meshes from a specified directory, and renders a depth map using 
the settings from configs/inference.yaml.

Usage:
    python scripts/generate_depth_maps.py \
        --prompts_file data/prompts.txt \
        --meshes_dir output/meshes \
        --output_dir output/depth_maps \
        --config configs/inference.yaml
"""

import sys
import os
import json
import argparse
import traceback
import sys
import os
import torch
from torchvision.transforms.functional import to_pil_image

# Ensure the src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage1_mesh.objects_extractors import get_objects_extractor
from src.stage2_scene.config import SceneConfig
from src.stage2_scene.pipelines.depth import SceneDepthPipeline


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

def depth_filename(idx: int) -> str:
    return f"depth_{idx + 1:03d}.png"


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def run(args):
    print("=" * 70)
    print("Standalone 3D-to-Depth Rendering (Stage 2)")
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

    # 1. Setup Extractor (Needed to reconstruct filenames)
    extractor_type = args.extractor
    print(f"Loading object extractor ({extractor_type}) to map prompts to filenames...")
    extractor = get_objects_extractor(extractor_type)

    # 2. Setup Scene Pipeline
    print("Initializing SceneDepthPipeline...")
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
        include_mirror_frame=not args.no_mirror_frame,
        include_mirror_surface=args.include_mirror_surface,
        include_mirror_wall=not args.no_mirror_wall,
    )
    
    pipeline = SceneDepthPipeline(scene_cfg, device=device)

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
        dm_path = manifest.get(key, {}).get("depth_map_path", "")
        if dm_path and os.path.exists(dm_path):
            print(f"\n[{idx + 1}/{len(prompts)}] Skipping: Depth map already exists for '{prompt[:50]}...'")
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
            results = pipeline.run(mesh_paths)
            depth_tensor = results["depth_map"]  # [1, H, W]

            # Save Output
            save_path = os.path.join(args.output_dir, depth_filename(idx))
            to_pil_image(depth_tensor.squeeze().cpu()).save(save_path)

            # Update Manifest
            manifest[key] = {
                "prompt": prompt,
                "depth_map_path": save_path,
            }
            save_manifest(args.output_dir, manifest)

            print(f"  -> Saved depth map: {save_path}")

        except Exception as e:
            print(f"  [ERROR] Rendering failed for index {idx}.")
            traceback.print_exc()

        # Keep VRAM clean between renders
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Done. Depth maps generated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render depth maps from a folder of pre-generated 3D meshes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompts_file", "-p", type=str, required=True,
                        help="Path to the text file containing prompts.")
    parser.add_argument("--meshes_dir", "-m", type=str, required=True,
                        help="Directory where the generated .glb or .obj meshes are stored.")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Directory to save the resulting depth map PNGs.")
    parser.add_argument("--camera_file", type=str, required=True,
                        help="Text file containing camera parameters per prompt (dist, elevation, look at height, azim).")
    
    # Scene configuration arguments with defaults from configs/inference.yaml
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
    parser.add_argument("--include_mirror_surface", action="store_true", help="Enable mirror surface rendering (defaults to false).")
    parser.add_argument("--no_mirror_wall", action="store_true", help="Disable mirror wall rendering (defaults to include).")
    
    parser.add_argument("--extractor", type=str, default="simple2", help="Object extractor type to use.")
    
    args = parser.parse_args()
    run(args)