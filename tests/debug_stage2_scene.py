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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage2_scene.config import SceneConfig
from src.stage2_scene.pipelines.correspondence import SceneCorrespondencePipeline

# -----------------------------------------------------------------------------
# 2. HELPER: Create Dummy Object
# -----------------------------------------------------------------------------
def create_dummy_object(output_dir: str, shape: str = "cube", size: float = 0.8):
    """Creates a single test object."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"obj_{shape}.obj")

    if not os.path.exists(path):
        if shape == "cube":
            mesh = trimesh.creation.box(extents=[size, size, size])
        elif shape == "sphere":
            mesh = trimesh.creation.icosphere(radius=size / 2)
        elif shape == "cylinder":
            mesh = trimesh.creation.cylinder(radius=size / 2, height=size)
        else:
            raise ValueError(f"Unknown shape: {shape}. Use cube, sphere, or cylinder.")
        mesh.apply_translation([0.0, size / 2, 0.0])
        mesh.export(path)

    return [path]

# -----------------------------------------------------------------------------
# 3. MAIN DEBUG LOGIC
# -----------------------------------------------------------------------------
def debug_scene(args):
    print("=" * 60)
    print("DEBUG: Stage 2 Scene Composition")
    print("=" * 60)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Prepare Object
    print(f"[Setup] Creating {args.shape} (size={args.obj_size}) in {output_dir}...")
    object_paths = create_dummy_object(output_dir, shape=args.shape, size=args.obj_size)

    # 2. Configure Scene from CLI args
    cfg = SceneConfig(
        # Object placement
        gap=args.gap,
        include_object_random_rotation=args.random_rotation,

        # Mirror
        mirror_gap_ahead=args.mirror_gap_ahead,
        mirror_gap_side=args.mirror_gap_side,
        mirror_gap_top=args.mirror_gap_top,
        mirror_thickness=args.mirror_thickness,
        mirror_height=args.mirror_height,

        # Camera
        camera_method=args.camera_method,
        camera_dist_multiplier=args.camera_dist,
        camera_elevation=args.camera_elevation,
        camera_look_at_height=args.camera_look_at_height,
        camera_azim_min=args.camera_azim_min,
        camera_azim_max=args.camera_azim_max,

        # Rendering
        render_size=args.render_size,
        renderer_type=args.renderer_type,

        # Scene elements
        include_floor=args.include_floor,
        include_walls=args.include_walls,
        include_mirror_frame=args.include_mirror_frame,
        include_mirror_surface=args.include_mirror_surface,
    )

    # Print active config
    print(f"\n[Config]")
    for field, value in vars(cfg).items():
        print(f"  {field}: {value}")
    print()

    # 3. Initialize Pipeline
    print(f"[Pipeline] Initializing on {device}...")
    try:
        pipeline = SceneCorrespondencePipeline(cfg, device=device)
    except Exception as e:
        print(f"[Error] Failed to init pipeline: {e}")
        return

    # 4. Run Pipeline
    print(f"[Run] Processing with {args.num_pairs} correspondence pairs...")
    try:
        result_img = pipeline.run(
            object_paths=object_paths,
            num_pairs=args.num_pairs,
            draw_intersections=False,
        )

        save_path = os.path.join(output_dir, "scene_result.png")
        cv2.imwrite(save_path, result_img)

        print(f"\n[Success] Saved to: {save_path}")
        print("-" * 50)
        print("Visual Check Guide:")
        print("  RED dots    = object surface points")
        print("  GREEN dots  = reflected points")
        print("  CYAN lines  = correspondence pairs")
        print("-" * 50)

    except Exception as e:
        print(f"[Error] Execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug Stage 2 Scene Composition with configurable parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Object ---
    g = parser.add_argument_group("Object")
    g.add_argument("--shape", type=str, default="cube", choices=["cube", "sphere", "cylinder"],
                    help="Shape of the test object.")
    g.add_argument("--obj_size", type=float, default=0.8,
                    help="Size of the test object (side length / diameter).")

    # --- Object placement ---
    g = parser.add_argument_group("Object Placement")
    g.add_argument("--gap", type=float, default=0.5,
                    help="Horizontal gap between objects.")
    g.add_argument("--random_rotation", action="store_true",
                    help="Apply random Y-axis rotation to objects.")

    # --- Mirror ---
    g = parser.add_argument_group("Mirror")
    g.add_argument("--mirror_gap_ahead", type=float, default=3.0,
                    help="Distance from objects to the mirror plane (Z-axis).")
    g.add_argument("--mirror_gap_side", type=float, default=2.0,
                    help="Extra mirror width beyond objects on each side.")
    g.add_argument("--mirror_gap_top", type=float, default=2.0,
                    help="Extra mirror height above objects.")
    g.add_argument("--mirror_thickness", type=float, default=0.1,
                    help="Thickness of the mirror frame.")
    g.add_argument("--mirror_height", type=float, default=None,
                    help="Fix the total mirror height. Objects are auto-scaled to fill it (mirror_height - mirror_gap_top). Overrides dynamic sizing.")

    # --- Camera ---
    g = parser.add_argument_group("Camera")
    g.add_argument("--camera_method", type=str, default="random_side",
                    choices=["random_side", "fixed_front"],
                    help="Camera placement strategy.")
    g.add_argument("--camera_dist", type=float, default=1.2,
                    help="Camera distance multiplier. Lower = closer.")
    g.add_argument("--camera_elevation", type=float, default=15.0,
                    help="Camera elevation in degrees (0 = horizontal).")
    g.add_argument("--camera_look_at_height", type=float, default=0.5,
                    help="Height the camera looks at (meters off floor).")
    g.add_argument("--camera_azim_min", type=float, default=20.0,
                    help="Minimum azimuth angle for random_side camera.")
    g.add_argument("--camera_azim_max", type=float, default=25.0,
                    help="Maximum azimuth angle for random_side camera.")

    # --- Rendering ---
    g = parser.add_argument_group("Rendering")
    g.add_argument("--render_size", type=int, default=512,
                    help="Output image resolution (square).")
    g.add_argument("--renderer_type", type=str, default="depth",
                    choices=["depth", "segmentation"],
                    help="Renderer type.")
    g.add_argument("--num_pairs", type=int, default=50,
                    help="Number of correspondence point pairs to sample.")

    # --- Scene elements ---
    g = parser.add_argument_group("Scene Elements")
    g.add_argument("--include_floor", action="store_true", default=True,
                    help="Include floor plane.")
    g.add_argument("--no_floor", dest="include_floor", action="store_false",
                    help="Remove floor plane.")
    g.add_argument("--include_walls", action="store_true", default=True,
                    help="Include room walls.")
    g.add_argument("--no_walls", dest="include_walls", action="store_false",
                    help="Remove room walls.")
    g.add_argument("--include_mirror_frame", action="store_true", default=True,
                    help="Include visible mirror frame.")
    g.add_argument("--no_mirror_frame", dest="include_mirror_frame", action="store_false",
                    help="Remove mirror frame.")
    g.add_argument("--include_mirror_surface", action="store_true", default=False,
                    help="Include the mirror glass surface.")

    # --- Output ---
    parser.add_argument("--output_dir", type=str, default="outputs/debug_scene",
                        help="Directory to save results.")

    args = parser.parse_args()
    debug_scene(args)
