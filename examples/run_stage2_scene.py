"""
PhysMirror — Stage 2 Scene Composition Example

Demonstrates the scene composition and depth rendering stage using a
simple geometric primitive (cube, sphere, or cylinder) as a stand-in object.
No GPU-heavy model loading is required — useful for testing scene layout,
camera angles, and mirror placement.

Usage:
    # Basic run with default settings:
    python examples/run_stage2_scene.py

    # Custom camera and mirror settings:
    python examples/run_stage2_scene.py \
        --shape sphere \
        --mirror_gap_ahead 1.7 \
        --camera_dist 2.2 \
        --camera_elevation 26 \
        --render_size 1024
"""

import sys
import os
import argparse
import torch
import trimesh
from torchvision.transforms.functional import to_pil_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage2_scene.config import SceneConfig
from src.stage2_scene.pipelines.depth import SceneDepthPipeline


def create_dummy_object(output_dir: str, shape: str = "cube", size: float = 0.8) -> str:
    """Creates a simple test mesh (cube, sphere, or cylinder)."""
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
    return path


def run(args):
    print("=" * 60)
    print("PhysMirror — Stage 2 Scene Composition Example")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a dummy mesh
    print(f"\n[Setup] Creating {args.shape} object (size={args.obj_size})...")
    mesh_path = create_dummy_object(args.output_dir, shape=args.shape, size=args.obj_size)

    # Configure the scene
    cfg = SceneConfig(
        gap=0.5,
        object_scale=1.5,
        object_base_rotation=180.0,
        include_object_random_rotation=args.random_rotation,
        mirror_gap_ahead=args.mirror_gap_ahead,
        mirror_gap_side=args.mirror_gap_side,
        mirror_gap_top=args.mirror_gap_top,
        mirror_thickness=0.1,
        camera_method=args.camera_method,
        camera_dist_multiplier=args.camera_dist,
        camera_elevation=args.camera_elevation,
        camera_look_at_height=args.camera_look_at_height,
        camera_azim_min=args.camera_azim_min,
        camera_azim_max=args.camera_azim_max,
        render_size=args.render_size,
        include_floor=not args.no_floor,
        include_walls=not args.no_walls,
        include_mirror_frame=True,
        include_mirror_surface=False,
        include_mirror_wall=True,
    )

    print(f"[Scene] Mirror gap: {args.mirror_gap_ahead}")
    print(f"[Scene] Camera: dist={args.camera_dist}, elev={args.camera_elevation}")

    # Render
    print("\n[Rendering] Composing scene and rendering depth map...")
    pipeline = SceneDepthPipeline(cfg, device=device)
    results = pipeline.run([mesh_path])
    depth_tensor = results["depth_map"]

    save_path = os.path.join(args.output_dir, "depth_map.png")
    to_pil_image(depth_tensor.squeeze().cpu()).save(save_path)

    print(f"\n[Done] Depth map saved to: {save_path}")
    print(f"  Shape: {depth_tensor.shape}")
    print(f"  Value range: [{depth_tensor.min():.3f}, {depth_tensor.max():.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PhysMirror: Stage 2 scene composition example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Object
    parser.add_argument("--shape", type=str, default="cube", choices=["cube", "sphere", "cylinder"],
                        help="Shape of the test object.")
    parser.add_argument("--obj_size", type=float, default=0.8, help="Size of the test object.")
    parser.add_argument("--random_rotation", action="store_true", help="Apply random rotation to the object.")

    # Mirror
    parser.add_argument("--mirror_gap_ahead", type=float, default=1.7, help="Distance from object to mirror.")
    parser.add_argument("--mirror_gap_side", type=float, default=2.0, help="Side gap for mirror.")
    parser.add_argument("--mirror_gap_top", type=float, default=2.0, help="Top gap for mirror.")

    # Camera
    parser.add_argument("--camera_method", type=str, default="random_side", choices=["random_side", "fixed_front"])
    parser.add_argument("--camera_dist", type=float, default=2.2, help="Camera distance multiplier.")
    parser.add_argument("--camera_elevation", type=float, default=26.0, help="Camera elevation angle.")
    parser.add_argument("--camera_look_at_height", type=float, default=1.8, help="Camera look-at height.")
    parser.add_argument("--camera_azim_min", type=float, default=-10.0, help="Min azimuth angle.")
    parser.add_argument("--camera_azim_max", type=float, default=10.0, help="Max azimuth angle.")

    # Rendering
    parser.add_argument("--render_size", type=int, default=1024, help="Render resolution.")
    parser.add_argument("--no_floor", action="store_true", help="Disable floor.")
    parser.add_argument("--no_walls", action="store_true", help="Disable walls.")

    # Output
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/stage2_demo",
                        help="Directory to save outputs.")

    args = parser.parse_args()
    run(args)
