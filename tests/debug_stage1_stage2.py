"""
Debug script: Stage 1 (Text -> Meshes) + Stage 2 (Depth + Segmentation).

Both maps are rendered from the EXACT same scene:
  - same object positions, rotations, mirror geometry
  - same camera pose (R, T)
The scene is built once; depth is rendered from the unpainted geometry,
then meshes are painted and segmentation is rendered with the same camera.

Config is loaded from configs/inference.yaml (override with --config).
"""

import sys
import os
import argparse
import json
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage1_mesh.objects_extractors import get_objects_extractor
from src.stage1_mesh.factory import get_text_to_3d_model
from src.stage2_scene.config import SceneConfig
from src.stage2_scene.builder import SceneBuilder
from src.stage2_scene.cameras import get_camera_strategy
from src.stage2_scene.renderers.depth import DepthRenderer
from src.stage2_scene.renderers.segmentation import SegmentationRenderer
from src.stage2_scene.scene_utils.colors import ColorPalette
from src.stage2_scene.scene_utils.metadata import Seg2AnyFormatter
from src.stage2_scene.utils import MeshUtils


def _paint_and_track(mesh_list, prompt_getter, json_prompts, json_colors, palette, color_idx, device):
    """Assign a unique color to each mesh, paint it in-place, and record metadata."""
    for i, mesh in enumerate(mesh_list):
        color_255 = palette.get_color(color_idx)
        color_norm = palette.get_normalized_color(color_idx)
        color_idx += 1
        mesh_list[i] = MeshUtils.paint_mesh(mesh, color=color_norm, device=device)
        json_prompts.append(prompt_getter(i))
        json_colors.append(color_255)
    return color_idx


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_scene_config(stage2_cfg):
    return SceneConfig(
        gap=stage2_cfg.get("gap", 0.5),
        object_scale=stage2_cfg.get("object_scale", 1.0),
        object_base_rotation=stage2_cfg.get("object_base_rotation", 0.0),
        include_object_random_rotation=stage2_cfg.get("random_rotation", False),
        mirror_gap_ahead=stage2_cfg.get("mirror_gap_ahead", 3.0),
        mirror_gap_side=stage2_cfg.get("mirror_gap_side", 2.0),
        mirror_gap_top=stage2_cfg.get("mirror_gap_top", 2.0),
        mirror_thickness=stage2_cfg.get("mirror_thickness", 0.1),
        mirror_height=stage2_cfg.get("mirror_height", None),
        camera_method=stage2_cfg.get("camera_method", "random_side"),
        camera_dist_multiplier=stage2_cfg.get("camera_dist", 1.2),
        camera_elevation=stage2_cfg.get("camera_elevation", 15.0),
        camera_look_at_height=stage2_cfg.get("camera_look_at_height", 0.5),
        camera_azim_min=stage2_cfg.get("camera_azim_min", 20.0),
        camera_azim_max=stage2_cfg.get("camera_azim_max", 25.0),
        render_size=stage2_cfg.get("render_size", 512),
        include_floor=stage2_cfg.get("include_floor", True),
        include_walls=stage2_cfg.get("include_walls", True),
        include_mirror_frame=stage2_cfg.get("include_mirror_frame", True),
        include_mirror_surface=False,  # not used in segmentation pipeline
        include_mirror_wall=stage2_cfg.get("include_mirror_wall", False),
    )


def run(args):
    print("=" * 60)
    print("DEBUG: Stage 1 (Text->Mesh) + Stage 2 (Depth + Segmentation)")
    print("=" * 60)

    # Load YAML config
    yaml_cfg = load_config(args.config)
    stage1_cfg = yaml_cfg.get("stage1", {})
    stage2_cfg = yaml_cfg.get("stage2", {})

    device = args.device or yaml_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    mesh_dir = os.path.join(args.output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    print(f"[Config] Loaded from: {args.config}")
    print(f"[Config] Device: {device}")

    # =========================================================================
    # STAGE 1: Text -> 3D Meshes
    # =========================================================================
    print(f"\n[Stage 1] Prompt: '{args.prompt}'")

    mesh_model_name = stage1_cfg.get("mesh_model", "shap_e")
    extractor_name = stage1_cfg.get("extractor", "simple2")

    extractor = get_objects_extractor(extractor_name)
    object_descriptions = extractor.extract(args.prompt)
    print(f"[Stage 1] Extracted objects: {object_descriptions}")

    mesh_ext = ".glb" if mesh_model_name == "trellis" else ".obj"
    mesh_kwargs = {"device": device}
    if mesh_model_name == "trellis":
        mesh_kwargs["model_name"] = stage1_cfg.get("trellis_model_name", "microsoft/TRELLIS-text-large")

    mesh_model = get_text_to_3d_model(mesh_model_name, **mesh_kwargs)

    generated_mesh_paths = []
    for i, obj_prompt in enumerate(object_descriptions):
        safe_name = obj_prompt.replace(" ", "_")
        save_path = os.path.join(mesh_dir, f"{i}_{safe_name}{mesh_ext}")
        print(f"[Stage 1] Generating mesh for: '{obj_prompt}'...")
        mesh_model.generate(obj_prompt, save_path)
        generated_mesh_paths.append(save_path)

    print(f"[Stage 1] Done. {len(generated_mesh_paths)} mesh(es) saved to {mesh_dir}")

    del mesh_model, extractor
    torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 2: Build scene ONCE, render Depth + Segmentation from same pose
    # =========================================================================
    print("\n[Stage 2] Building scene (shared geometry + camera)...")

    cfg = build_scene_config(stage2_cfg)

    print("[Stage 2] SceneConfig:")
    for field, value in vars(cfg).items():
        print(f"  {field}: {value}")

    # Build geometry ONCE
    builder = SceneBuilder(cfg, device=device)
    scene_dict = builder.build(generated_mesh_paths)

    # Compute camera pose ONCE from the unpainted scene
    full_scene_unpainted = builder.get_complete_scene(scene_dict)
    cam_strategy = get_camera_strategy(cfg.camera_method)(cfg, device)
    R, T = cam_strategy.calculate_pose(full_scene_unpainted)
    print(f"[Stage 2] Camera computed. R: {R.shape}, T: {T.shape}")

    from torchvision.transforms.functional import to_pil_image

    # --- Depth: render from unpainted scene with the computed R, T ---
    print("\n[Stage 2] Rendering depth map...")
    depth_renderer = DepthRenderer(image_size=cfg.render_size, device=device)
    depth_map = depth_renderer.render(full_scene_unpainted, R, T)  # (1, H, W)

    depth_save_path = os.path.join(args.output_dir, "depth_map.png")
    to_pil_image(depth_map.squeeze().cpu()).save(depth_save_path)
    print(f"[Success] Depth map saved to: {depth_save_path}")
    print(f"  Shape: {depth_map.shape}")
    print(f"  Value range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")

    del depth_renderer, depth_map, full_scene_unpainted
    torch.cuda.empty_cache()

    # --- Segmentation: paint scene_dict meshes in-place, render with SAME R, T ---
    print("\n[Stage 2] Painting scene and rendering segmentation map...")

    palette = ColorPalette()
    json_prompts = []
    json_colors = []
    color_idx = 0

    color_idx = _paint_and_track(
        scene_dict["objects"],
        lambda i: object_descriptions[i],
        json_prompts, json_colors, palette, color_idx, device,
    )
    color_idx = _paint_and_track(
        scene_dict["reflections"],
        lambda i: f"Reflection of {object_descriptions[i]}",
        json_prompts, json_colors, palette, color_idx, device,
    )
    if cfg.include_mirror_frame:
        color_idx = _paint_and_track(
            scene_dict["mirror_frame"],
            lambda i: "A decorative mirror frame",
            json_prompts, json_colors, palette, color_idx, device,
        )
    if cfg.include_floor:
        if cfg.paint_floor:
            color_idx = _paint_and_track(
                scene_dict["floor"],
                lambda i: "A floor",
                json_prompts, json_colors, palette, color_idx, device,
            )
        else:
            scene_dict["floor"] = [
                MeshUtils.paint_mesh(m, [0.0, 0.0, 0.0], device) for m in scene_dict["floor"]
            ]
    if cfg.include_mirror_wall:
        if cfg.paint_mirror_wall:
            color_idx = _paint_and_track(
                scene_dict["mirror_wall"],
                lambda i: "A wall behind the mirror",
                json_prompts, json_colors, palette, color_idx, device,
            )
        else:
            scene_dict["mirror_wall"] = [
                MeshUtils.paint_mesh(m, [0.0, 0.0, 0.0], device) for m in scene_dict["mirror_wall"]
            ]
    if cfg.include_walls:
        if cfg.paint_walls:
            color_idx = _paint_and_track(
                scene_dict["walls"],
                lambda i: "Walls",
                json_prompts, json_colors, palette, color_idx, device,
            )
        else:
            scene_dict["walls"] = [
                MeshUtils.paint_mesh(m, [0.0, 0.0, 0.0], device) for m in scene_dict["walls"]
            ]

    full_scene_painted = builder.get_complete_scene(scene_dict)
    seg_renderer = SegmentationRenderer(image_size=cfg.render_size, device=device)
    seg_map = seg_renderer.render(full_scene_painted, R, T)  # (1, 3, H, W)

    seg_save_path = os.path.join(args.output_dir, "segmentation_map.png")
    to_pil_image(seg_map.squeeze(0).cpu()).save(seg_save_path)
    print(f"[Success] Segmentation map saved to: {seg_save_path}")
    print(f"  Shape: {seg_map.shape}")

    # --- JSON ---
    formatter = Seg2AnyFormatter()
    json_data = formatter.format(
        caption=args.prompt,
        seed=0,
        object_prompts=json_prompts,
        object_colors=json_colors,
    )
    json_save_path = os.path.join(args.output_dir, "segmentation.json")
    with open(json_save_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"[Success] Segmentation JSON saved to: {json_save_path}")
    print(f"  Segments: {len(json_data['segments_info'])}")
    for seg in json_data["segments_info"]:
        print(f"    color={seg['color']}  text='{seg['text']}'")

    del seg_renderer, seg_map, full_scene_painted
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1 + Stage 2: Text prompt -> Depth map + Segmentation map (same scene/camera)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", "-p", type=str, required=True,
                        help="Text prompt describing the scene.")
    parser.add_argument("--config", "-c", type=str, default="configs/inference.yaml",
                        help="Path to the YAML config file.")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/debug_stage1_stage2",
                        help="Directory to save results.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Overrides config if set.")

    args = parser.parse_args()
    run(args)
