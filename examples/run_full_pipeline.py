"""
PhysMirror — End-to-End Single-Prompt Example

Runs the full three-stage pipeline on a single text prompt:
  Stage 1: Text → 3D Mesh  (via Trellis)
  Stage 2: 3D Mesh → Depth Map  (scene composition + rendering)
  Stage 3: Depth Map → Image  (via FLUX Omini depth conditioning)

Usage:
    python examples/run_full_pipeline.py \
        --prompt "A wooden chair" \
        --output_dir outputs/full_pipeline

    # Use a different config:
    python examples/run_full_pipeline.py \
        --prompt "A red vase and a blue cup" \
        --config configs/inference.yaml \
        --output_dir outputs/full_pipeline
"""

import sys
import os
import argparse
import yaml
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage1_mesh.objects_extractors import get_objects_extractor
from src.stage1_mesh.factory import get_text_to_3d_model
from src.stage2_scene.config import SceneConfig
from src.stage2_scene.pipelines.depth import SceneDepthPipeline
from src.stage3_generation.factory import get_t2i_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(args):
    cfg = load_config(args.config)
    s1 = cfg["stage1"]
    s2 = cfg["stage2"]
    s3 = cfg["stage3"]

    device = args.device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    mesh_dir = os.path.join(args.output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    print("=" * 60)
    print("PhysMirror — Full Pipeline Example")
    print(f"  Prompt:  {args.prompt}")
    print(f"  Device:  {device}")
    print(f"  Config:  {args.config}")
    print(f"  Output:  {args.output_dir}")
    print("=" * 60)

    # =====================================================================
    # Stage 1: Text → 3D Meshes
    # =====================================================================
    print("\n[Stage 1] Generating 3D meshes from text...")

    mesh_model_name = s1.get("mesh_model", "trellis")
    extractor = get_objects_extractor(s1.get("extractor", "simple2"))
    object_descriptions = extractor.extract(args.prompt)
    print(f"  Extracted objects: {object_descriptions}")

    mesh_ext = ".glb" if mesh_model_name == "trellis" else ".obj"
    mesh_kwargs = {"device": device}
    if mesh_model_name == "trellis":
        mesh_kwargs["model_name"] = s1.get("trellis_model_name", "microsoft/TRELLIS-text-large")

    mesh_model = get_text_to_3d_model(mesh_model_name, **mesh_kwargs)

    mesh_paths = []
    for i, obj_prompt in enumerate(object_descriptions):
        safe_name = obj_prompt.replace(" ", "_")[:64]
        save_path = os.path.join(mesh_dir, f"{i}_{safe_name}{mesh_ext}")
        print(f"  [{i+1}/{len(object_descriptions)}] Generating: '{obj_prompt}'")
        mesh_model.generate(obj_prompt, save_path)
        mesh_paths.append(save_path)

    print(f"  Done — {len(mesh_paths)} mesh(es) saved to {mesh_dir}")
    del mesh_model, extractor
    torch.cuda.empty_cache()

    # =====================================================================
    # Stage 2: 3D Meshes → Depth Map
    # =====================================================================
    print("\n[Stage 2] Composing mirror scene and rendering depth map...")

    scene_cfg = SceneConfig(
        gap=s2["gap"],
        object_scale=s2["object_scale"],
        object_base_rotation=s2["object_base_rotation"],
        include_object_random_rotation=s2["random_rotation"],
        mirror_gap_ahead=s2["mirror_gap_ahead"],
        mirror_gap_side=s2["mirror_gap_side"],
        mirror_gap_top=s2["mirror_gap_top"],
        mirror_thickness=s2["mirror_thickness"],
        mirror_height=s2.get("mirror_height"),
        camera_method=s2["camera_method"],
        camera_dist_multiplier=s2["camera_dist"],
        camera_elevation=s2["camera_elevation"],
        camera_look_at_height=s2["camera_look_at_height"],
        camera_azim_min=s2["camera_azim_min"],
        camera_azim_max=s2["camera_azim_max"],
        render_size=s2["render_size"],
        include_floor=s2["include_floor"],
        include_walls=s2["include_walls"],
        include_mirror_frame=s2["include_mirror_frame"],
        include_mirror_surface=s2["include_mirror_surface"],
        include_mirror_wall=s2["include_mirror_wall"],
    )

    pipeline_s2 = SceneDepthPipeline(scene_cfg, device=device)
    results = pipeline_s2.run(mesh_paths)
    depth_tensor = results["depth_map"]  # [1, H, W]

    depth_save_path = os.path.join(args.output_dir, "depth_map.png")
    to_pil_image(depth_tensor.squeeze().cpu()).save(depth_save_path)
    print(f"  Saved depth map: {depth_save_path}")
    print(f"  Shape: {depth_tensor.shape}, range: [{depth_tensor.min():.3f}, {depth_tensor.max():.3f}]")

    del pipeline_s2
    torch.cuda.empty_cache()

    # =====================================================================
    # Stage 3: Depth Map → Final Image
    # =====================================================================
    print("\n[Stage 3] Generating final image from depth map...")

    model_s3 = get_t2i_model("flux_omini")
    model_s3.load_model(
        model_id=s3["model_id"],
        lora_repo=s3["lora_repo"],
        lora_weight_name=s3["lora_weight_name"],
        adapter_name="depth",
    )

    depth_image = Image.open(depth_save_path)
    final_image = model_s3.generate(
        prompt=args.prompt,
        condition_image=depth_image,
        width=s3["width"],
        height=s3["height"],
        num_steps=s3["steps"],
        guidance_scale=s3["guidance_scale"],
        seed=s3["seed"],
    )

    final_save_path = os.path.join(args.output_dir, "final_result.png")
    final_image.save(final_save_path)
    print(f"  Saved final image: {final_save_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete. Outputs:")
    print(f"  Meshes:     {mesh_dir}/")
    print(f"  Depth map:  {depth_save_path}")
    print(f"  Final image: {final_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PhysMirror: Full pipeline example (Text → Mesh → Depth → Image)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", "-p", type=str, required=True,
                        help="Text prompt describing objects in the scene.")
    parser.add_argument("--config", "-c", type=str, default="configs/inference.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/full_pipeline",
                        help="Directory to save all outputs.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Auto-detected if not set.")

    args = parser.parse_args()
    run(args)
