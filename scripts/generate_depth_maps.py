"""
Stage 1 + Stage 2 batch inference script.

Reads prompts from a text file, generates 3D meshes for all prompts using
Trellis (loaded once), then renders a depth map for each prompt's scene.
Meshes are written to a temporary directory and deleted automatically after
depth rendering — only depth maps are kept on disk.

Output files: depth_maps_dir/depth_001.png, depth_002.png, ...  (1-based index)
Manifest:     depth_maps_dir/manifest.json  {index -> {prompt, depth_map_path}}

All parameters are read from a YAML config file (default: configs/inference.yaml).
Already-completed depth maps are skipped — safe to resume.

Usage:
    python scripts/generate_depth_maps.py
    python scripts/generate_depth_maps.py --config configs/inference.yaml
    python scripts/generate_depth_maps.py --start_idx 0 --end_idx 120
"""

import sys
import os
import json
import argparse
import traceback
import tempfile
import yaml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage1_mesh.objects_extractors import get_objects_extractor
from src.stage1_mesh.factory import get_text_to_3d_model
from src.stage2_scene.config import SceneConfig
from src.stage2_scene.pipelines.depth import SceneDepthPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    """1-based filename: depth_001.png, depth_002.png, ..."""
    return f"depth_{idx + 1:03d}.png"


# ---------------------------------------------------------------------------
# Phase A: Generate all meshes into a temporary directory (model loaded once)
# ---------------------------------------------------------------------------

def phase_a_generate_meshes(prompts, cfg, device, tmp_mesh_dir, manifest) -> dict:
    """
    Generates 3D meshes for every prompt that still needs a depth map.
    Returns a dict {global_idx: [mesh_path, ...]} for use in Phase B.
    Mesh files live in tmp_mesh_dir and will be auto-deleted by the caller.
    """
    s1 = cfg["stage1"]
    extractor = get_objects_extractor(s1["extractor"])

    mesh_model_name = s1["mesh_model"]
    mesh_ext = ".glb" if mesh_model_name == "trellis" else ".obj"
    mesh_kwargs = {"device": device}
    if mesh_model_name == "trellis":
        mesh_kwargs["model_name"] = s1["trellis_model_name"]

    mesh_model = None  # lazy-load on the first prompt that needs it
    mesh_data: dict = {}  # idx -> list of temp mesh paths

    for idx, prompt in enumerate(prompts):
        if prompt is None:
            continue

        key = str(idx)
        dm_path = manifest.get(key, {}).get("depth_map_path", "")
        if dm_path and os.path.exists(dm_path):
            print(f"[Phase A] [{idx}] Depth map already exists, skipping mesh gen: '{prompt[:60]}'")
            continue

        print(f"\n[Phase A] [{idx}] '{prompt}'")

        try:
            object_descriptions = extractor.extract(prompt)
            print(f"  Objects: {object_descriptions}")

            if mesh_model is None:
                print(f"  Loading mesh model: {mesh_model_name} ...")
                mesh_model = get_text_to_3d_model(mesh_model_name, **mesh_kwargs)

            prompt_mesh_dir = os.path.join(tmp_mesh_dir, f"{idx:04d}")
            os.makedirs(prompt_mesh_dir, exist_ok=True)

            paths = []
            for obj_idx, obj_prompt in enumerate(object_descriptions):
                safe_name = obj_prompt.replace(" ", "_")[:64]
                save_path = os.path.join(prompt_mesh_dir, f"{obj_idx}_{safe_name}{mesh_ext}")
                print(f"  [{obj_idx + 1}/{len(object_descriptions)}] Generating: '{obj_prompt}'")
                mesh_model.generate(obj_prompt, save_path)
                paths.append(save_path)

            mesh_data[idx] = paths

        except Exception:
            print(f"  [ERROR] Mesh generation failed for index {idx}.")
            traceback.print_exc()

    if mesh_model is not None:
        print("\n[Phase A] Unloading mesh model to free GPU memory...")
        del mesh_model
        torch.cuda.empty_cache()

    del extractor
    return mesh_data


# ---------------------------------------------------------------------------
# Phase B: Render depth maps using the temporary meshes
# ---------------------------------------------------------------------------

def phase_b_render_depth_maps(prompts, cfg, device, output_dir, manifest, mesh_data) -> dict:
    """
    For each prompt that has meshes in mesh_data, runs Stage 2 and saves
    the depth map as depth_NNN.png inside output_dir.
    """
    from torchvision.transforms.functional import to_pil_image

    s2 = cfg["stage2"]
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

    for idx, prompt in enumerate(prompts):
        if prompt is None:
            continue

        key = str(idx)
        dm_path = manifest.get(key, {}).get("depth_map_path", "")
        if dm_path and os.path.exists(dm_path):
            print(f"[Phase B] [{idx}] Depth map already exists, skipping: '{prompt[:60]}'")
            continue

        if idx not in mesh_data:
            print(f"[Phase B] [{idx}] No meshes available (Phase A may have failed), skipping.")
            continue

        mesh_paths = mesh_data[idx]
        missing = [p for p in mesh_paths if not os.path.exists(p)]
        if missing:
            print(f"[Phase B] [{idx}] Missing mesh files {missing}, skipping.")
            continue

        print(f"\n[Phase B] [{idx}] Rendering: '{prompt}'")

        try:
            pipeline = SceneDepthPipeline(scene_cfg, device=device)
            results = pipeline.run(mesh_paths)
            depth_tensor = results["depth_map"]  # [1, H, W]

            save_path = os.path.join(output_dir, depth_filename(idx))
            to_pil_image(depth_tensor.squeeze().cpu()).save(save_path)

            manifest[key] = {
                "prompt": prompt,
                "depth_map_path": save_path,
            }
            save_manifest(output_dir, manifest)

            print(f"  Saved: {save_path}  "
                  f"shape={depth_tensor.shape}  "
                  f"range=[{depth_tensor.min():.3f}, {depth_tensor.max():.3f}]")

            del pipeline, results, depth_tensor
            torch.cuda.empty_cache()

        except Exception:
            print(f"  [ERROR] Stage 2 failed for index {idx}.")
            traceback.print_exc()

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    cfg = load_config(args.config)

    print("=" * 70)
    print("Batch Depth Map Generation  (Stage 1 + Stage 2)")
    print(f"Config: {args.config}")
    print("=" * 70)

    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = cfg["depth_maps_dir"]
    os.makedirs(output_dir, exist_ok=True)

    all_prompts = load_prompts(cfg["prompts_file"])
    print(f"Loaded {len(all_prompts)} prompts from {cfg['prompts_file']}")

    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(all_prompts)
    print(f"Processing indices {start}–{end - 1}  ({end - start} prompts)\n")

    # Build list aligned to global indices (None = out of range, skip)
    global_prompts = [None] * len(all_prompts)
    for i in range(start, end):
        global_prompts[i] = all_prompts[i]

    manifest = load_manifest(output_dir)

    # Meshes live in a temp dir for the duration of the run, then auto-deleted
    with tempfile.TemporaryDirectory(prefix="mirror_t2i_meshes_") as tmp_mesh_dir:
        print(f"Temporary mesh dir: {tmp_mesh_dir}\n")

        print("=" * 70)
        print("PHASE A  —  Text → 3D Meshes  (model loaded once for all prompts)")
        print("=" * 70)
        mesh_data = phase_a_generate_meshes(global_prompts, cfg, device, tmp_mesh_dir, manifest)

        print("\n" + "=" * 70)
        print("PHASE B  —  Meshes → Depth Maps")
        print("=" * 70)
        manifest = phase_b_render_depth_maps(global_prompts, cfg, device, output_dir, manifest, mesh_data)

    # tmp_mesh_dir is now deleted
    print("\n[Info] Temporary mesh files deleted.")

    completed = sum(
        1 for i in range(start, end)
        if manifest.get(str(i), {}).get("depth_map_path")
        and os.path.exists(manifest[str(i)]["depth_map_path"])
    )
    print(f"\n{'=' * 70}")
    print(f"Done.  {completed}/{end - start} depth maps generated.")
    print(f"Manifest: {manifest_path(output_dir)}")
    print("Run generate_images.py next to produce final images.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch Stage 1+2: prompts → depth maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, default="configs/inference.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="First prompt index to process (inclusive, 0-based).")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Last prompt index to process (exclusive). Default: all.")
    args = parser.parse_args()
    run(args)
