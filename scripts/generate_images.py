"""
Stage 3 batch inference script.

Reads the manifest.json produced by generate_depth_maps.py, loads Flux Omini
once, then generates a final image for every depth map in the manifest.

Output files: images_dir/image_001.png, image_002.png, ...  (1-based index)

All parameters are read from a YAML config file (default: configs/inference.yaml).
Already-generated images are skipped — safe to resume.

Usage:
    python scripts/generate_images.py
    python scripts/generate_images.py --config configs/inference.yaml
    python scripts/generate_images.py --start_idx 0 --end_idx 120
"""

import sys
import os
import json
import argparse
import traceback
import yaml
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage3_generation.factory import get_t2i_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest(depth_maps_dir: str) -> dict:
    path = os.path.join(depth_maps_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"manifest.json not found in '{depth_maps_dir}'.\n"
            "Run generate_depth_maps.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def image_filename(idx: int) -> str:
    """1-based filename: image_001.png, image_002.png, ..."""
    return f"image_{idx + 1:03d}.png"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    cfg = load_config(args.config)

    print("=" * 70)
    print("Batch Image Generation  (Stage 3 — Flux Omini)")
    print(f"Config: {args.config}")
    print("=" * 70)

    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    depth_maps_dir = cfg["depth_maps_dir"]
    images_dir = cfg["images_dir"]
    s3 = cfg["stage3"]

    manifest = load_manifest(depth_maps_dir)
    all_keys = sorted(manifest.keys(), key=lambda k: int(k))
    print(f"Manifest contains {len(all_keys)} entries.")

    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else max(int(k) for k in all_keys) + 1
    keys_to_process = [k for k in all_keys if start <= int(k) < end]
    print(f"Processing indices {start}–{end - 1}  ({len(keys_to_process)} entries)\n")

    os.makedirs(images_dir, exist_ok=True)

    # Load Flux Omini once for all prompts
    print("[Stage 3] Loading Flux Omini (this may take a while)...")
    try:
        model = get_t2i_model("flux_omini")
        model.load_model(
            model_id=s3["model_id"],
            lora_repo=s3["lora_repo"],
            lora_weight_name=s3["lora_weight_name"],
            adapter_name="depth",
        )
    except Exception:
        print("[ERROR] Failed to load Flux Omini model.")
        traceback.print_exc()
        return

    print("[Stage 3] Model loaded.\n")

    completed = skipped = errors = 0
    total = len(keys_to_process)

    for key in keys_to_process:
        entry = manifest[key]
        idx = int(key)
        prompt = entry.get("prompt", "")
        dm_path = entry.get("depth_map_path", "")

        out_path = os.path.join(images_dir, image_filename(idx))

        if os.path.exists(out_path):
            print(f"[{idx}] Already exists, skipping: '{prompt[:60]}'")
            skipped += 1
            continue

        if not dm_path or not os.path.exists(dm_path):
            print(f"[{idx}] No depth map found, skipping: '{prompt[:60]}'")
            errors += 1
            continue

        print(f"\n[{completed + skipped + errors + 1}/{total}] idx={idx}  '{prompt}'")
        print(f"  Depth map: {dm_path}")

        try:
            depth_image = Image.open(dm_path)

            result_img = model.generate(
                prompt=prompt,
                condition_image=depth_image,
                width=s3["width"],
                height=s3["height"],
                num_steps=s3["steps"],
                guidance_scale=s3["guidance_scale"],
                seed=s3["seed"],
            )

            result_img.save(out_path)
            print(f"  Saved: {out_path}")
            completed += 1

        except Exception:
            print(f"  [ERROR] Generation failed for index {idx}.")
            traceback.print_exc()
            errors += 1

    print(f"\n{'=' * 70}")
    print(f"Done.  completed={completed}  skipped={skipped}  errors={errors}")
    print(f"Images saved to: {images_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch Stage 3: depth maps → images via Flux Omini",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, default="configs/inference.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="First manifest index to process (inclusive, 0-based).")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Last manifest index to process (exclusive). Default: all.")
    args = parser.parse_args()
    run(args)
