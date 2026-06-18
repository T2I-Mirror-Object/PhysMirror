import sys
import os
import torch
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

# -----------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP
# -----------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage3_generation.factory import get_t2i_model

def load_data(seg_map_path: str, json_path: str):
    if not os.path.exists(seg_map_path):
        raise FileNotFoundError(f"Segmentation map not found: {seg_map_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON metadata not found: {json_path}")

    print(f"[Setup] Loading segmentation map: {seg_map_path}")
    print(f"[Setup] Loading JSON metadata:    {json_path}")

    seg_img = Image.open(seg_map_path).convert("RGB")
    with open(json_path, "r") as f:
        meta_data = json.load(f)

    return seg_img, meta_data

# -----------------------------------------------------------------------------
# 2. MAIN DEBUG LOGIC
# -----------------------------------------------------------------------------
def run(args):
    print("=" * 60)
    print("DEBUGGING STAGE 3: Seg2Any Generation")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    try:
        seg_img, meta_data = load_data(args.seg_map, args.json)
    except Exception as e:
        print(f"[Error] Data loading failed: {e}")
        return

    # 2. Initialize Wrapper
    try:
        print("\n[1] Initializing Seg2Any Wrapper...")
        model = get_t2i_model("seg2any")
    except Exception as e:
        print(f"[Error] Failed to init wrapper: {e}")
        return

    # 3. Load Weights
    try:
        print(f"\n[2] Loading weights...")
        print(f"   - Flux: {args.flux_path}")
        print(f"   - LoRA: {args.lora_path}")

        model.load_model(
            pretrained_model_path=args.flux_path,
            lora_path=args.lora_path,
            weight_dtype_str="bf16"
        )
    except Exception as e:
        print(f"[Error] Loading weights failed: {e}")
        return

    # 4. Generate
    prompt = args.prompt or meta_data.get("caption", "")
    print(f"\n[3] Generating image with prompt: '{prompt}'")
    try:
        final_image = model.generate(
            prompt=prompt,
            condition_image=seg_img,
            meta_data=meta_data,
            width=args.width,
            height=args.height,
            num_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )

        save_path = os.path.join(args.output_dir, "seg2any_result.png")
        final_image.save(save_path)
        print(f"\n[Success] Result saved to: {save_path}")

    except Exception as e:
        print(f"[Error] Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug Stage 3: Segmentation map -> Image via Seg2Any",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input ---
    parser.add_argument("--seg_map", "-s", type=str, required=True,
                        help="Path to the input segmentation map image.")
    parser.add_argument("--json", "-j", type=str, required=True,
                        help="Path to the segmentation JSON metadata file.")
    parser.add_argument("--prompt", "-p", type=str, default=None,
                        help="Text prompt. Defaults to 'caption' field in the JSON.")

    # --- Model ---
    g = parser.add_argument_group("Model")
    g.add_argument("--flux_path", type=str, default="black-forest-labs/FLUX.1-dev",
                   help="Path to local Flux model or HF Model ID.")
    g.add_argument("--lora_path", type=str, required=True,
                   help="Path to the Seg2Any 'seg_flux_dev' folder containing 'default' and 'cond' subfolders.")

    # --- Generation ---
    g = parser.add_argument_group("Generation")
    g.add_argument("--width", type=int, default=1024)
    g.add_argument("--height", type=int, default=1024)
    g.add_argument("--steps", type=int, default=25, help="Number of diffusion steps.")
    g.add_argument("--guidance_scale", type=float, default=3.5)
    g.add_argument("--seed", type=int, default=42)

    # --- Output ---
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/debug_stage3_seg2any",
                        help="Directory to save results.")

    args = parser.parse_args()
    run(args)
