"""
PhysMirror — Stage 3 Image Generation Example

Demonstrates the depth-conditioned image generation stage.
Takes a pre-rendered depth map and a text prompt, then generates a
photorealistic image using FLUX with OminiControl depth conditioning.

Usage:
    python examples/run_stage3_generation.py \
        --depth_map examples/sample_depth_map.png \
        --prompt "A wooden chair in front of a mirror"

    # Custom generation settings:
    python examples/run_stage3_generation.py \
        --depth_map examples/sample_depth_map.png \
        --prompt "A red vase on a table in front of a mirror" \
        --steps 28 \
        --guidance_scale 3.5 \
        --seed 42
"""

import sys
import os
import argparse
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage3_generation.factory import get_t2i_model


def run(args):
    print("=" * 60)
    print("PhysMirror — Stage 3 Image Generation Example")
    print(f"  Depth map: {args.depth_map}")
    print(f"  Prompt:    {args.prompt}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load depth map
    print(f"\n[1] Loading depth map from: {args.depth_map}")
    depth_image = Image.open(args.depth_map)

    # 2. Initialize model
    print("[2] Initializing Flux Omini model (this may take a while)...")
    model = get_t2i_model("flux_omini")
    model.load_model(
        model_id=args.model_id,
        lora_repo=args.lora_repo,
        lora_weight_name=args.lora_weight_name,
        adapter_name="depth",
    )

    # 3. Generate image
    print(f"[3] Generating image with prompt: '{args.prompt}'")
    result_img = model.generate(
        prompt=args.prompt,
        condition_image=depth_image,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    # 4. Save result
    save_path = os.path.join(args.output_dir, "generated_image.png")
    result_img.save(save_path)
    print(f"\n[Done] Generated image saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PhysMirror: Stage 3 image generation example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--depth_map", "-d", type=str, required=True,
                        help="Path to the input depth map image.")
    parser.add_argument("--prompt", "-p", type=str,
                        default="A wooden chair in front of a mirror",
                        help="Text prompt for image generation.")

    # Model
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="Base FLUX model ID.")
    parser.add_argument("--lora_repo", type=str, default="Yuanshi/OminiControl",
                        help="OminiControl LoRA repository.")
    parser.add_argument("--lora_weight_name", type=str, default="experimental/depth.safetensors",
                        help="LoRA weight filename.")

    # Generation
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--steps", type=int, default=28, help="Number of diffusion steps.")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Output
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/stage3_demo",
                        help="Directory to save results.")

    args = parser.parse_args()
    run(args)
