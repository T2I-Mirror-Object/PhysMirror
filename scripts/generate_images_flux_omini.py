"""
Standalone Stage 3 Generation inference script using FLUX Omini + Custom LoRA.

Reads prompts from a text file, finds the matching depth maps (.png) 
from the input directory, and generates final images using 
FLUX Omini and your custom-trained LoRA (loaded from HF repo).

Usage:
    python scripts/generate_images_flux_omini.py \
        --prompts_file data/prompts.txt \
        --depth_maps_dir output/depth_maps \
        --output_dir output/final_images \
        --model_id black-forest-labs/FLUX.1-dev \
        --lora_repo your-username/flux-omini-lora \
        --lora_weight_name flux_omini.safetensors
"""

import sys
import os
import json
import argparse
import traceback
import torch
from PIL import Image

# Ensure the src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Assuming you registered your new class as "flux_omini" in factory.py
from src.stage3_generation.factory import get_t2i_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    print("Standalone Image Generation (Stage 3 — Flux Omini + Depth LoRA)")
    print(f"Prompts:         {args.prompts_file}")
    print(f"Depth Maps Dir:  {args.depth_maps_dir}")
    print(f"Output Dir:      {args.output_dir}")
    print(f"Base Model:      {args.model_id}")
    print(f"LoRA Repo:       {args.lora_repo}")
    print(f"LoRA Weight:     {args.lora_weight_name}")
    print(f"Adapter Name:    {args.adapter_name}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = load_manifest(args.output_dir)
    prompts = load_prompts(args.prompts_file)

    # 1. Initialize Model Wrapper
    print("\nInitializing Flux Omini Model...")
    try:
        # Note: adjust "flux_omini" if you used a different key in your factory.py
        model_s3 = get_t2i_model("flux_omini")
        model_s3.load_model(
            model_id=args.model_id,
            lora_repo=args.lora_repo,
            lora_weight_name=args.lora_weight_name,
            adapter_name=args.adapter_name,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Process Prompts
    for idx, prompt in enumerate(prompts):
        if not prompt:
            continue

        key = str(idx)
        # Resumability check
        final_path = manifest.get(key, {}).get("image_path", "")
        if final_path and os.path.exists(final_path):
            print(f"\n[{idx + 1}/{len(prompts)}] Skipping: Final image already exists for '{prompt[:50]}...'")
            continue

        print(f"\n[{idx + 1}/{len(prompts)}] Generating: '{prompt}'")

        # Reconstruct expected input filename from Stage 2 logic (depth_001.png)
        depth_filename = f"depth_{idx + 1:03d}.png"
        depth_img_path = os.path.join(args.depth_maps_dir, depth_filename)

        # Validate input exists
        if not os.path.exists(depth_img_path):
            print(f"  [WARNING] Missing depth map for this prompt. Skipping.")
            print(f"    -> Looked for: {depth_img_path}")
            continue

        try:
            # Load depth map as PIL Image (our wrapper's _ensure_pil handles the rest)
            depth_image = Image.open(depth_img_path)

            # Generate Final Image
            final_image = model_s3.generate(
                prompt=prompt,
                condition_image=depth_image, 
                width=args.width,
                height=args.height,
                num_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed
            )

            # Save Output
            save_img_path = os.path.join(args.output_dir, f"image_{idx + 1:03d}.png")
            final_image.save(save_img_path)

            # Update Manifest
            manifest[key] = {
                "prompt": prompt,
                "image_path": save_img_path,
                "depth_map_path": depth_img_path
            }
            save_manifest(args.output_dir, manifest)

            print(f"  -> Saved final image: {save_img_path}")

        except Exception as e:
            print(f"  [ERROR] Generation failed for index {idx}.")
            traceback.print_exc()

        # Keep VRAM clean between generations
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Done. All final images generated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate final images using FLUX Omini and a custom LoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # File Paths
    parser.add_argument("--prompts_file", "-p", type=str, required=True,
                        help="Path to the text file containing prompts.")
    parser.add_argument("--depth_maps_dir", "-d", type=str, required=True,
                        help="Directory containing the Stage 2 depth_NNN.png files.")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Directory to save the resulting final images.")
    
    # Model Paths
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev", 
                        help="HuggingFace ID or path to base Flux Omini model.")
    parser.add_argument("--lora_repo", type=str, required=True, 
                        help="HuggingFace repo ID for your custom LoRA (e.g. username/flux-omini-lora).")
    parser.add_argument("--lora_weight_name", type=str, required=True, 
                        help="Name of the LoRA weight file in the repo (e.g. flux_omini.safetensors).")
    parser.add_argument("--adapter_name", type=str, default="depth",
                        help="Adapter name passed to load_model (usually 'depth' for Omini).")
    
    # Generation Settings
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    run(args)