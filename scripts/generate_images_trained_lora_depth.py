"""
Standalone Stage 3 Generation inference script using native Flux Depth + Custom LoRA.

Reads prompts from a text file, finds the matching depth maps (.png) 
from the input directory, and generates final photorealistic images using 
FLUX.1-Depth-dev and your custom-trained SynMirror LoRA.

Usage:
    python scripts/generate_images_trained_lora_depth.py \
        --prompts_file data/prompts.txt \
        --depth_maps_dir output/depth_maps \
        --output_dir output/final_images \
        --lora_path path/to/your/custom_lora.safetensors
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

# Assuming you registered your new class as "flux_depth" in factory.py
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
    print("Standalone Image Generation (Stage 3 - Flux Depth + Custom LoRA)")
    print(f"Prompts:        {args.prompts_file}")
    print(f"Depth Maps Dir: {args.depth_maps_dir}")
    print(f"Output Dir:     {args.output_dir}")
    print(f"Base Model:     {args.model_id}")
    print(f"LoRA Path:      {args.lora_path}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = load_manifest(args.output_dir)
    prompts = load_prompts(args.prompts_file)

    # 1. Initialize Model Wrapper
    print("\nInitializing Flux Depth Model...")
    try:
        # Note: adjust "flux_depth" if you used a different key in your factory.py
        model_s3 = get_t2i_model("flux_depth")
        model_s3.load_model(
            model_id=args.model_id,
            lora_path=args.lora_path
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
                guidance_scale=args.guidance,
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
        description="Generate final images using FLUX.1-Depth-dev and a custom LoRA.",
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
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-Depth-dev", 
                        help="Path to base Flux Depth model or HuggingFace ID.")
    parser.add_argument("--lora_path", type=str, default=None, 
                        help="Path to your custom SynMirror LoRA weights (.safetensors).")
    
    # Generation Settings
    parser.add_argument("--height", type=int, default=512, help="Output image height.")
    parser.add_argument("--width", type=int, default=512, help="Output image width.")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    run(args)