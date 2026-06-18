"""
Standalone Stage 3 Generation inference script using Seg2Any.

Reads prompts from a text file, finds the matching segmentation maps (.png) 
and metadata (.json) from the input directory, and generates final photorealistic 
images using the FLUX model guided by the Seg2Any LoRA.

Usage:
    python scripts/generate_images_seg2any.py \
        --prompts_file data/prompts.txt \
        --input_dir output/segmentation_maps \
        --output_dir output/final_images \
        --lora_path path/to/seg_flux_dev
"""

import sys
import os
import json
import argparse
import traceback
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

# Ensure the src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    print("Standalone Image Generation (Stage 3 - Seg2Any)")
    print(f"Prompts:     {args.prompts_file}")
    print(f"Input Dir:   {args.input_dir} (Seg Maps & JSONs)")
    print(f"Output Dir:  {args.output_dir}")
    print(f"Base Model:  {args.flux_path}")
    print(f"LoRA Path:   {args.lora_path}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = load_manifest(args.output_dir)
    prompts = load_prompts(args.prompts_file)

    # 1. Initialize Seg2Any Model
    print("\nInitializing Seg2Any Model...")
    try:
        model_s3 = get_t2i_model("seg2any")
        model_s3.load_model(
            pretrained_model_path=args.flux_path,
            lora_path=args.lora_path,
            weight_dtype_str="bf16"
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load Seg2Any model: {e}")
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

        # Reconstruct expected input filenames from Stage 2
        base_filename = f"seg_{idx + 1:03d}"
        seg_img_path = os.path.join(args.input_dir, f"{base_filename}.png")
        seg_json_path = os.path.join(args.input_dir, f"{base_filename}.json")

        # Validate inputs exist
        if not os.path.exists(seg_img_path) or not os.path.exists(seg_json_path):
            print(f"  [WARNING] Missing inputs for this prompt. Skipping.")
            print(f"    -> Looked for: {seg_img_path} and {seg_json_path}")
            continue

        try:
            # Load JSON metadata
            with open(seg_json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Load PNG and convert to the tensor format expected by Seg2Any [1, 3, H, W]
            pil_img = Image.open(seg_img_path).convert("RGB")
            # to_tensor converts PIL (H, W, C) [0, 255] -> Tensor (C, H, W) [0.0, 1.0]
            seg_map_tensor = to_tensor(pil_img).unsqueeze(0).to(device) 

            # Generate Final Image
            final_image = model_s3.generate(
                prompt=prompt,
                condition_image=seg_map_tensor, 
                meta_data=json_data,            
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
                "seg_map_path": seg_img_path,
                "metadata_path": seg_json_path
            }
            save_manifest(args.output_dir, manifest)

            print(f"  -> Saved final image: {save_img_path}")

        except Exception as e:
            print(f"  [ERROR] Generation failed for index {idx}.")
            traceback.print_exc()

        # Keep VRAM clean between generations, FLUX is very heavy
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Done. All final images generated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate final images using Seg2Any from segmentation maps and metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # File Paths
    parser.add_argument("--prompts_file", "-p", type=str, required=True,
                        help="Path to the text file containing prompts.")
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="Directory containing the Stage 2 .png and .json files.")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Directory to save the resulting final images.")
    
    # Model Paths
    parser.add_argument("--flux_path", type=str, default="black-forest-labs/FLUX.1-dev", 
                        help="Path to base Flux model or HuggingFace ID.")
    parser.add_argument("--lora_path", type=str, required=True, 
                        help="Path to the Seg2Any 'seg_flux_dev' folder containing 'default' and 'cond' subfolders.")
    
    # Generation Settings
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    run(args)