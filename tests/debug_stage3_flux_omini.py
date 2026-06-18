import sys
import os
import argparse
import torch
from PIL import Image

# -----------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP
# -----------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage3_generation.factory import get_t2i_model

# -----------------------------------------------------------------------------
# 2. MAIN DEBUG LOGIC
# -----------------------------------------------------------------------------
def run(args):
    print("="*60)
    print("DEBUGGING STAGE 3: Flux Omini Generation")
    print("="*60)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Depth Map
    print(f"\n[1] Loading depth map from: {args.depth_map}")
    depth_image = Image.open(args.depth_map)

    # 2. Initialize Model via Factory
    try:
        print("\n[2] Initializing Model Wrapper...")
        model = get_t2i_model("flux_omini")
    except ValueError as e:
        print(f"[Error] Factory failed: {e}")
        return

    # 3. Load Weights
    try:
        print("\n[3] Loading Weights (This may take time)...")
        model.load_model(
            model_id=args.model_id,
            lora_repo=args.lora_repo,
            lora_weight_name=args.lora_weight_name,
            adapter_name="depth"
        )
    except Exception as e:
        print(f"[Error] Loading weights failed: {e}")
        return

    # 4. Generate Image
    print(f"\n[4] Generating with prompt: '{args.prompt}'")

    try:
        result_img = model.generate(
            prompt=args.prompt,
            condition_image=depth_image,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )

        # 5. Save Result
        save_path = os.path.join(args.output_dir, "flux_omini_result.png")
        result_img.save(save_path)
        print(f"\n[Success] Generated image saved to: {save_path}")

    except Exception as e:
        print(f"[Error] Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug Stage 3: Depth map -> Image via Flux Omini",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--depth_map", "-d", type=str, required=True,
                        help="Path to the input depth map image.")
    parser.add_argument("--prompt", "-p", type=str,
                        default="A high quality photo of a long hallway, depth of field, 4k, realistic texture",
                        help="Text prompt for generation.")

    # Model
    g = parser.add_argument_group("Model")
    g.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev",
                   help="Base Flux model ID.")
    g.add_argument("--lora_repo", type=str, default="Yuanshi/OminiControl",
                   help="LoRA repository.")
    g.add_argument("--lora_weight_name", type=str, default="experimental/depth.safetensors",
                   help="LoRA weight filename.")

    # Generation
    g = parser.add_argument_group("Generation")
    g.add_argument("--width", type=int, default=512)
    g.add_argument("--height", type=int, default=512)
    g.add_argument("--steps", type=int, default=28,
                   help="Number of diffusion steps.")
    g.add_argument("--guidance_scale", type=float, default=3.5)
    g.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/debug_stage3",
                        help="Directory to save results.")

    args = parser.parse_args()
    run(args)