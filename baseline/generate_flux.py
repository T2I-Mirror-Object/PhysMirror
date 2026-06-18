import argparse
import os
import torch
from diffusers import FluxPipeline
from tqdm.auto import tqdm
from pathlib import Path
from typing import List

def load_prompts(prompt_file_path: str) -> List[str]:
    """Reads prompts from a text file, one per line."""
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt file not found at: {prompt_file_path}")
    
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        # Read lines and strip whitespace, ignore empty lines
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(prompts)} prompts from {prompt_file_path}")
    return prompts

def setup_pipeline(model_id: str, device: str, offload: bool):
    """Initializes the FLUX pipeline with appropriate optimizations."""
    print(f"Loading model ID: {model_id}...")
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using precision: {dtype}")

    try:
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype
        )
    except Exception as e:
        print(f"\nError loading model. Ensure you have access to {model_id} and have run 'huggingface-cli login'.")
        raise e

    if offload:
        # Best for 12GB-24GB VRAM GPU to prevent OOM
        print("Enabling model CPU offload for VRAM efficiency.")
        pipe.enable_model_cpu_offload()
    else:
        # Best if you have A100/H100 (80GB+) for maximum speed
        pipe.to(device)
        
    # Optional: Compile for speed boost (can take time on first run)
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    
    return pipe

def generate_images(
    pipe: FluxPipeline,
    prompts: List[str],
    output_dir: str,
    height: int,
    width: int,
    steps: int,
    guidance: float,
    seed: int
):
    """Main generation loop."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    generator = torch.Generator("cuda").manual_seed(seed)
    print(f"Global seed set to: {seed}")

    # Iterate through prompts with index for filename
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating Images")):
        
        # Format output filename: image_000.png, image_001.png, etc.
        # Using indices ensures precise matching with the line numbers in prompt.txt
        filename = f"image_{str(idx).zfill(3)}.png"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            # Optional: skip if already exists to resume interrupted runs
            print(f"Skipping {filename}, already exists.")
            continue

        with torch.no_grad():
            # FLUX uses 'guidance_scale' differently than SDXL. 
            # Around 3.5 is standard for realism.
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]

        image.save(filepath)
        
    print(f"\nGeneration complete. Images saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate baseline images using FLUX.1-dev for research evaluation.")

    # Required arguments
    parser.add_argument(
        "--prompt_file", 
        type=str, 
        required=True, 
        help="Path to the text file containing prompts, one per line."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save generated images."
    )

    # Optional arguments (with sensible research defaults for FLUX)
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev", help="Hugging Face model ID.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu).")
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    # FLUX is fast, 28 steps is usually sufficient for dev version
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps.")
    # FLUX guidance is typically lower than SDXL, around 3.5 is good for realism
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale (CFG).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--no_offload", 
        action="store_true", 
        help="Disable CPU offload. Use only if you have massive VRAM (e.g., A100 80GB) for faster speeds."
    )

    args = parser.parse_args()

    # 1. Load Prompts
    prompts_list = load_prompts(args.prompt_file)

    # 2. Setup Model
    # Offload by default unless --no_offload is passed
    use_offload = not args.no_offload
    pipe = setup_pipeline(args.model_id, args.device, use_offload)

    # 3. Run Generation
    generate_images(
        pipe=pipe,
        prompts=prompts_list,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed
    )

if __name__ == "__main__":
    main()