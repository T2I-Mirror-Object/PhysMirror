import argparse
import os
import torch
from diffusers import StableDiffusionXLPipeline
from tqdm.auto import tqdm
from typing import List

def load_prompts(prompt_file_path: str) -> List[str]:
    """Reads prompts from a text file, one per line."""
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt file not found at: {prompt_file_path}")
    
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(prompts)} prompts from {prompt_file_path}")
    return prompts

def setup_pipeline(model_id: str, device: str, offload: bool):
    """Initializes the SDXL pipeline."""
    print(f"Loading SDXL model: {model_id}...")
    
    # SDXL runs best in float16 (unlike FLUX which prefers bfloat16)
    dtype = torch.float16
    print(f"Using precision: {dtype}")

    try:
        # Load the pipeline
        # variant="fp16" loads the smaller, optimized weights if available
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16", 
            use_safetensors=True
        )
    except Exception as e:
        print(f"\nError loading model {model_id}. Check your internet connection or Hugging Face token.")
        raise e

    if offload:
        print("Enabling model CPU offload for VRAM efficiency.")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    
    # Optional: Enable VAE slicing to save more VRAM if needed
    # pipe.enable_vae_slicing()
    
    return pipe

def generate_images(
    pipe: StableDiffusionXLPipeline,
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

    # Create a generator for reproducibility
    generator = torch.Generator("cuda").manual_seed(seed)
    print(f"Global seed set to: {seed}")

    for idx, prompt in enumerate(tqdm(prompts, desc="Generating SDXL Images")):
        
        # Match filenames exactly to the FLUX script (image_000.png)
        filename = f"image_{str(idx).zfill(3)}.png"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"Skipping {filename}, already exists.")
            continue

        with torch.no_grad():
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
    parser = argparse.ArgumentParser(description="Generate baseline images using SDXL 1.0.")

    # Required arguments
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to prompts file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save images.")

    # Optional arguments
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="HF Model ID.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu).")
    
    # SDXL Defaults
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    parser.add_argument("--steps", type=int, default=40, help="Inference steps (SDXL usually needs 30-50).")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (CFG).") # SDXL standard is higher than FLUX
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no_offload", action="store_true", help="Disable CPU offload.")

    args = parser.parse_args()

    # 1. Load Prompts
    prompts_list = load_prompts(args.prompt_file)

    # 2. Setup Model
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