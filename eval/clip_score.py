import argparse
import os
import json
import csv
import torch
import numpy as np
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Average CLIP Score for images in a folder.")
    
    parser.add_argument(
        "--input_dir", type=str, required=True, 
        help="Path to the folder containing images."
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="A single prompt to use for ALL images. (Use either this or --prompts_json)"
    )
    parser.add_argument(
        "--prompts_json", type=str, default=None,
        help="Path to a JSON file mapping filenames to prompts (e.g., {'img1.png': 'a cat'})."
    )
    parser.add_argument(
        "--model_name", type=str, default="openai/clip-vit-base-patch16",
        help="HuggingFace model ID for CLIP."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Number of images to process at once."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run calculation on (cuda or cpu)."
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Path to save the individual scores in a CSV file."
    )
    
    return parser.parse_args()

def load_image_as_tensor(path):
    """Loads an image and converts it to the format expected by torchmetrics (C, H, W) uint8."""
    try:
        img = Image.open(path).convert("RGB")
        # Convert PIL to Numpy array (H, W, C)
        img_np = np.array(img)
        # Convert to Tensor (C, H, W)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    args = parse_args()
    
    # 1. Prepare Partial Function
    # We fix the model name here to avoid reloading it constantly
    clip_score_fn = partial(clip_score, model_name_or_path=args.model_name)

    # 2. Get Image Files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    # 3. Load Prompts
    prompt_map = {}
    if args.prompts_json:
        with open(args.prompts_json, 'r') as f:
            prompt_map = json.load(f)
    elif args.prompt:
        # Use the same prompt for all images
        prompt_map = {f: args.prompt for f in image_files}
    else:
        raise ValueError("You must provide either --prompt or --prompts_json")

    # 4. Processing Loop
    scores = []
    
    # Process in batches
    # If output_csv is set, we open the file and write header
    csv_file = None
    writer = None
    if args.output_csv:
        csv_file = open(args.output_csv, 'w', newline='', encoding='utf-8')
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "score"])

    try:
        for i in tqdm(range(0, len(image_files), args.batch_size), desc="Calculating CLIP Scores"):
            batch_files = image_files[i : i + args.batch_size]
            
            batch_images = []
            batch_prompts = []
            batch_filenames = []
            
            for fname in batch_files:
                if fname not in prompt_map:
                    continue # Skip if no prompt found for this image
                    
                img_tensor = load_image_as_tensor(os.path.join(args.input_dir, fname))
                if img_tensor is not None:
                    batch_images.append(img_tensor)
                    batch_prompts.append(prompt_map[fname])
                    batch_filenames.append(fname)
            
            if not batch_images:
                continue

            if args.output_csv:
                # Calculate score for each image individually to log to CSV
                # This might be slower but ensures accurate mapping
                for img, txt, fname in zip(batch_images, batch_prompts, batch_filenames):
                     # clip_score expects (N, C, H, W)
                     img_in = img.unsqueeze(0).to(args.device)
                     with torch.no_grad():
                         s = clip_score_fn(img_in, [txt]).detach().item()
                     scores.append(s)
                     writer.writerow([fname, s])
            else:
                # Stack images into (N, C, H, W)
                images_tensor = torch.stack(batch_images).to(args.device)
                
                # Calculate Score
                # torchmetrics handles the tokenization internally
                with torch.no_grad():
                    batch_score = clip_score_fn(images_tensor, batch_prompts).detach()
                    # If batch_score is a single value (scalar), wrap it in a list
                    if batch_score.ndim == 0:
                        scores.append(float(batch_score))
                    else:
                        # Assuming older versions might return a tensor of scores, 
                        # but usually clip_score returns a scalar mean over the batch unless configured otherwise.
                        # However, for safety, let's treat the result as the MEAN of this batch.
                        scores.append(float(batch_score))
    finally:
        if csv_file:
            csv_file.close()

    # 5. Final Calculation
    if scores:
        # Since torchmetrics clip_score usually returns the mean of the batch,
        # we can average the batch means (assuming equal batch sizes roughly)
        # or better yet, accumulate sum and count if you want exact precision.
        # For simplicity here, we average the batch results.
        final_avg = sum(scores) / len(scores)
        print(f"\nProcessed {len(image_files)} images.")
        print(f"Average CLIP Score: {final_avg:.4f}")
    else:
        print("Could not calculate score (no valid images/prompts pairs).")

if __name__ == "__main__":
    main()