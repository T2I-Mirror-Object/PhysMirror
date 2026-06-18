"""
Standalone Stage 1 inference script.

Reads object names from a text file and generates 3D meshes 
using the TRELLIS or Shap-E model. All meshes are saved directly into a single output 
directory, named after the object name.

Usage:
    python scripts/generate_meshes.py --objects_file data/objects.txt --output_dir output/meshes
"""

import sys
import os
import argparse
import traceback
import torch

# Ensure the src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage1_mesh.factory import get_text_to_3d_model


def load_objects(path: str) -> list:
    """Reads object names from a text file, ignoring empty lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run(args):
    print("=" * 70)
    print("Standalone Text-to-3D Mesh Generation")
    print(f"Objects File: {args.objects_file}")
    print(f"Output Dir:   {args.output_dir}")
    print(f"Mesh Model:   {args.mesh_model}")
    if args.mesh_model == "trellis":
        print(f"Trellis Model:{args.model_name}")
    print("=" * 70)

    # 1. Setup Device and Directories
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load Objects
    if not os.path.exists(args.objects_file):
        print(f"[ERROR] Objects file not found: {args.objects_file}")
        sys.exit(1)
        
    objects = load_objects(args.objects_file)
    print(f"Loaded {len(objects)} object names.\n")

    if not objects:
        print("No objects to process. Exiting.")
        return

    # 3. Initialize Model
    print(f"Loading {args.mesh_model.upper()} model...")
    if args.mesh_model == "trellis":
        mesh_model = get_text_to_3d_model(
            "trellis", 
            device=device, 
            model_name=args.model_name
        )
        mesh_ext = ".glb"  # TRELLIS outputs GLB format
    else:
        mesh_model = get_text_to_3d_model("shap_e", device=device)
        mesh_ext = ".obj"

    # 4. Process Objects

    for idx, obj_name in enumerate(objects):
        print(f"\n[{idx + 1}/{len(objects)}] Processing: '{obj_name}'")

        try:
            # Sanitize the string to make it a safe, valid filename
            safe_name = "".join(c for c in obj_name if c.isalnum() or c in " -_").strip()
            safe_name = safe_name.replace(" ", "_")[:64]
            
            if not safe_name:
                safe_name = "unknown_object"

            filename = f"{safe_name}{mesh_ext}"
            save_path = os.path.join(args.output_dir, filename)
            
            # Check if this exact object was already generated
            if os.path.exists(save_path):
                print(f"  -> Skipping: '{filename}' already exists.")
                continue
            
            print(f"  -> Generating mesh for: '{obj_name}'")
            mesh_model.generate(obj_name, save_path)
            print(f"  -> Saved to: {save_path}")

        except Exception as e:
            print(f"  [ERROR] Generation failed for object: '{obj_name}'")
            traceback.print_exc()

    # 5. Cleanup
    print("\nGeneration complete. Unloading model and clearing VRAM...")
    del mesh_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 3D meshes from object names using TRELLIS or Shap-E.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required Arguments
    parser.add_argument("--objects_file", "-f", type=str, required=True,
                        help="Path to the text file containing object names (one per line).")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Directory where the generated meshes will be saved.")
    
    # Optional Overrides
    parser.add_argument("--mesh_model", type=str, choices=["shap_e", "trellis"], default="trellis",
                        help="3D mesh generation model to use.")
    parser.add_argument("--model_name", "-m", type=str, default="microsoft/TRELLIS-text-large",
                        help="HuggingFace model ID for TRELLIS (only used if mesh_model=trellis).")
    
    args = parser.parse_args()
    run(args)