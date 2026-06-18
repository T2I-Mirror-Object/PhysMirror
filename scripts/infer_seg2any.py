import sys
import os
import torch
import json
import argparse
import numpy as np
from PIL import Image

# Add project root to path so we can import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage1_mesh.objects_extractors import get_objects_extractor
from src.stage1_mesh.factory import get_text_to_3d_model

from src.stage2_scene.config import SceneConfig
from src.stage2_scene.pipelines.segmentation import SceneSegmentationPipeline

from src.stage3_generation.factory import get_t2i_model

def parse_args():
    parser = argparse.ArgumentParser(description="Mirror-T2I: Full Pipeline Inference with Seg2Any")

    # --- INPUT / OUTPUT ---
    parser.add_argument("--prompt", type=str, required=True, help="The text prompt describing the scene.")
    parser.add_argument("--output_dir", type=str, default="outputs/inference_result", help="Directory to save results.")
    
    # --- MODEL PATHS ---
    parser.add_argument("--flux_path", type=str, default="black-forest-labs/FLUX.1-dev", help="Path to base Flux model or HF ID.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to Seg2Any 'seg_flux_dev' folder containing 'default' and 'cond' subfolders.")
    
    # --- GENERATION SETTINGS ---
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # --- SCENE SETTINGS ---
    parser.add_argument("--gap", type=float, default=0.3, help="Gap between objects in the scene.")
    parser.add_argument("--mirror_dist", type=float, default=3.0, help="Distance from objects to the mirror.")
    parser.add_argument("--camera_method", type=str, default="random_side", help="Camera placement strategy.")
    parser.add_argument("--camera_elevation", type=float, default=10.0, help="Camera elevation angle.")
    parser.add_argument("--camera_look_at_height", type=float, default=1.0, help="Height of the camera's target point.")
    parser.add_argument("--camera_dist_multiplier", type=float, default=1.2, help="Multiplier for camera distance based on mirror gap.")
    parser.add_argument("--camera_azim_min", type=float, default=20.0, help="Minimum azimuth angle for random side.")
    parser.add_argument("--camera_azim_max", type=float, default=25.0, help="Maximum azimuth angle for random side.")
    
    # --- FLOOR/WALL PAINTING ---
    parser.add_argument("--paint_floor", action="store_true", help="If set, floor will be colored and visible in segmentation.")
    parser.add_argument("--paint_walls", action="store_true", help="If set, walls will be colored and visible in segmentation.")

    # --- STAGE 1 MODEL SELECTION ---
    parser.add_argument("--mesh_model", type=str, default="shap_e", choices=["shap_e", "trellis"],
                        help="Text-to-3D model to use: 'shap_e' or 'trellis' (default: shap_e)")
    parser.add_argument("--trellis_model_name", type=str, default="microsoft/TRELLIS-text-large",
                        help="Trellis model variant (only used if --mesh_model=trellis)")

    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup Directories
    os.makedirs(args.output_dir, exist_ok=True)
    mesh_dir = os.path.join(args.output_dir, "stage1_meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    print(f"=== STARTING INFERENCE PIPELINE ===")
    print(f"Prompt: '{args.prompt}'")
    print(f"Output: {args.output_dir}")

    # =========================================================================
    # STAGE 1: TEXT TO 3D MESH
    # =========================================================================
    print("\n[Stage 1] Extracting objects and generating 3D meshes...")
    
    # 1. Extract Objects
    extractor = get_objects_extractor("simple2")
    object_prompts = extractor.extract(args.prompt)
    
    if not object_prompts:
        print("[Error] No objects extracted from prompt! Exiting.")
        return

    print(f" > Extracted Objects: {object_prompts}")

    # 2. Generate Meshes (ShapE or Trellis)
    print(f" > Using mesh model: {args.mesh_model}")

    if args.mesh_model == "trellis":
        mesh_model = get_text_to_3d_model(
            "trellis",
            device=device,
            model_name=args.trellis_model_name,
            seed=args.seed
        )
        mesh_ext = ".glb"
    else:
        mesh_model = get_text_to_3d_model("shap_e", device=device)
        mesh_ext = ".obj"

    generated_mesh_paths = []
    for i, obj_prompt in enumerate(object_prompts):
        safe_name = obj_prompt.replace(" ", "_")[:20]
        save_path = os.path.join(mesh_dir, f"{i}_{safe_name}{mesh_ext}")
        
        # Generate mesh (model handles caching internally)
        print(f" > Generating mesh for: '{obj_prompt}'...")
        mesh_model.generate(obj_prompt, save_path)
            
        generated_mesh_paths.append(save_path)

    print(f"[Stage 1] Completed.")

    # =========================================================================
    # STAGE 2: SCENE SEGMENTATION
    # =========================================================================
    print("\n[Stage 2] Composing scene and rendering segmentation map...")

    # 1. Configuration
    s2_config = SceneConfig(
        gap=args.gap,
        mirror_gap_ahead=args.mirror_dist,
        camera_method=args.camera_method,
        camera_elevation=args.camera_elevation,
        camera_look_at_height=args.camera_look_at_height,
        camera_dist_multiplier=args.camera_dist_multiplier,
        camera_azim_min=args.camera_azim_min,
        camera_azim_max=args.camera_azim_max,
        render_size=max(args.width, args.height), # Render at least as big as output
        
        # Force Segmentation Flags
        renderer_type="segmentation",
        include_mirror_surface=False, 
        include_mirror_frame=True,
        include_floor=True,
        include_walls=True
    )

    # 2. Initialize Pipeline
    pipeline_s2 = SceneSegmentationPipeline(s2_config, device=device)

    # 3. Run Pipeline
    s2_results = pipeline_s2.run(
        object_paths=generated_mesh_paths,
        object_prompts=object_prompts,
        global_caption=args.prompt
    )
    
    # 4. Retrieve Outputs
    seg_map_tensor = s2_results["segmentation_map"] # Shape [1, 3, H, W]
    json_data = s2_results["json_data"]             # Dict
    
    # 5. Save Debug Metadata
    json_path = os.path.join(args.output_dir, "seg_metadata.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save Debug Image
    seg_img_np = seg_map_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    if seg_img_np.max() <= 1.0: seg_img_np = (seg_img_np * 255).astype(np.uint8)
    Image.fromarray(seg_img_np).save(os.path.join(args.output_dir, "debug_seg_map.png"))
    
    print(f"[Stage 2] Completed. Metadata saved to {json_path}")

    # =========================================================================
    # STAGE 3: GENERATION (SEG2ANY)
    # =========================================================================
    print("\n[Stage 3] Generating final image with Seg2Any...")

    try:
        # 1. Initialize Wrapper
        model_s3 = get_t2i_model("seg2any")

        # 2. Load Models
        model_s3.load_model(
            pretrained_model_path=args.flux_path,
            lora_path=args.lora_path,
            weight_dtype_str="bf16"
        )

        # 3. Generate
        final_image = model_s3.generate(
            prompt=args.prompt,
            condition_image=seg_map_tensor, 
            meta_data=json_data,            
            width=args.width,
            height=args.height,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )

        # 4. Save Final Output
        final_output_path = os.path.join(args.output_dir, "final_result.png")
        final_image.save(final_output_path)

        print(f"\n=== SUCCESS ===")
        print(f"Image saved to: {final_output_path}")

    except Exception as e:
        print(f"\n[Stage 3 Error] Failed to generate image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()