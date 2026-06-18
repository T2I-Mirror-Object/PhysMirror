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
from src.stage2_scene.pipelines.depth import SceneDepthPipeline

from src.stage3_generation.factory import get_t2i_model

def parse_args():
    parser = argparse.ArgumentParser(description="Mirror-T2I: Multi-Condition Single Prompt Inference")

    # --- INPUT / OUTPUT ---
    parser.add_argument("--prompt", type=str, required=True, help="The text prompt describing the scene.")
    parser.add_argument("--output_dir", type=str, default="outputs/inference_result", help="Directory to save results.")
    
    # --- MODEL PATHS ---
    parser.add_argument("--flux_path", type=str, default="black-forest-labs/FLUX.1-dev", help="Path to base Flux model or HF ID.")
    parser.add_argument("--seg_lora_path", type=str, required=True, help="Path to Seg2Any 'seg_flux_dev' folder.")
    parser.add_argument("--omini_repo", type=str, default="Yuanshi/OminiControl", help="HuggingFace repo for OminiControl.")
    parser.add_argument("--omini_weight", type=str, default="experimental/depth.safetensors", help="OminiControl LoRA weight file.")
    
    # --- GENERATION SETTINGS ---
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # --- SCENE SETTINGS (STAGE 2) ---
    
    # Object placement
    parser.add_argument("--gap", type=float, default=0.5, help="Gap between objects in the scene.")
    parser.add_argument("--object_scale", type=float, default=1.5, help="Global scale applied to objects.")
    parser.add_argument("--object_base_rotation", type=float, default=180.0, help="Base rotation applied to objects.")
    parser.add_argument("--enable_random_rotation", action="store_true", 
                        help="Pass this flag to enable random rotation of objects (defaults to True).")

    # Mirror
    parser.add_argument("--mirror_gap_ahead", type=float, default=1.7, help="Distance from objects to the mirror.")
    parser.add_argument("--mirror_gap_side", type=float, default=2.0, help="Side gap for mirror placement.")
    parser.add_argument("--mirror_gap_top", type=float, default=2.0, help="Top gap for mirror placement.")
    parser.add_argument("--mirror_thickness", type=float, default=0.1, help="Thickness of the mirror.")
    parser.add_argument("--mirror_height", type=float, default=None, 
                        help="Set a specific height to fix mirror height and auto-scale objects. (null/None = dynamic)")

    # Camera
    parser.add_argument("--camera_method", type=str, default="random_side", choices=["random_side", "fixed_front"], 
                        help="Camera placement strategy.")
    parser.add_argument("--camera_dist", type=float, default=2.5, help="Multiplier for camera distance.")
    parser.add_argument("--camera_elevation", type=float, default=26.0, help="Camera elevation angle.")
    parser.add_argument("--camera_look_at_height", type=float, default=1.8, help="Height of the camera's target point.")
    parser.add_argument("--camera_azim_min", type=float, default=-10.0, help="Minimum azimuth angle for random side.")
    parser.add_argument("--camera_azim_max", type=float, default=10.0, help="Maximum azimuth angle for random side.")

    # Rendering
    parser.add_argument("--render_size", type=int, default=1024, help="Render size for the spatial prior maps.")

    # Scene elements (Booleans)
    parser.add_argument("--no_floor", action="store_false", dest="include_floor", 
                        help="Pass this flag to exclude the floor (defaults to True).")
    parser.add_argument("--no_walls", action="store_false", dest="include_walls", 
                        help="Pass this flag to exclude walls (defaults to True).")
    parser.add_argument("--no_mirror_frame", action="store_false", dest="include_mirror_frame", 
                        help="Pass this flag to exclude the mirror frame (defaults to True).")
    parser.add_argument("--include_mirror_surface", action="store_true", 
                        help="Pass this flag to include the mirror surface (defaults to False).")
    parser.add_argument("--no_mirror_wall", action="store_false", dest="include_mirror_wall", 
                        help="Pass this flag to exclude the mirror wall (defaults to True).")

    # --- STAGE 1 MODEL SELECTION ---
    parser.add_argument("--mesh_model", type=str, default="shap_e", choices=["shap_e", "trellis"])
    parser.add_argument("--trellis_model_name", type=str, default="microsoft/TRELLIS-text-large")

    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup Directories
    os.makedirs(args.output_dir, exist_ok=True)
    mesh_dir = os.path.join(args.output_dir, "stage1_meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    print(f"=== STARTING MULTI-CONDITION INFERENCE PIPELINE ===")
    print(f"Prompt: '{args.prompt}'")

    # =========================================================================
    # STAGE 1: TEXT TO 3D MESH
    # =========================================================================
    print("\n[Stage 1] Extracting objects and generating 3D meshes...")
    
    extractor = get_objects_extractor("simple2")
    object_prompts = extractor.extract(args.prompt)
    
    if not object_prompts:
        print("[Error] No objects extracted from prompt! Exiting.")
        return

    print(f" > Extracted Objects: {object_prompts}")

    if args.mesh_model == "trellis":
        mesh_model = get_text_to_3d_model("trellis", device=device, model_name=args.trellis_model_name, seed=args.seed)
        mesh_ext = ".glb"
    else:
        mesh_model = get_text_to_3d_model("shap_e", device=device)
        mesh_ext = ".obj"

    generated_mesh_paths = []
    for i, obj_prompt in enumerate(object_prompts):
        safe_name = obj_prompt.replace(" ", "_")[:20]
        save_path = os.path.join(mesh_dir, f"{i}_{safe_name}{mesh_ext}")
        
        print(f" > Generating mesh for: '{obj_prompt}'...")
        mesh_model.generate(obj_prompt, save_path)
        generated_mesh_paths.append(save_path)

    print(f"[Stage 1] Completed.")
    del mesh_model
    del extractor
    torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 2: SCENE RENDERING (SEGMENTATION + DEPTH)
    # =========================================================================
    print("\n[Stage 2] Composing scene and rendering spatial priors...")

    # 1. Config for Segmentation Map
    s2_config_seg = SceneConfig(
        gap=args.gap,
        object_scale=args.object_scale,
        object_base_rotation=args.object_base_rotation,
        include_object_random_rotation=args.enable_random_rotation,
        mirror_gap_ahead=args.mirror_gap_ahead,
        mirror_gap_side=args.mirror_gap_side,
        mirror_gap_top=args.mirror_gap_top,
        mirror_thickness=args.mirror_thickness,
        mirror_height=args.mirror_height,
        camera_method=args.camera_method,
        camera_dist_multiplier=args.camera_dist,
        camera_elevation=args.camera_elevation,
        camera_look_at_height=args.camera_look_at_height,
        camera_azim_min=args.camera_azim_min,
        camera_azim_max=args.camera_azim_max,
        render_size=args.render_size,
        include_floor=args.include_floor,
        include_walls=args.include_walls,
        include_mirror_frame=args.include_mirror_frame,
        include_mirror_surface=args.include_mirror_surface,
        include_mirror_wall=args.include_mirror_wall,
        renderer_type="segmentation",
    )

    pipeline_seg = SceneSegmentationPipeline(s2_config_seg, device=device)
    s2_seg_results = pipeline_seg.run(
        object_paths=generated_mesh_paths,
        object_prompts=object_prompts,
        global_caption=args.prompt
    )
    
    seg_map_tensor = s2_seg_results["segmentation_map"] 
    json_data = s2_seg_results["json_data"]
    camera_pose = s2_seg_results.get("camera_pose", None) # Crucial: Extract camera to align depth

    # 2. Config for Depth Map
    s2_config_depth = SceneConfig(
        gap=args.gap,
        object_scale=args.object_scale,
        object_base_rotation=args.object_base_rotation,
        include_object_random_rotation=args.enable_random_rotation,
        mirror_gap_ahead=args.mirror_gap_ahead,
        mirror_gap_side=args.mirror_gap_side,
        mirror_gap_top=args.mirror_gap_top,
        mirror_thickness=args.mirror_thickness,
        mirror_height=args.mirror_height,
        camera_method=args.camera_method,
        camera_dist_multiplier=args.camera_dist,
        camera_elevation=args.camera_elevation,
        camera_look_at_height=args.camera_look_at_height,
        camera_azim_min=args.camera_azim_min,
        camera_azim_max=args.camera_azim_max,
        render_size=args.render_size,
        include_floor=args.include_floor,
        include_walls=args.include_walls,
        include_mirror_frame=args.include_mirror_frame,
        include_mirror_surface=args.include_mirror_surface,
        include_mirror_wall=args.include_mirror_wall,
        renderer_type="depth",
    )

    pipeline_depth = SceneDepthPipeline(s2_config_depth, device=device)
    s2_depth_results = pipeline_depth.run(
        object_paths=generated_mesh_paths,
    )
    
    depth_map_tensor = s2_depth_results["depth_map"]

    # --- Save Debug Outputs ---
    json_path = os.path.join(args.output_dir, "seg_metadata.json")
    with open(json_path, 'w') as f: json.dump(json_data, f, indent=2)
    
    seg_img_np = seg_map_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    if seg_img_np.max() <= 1.0: seg_img_np = (seg_img_np * 255).astype(np.uint8)
    Image.fromarray(seg_img_np).save(os.path.join(args.output_dir, "debug_seg_map.png"))

    depth_img_np = depth_map_tensor.squeeze().cpu().numpy()
    if depth_img_np.max() <= 1.0: depth_img_np = (depth_img_np * 255).astype(np.uint8)
    Image.fromarray(depth_img_np, mode='L').save(os.path.join(args.output_dir, "debug_depth_map.png"))
    
    print(f"[Stage 2] Completed. Spatial priors rendered and aligned.")
    del pipeline_seg
    del pipeline_depth
    torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 3: GENERATION (MULTI-CONDITION)
    # =========================================================================
    print("\n[Stage 3] Generating final image with Multi-Condition Wrapper...")

    try:
        model_s3 = get_t2i_model("multi_cond") # Assuming you register our new class as "multi_cond"

        model_s3.load_model(
            pretrained_model_path=args.flux_path,
            seg_lora_path=args.seg_lora_path,
            omini_lora_repo=args.omini_repo,
            omini_lora_weight=args.omini_weight,
            weight_dtype_str="bf16"
        )

        final_image = model_s3.generate(
            prompt=args.prompt,
            seg_image=seg_map_tensor, 
            depth_image=depth_map_tensor, 
            meta_data=json_data,            
            width=args.width,
            height=args.height,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )

        final_output_path = os.path.join(args.output_dir, "final_result_multicond.png")
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