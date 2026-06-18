import sys
import os
import argparse
import time
import json
import yaml
import torch
from pathlib import Path
from torchvision.transforms.functional import to_pil_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.stage1_mesh.objects_extractors import get_objects_extractor
from src.stage1_mesh.factory import get_text_to_3d_model

from src.stage2_scene.config import SceneConfig
from src.stage2_scene.pipelines.depth import SceneDepthPipeline

from src.stage3_generation.factory import get_t2i_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full Mirror-T2I pipeline with Flux Omini depth conditioning"
    )
    
    # Global settings
    parser.add_argument(
        "--prompts_file", "-p",
        type=str,
        default="data/prompts.txt",
        help="Path to text file containing text prompts"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of prompts to process"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="outputs/pipeline_run_001",
        help="Output directory for all generated files"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/inference.yaml",
        help="Path to YAML config file for Stage 2 setting"
    )
    parser.add_argument(
        "--start_stage",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Stage to start pipeline from (1=Mesh, 2=Depth, 3=Generate)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified"
    )
    
    # Stage 1: Mesh Generation
    parser.add_argument(
        "--mesh_model",
        type=str,
        choices=["shap_e", "trellis"],
        default="trellis",
        help="3D mesh generation model to use"
    )
    parser.add_argument(
        "--trellis_model_name",
        type=str,
        default="microsoft/TRELLIS-text-large",
        help="Trellis model name (only used if mesh_model=trellis)"
    )
    
    # Stage 3: Flux Omini
    parser.add_argument(
        "--flux_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Flux model ID from HuggingFace"
    )
    parser.add_argument(
        "--omini_repo",
        type=str,
        default="Yuanshi/OminiControl",
        help="OminiControl repository"
    )
    parser.add_argument(
        "--omini_weight",
        type=str,
        default="experimental/depth.safetensors",
        help="OminiControl LoRA weight file"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_timings(data_list: list, path: str):
    timing_info = []
    for data in data_list:
        timing_info.append({
            "prompt_idx": data['idx'],
            "prompt": data['prompt'],
            "stage1_time": data.get("stage1_time", 0.0),
            "stage2_time": data.get("stage2_time", 0.0),
            "stage3_time": data.get("stage3_time", 0.0),
            "total_time": data.get("total_time", 0.0)
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timing_info, f, indent=4)


def run_pipeline(args):
    # Device setup
    DEVICE = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Stage 2 config
    cfg = load_config(args.config)
    s2 = cfg["stage2"]
    
    S2_CONFIG = SceneConfig(
        gap=s2["gap"],
        object_scale=s2["object_scale"],
        object_base_rotation=s2["object_base_rotation"],
        include_object_random_rotation=s2["random_rotation"],
        mirror_gap_ahead=s2["mirror_gap_ahead"],
        mirror_gap_side=s2["mirror_gap_side"],
        mirror_gap_top=s2["mirror_gap_top"],
        mirror_thickness=s2["mirror_thickness"],
        mirror_height=s2.get("mirror_height"),
        camera_method=s2["camera_method"],
        camera_dist_multiplier=s2["camera_dist"],
        camera_elevation=s2["camera_elevation"],
        camera_look_at_height=s2["camera_look_at_height"],
        camera_azim_min=s2["camera_azim_min"],
        camera_azim_max=s2["camera_azim_max"],
        render_size=s2["render_size"],
        include_floor=s2["include_floor"],
        include_walls=s2["include_walls"],
        include_mirror_frame=s2["include_mirror_frame"],
        include_mirror_surface=s2["include_mirror_surface"],
        include_mirror_wall=s2["include_mirror_wall"],
    )

    # Setup Directories
    os.makedirs(args.output_dir, exist_ok=True)
    timing_path = os.path.join(args.output_dir, "inference_times.json")
    
    # Load prompts
    if not os.path.exists(args.prompts_file):
        print(f"[ERROR] Prompts file not found: {args.prompts_file}")
        return
        
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
        
    if args.limit is not None:
        prompts = prompts[:args.limit]
        
    print(f"=== STARTING FULL PIPELINE: {len(prompts)} prompts ===")
    
    prompt_data_list = []
    existing_timing = {}
    if os.path.exists(timing_path):
        with open(timing_path, "r", encoding="utf-8") as f:
            try:
                loaded = json.load(f)
                for t in loaded:
                    existing_timing[t["prompt_idx"]] = t
            except:
                pass
    
    for prompt_idx, prompt in enumerate(prompts):
        prompt_dir = os.path.join(args.output_dir, f"prompt_{prompt_idx:03d}")
        os.makedirs(prompt_dir, exist_ok=True)
        mesh_dir = os.path.join(prompt_dir, "stage1_meshes")
        os.makedirs(mesh_dir, exist_ok=True)
        
        prev_t = existing_timing.get(prompt_idx, {})
        
        prompt_data_list.append({
            "idx": prompt_idx,
            "prompt": prompt,
            "dir": prompt_dir,
            "mesh_dir": mesh_dir,
            "mesh_paths": [],
            "stage1_time": prev_t.get("stage1_time", 0.0),
            "stage2_time": prev_t.get("stage2_time", 0.0),
            "stage3_time": prev_t.get("stage3_time", 0.0),
            "total_time": prev_t.get("total_time", 0.0)
        })

    # =========================================================================
    # STAGE 1: TEXT TO 3D MESH
    # =========================================================================
    if args.start_stage <= 1:
        print("\n[Stage 1] Extracting objects and generating 3D meshes...")
        
        extractor = get_objects_extractor("simple2")
        if args.mesh_model == "trellis":
            mesh_model = get_text_to_3d_model(
                "trellis",
                device=DEVICE,
                model_name=args.trellis_model_name
            )
            mesh_ext = ".glb"
        else:
            mesh_model = get_text_to_3d_model("shap_e", device=DEVICE)
            mesh_ext = ".obj"

        for data in prompt_data_list:
            print(f"\n > Prompt [{data['idx']+1}/{len(prompts)}]: '{data['prompt']}'")
            t0 = time.time()
            
            object_descriptions = extractor.extract(data["prompt"])
            print(f"   - Found objects: {object_descriptions}")
            
            generated_mesh_paths = []
            for i, obj_prompt in enumerate(object_descriptions):
                safe_name = obj_prompt.replace(" ", "_")
                safe_name = "".join(c for c in safe_name if c.isalnum() or c in " -_").strip()[:64]
                save_path = os.path.join(data["mesh_dir"], f"{i}_{safe_name}{mesh_ext}")

                print(f"   - Generating mesh for: '{obj_prompt}'...")
                mesh_model.generate(obj_prompt, save_path)
                generated_mesh_paths.append(save_path)
                
            t1 = time.time()
            data["mesh_paths"] = generated_mesh_paths
            data["stage1_time"] = t1 - t0
            data["total_time"] = data["stage1_time"] + data["stage2_time"] + data["stage3_time"]
            save_timings(prompt_data_list, timing_path)

        print("\n > Stage 1 Complete. Unloading mesh model to free memory...")
        del mesh_model
        del extractor
        torch.cuda.empty_cache()
    else:
        print("\n[Stage 1] Skipped (assuming meshes exist in output directories)...")
        # Load existing meshes
        import glob
        for data in prompt_data_list:
            existing_meshes = glob.glob(os.path.join(data["mesh_dir"], "*.*"))
            existing_meshes = [m for m in existing_meshes if m.endswith(".glb") or m.endswith(".obj")]
            data["mesh_paths"] = sorted(existing_meshes)

    # =========================================================================
    # STAGE 2: SCENE COMPOSITION & DEPTH RENDERING
    # =========================================================================
    if args.start_stage <= 2:
        print("\n[Stage 2] Composing scenes and rendering depth maps...")
        
        pipeline_s2 = SceneDepthPipeline(S2_CONFIG, device=DEVICE)
        
        for data in prompt_data_list:
            print(f"\n > Prompt [{data['idx']+1}/{len(prompts)}]: '{data['prompt']}'")
            if not data["mesh_paths"]:
                print(f"   [WARNING] No meshes found for prompt {data['idx']}, skipping Scene generation.")
                continue

            t0 = time.time()
            
            s2_results = pipeline_s2.run(data["mesh_paths"])
            depth_map_tensor = s2_results["depth_map"] # Shape [1, 512, 512]
            
            # Save debug depth map
            depth_image = to_pil_image(depth_map_tensor.squeeze().cpu())
            debug_depth_path = os.path.join(data["dir"], "stage2_depth.png")
            depth_image.save(debug_depth_path)
            
            t1 = time.time()
            
            data["stage2_time"] = t1 - t0
            data["depth_map_path"] = debug_depth_path
            data["depth_tensor"] = depth_map_tensor.cpu()
            
            data["total_time"] = data["stage1_time"] + data["stage2_time"] + data["stage3_time"]
            save_timings(prompt_data_list, timing_path)
            
        print("\n > Stage 2 Complete. Unloading scene pipeline to free memory...")
        del pipeline_s2
        torch.cuda.empty_cache()
    else:
        print("\n[Stage 2] Skipped (assuming depth maps exist in output directories)...")

    # =========================================================================
    # STAGE 3: CONDITIONED IMAGE GENERATION (FLUX OMINI)
    # =========================================================================
    if args.start_stage <= 3:
        print("\n[Stage 3] Generating final images with Flux Omini...")
        
        model_s3 = get_t2i_model("flux_omini")
        model_s3.load_model(
            model_id=args.flux_id,
            lora_repo=args.omini_repo,
            lora_weight_name=args.omini_weight,
            adapter_name="depth"
        )
        
        for data in prompt_data_list:
            print(f"\n > Prompt [{data['idx']+1}/{len(prompts)}]: '{data['prompt']}'")
            t0 = time.time()
            
            if "depth_tensor" in data:
                depth_tensor = data["depth_tensor"].to(DEVICE)
            else:
                depth_path = os.path.join(data["dir"], "stage2_depth.png")
                if not os.path.exists(depth_path):
                    print(f"   [WARNING] Missing depth map for prompt {data['idx']} at {depth_path}, skipping Stage 3.")
                    continue
                from PIL import Image
                import torchvision.transforms.functional as TF
                img = Image.open(depth_path).convert("L")
                depth_tensor = TF.to_tensor(img).to(DEVICE)
            
            final_image = model_s3.generate(
                prompt=data["prompt"],
                condition_image=depth_tensor,
                width=S2_CONFIG.render_size,
                height=S2_CONFIG.render_size,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed
            )
            
            final_output_path = os.path.join(data["dir"], "final_result.png")
            final_image.save(final_output_path)
            
            t1 = time.time()
            
            data["stage3_time"] = t1 - t0
            data["total_time"] = data["stage1_time"] + data["stage2_time"] + data["stage3_time"]
            
            print(f"   - Saved final image: {final_output_path}")
            save_timings(prompt_data_list, timing_path)

    print(f"\n=== PIPELINE FINISHED FOR {len(prompts)} PROMPTS ===")
    print(f"Inference times saved to: {timing_path}")


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.empty_cache()
    run_pipeline(args)
