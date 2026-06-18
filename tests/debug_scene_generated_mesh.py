import sys
import os
import torch
import argparse
import shutil
from pytorch3d.io import save_obj

# -----------------------------------------------------------------------------
# 1. SETUP & IMPORTS
# -----------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stage 1
from src.stage1_mesh.objects_extractors import SimpleSplitObjectsExtractor2
from src.stage1_mesh.shap_e_model import ShapEModel

# Stage 2 (Builder Only)
from src.stage2_scene.config import SceneConfig
from src.stage2_scene.builder import SceneBuilder

# -----------------------------------------------------------------------------
# 2. MAIN DEBUG LOGIC
# -----------------------------------------------------------------------------
def debug_geometry_integration(args):
    print("="*60)
    print("DEBUG: Stage 1 + Stage 2 (Geometry Inspection Only)")
    print("="*60)

    # Configs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "outputs/debug_mesh_integration"
    mesh_dir = os.path.join(output_dir, "meshes")
    
    # Clean up previous run if force is enabled to ensure we see new meshes
    if args.force and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    os.makedirs(mesh_dir, exist_ok=True)
    
    prompt = "A wooden chair in front of the mirror"
    
    # Parse orientation
    try:
        orientation_list = [float(x) for x in args.orientation.split(',')]
        if len(orientation_list) != 3: raise ValueError
    except:
        print(f"[Error] Invalid orientation: '{args.orientation}'. Use 'x,y,z' (e.g., '270,0,0')")
        return

    # =========================================================================
    # STAGE 1: GENERATE MESHES
    # =========================================================================
    print(f"\n[Stage 1] Generating Meshes from Prompt: '{prompt}'")
    print(f" > Applied Orientation: {orientation_list}")
    
    # 1. Extract
    extractor = SimpleSplitObjectsExtractor2()
    object_prompts = extractor.extract(prompt)
    print(f" > Extracted Objects: {object_prompts}")
    
    # 2. Generate (Shap-E)
    shape_model = ShapEModel(device=device, orientation=orientation_list)
    generated_paths = []
    
    for i, obj_prompt in enumerate(object_prompts):
        safe_name = obj_prompt.replace(" ", "_")
        save_path = os.path.join(mesh_dir, f"{i}_{safe_name}.obj")
        
        if not os.path.exists(save_path):
            print(f" > Generating: '{obj_prompt}'...")
            shape_model.generate(obj_prompt, save_path)
        else:
            print(f" > Using cached: {save_path}")
            
        generated_paths.append(save_path)

    # =========================================================================
    # STAGE 2: BUILD SCENE (No Rendering)
    # =========================================================================
    print("\n[Stage 2] Building Scene Geometry...")
    
    # 1. Config (Just for layout)
    cfg = SceneConfig(
        gap=0.5,
        mirror_gap_ahead=3.0,
        include_mirror_surface=True, # We want to see the glass in the debug mesh
        include_mirror_frame=True,
        include_floor=True,
        include_walls=True
    )
    
    # 2. Init Builder
    builder = SceneBuilder(cfg, device=device)
    
    # 3. Build & Arrange
    scene_dict = builder.build(generated_paths)
    
    # 4. Merge Everything for Inspection
    # This combines objects, mirror, reflections, floor, walls into one massive mesh
    full_mesh = builder.get_complete_scene(scene_dict)
    
    # 5. Save Final OBJ
    scene_save_path = os.path.join(output_dir, "full_scene_debug.obj")
    save_obj(scene_save_path, full_mesh.verts_packed(), full_mesh.faces_packed())
    
    print(f"\n[Success] Scene saved to: {scene_save_path}")
    print("Instructions:")
    print("1. Open 'outputs/debug_mesh_integration/full_scene_debug.obj' in MeshLab or Blender.")
    print("2. Check if the chair is standing upright on the floor.")
    print("3. Check if the reflection (behind the mirror) matches the object.")
    print("4. If orientation is wrong, re-run with different --orientation angles.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orientation", type=str, default="0,0,0", help="Rotation (x,y,z) in degrees.")
    parser.add_argument("--force", action="store_true", help="Delete previous outputs and regenerate.")
    args = parser.parse_args()
    
    debug_geometry_integration(args)