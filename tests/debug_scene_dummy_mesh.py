import sys
import os
import torch
from pytorch3d.io import save_obj

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage2_scene.config import SceneConfig
from src.stage2_scene.builder import SceneBuilder

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Config
    cfg = SceneConfig(
        gap=0.5,
        mirror_gap_ahead=3.0
    )
    
    # 2. Init Builder
    builder = SceneBuilder(cfg, device=device)
    
    # 3. Create a dummy .obj file to test loading
    dummy_path = "outputs/debug_cube.obj"
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists(dummy_path):
        print("Creating dummy cube...")
        # (Simple code to write a cube .obj manually or use trimesh)
        import trimesh
        trimesh.creation.box().export(dummy_path)

    # 4. Build Scene
    scene_dict = builder.build([dummy_path, dummy_path]) # Test with 2 objects
    
    # 5. Merge and Save for Visual Inspection
    full_mesh = builder.get_complete_scene(scene_dict)
    save_path = "outputs/debug_stage2_scene.obj"
    
    # Save final mesh
    save_obj(save_path, full_mesh.verts_packed(), full_mesh.faces_packed())
    print(f"Scene saved to {save_path}. Open this in MeshLab/Blender to verify!")

if __name__ == "__main__":
    main()