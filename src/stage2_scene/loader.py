import torch
from pathlib import Path
from typing import List
import trimesh
from pytorch3d.io import load_obj, IO
from pytorch3d.structures import Meshes
from .utils import MeshUtils

class MeshLoader:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.io = IO()

    def load_meshes(self, object_paths: List[str]) -> List[Meshes]:
        """
        Universal mesh loader (OBJ, PLY, GLB/GLTF).
        Returns a list of PyTorch3D Meshes on the correct device.
        """
        loaded_meshes = []
        
        for path_str in object_paths:
            path = Path(path_str)
            suffix = path.suffix.lower()

            try:
                if suffix == '.obj':
                    # PyTorch3D native OBJ loader
                    verts, faces, aux = load_obj(path, device=self.device)
                    mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
                    
                elif suffix == '.ply':
                    # PyTorch3D native PLY loader
                    mesh = self.io.load_mesh(path, device=self.device)
                    
                else:
                    # Fallback to Trimesh (GLB, STL, etc.)
                    # This handles the "import trimesh" logic you had
                    print(f"[Loader] {suffix} format detected. Using Trimesh fallback.")
                    mesh = self._load_via_trimesh(path)

                mesh = MeshUtils.paint_mesh(mesh, color=[1.0, 1.0, 1.0], device=self.device)

                loaded_meshes.append(mesh)
                
            except Exception as e:
                print(f"[Loader] Error loading {path}: {e}")
                # You might want to skip or raise depending on strictness
                raise e

        return loaded_meshes

    def _load_via_trimesh(self, path: Path) -> Meshes:
        """Internal helper for non-native formats."""
        tri_mesh = trimesh.load(path, force='mesh')
        
        verts = torch.tensor(tri_mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(tri_mesh.faces, dtype=torch.int64, device=self.device)
        
        return Meshes(verts=[verts], faces=[faces])