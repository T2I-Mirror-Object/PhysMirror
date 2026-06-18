import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from typing import List

class MeshUtils:
    @staticmethod
    def get_bounds(mesh: Meshes) -> torch.Tensor:
        """Returns [2, 3] tensor (min_xyz, max_xyz)."""
        verts = mesh.verts_packed()
        return torch.stack([verts.min(dim=0)[0], verts.max(dim=0)[0]])

    @staticmethod
    def get_centroid(mesh: Meshes) -> torch.Tensor:
        verts = mesh.verts_packed()
        return verts.mean(dim=0)

    @staticmethod
    def translate(mesh: Meshes, translation: torch.Tensor) -> Meshes:
        """Applies translation [x, y, z]."""
        verts = mesh.verts_packed()
        new_verts = verts + translation
        return Meshes(verts=[new_verts], faces=mesh.faces_list())

    @staticmethod
    def rotate_y(mesh: Meshes, angle_rad: float, device="cuda") -> Meshes:
        """Rotates mesh around its OWN centroid along Y-axis."""
        centroid = MeshUtils.get_centroid(mesh)
        
        # Create rotation matrix
        cos_a = torch.cos(torch.tensor(angle_rad))
        sin_a = torch.sin(torch.tensor(angle_rad))
        
        R = torch.tensor([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=torch.float32, device=device)

        # 1. Center -> 2. Rotate -> 3. Move back
        verts = mesh.verts_packed()
        centered = verts - centroid
        rotated = torch.matmul(centered, R.t())
        final_verts = rotated + centroid
        
        return Meshes(verts=[final_verts], faces=mesh.faces_list())
    
    @staticmethod
    def scale(mesh: Meshes, factor: float) -> Meshes:
        """Scales mesh uniformly around its centroid."""
        centroid = MeshUtils.get_centroid(mesh)
        verts = mesh.verts_packed()
        new_verts = (verts - centroid) * factor + centroid
        return Meshes(verts=[new_verts], faces=mesh.faces_list())

    @staticmethod
    def get_group_bounds(meshes: List[Meshes]) -> torch.Tensor:
        """Returns the min/max XYZ for the whole group of meshes."""
        if not meshes:
            return torch.zeros((2, 3))
            
        min_vals = []
        max_vals = []
        
        for m in meshes:
            bounds = MeshUtils.get_bounds(m)
            min_vals.append(bounds[0])
            max_vals.append(bounds[1])
            
        # Stack and find absolute min/max
        group_min = torch.stack(min_vals).min(dim=0)[0]
        group_max = torch.stack(max_vals).max(dim=0)[0]
        
        return torch.stack([group_min, group_max])

    @staticmethod
    def get_group_dims(meshes: List[Meshes]):
        """Returns (width, height, depth) of the group."""
        bounds = MeshUtils.get_group_bounds(meshes)
        dims = bounds[1] - bounds[0]
        return dims[0].item(), dims[1].item(), dims[2].item()

    @staticmethod
    def paint_mesh(mesh: Meshes, color: list = [1.0, 1.0, 1.0], device="cuda") -> Meshes:
        """
        Force assigns a solid VertexTexture to the mesh.
        Handles both single meshes and batches safely.
        """
        # 1. Extract Geometry
        verts_list = mesh.verts_list()
        faces_list = mesh.faces_list()
        
        # 2. Create Texture for each mesh in the batch
        new_features = []
        color_tensor = torch.tensor(color, dtype=torch.float32, device=device)
        
        for v in verts_list:
            # Create (N_verts, 3) color block
            vertex_colors = color_tensor[None].repeat(v.shape[0], 1)
            new_features.append(vertex_colors)
            
        textures = TexturesVertex(verts_features=new_features)
        
        # 3. Return Reconstructed Mesh
        return Meshes(verts=verts_list, faces=faces_list, textures=textures)