import torch
from typing import List
from pytorch3d.structures import Meshes
from .utils import MeshUtils

class ScenePhysics:
    def __init__(self, device="cuda"):
        self.device = device

    def create_reflections(self, meshes: List[Meshes], mirror_z_pos: float) -> List[Meshes]:
        """
        Creates reflected copies of objects across a mirror plane at Z = mirror_z_pos.
        """
        reflected_meshes = []

        for mesh in meshes:
            verts = mesh.verts_packed().clone()
            
            verts[:, 2] = -verts[:, 2]
            
            translation_z = 2 * mirror_z_pos
            verts[:, 2] = verts[:, 2] + translation_z

            faces = mesh.faces_list()[0].flip(dims=[1]) 

            new_mesh = Meshes(verts=[verts], faces=[faces])

            new_mesh = MeshUtils.paint_mesh(new_mesh, color=[0.9, 0.9, 0.9], device=self.device)
            
            reflected_meshes.append(new_mesh)

        return reflected_meshes