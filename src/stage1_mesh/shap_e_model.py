import os
import torch
import numpy as np
import trimesh
from trimesh import transformations

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

# Import the contract we defined earlier
from .base import BaseTextTo3D

class ShapEModel(BaseTextTo3D):
    def __init__(
        self, 
        # Base args
        device: str = "cuda", 
        output_dir: str = "outputs/meshes",
        # Shap-E specific args
        seed: int = 42, 
        guidance: float = 15.0, 
        fp16: bool = True, 
        karras_steps: int = 64,
        sigma_min: float = 1e-3,
        sigma_max: float = 160,
        s_churn: float = 0,
        orientation: list = [-90.0, 180.0, 0.0]  # [x, y, z] in degrees
    ):
        # 1. Initialize the Base Class
        super().__init__(device=device, output_dir=output_dir)
        
        # 2. Store Shap-E configs
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.seed = seed
        self.guidance = guidance
        self.fp16 = fp16
        self.karras_steps = karras_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.s_churn = s_churn
        self.orientation = orientation
        
        # 3. Lazy Loading State (Don't load weights yet!)
        self.xm = None
        self.model = None
        self.diffusion = None

    def _load_model(self):
        """Internal helper: Load weights only when absolutely necessary."""
        if self.model is not None:
            return

        print(f"[ShapE] Loading weights on {self.device}...")
        torch.manual_seed(self.seed)
        
        self.xm = load_model("transmitter", device=self.device)
        self.model = load_model("text300M", device=self.device)
        self.diffusion = diffusion_from_config(load_config("diffusion"))

    def generate(self, prompt: str, save_path: str) -> str:
        """
        Generates 3D mesh. Inherited from BaseTextTo3D.
        
        Args:
            prompt: Full description (e.g., "a red chair")
            save_path: Path to save the mesh (e.g., "outputs/meshes/red_chair.obj")
        """
        # 1. CACHING CHECK (Crucial for speed)
        if os.path.exists(save_path):
            print(f"[ShapE] Found cached mesh: {save_path}. Skipping generation.")
            return save_path

        # 2. If not cached, Load Model & Generate
        self._load_model()

        print(f"[ShapE] Generating: '{prompt}'")
        latents = sample_latents(
            batch_size=1,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=self.guidance,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=self.fp16,
            use_karras=True,
            karras_steps=self.karras_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            s_churn=self.s_churn,
            device=self.device,
        )

        # 3. Decode to Mesh
        tri = decode_latent_mesh(self.xm, latents[0]).tri_mesh()

        # 4. Convert to Trimesh & Rotate
        mesh = trimesh.Trimesh(vertices=tri.verts, faces=tri.faces)
        
        if any(angle != 0.0 for angle in self.orientation):
            self._apply_rotation(mesh)

        parent_dir = os.path.dirname(save_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        if not save_path.endswith(".obj"):
            save_path += ".obj"

        # 5. Save and Return Path
        mesh.export(save_path)
        print(f"[ShapE] Saved to {save_path}")
        
        return save_path

    def _apply_rotation(self, mesh):
        """Helper to keep the main logic clean."""
        rads = np.deg2rad(self.orientation)
        rotation_matrix = transformations.euler_matrix(
            rads[0], rads[1], rads[2], axes='sxyz'
        )
        mesh.apply_transform(rotation_matrix)