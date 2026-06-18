import os
import sys

# Set SPCONV_ALGO to 'native' before importing TRELLIS modules
# 'auto' mode does benchmarking which can cause floating point exceptions
os.environ['SPCONV_ALGO'] = 'native'

import torch

# Add TRELLIS to path
# __file__ is src/stage1_mesh/trellis_model.py
# Go up 3 levels to Mirror-T2I/, then into TRELLIS/
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "TRELLIS")
if TRELLIS_PATH not in sys.path:
    sys.path.insert(0, TRELLIS_PATH)

from .base import BaseTextTo3D


class TrellisModel(BaseTextTo3D):
    """
    Text-to-3D generation using Microsoft TRELLIS.

    TRELLIS provides high-quality 3D asset generation from text prompts
    using a structured latent representation (SLAT).
    """

    def __init__(
        self,
        # Base args
        device: str = "cuda",
        output_dir: str = "outputs/meshes",
        # Trellis specific args
        model_name: str = "microsoft/TRELLIS-text-large",
        seed: int = 42,
        simplify: float = 0.95,
        texture_size: int = 1024,
    ):
        """
        Initialize TRELLIS text-to-3D generator.

        Args:
            device: Device for inference ('cuda' or 'cpu')
            output_dir: Directory to save output meshes
            model_name: Pretrained model name. Options:
                - "microsoft/TRELLIS-text-base" (342M params)
                - "microsoft/TRELLIS-text-large" (1.1B params)
                - "microsoft/TRELLIS-text-xlarge" (2.0B params)
            seed: Random seed for reproducibility
            simplify: Mesh simplification ratio (0.0-1.0)
            texture_size: Texture resolution for exported mesh
        """
        super().__init__(device=device, output_dir=output_dir)

        self.model_name = model_name
        self.seed = seed
        self.simplify = simplify
        self.texture_size = texture_size

        # Lazy loading state
        self.pipeline = None

    def _load_model(self):
        """Internal helper: Load weights only when necessary."""
        if self.pipeline is not None:
            return

        from trellis.pipelines import TrellisTextTo3DPipeline

        print(f"[Trellis] Loading model: {self.model_name}...")
        self.pipeline = TrellisTextTo3DPipeline.from_pretrained(self.model_name)
        if self.device == "cuda":
            self.pipeline.cuda()
        print("[Trellis] Model loaded successfully")

    def generate(self, prompt: str, save_path: str) -> str:
        """
        Generates 3D mesh from text prompt.

        Args:
            prompt: Full description (e.g., "a red chair")
            save_path: Path to save the mesh (e.g., "outputs/meshes/red_chair.glb")

        Returns:
            str: The absolute path to the saved .glb file.
        """
        # Ensure save_path has .glb extension (Trellis outputs GLB format)
        if not save_path.endswith(".glb"):
            # Replace extension or add .glb
            base, ext = os.path.splitext(save_path)
            save_path = base + ".glb"

        # Caching check
        if os.path.exists(save_path):
            print(f"[Trellis] Found cached mesh: {save_path}. Skipping generation.")
            return save_path

        # Load model if not already loaded
        self._load_model()

        from trellis.utils import postprocessing_utils

        print(f"[Trellis] Generating: '{prompt}'")

        # Generate 3D representations
        outputs = self.pipeline.run(prompt, seed=self.seed)

        # Ensure parent directory exists
        parent_dir = os.path.dirname(save_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # Convert to GLB with mesh simplification and texturing
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=self.simplify,
            texture_size=self.texture_size,
        )
        glb.export(save_path)

        print(f"[Trellis] Saved to {save_path}")
        return save_path

    def unload_model(self):
        """Unload model to free GPU memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            print("[Trellis] Model unloaded, GPU memory freed")
