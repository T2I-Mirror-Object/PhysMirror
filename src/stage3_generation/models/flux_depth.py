import torch
from PIL import Image
from diffusers import FluxControlPipeline
from typing import Union

from .base import BaseT2IModel

class FluxDepthWrapper(BaseT2IModel):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)

    def load_model(self, 
                   model_id: str = "black-forest-labs/FLUX.1-Depth-dev", 
                   lora_path: str = None, 
                   **kwargs):
        """
        Loads the native Flux Depth model and optionally applies a custom LoRA.
        
        Args:
            model_id: The base Flux Depth model.
            lora_path: Path to your custom-trained SynMirror LoRA weights.
        """
        print(f"[Stage3] Loading Flux Depth pipeline: {model_id}...")
        
        # 1. Load Base Flux Control Pipeline
        self.pipe = FluxControlPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        ).to(self.device)

        # 2. Load Custom LoRA (if provided)
        if lora_path:
            print(f"[Stage3] Loading custom LoRA from: {lora_path}")
            self.pipe.load_lora_weights(lora_path)
            
        print("[Stage3] Flux Depth loaded successfully.")

    def generate(
        self, 
        prompt: str, 
        condition_image: Union[Image.Image, torch.Tensor], 
        negative_prompt: str = "", # Ignored by Flux
        seed: int = 42,
        num_steps: int = 28,
        guidance_scale: float = 3.5,
        width: int = 512,
        height: int = 512,
        **kwargs
    ) -> Image.Image:
        
        # 1. Standardize Input (Tensor -> PIL, ensure RGB)
        img_pil = self._ensure_pil(condition_image)
        
        # 2. Resize Logic
        if img_pil.size != (width, height):
            print(f"[FluxDepth] Resizing depth map from {img_pil.size} to ({width}, {height})")
            img_pil = img_pil.resize((width, height), Image.Resampling.LANCZOS)
            
        # 3. Setup Generator for Reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 4. Generate
        print(f"[FluxDepth] Generating with prompt: '{prompt[:50]}...'")
        result = self.pipe(
            prompt=prompt,
            control_image=img_pil,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        return result

    def _ensure_pil(self, image: Union[Image.Image, torch.Tensor]) -> Image.Image:
        """Helper to convert Tensor to PIL and ensure RGB format for Flux Depth."""
        pil_image = None
        
        # Case 1: Already PIL
        if isinstance(image, Image.Image):
            pil_image = image
            
        # Case 2: Tensor
        elif isinstance(image, torch.Tensor):
            if image.ndim == 4: image = image[0]  # Remove batch
            if image.ndim == 3 and image.shape[0] == 1: 
                image = image.squeeze(0) # Remove channel dim if 1
            
            # If shape is now [H, W], convert to numpy
            arr = (image.cpu().numpy() * 255).astype('uint8')
            pil_image = Image.fromarray(arr)
            
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Flux.1-Depth-dev natively expects 3-channel RGB maps even for depth
        return pil_image.convert("RGB")