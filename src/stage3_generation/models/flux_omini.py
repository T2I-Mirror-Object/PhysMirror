import os
import torch
from PIL import Image
from diffusers import FluxPipeline
from typing import Union, Optional

from ..utils.flux_omini import Condition, generate, seed_everything

from .base import BaseT2IModel

class FluxOminiWrapper(BaseT2IModel):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.adapter_name = "depth" # Default, can be updated in load_model

    def load_model(self, 
                   model_id: str, 
                   lora_repo: str = None, 
                   lora_weight_name: str = None, 
                   adapter_name: str = "depth",
                   **kwargs):
        """
        Args:
            model_id: The base Flux model (e.g. "black-forest-labs/FLUX.1-schnell")
            lora_repo: The OminiControl LoRA repo
            lora_weight_name: The .safetensors filename
            adapter_name: Internal adapter name (e.g. "depth")
        """
        print(f"[Stage3] Loading Flux Omini: {model_id} + {lora_repo}...")
        self.adapter_name = adapter_name

        # 1. Load Base Flux Pipeline
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 # Flux usually likes bf16
        ).to(self.device)

        # 2. Load Omini LoRA
        if lora_repo and lora_weight_name:
            print(f"[Stage3] Loading LoRA: {lora_weight_name}")
            self.pipe.unload_lora_weights() # Safety cleanup
            self.pipe.load_lora_weights(
                lora_repo,
                weight_name=lora_weight_name,
                adapter_name=adapter_name,
            )
            self.pipe.set_adapters([adapter_name])
        
        print("[Stage3] Flux Omini loaded successfully.")

    def generate(
        self, 
        prompt: str, 
        condition_image: Union[Image.Image, torch.Tensor], 
        negative_prompt: str = "", # Flux usually ignores this, but we keep signature
        seed: int = 42,
        num_steps: int = 28,
        guidance_scale: float = 3.5,
        width: int = 512,
        height: int = 512,
        **kwargs
    ) -> Image.Image:
        
        # 1. Standardize Input (Tensor -> PIL)
        img_pil = self._ensure_pil(condition_image)
        
        # 2. Resize Logic (Copied from your code)
        # Omini needs exact dimensions usually
        if img_pil.size != (width, height):
            print(f"[FluxOmini] Resizing condition from {img_pil.size} to ({width}, {height})")
            img_pil = img_pil.resize((width, height), Image.Resampling.LANCZOS)
        
        # 3. Create Omini Condition
        # This is the specific part we hide from the main pipeline!
        condition = Condition(img_pil, self.adapter_name)
        
        # 4. Set Seed
        seed_everything(seed)
        
        # 5. Generate (Using the external utility function)
        print(f"[FluxOmini] Generating with prompt: '{prompt}'")
        result = generate(
            self.pipe,
            prompt=prompt,
            conditions=[condition],
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images[0]
        
        return result

    def _ensure_pil(self, image):
        """Helper to convert Tensor to PIL and ensure RGB format."""
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

        # The VAE always requires 3 channels (RGB), even for depth maps.
        return pil_image.convert("RGB")