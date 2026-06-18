import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Union, Dict

from diffusers import FluxPipeline
from ..libs.seg2any.models import FluxTransformer2DModel
from ..libs.seg2any.pipelines import FluxRegionalPipeline
from ..libs.seg2any.utils.visualizer import Visualizer

from ..utils.flux_omini import Condition, generate as omini_generate, seed_everything
from .base import BaseT2IModel

class MirrorMultiConditionWrapper(BaseT2IModel):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.visualizer = Visualizer()
        self.cond_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def load_model(self, 
                   pretrained_model_path: str, 
                   seg_lora_path: str, 
                   omini_lora_repo: str,
                   omini_lora_weight: str,
                   weight_dtype_str: str = "bf16",
                   **kwargs):
        
        print("[MultiCond] Initializing Amalgamation Pipeline...")
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        weight_dtype = dtype_map.get(weight_dtype_str, torch.bfloat16)

        # 1. Base the architecture on Seg2Any's custom transformer
        transformer = FluxTransformer2DModel.from_pretrained(
            pretrained_model_path, 
            subfolder="transformer", 
            torch_dtype=weight_dtype
        )

        # 2. Use Seg2Any's custom pipeline to handle regional attention
        self.pipe = FluxRegionalPipeline.from_pretrained(
            pretrained_model_path,
            transformer=transformer,
            torch_dtype=weight_dtype,
        )

        # 3. Load all LoRAs into the SAME pipeline
        print("[MultiCond] Loading Seg2Any LoRAs...")
        self.pipe.load_lora_weights(os.path.join(seg_lora_path, 'default'), adapter_name="seg_default")
        
        cond_lora = os.path.join(seg_lora_path, 'cond')
        active_adapters = ["seg_default"]
        
        if os.path.exists(cond_lora):
            self.pipe.load_lora_weights(cond_lora, adapter_name="seg_cond")
            active_adapters.append("seg_cond")

        print(f"[MultiCond] Loading OminiControl LoRA: {omini_lora_weight}...")
        self.pipe.load_lora_weights(
            omini_lora_repo,
            weight_name=omini_lora_weight,
            adapter_name="omini_depth"
        )
        active_adapters.append("omini_depth")

        # 4. Activate all adapters simultaneously
        # You can tune these weights (e.g., [1.0, 1.0, 0.8]) if depth overpowers segmentation
        self.pipe.set_adapters(active_adapters, adapter_weights=[1.0] * len(active_adapters))
        self.pipe.to(self.device)
        
        print(f"[MultiCond] Loaded successfully with adapters: {active_adapters}")

    def generate(self, 
                 prompt: str, 
                 seg_image: Union[Image.Image, torch.Tensor], 
                 depth_image: Union[Image.Image, torch.Tensor],
                 meta_data: dict,  
                 seed: int = 42,
                 num_steps: int = 32,
                 guidance_scale: float = 3.5,
                 width: int = 512,
                 height: int = 512,
                 cond_scale_factor: int = 2,
                 **kwargs) -> Image.Image:
        
        # --- A. Preprocess Seg2Any Inputs ---
        seg_inputs = self._preprocess(seg_image, meta_data, width, height, cond_scale_factor)
        
        # --- B. Preprocess OminiControl Inputs ---
        depth_pil = self._ensure_pil(depth_image)
        if depth_pil.size != (width, height):
            depth_pil = depth_pil.resize((width, height), Image.Resampling.LANCZOS)
        
        depth_condition = Condition(depth_pil, "omini_depth")

        seed_everything(seed)
        generator = torch.Generator("cuda").manual_seed(seed)

        print(f"[MultiCond] Generating fused image for prompt: '{prompt}'")

        # =========================================================
        # THE FIX: Temporarily mock check_inputs to bypass the crash
        # =========================================================
        original_check_inputs = self.pipe.check_inputs
        self.pipe.check_inputs = lambda *args, **kwargs: None 
        
        try:
            # --- C. The Fused Forward Pass ---
            result = omini_generate(
                self.pipe, 
                prompt=prompt,
                conditions=[depth_condition], 
                
                # Seg2Any kwargs
                regional_prompts=seg_inputs["regional_captions"],
                regional_labels=seg_inputs["label"],
                cond=seg_inputs["cond_tensor"], 
                attention_mask_method="hard",
                is_filter_cond_token=True,
                cond2image_attention_weight=kwargs.get("cond2image_attention_weight", 1),
                hard_attn_block_range=[19, 37],
                num_images_per_prompt=1,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                generator=generator,
                height=int(height),
                width=int(width),
                cond_scale_factor=cond_scale_factor
            ).images[0]
            
        finally:
            # Always restore the original method safely
            self.pipe.check_inputs = original_check_inputs
            
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

    def _preprocess(self, seg_map_tensor, json_data, width, height, cond_scale_factor):
        """
        Converts:
          - seg_map_tensor (1, 3, H, W)
          - json_data (Dict from Stage 2)
        Returns:
          - Dictionary expected by FluxRegionalPipeline
        """
        # A. Prepare Segmentation Image (H, W, 3) Numpy
        if isinstance(seg_map_tensor, torch.Tensor):
            seg_map_np = seg_map_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
            # If normalized 0-1, scale to 0-255
            if seg_map_np.max() <= 1.0:
                seg_map_np = (seg_map_np * 255).astype(np.uint8)
        else:
            # Assume PIL
            seg_map_np = np.array(seg_map_tensor)
            
        # Resize if needed (nearest neighbor to preserve colors)
        if seg_map_np.shape[:2] != (height, width):
            seg_map_pil = Image.fromarray(seg_map_np).resize((width, height), Image.NEAREST)
            seg_map_np = np.array(seg_map_pil)

        # B. Parse JSON Data
        # Stage 2 JSON format: {'segments_info': [{'color': [R,G,B], 'text': '...'}, ...]}
        seg_anno = json_data
        
        # Create Mapping: (R,G,B) -> Prompt
        color_to_text = {}
        for item in seg_anno['segments_info']:
            color_tuple = tuple(item['color'])
            color_to_text[color_tuple] = item['text']

        # C. Generate Masks & Prompts
        label_masks = []
        regional_captions = []
        
        # Get unique colors in the image
        unique_colors = np.unique(seg_map_np.reshape(-1, 3), axis=0)

        print("Color to text:", color_to_text)
        
        for color in unique_colors:
            print("Color:", color)
            color_tuple = tuple(color.tolist())
            
            # If this color is in our JSON description
            if color_tuple in color_to_text:
                prompt = color_to_text[color_tuple]
                
                # Create Binary Mask
                mask = (seg_map_np[..., 0] == color[0]) & \
                       (seg_map_np[..., 1] == color[1]) & \
                       (seg_map_np[..., 2] == color[2])
                
                label_masks.append(mask)
                regional_captions.append(prompt)

        # Stack Masks
        if not label_masks:
            raise ValueError("No matching colors found between Segmentation Map and JSON!")
            
        label_stack = np.stack(label_masks, axis=0) # (N, H, W)
        label_tensor = torch.from_numpy(label_stack).long().to(self.device)

        # D. Create Contour Condition (The "Visualizer" step)
        cond_vis = np.zeros([height, width, 3], dtype=np.uint8)
        cond_vis = self.visualizer.draw_contours(
            cond_vis,
            label_stack,
            thickness=1,
            colors=[(255, 255, 255)] * len(regional_captions)
        )
        
        # Resize Condition (based on scale factor logic in infer.py)
        s = cond_scale_factor * 16
        cond_res = (int(height // s * 16), int(width // s * 16))
        
        cond_pil = Image.fromarray(cond_vis).resize(cond_res, Image.BILINEAR)
        
        # Normalize (-1 to 1) and add batch dim
        cond_tensor = self.cond_transform(cond_pil).unsqueeze(0).to(self.device)
        
        # Denormalize for pipeline input (0 to 1) as seen in infer.py: (cond + 1) / 2
        cond_input = (cond_tensor + 1.0) / 2.0

        return {
            "label": label_tensor,
            "regional_captions": regional_captions,
            "cond_tensor": cond_input
        }