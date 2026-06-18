import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from ..libs.seg2any.models import FluxTransformer2DModel
from ..libs.seg2any.pipelines import FluxRegionalPipeline
from ..libs.seg2any.utils.visualizer import Visualizer

from .base import BaseT2IModel

class Seg2AnyWrapper(BaseT2IModel):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.visualizer = Visualizer()
        
        # Pre-processor for the condition image (copied from infer.py)
        self.cond_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def load_model(self, 
                   pretrained_model_path: str, 
                   lora_path: str, 
                   weight_dtype_str: str = "bf16",
                   **kwargs):
        
        print(f"[Seg2Any] Loading model from {pretrained_model_path}...")
        
        # Determine Dtype
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        weight_dtype = dtype_map.get(weight_dtype_str, torch.bfloat16)

        # 1. Load Transformer
        transformer = FluxTransformer2DModel.from_pretrained(
            pretrained_model_path, 
            subfolder="transformer", 
            torch_dtype=weight_dtype
        )

        # 2. Load Pipeline
        self.pipe = FluxRegionalPipeline.from_pretrained(
            pretrained_model_path,
            transformer=transformer,
            torch_dtype=weight_dtype,
        )

        # 3. Load LoRAs
        # Assuming lora_path contains 'default' and 'cond' folders as per their structure
        print(f"[Seg2Any] Loading LoRAs from {lora_path}...")
        self.pipe.load_lora_weights(os.path.join(lora_path, 'default'), adapter_name="default")
        
        cond_lora = os.path.join(lora_path, 'cond')
        if os.path.exists(cond_lora):
            self.pipe.load_lora_weights(cond_lora, adapter_name="cond")
            self.pipe.set_adapters(['cond', 'default'])
        else:
            self.pipe.set_adapters("default")

        self.pipe.to(self.device)
        print("[Seg2Any] Loaded successfully.")

    def generate(self, 
                 prompt: str, 
                 condition_image, 
                 meta_data: dict,  # <--- CRITICAL: Stage 2 JSON Data
                 seed: int = 42,
                 num_steps: int = 32,
                 guidance_scale: float = 3.5,
                 width: int = 512,
                 height: int = 512,
                 cond_scale_factor: int = 2,
                 **kwargs):
        
        # 1. Preprocess Inputs (The hard part)
        # We need to convert Stage 2 outputs to Seg2Any internal format
        processed_inputs = self._preprocess(condition_image, meta_data, width, height, cond_scale_factor)
        
        # 2. Setup Generator
        generator = torch.Generator("cuda").manual_seed(seed)

        print("\n=== DEBUGGING SEG2ANY INPUTS ===")
        print(f"1. Width: {width} (Type: {type(width)})")
        print(f"2. Height: {height} (Type: {type(height)})")
        
        cond_tensor = processed_inputs["cond_tensor"]
        print(f"3. Cond Tensor Shape: {cond_tensor.shape}")
        print(f"4. Cond Tensor Type: {cond_tensor.dtype}")
        
        w_int = int(width)
        h_int = int(height)
        print(f"5. Passing to pipe -> height={h_int}, width={w_int}")
        print("================================\n")

        # 3. Run Inference
        print(f"[Seg2Any] Generating with global prompt: '{prompt}'")
        
        result = self.pipe(
            global_prompt=prompt,
            regional_prompts=processed_inputs["regional_captions"],
            regional_labels=processed_inputs["label"],
            cond=processed_inputs["cond_tensor"],
            attention_mask_method="hard",
            is_filter_cond_token=True,
            cond2image_attention_weight=kwargs.get("cond2image_attention_weight", 1),
            hard_attn_block_range=[19, 37],
            height=h_int,
            width=w_int,
            cond_scale_factor=cond_scale_factor,
            num_images_per_prompt=1,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
        ).images[0]
        
        return result

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