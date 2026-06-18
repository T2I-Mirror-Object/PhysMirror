import json
import os
from typing import List, Dict

class Seg2AnyFormatter:
    """Handles the creation of the metadata JSON for Seg2Any."""
    
    def format(self, 
               caption: str, 
               seed: int, 
               object_prompts: List[str], 
               object_colors: List[list]) -> Dict:
        
        segments_info = []
        for prompt, color in zip(object_prompts, object_colors):
            segments_info.append({
                "color": color, # Expects [0-255, 0-255, 0-255]
                "text": prompt
            })

        return {
            "caption": caption,
            "seed": seed,
            "segments_info": segments_info
        }

    def save_json(self, data: Dict, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)