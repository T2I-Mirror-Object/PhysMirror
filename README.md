
## Batch Inference (full dataset)

Edit `configs/inference.yaml` to configure all parameters, then run the two scripts in order.

### Step 1 — Generate depth maps (Stage 1 + Stage 2)

```bash
python scripts/generate_depth_maps.py
# or point to a different config:
python scripts/generate_depth_maps.py --config configs/inference.yaml
# process only a slice (useful for parallelising across GPUs):
python scripts/generate_depth_maps.py --start_idx 0 --end_idx 120
```

Trellis is **loaded once** for all prompts, then unloaded before Stage 2 runs.
Meshes are written to a **temporary directory** and deleted automatically after depth rendering — only depth maps are kept.
Outputs land in `inference_result/depth_maps/` (configurable via `depth_maps_dir`):

```
inference_result/depth_maps/
├── manifest.json    ← index → {prompt, depth_map_path}
├── depth_001.png
├── depth_002.png
└── ...
```

### Step 2 — Generate images (Stage 3)

```bash
python scripts/generate_images.py
# or point to a different config:
python scripts/generate_images.py --config configs/inference.yaml
# process only a slice:
python scripts/generate_images.py --start_idx 0 --end_idx 120
```

Flux Omini is **loaded once** for all depth maps.
Outputs land in `inference_result/images/` (configurable via `images_dir`):

```
inference_result/images/
├── image_001.png
├── image_002.png
└── ...
```

Both scripts **resume safely** — already-processed entries are skipped.

---

## Change objects position
## Change Camera position

cd /raid/ltnghia02/mxbach_ndphuc/Mirror-T2I/
conda activate mirrorgen
export HF_HOME=/raid/ltnghia02/mxbach_ndphuc/hf_cache/
export CUDA_VISIBLE_DEVICES=2
scp -r 938b4e1db6a348bfb4c84aeee9794aef@ssh.axisapps.io:/raid/ltnghia02/mxbach_ndphuc/Mirror-T2I/inference_result/depth_maps ./

## Choose Image 
### Pipeline Trellis + FLux Omini Depth
#### Single Object
003, 006, 007, 011, 015, 021, 027, 033, 046, 052, 073, 078, 079, 081, 093, 095, 103
#### Second Objects

#### Third Objects
