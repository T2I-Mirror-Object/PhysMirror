# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Mirror-T2I** is a three-stage text-to-image generation pipeline that creates photorealistic images with mirror reflections:

1. **Stage 1: Text → 3D Mesh** — Converts text prompts to 3D models using Trellis or SHAP-E
2. **Stage 2: 3D Mesh → Depth Map** — Renders meshes into depth maps with scene composition (mirror, camera, lighting)
3. **Stage 3: Depth Map → Image** — Uses depth-conditioned Flux Omini to generate final images

The pipeline supports batch processing, parallelization across GPUs, and safe resumption of interrupted runs.

## Architecture

### Core Modules

**`src/stage1_mesh/`** — Text-to-3D mesh generation
- `factory.py` — Factory for loading Trellis or SHAP-E models
- `trellis_model.py` / `shap_e_model.py` — Model implementations
- `objects_extractors/` — Extract individual objects from prompts (simple, simple2, heuristic)
- All mesh models implement `BaseTextTo3D`

**`src/stage2_scene/`** — Scene composition and depth rendering
- `builder.py` — Constructs 3D scenes (objects, mirrors, cameras, lights)
- `layout.py` — Positions objects relative to mirrors and cameras
- `renderers/` — Render depth maps and segmentation maps
- `pipelines/depth.py` — Main depth rendering pipeline
- `config.py` — `SceneConfig` dataclass with all scene parameters
- `scene_utils/` — Helpers for correspondence, metadata, colors

**`src/stage3_generation/`** — Depth-to-image generation
- `models/` — Image generation models (Flux Omini with depth control)
- `factory.py` — Factory for loading image models

### Configuration

All batch inference parameters are in `configs/inference.yaml`. Key sections:
- `stage1` — Mesh model choice (trellis/shap_e), object extractor
- `stage2` — Scene parameters (mirror gaps, camera distance, render size, scene elements)
- `stage3` — Image generation (model, LoRA, diffusion steps, guidance scale)

## Common Commands

### Full Pipeline (Batch Inference)

```bash
# Generate depth maps (Stages 1 + 2)
python scripts/generate_depth_maps.py

# Generate final images (Stage 3)
python scripts/generate_images.py
```

Both scripts read parameters from `configs/inference.yaml` and safely resume from already-completed entries. Use `--start_idx` and `--end_idx` to parallelize across GPUs:

```bash
# GPU 0 processes prompts 0-99
CUDA_VISIBLE_DEVICES=0 python scripts/generate_depth_maps.py --start_idx 0 --end_idx 100

# GPU 1 processes prompts 100-199
CUDA_VISIBLE_DEVICES=1 python scripts/generate_depth_maps.py --start_idx 100 --end_idx 200
```

### Debug Individual Stages

All debug scripts output visualization files to `tests/debug_outputs/`.

**Stage 1 + Stage 2 (Text → Depth):**
```bash
python tests/debug_stage1_stage2.py -p "A wooden chair" --mesh_model trellis \
  --mirror_gap_ahead 1.7 --camera_dist 2.2 --camera_elevation 26 --render_size 1024
```

**Stage 2 Only (Depth rendering with a dummy mesh):**
```bash
# Test scene composition, camera, lighting
python tests/debug_stage2_scene.py --camera_dist 0.8 --camera_elevation 30 --render_size 1024
```

**Stage 3 Only (Depth → Image):**
```bash
python tests/debug_stage3_flux_omini.py -d tests/mock_data/depth_map.png \
  -p "A wooden chair in front of a mirror"
```

See `tests/README.md` for full parameter documentation.

### Training

**ControlNet (depth → mask):**
```bash
python train_controlnet_flux.py --config configs/train_controlnet.yaml
```

**LoRA (fine-tuned depth→image):**
```bash
python train_control_lora_flux.py --config configs/train_lora.yaml
```

## Key Design Patterns

### Manifest-Based Output

Batch scripts create a `manifest.json` that maps indices to results:

```json
{
  "0": {"prompt": "A red ball", "depth_map_path": "inference_result/depth_maps/depth_001.png"},
  "1": {"prompt": "A wooden chair", "depth_map_path": "inference_result/depth_maps/depth_002.png"}
}
```

This enables safe resumption and downstream processing without re-scanning directories.

### Model Loading Pattern

Models are loaded once and reused across all prompts in a batch for efficiency:

```python
# Phase A: Load model once, generate all meshes
mesh_model = get_text_to_3d_model(...)  # lazy-loaded
for prompt in prompts:
    mesh_paths = mesh_model.generate(...)  # reuse same model

# Phase B: Unload, then load image model
mesh_model = None
image_model = get_t2i_model(...)  # new model for Stage 3
```

### Config-Driven Behavior

Parameters from `configs/inference.yaml` drive the pipeline. To change scene composition, camera angles, or model choices, edit the config file rather than code.

## Scene Parameters (Key for Stage 2)

From `src/stage2_scene/config.py`:

- **Mirror placement** — `mirror_gap_ahead`, `mirror_gap_side`, `mirror_gap_top` (distance from object)
- **Object scaling** — `object_scale`, `object_base_rotation` (180° for camera-facing view)
- **Camera** — `camera_dist`, `camera_elevation`, `camera_look_at_height`, `camera_azim_min/max`
- **Scene elements** — `include_floor`, `include_walls`, `include_mirror_frame`, `include_mirror_surface`
- **Rendering** — `render_size` (typically 1024 for final output)

## Testing & Debugging

Run individual stage debug scripts to isolate issues:

1. **No output from Stage 1?** → Run `debug_stage1.py` to test mesh generation alone
2. **Depth map looks wrong?** → Run `debug_stage2_scene.py` to tweak scene/camera parameters
3. **Image quality poor?** → Check depth map quality and Stage 3 parameters (guidance_scale, steps)

Debug scripts save outputs to `tests/debug_outputs/` with filenames like `depth_debug.png`, `mesh_debug.glb`.

## Dependencies

Key packages (see `requirements.txt`):
- **Mesh generation** — Trellis, SHAP-E, trimesh, PyMCubes
- **Rendering** — nvdiffrast, PyOpenGL, imageio
- **Image generation** — diffusers, transformers, torch
- **Training** — pytorch-lightning, accelerate
- **Scene** — pydantic, omegaconf

GPU memory requirement: ~24GB for full pipeline (Trellis + Flux Omini). Reduce model sizes or batch size as needed.

## Common Issues & Solutions

**Out of memory during mesh generation?**
- Reduce `object_scale` in Stage 2
- Use a smaller Trellis variant (not `large`)

**Depth map has no content?**
- Check `camera_dist` is not too far
- Verify `include_mirror_surface=false` (glass is invisible)
- Adjust `camera_elevation` if object is out of frame

**Image generation fails?**
- Ensure manifest.json exists from `generate_depth_maps.py`
- Check depth maps are valid PNG files (8-bit grayscale)
- Try a different `guidance_scale` (lower = more flexible, higher = more adherence to depth)

**Resume not working?**
- Verify manifest.json is readable
- Check file permissions in output directories
- Incomplete files are detected by checking if depth_map_path exists

