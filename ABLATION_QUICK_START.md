# Camera Sensitivity Ablation — Quick Start Guide

## TL;DR

Run a camera sensitivity ablation study on your Mirror-T2I pipeline in **3 commands**:

```bash
# 1. Run ablation (25 camera configs, 3 sample prompts, ~1 hour)
python scripts/ablation_camera_sensitivity.py --preset standard --sample_only

# 2. Analyze results
python scripts/analyze_ablation.py ablation_results/camera_sensitivity --all_plots

# 3. View results
python scripts/inspect_ablation_results.py ablation_results/camera_sensitivity --list
```

## What Gets Tested

| Dimension | Variations |
|-----------|-----------|
| **Azimuth** (left/right) | 0°, ±5°, ±15°, ±25°, ±45° |
| **Elevation** (up/down tilt) | -10°, 0°, 10°, 20°, 26°*, 45°, 60° |
| **Distance** (camera distance) | 0.8x, 1.0x, 1.2x*, 1.5x, 2.0x |
| **Look-At Height** | 0.5m, 1.2m, 1.8m*, 2.2m |

*Default values

**Plus 3 extreme combinations:**
- Extreme steep angle + extreme close + low look-at
- Extreme upward angle + extreme far + high look-at
- Moderate extreme on all dimensions

## Output Files

```
ablation_results/camera_sensitivity/
├── ablation_results.json           ← Raw data (load in Python/Excel)
├── ablation_comparison.csv         ← Table format (open in Excel)
├── depth_map_comparison.png        ← Visual comparison of depth maps
├── parameter_sensitivity.png       ← Charts showing which parameters matter
└── results/
    ├── baseline_default/
    │   ├── depth_maps/depth_001.png
    │   └── images/image_001.png
    ├── azim_frontal/
    │   ├── depth_maps/...
    │   └── images/...
    └── ... (one folder per configuration)
```

## Key Insights You'll Get

### 1. Failure Modes
Which camera angles cause the pipeline to fail?

Example findings:
- ❌ 90° side view → Object fully clipped from view
- ⚠️ Looking straight down (89°) → Mirror reflection not visible
- ⚠️ Very far camera (3.0x) → Objects too small for detail
- ✓ ±45° azimuth works, but image quality degrades

### 2. Robustness of Diffusion Models
Is FLUX.1-dev + OminiControl control robust to unusual depth maps?

Example findings:
- Depth maps are valid → Images usually generate, but quality varies
- Extreme angles → May see artifacts or unnatural perspective
- Which parameters (azimuth vs elevation vs distance) matter most for diffusion?

### 3. Optimal Operating Range
What's the "sweet spot" for camera parameters?

Example findings:
- Elevation 20-30° → Ideal object/reflection balance
- Azimuth ±10-15° → Good off-axis perspective
- Distance 1.0-1.5x → Best detail and composition

## Manual Inspection Workflow

After running ablation, inspect results visually:

```bash
# List all configurations
python scripts/inspect_ablation_results.py ablation_results/camera_sensitivity --list

# Compare 2-3 specific configurations (same prompt, different cameras)
python scripts/inspect_ablation_results.py ablation_results/camera_sensitivity \
  -c baseline_default azim_frontal elev_extreme_60 \
  -s 0 --type depth

# Compare all successful configurations
python scripts/inspect_ablation_results.py ablation_results/camera_sensitivity \
  --compare_all --type image -o all_configurations_sample1.png

# Look at all depth maps for one configuration
python scripts/inspect_ablation_results.py ablation_results/camera_sensitivity \
  -c elev_extreme_60 --type depth
```

## What to Look For

When inspecting depth maps and images:

### Depth Map Checks
- [ ] Is the object fully visible (not clipped)?
- [ ] Is the mirror reflection visible?
- [ ] Are depth values reasonable (grayscale, not blown out)?
- [ ] Do you see artifacts or numerical issues?

### Generated Image Checks
- [ ] Does the prompt match the image?
- [ ] Is the object realistic (lighting, shadows)?
- [ ] Is the reflection visible in the mirror?
- [ ] Are there distortions or artifacts?
- [ ] Does the composition look natural for that camera angle?

### Diffusion Model Sensitivity
- [ ] Do extreme depth maps still generate valid images?
- [ ] Which depth map issues cause generation failures?
- [ ] Do unusual camera angles cause systematic artifacts?

## Common Findings

### Expected: Azimuth is Robust
- Depth maps look good at all azimuth angles
- Image generation works at all offsets
- Even ±45° might work (though perspective looks unusual)

### Expected: Elevation is Critical
- 0° (eye level) → Object at bottom, large in frame
- 26° (default) → Good balance of object and reflection
- 60° (steep down) → Reflection might be missed
- -10° (looking up) → Object partially clipped

### Expected: Distance is Moderate Factor
- 0.3x → Object too large, heavily clipped
- 1.2x (default) → Good detail and framing
- 3.0x → Object tiny in frame, may lose detail

### Expected: Look-At Height Matters
- Too low → Object bottom in frame, head clipped
- Too high → Object top in frame, feet clipped
- Sweet spot around 1.5-2.0m (center of typical objects)

## Deeper Analysis

After inspection, analyze with Python:

```python
import json
from pathlib import Path

# Load results
results_file = Path("ablation_results/camera_sensitivity/ablation_results.json")
with open(results_file) as f:
    results = json.load(f)

# Which configurations succeeded?
successful = {k: v for k, v in results.items() if v.get("success")}
print(f"Success rate: {len(successful)}/{len(results)}")

# Which were fastest?
fastest = sorted(
    [(k, v["total_time_sec"]) for k, v in successful.items()],
    key=lambda x: x[1]
)
print("Fastest configurations:")
for name, time_sec in fastest[:5]:
    print(f"  {name}: {time_sec:.1f}s")

# Group by parameter (e.g., which elevations succeeded?)
by_elevation = {}
for name, result in results.items():
    elev = result["config"]["camera_elevation"]
    if elev not in by_elevation:
        by_elevation[elev] = {"success": 0, "total": 0}
    by_elevation[elev]["total"] += 1
    if result.get("success"):
        by_elevation[elev]["success"] += 1

print("Success by elevation:")
for elev in sorted(by_elevation.keys()):
    stats = by_elevation[elev]
    success_rate = 100 * stats["success"] / stats["total"]
    print(f"  {elev:6.1f}°: {success_rate:6.1f}% ({stats['success']}/{stats['total']})")
```

## Advanced: Custom Ablations

Test specific camera angles:

1. Create a custom config file in `configs/ablation_custom.yaml`
2. Modify one or two parameters in `stage2`:
   ```yaml
   stage2:
     camera_elevation: 45.0          # Try 45° down
     camera_azim_min: -30.0
     camera_azim_max: 30.0
     camera_dist: 0.9
     # ... rest as default
   ```

3. Run single test:
   ```bash
   python scripts/generate_depth_maps.py --config configs/ablation_custom.yaml
   python scripts/generate_images.py --config configs/ablation_custom.yaml
   ```

## Reporting Results

Create a summary report:

### 1. Executive Summary
- What parameters were tested?
- Overall success rate?
- Any surprising findings?

### 2. Key Findings
- Which parameters are most sensitive?
- Which camera angles work best?
- Which combinations fail?

### 3. Visualizations
- Include `depth_map_comparison.png`
- Include `parameter_sensitivity.png`
- Show examples of depth maps and corresponding images for:
  - Best case (baseline)
  - Extreme cases (±45° azimuth, 60° elevation, etc.)
  - Failed cases (if any interesting failures)

### 4. Detailed Table
- Include `ablation_comparison.csv` or create a table from it
- Format: Configuration, Elevation, Azimuth, Distance, Look-At, Depth Count, Image Count, Time, Success

## Troubleshooting

**"Out of memory" during ablation?**
- Reduce `sample_size` (e.g., `--sample_size 1` for just 1 prompt)
- Use `--preset extreme` with just a few configs
- Run configurations one at a time

**"No results files generated"?**
- Check that `data/prompts.txt` exists
- Run a single configuration manually: `python scripts/generate_depth_maps.py --config configs/inference.yaml`
- Check GPU memory with `nvidia-smi`

**"analyze_ablation.py crashes"?**
- Install matplotlib: `pip install matplotlib`
- Check that results directory structure is correct

**"Missing depth maps for some configs"?**
- This is normal if generation failed for those configs
- Check the JSON results file for error messages
- Look at `ablation_results.json` to see which configs have `success=false`

## Related Documents

- **ABLATION_CAMERA_SENSITIVITY.md** — Detailed research guide
- **CLAUDE.md** — Overall pipeline documentation
- **README.md** — Basic usage instructions

## Support

If results are unexpected:

1. **Verify baseline works:** `python tests/debug_stage1_stage2.py -p "A red ball"`
2. **Check a simple ablation:** Run just `azim_frontal` and `baseline_default`
3. **Inspect depth maps manually:** `python scripts/inspect_ablation_results.py ablation_results/camera_sensitivity --list`
4. **Check GPU memory:** `nvidia-smi`

