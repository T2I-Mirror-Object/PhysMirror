# Camera Placement Sensitivity Ablation Study

This document describes a comprehensive ablation study that systematically evaluates how the Mirror-T2I pipeline is sensitive to different camera placement parameters and extreme camera angles.

## Overview

The camera placement in your pipeline uses three key parameters:

1. **Azimuth offset** (`camera_azim_min`, `camera_azim_max`) — How far left/right the camera is positioned
2. **Elevation tilt** (`camera_elevation`) — How much the camera tilts downward (or upward if negative)
3. **Distance multiplier** (`camera_dist`) — How far the camera is from the scene
4. **Look-at height** (`camera_look_at_height`) — What height the camera focuses on

The default settings are:
- Elevation: **26.0°** (moderate downward tilt to expose both object and reflection)
- Azimuth: **-10° to +10°** (mild left/right offset)
- Distance: **1.2x** the mirror gap
- Look-at height: **1.8m** (roughly object center)

This ablation study answers:
- **How sensitive is the pipeline to camera azimuth?** (Can we get extreme side angles?)
- **How sensitive is it to elevation tilt?** (What happens at 0°, 45°, 60°, or looking upward?)
- **How sensitive is it to distance?** (Close vs. far cameras)
- **What combinations break the pipeline?** (Extreme edge cases)

## Running the Ablation Study

### Quick Start (3 prompts, standard variations)

```bash
python scripts/ablation_camera_sensitivity.py \
  --preset standard \
  --sample_only \
  --output ablation_results/camera_sensitivity
```

This runs ~25 camera configurations on 3 sample prompts. Runtime: **30-60 minutes** depending on GPU.

### Full Study (All prompts, all variations)

```bash
python scripts/ablation_camera_sensitivity.py \
  --preset all \
  --output ablation_results/camera_sensitivity
```

This runs ~35 camera configurations on all prompts. Runtime: **Several hours** depending on dataset size.

### Extreme Only (Edge cases)

```bash
python scripts/ablation_camera_sensitivity.py \
  --preset extreme \
  --sample_only \
  --output ablation_results/camera_sensitivity
```

Focuses on extreme angles (looking up, straight down, 90° side views, etc.).

### Options

```
--preset {standard,extreme,all}       Which configurations to test
--sample_only                         Use only first N prompts (faster)
--sample_size N                       Number of prompts to sample (default: 3)
--output DIR                          Output directory for results
--config FILE                         Base config file (default: configs/inference.yaml)
```

## Camera Configurations Tested

### Standard Ablation (25 configurations)

**Azimuth Ablation** (left/right offset):
- `azim_frontal` — 0° (dead center)
- `azim_narrow_5` — ±5° (narrow offset)
- `azim_moderate_15` — ±15° (moderate offset)
- `azim_wide_25` — ±25° (wide offset)
- `azim_extreme_left_45` — -45° (extreme left)
- `azim_extreme_right_45` — +45° (extreme right)

**Elevation Ablation** (upward/downward tilt):
- `elev_eye_level` — 0° (perfectly level)
- `elev_slight_down_10` — 10° down
- `elev_moderate_20` — 20° down
- `elev_steep_45` — 45° down
- `elev_extreme_60` — 60° down (very steep)
- `elev_negative_10_up` — -10° (looking slightly up)

**Distance Ablation** (camera distance):
- `dist_close_0.8` — 0.8x multiplier (close)
- `dist_near_1.0` — 1.0x multiplier
- `dist_far_1.5` — 1.5x multiplier (far)
- `dist_very_far_2.0` — 2.0x multiplier (very far)

**Look-At Height Ablation**:
- `lookat_low_0.5` — 0.5m (very low)
- `lookat_medium_1.2` — 1.2m
- `lookat_high_2.2` — 2.2m (high)

**Extreme Combinations**:
- `extreme_combo_1` — 60° down, ±45° azimuth, 0.8x distance, 0.5m look-at (very extreme)
- `extreme_combo_2` — -10° up, ±30° azimuth, 2.0x distance, 2.5m look-at (opposite extreme)
- `extreme_combo_3` — Moderate extreme (45° down, ±20° azimuth, 1.5x distance)

### Extreme Configurations (10 additional)

- `extreme_straight_down_89` — Looking almost straight down
- `extreme_looking_up_45` — Looking upward at 45°
- `extreme_side_90_left` — 90° to the left
- `extreme_side_90_right` — 90° to the right
- `extreme_distance_very_close_0.3` — Very close camera
- `extreme_distance_far_3.0` — Very distant camera
- And more...

## Output Structure

After running, the results directory contains:

```
ablation_results/camera_sensitivity/
├── ablation_results.json           # Raw results for all configurations
├── ablation_comparison.csv         # Table of all results
├── depth_map_comparison.png        # Visual comparison of depth maps
├── parameter_sensitivity.png       # Charts showing parameter sensitivity
│
├── configs/
│   ├── baseline_default.yaml
│   ├── azim_frontal.yaml
│   ├── elev_extreme_60.yaml
│   └── ... (one config per camera setting)
│
└── results/
    ├── baseline_default/
    │   ├── depth_maps/
    │   │   ├── manifest.json
    │   │   ├── depth_001.png
    │   │   └── ...
    │   └── images/
    │       ├── image_001.png
    │       └── ...
    ├── azim_frontal/
    │   ├── depth_maps/
    │   └── images/
    └── ... (directories for each configuration)
```

## Analyzing Results

After running the ablation, generate analysis reports:

### Performance Summary

```bash
python scripts/analyze_ablation.py ablation_results/camera_sensitivity
```

Prints:
- Success/failure rates for each configuration
- Timing metrics (Stage 1+2, Stage 3, total)
- Parameter impact analysis (which parameters affect speed most?)

### Generate Visualizations

```bash
# All plots
python scripts/analyze_ablation.py ablation_results/camera_sensitivity --all_plots

# Just depth maps
python scripts/analyze_ablation.py ablation_results/camera_sensitivity --plot_depth_maps

# Just sensitivity charts
python scripts/analyze_ablation.py ablation_results/camera_sensitivity --plot_sensitivity
```

This generates:
- `depth_map_comparison.png` — Side-by-side depth maps for different camera angles
- `parameter_sensitivity.png` — Charts showing how each parameter affects pipeline performance
- `ablation_comparison.csv` — Detailed table for spreadsheet analysis

## Key Research Questions

### 1. Azimuth Sensitivity
**Question:** How sensitive is the pipeline to horizontal camera offset?

**Hypothesis:** Moderate offsets (±10-20°) should work well. Extreme angles (±45°+) may cause issues:
- Object might be cropped from view
- Reflection might be distorted or clipped
- Diffusion model might struggle with unusual perspective

**To investigate:** Compare depth maps from `azim_frontal` vs `azim_extreme_left_45` and examine:
- Does the object fully appear in the depth map?
- Is the mirror reflection visible?
- Does the generated image look natural or distorted?

### 2. Elevation Sensitivity
**Question:** How sensitive is the pipeline to camera tilt?

**Hypothesis:** Default 26° downward tilt is optimal to see both object and reflection. Extremes will fail:
- 0° (eye level) — Object and reflection both fully visible, but harsh lighting
- 60° (very steep) — Looking down at object, but missing upper body and reflection
- -10° (looking up) — Object below camera, reflection might be above, composition odd

**To investigate:** Compare depth maps and images across elevation values. Look for:
- Clipping of object parts
- Quality of reflection visibility
- Natural vs. unnatural composition

### 3. Distance Sensitivity
**Question:** How does camera distance affect depth quality and image generation?

**Hypothesis:** Too close (0.3x) will clip objects. Too far (3.0x) will show small objects in large space.

**To investigate:** Compare:
- Depth map detail levels (close camera = more detail)
- Object size in final image
- Image quality/realism

### 4. Extreme Combinations
**Question:** Which combinations of extreme parameters break the pipeline?

**To investigate:**
- `extreme_combo_1` (steep angle + very close + low look-at) — Likely to clip object
- `extreme_combo_2` (looking up + far + high look-at) — Likely to miss object
- How do failures manifest? (empty depth maps, tiny objects, clipped objects)

## Metrics to Evaluate

The ablation collects:

1. **Success/Failure** — Did both Stage 1+2 and Stage 3 complete?
2. **Depth Map Count** — How many valid depth maps were generated?
3. **Image Count** — How many final images were generated?
4. **Timing** — How long did each stage take?

Additional metrics you can manually evaluate:

1. **Depth Map Quality**
   - Is the object fully visible?
   - Is the reflection visible?
   - Are the depth values reasonable (not clipped to 0 or 255)?

2. **Image Quality**
   - Does the generated image look realistic?
   - Are the object and reflection both present?
   - Is the lighting/shading reasonable?

3. **Diffusion Model Robustness**
   - Does FLUX.1-dev + OminiControl work with unusual depth maps?
   - Are there systematic failures for certain camera angles?
   - Does Seg2Any (alternative) handle extreme angles better?

## Expected Results

Based on the pipeline design, we expect:

| Configuration | Expected Outcome |
|---|---|
| Baseline (26° down, ±10° azimuth) | ✓ High success rate, realistic images |
| Frontal (0° azimuth) | ✓ Works well, symmetric composition |
| Extreme azimuth (±45°) | ? Unknown; likely issues at ±90° |
| Moderate elevation (10-45° down) | ✓ Should work, varying object/reflection visibility |
| Extreme elevation (60°+ down or up) | ✗ Likely failures or distorted results |
| Very close (0.3x distance) | ✗ Object clipping likely |
| Very far (3.0x distance) | ? Tiny objects; may be too small for detail |
| Extreme combos | ✗ High failure rate expected |

## Interpreting Results

### Success Rate

- **> 90%:** Camera parameters are robust; pipeline handles wide range of angles
- **50-90%:** Pipeline works for "reasonable" angles but fails on extremes
- **< 50%:** Camera parameters are critical; only specific angles work well

### Timing Patterns

If **distance** strongly correlates with time:
- Farther cameras might render larger scenes
- Closer cameras are faster (smaller rendered area)

If **elevation** strongly correlates with time:
- More complex scene geometry at certain angles
- Or fewer occlusions at other angles

### Failure Modes

Common failures might include:

1. **Object Clipping** — Object goes outside view frustum
   - Symptom: Empty or mostly black depth maps
   - Solution: Adjust distance or look-at height

2. **Reflection Clipping** — Reflection not visible in depth map
   - Symptom: Depth map shows only object, no mirror
   - Solution: Adjust elevation or distance

3. **Diffusion Model Failure** — Depth map is valid but image generation fails
   - Symptom: Error in Stage 3, valid depth maps but no images
   - Indicates diffusion model sensitivity (not camera sensitivity)

4. **Numerical Issues** — Out-of-bounds parameters
   - Symptom: NaN or Inf values in depth rendering
   - Solution: Clamp parameters to valid ranges

## Advanced: Custom Ablation

To test your own camera configurations, edit `ablation_camera_sensitivity.py`:

```python
def my_custom_configs(self) -> List[CameraConfig]:
    return [
        CameraConfig("my_config_1", elevation=30.0, azim_min=-5.0, azim_max=5.0, dist=1.0, look_at=1.5),
        CameraConfig("my_config_2", elevation=45.0, azim_min=-20.0, azim_max=20.0, dist=1.5, look_at=1.8),
        # ... add more
    ]
```

Then run:

```bash
python scripts/ablation_camera_sensitivity.py --preset custom --sample_only
```

(You'd need to modify the script to add `custom` as a preset option.)

## Related Files

- `src/stage2_scene/cameras/strategies.py` — Camera placement logic
- `src/stage2_scene/config.py` — Camera configuration parameters
- `scripts/generate_depth_maps.py` — Stage 1+2 pipeline (uses camera config)
- `scripts/generate_images.py` — Stage 3 pipeline (takes depth maps as input)

## References

Camera positioning uses PyTorch3D's `look_at_view_transform`:
- Distance: Spherical radius (how far from origin)
- Elevation: Angle above horizon (0° = level, +90° = straight down)
- Azimuth: Angle in horizontal plane (0° = front, 90° = right, 180° = back)
- `at` (look-at point): Where the camera focuses

See: https://pytorch3d.org/docs/renderer_utils

