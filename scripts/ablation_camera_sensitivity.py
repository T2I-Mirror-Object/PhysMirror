#!/usr/bin/env python
"""
Camera Placement Ablation Study

Systematically varies camera parameters and evaluates their impact on:
1. Depth map quality and validity
2. Final image generation success rate
3. Visual quality (manual inspection)

Camera parameters ablated:
- camera_azimuth (left/right offset): 0°, ±10°, ±20°, ±30°, ±45°
- camera_elevation (downward tilt): 0°, 10°, 20°, 26° (default), 45°, 60°
- camera_dist (distance multiplier): 0.8, 1.0, 1.2 (default), 1.5, 2.0
- camera_look_at_height: 0.5, 1.0, 1.5, 1.8 (default), 2.2

Usage:
    python scripts/ablation_camera_sensitivity.py --output ablation_results/
    python scripts/ablation_camera_sensitivity.py --preset extreme --sample_only
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
import time
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image


@dataclass
class CameraConfig:
    """Represents a single camera configuration."""
    name: str
    camera_elevation: float
    camera_azim_min: float
    camera_azim_max: float
    camera_dist: float
    camera_look_at_height: float

    def description(self) -> str:
        """Human-readable description of this camera config."""
        return (
            f"{self.name}: "
            f"elev={self.camera_elevation:.0f}° "
            f"azim=[{self.camera_azim_min:.0f}°, {self.camera_azim_max:.0f}°] "
            f"dist={self.camera_dist:.2f}x "
            f"look_at={self.camera_look_at_height:.1f}m"
        )


class AblationStudy:
    def __init__(self, base_config_path: str, output_dir: str, sample_size: int = 3):
        """
        Initialize ablation study.

        Args:
            base_config_path: Path to base inference.yaml
            output_dir: Where to store ablation configs and results
            sample_size: Number of prompts to sample for testing
        """
        # Determine project root (script location's parent's parent)
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent

        self.base_config_path = Path(base_config_path)
        if not self.base_config_path.is_absolute():
            self.base_config_path = self.project_root / self.base_config_path

        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = self.project_root / self.output_dir

        self.sample_size = sample_size
        self.results_dir = self.output_dir / "results"
        self.configs_dir = self.output_dir / "configs"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        # Load base config
        with open(self.base_config_path, "r") as f:
            self.base_config = yaml.safe_load(f)

        self.results = {}

    def get_standard_configs(self) -> List[CameraConfig]:
        """Standard ablation: reasonable variations of camera parameters."""
        return [
            # Default baseline
            CameraConfig("baseline_default", 26.0, -10.0, 10.0, 1.2, 1.8),

            # === AZIMUTH ABLATION (left/right offset) ===
            CameraConfig("azim_frontal", 26.0, 0.0, 0.0, 1.2, 1.8),           # Dead center
            CameraConfig("azim_narrow_5", 26.0, -5.0, 5.0, 1.2, 1.8),         # Narrow offset
            CameraConfig("azim_moderate_15", 26.0, -15.0, 15.0, 1.2, 1.8),    # Moderate offset
            CameraConfig("azim_wide_25", 26.0, -25.0, 25.0, 1.2, 1.8),        # Wide offset
            CameraConfig("azim_extreme_left_45", 26.0, -45.0, -45.0, 1.2, 1.8),  # Extreme left
            CameraConfig("azim_extreme_right_45", 26.0, 45.0, 45.0, 1.2, 1.8),    # Extreme right

            # === ELEVATION ABLATION (upward/downward tilt) ===
            CameraConfig("elev_eye_level", 0.0, -10.0, 10.0, 1.2, 1.8),       # Eye level (0°)
            CameraConfig("elev_slight_down_10", 10.0, -10.0, 10.0, 1.2, 1.8), # Slight down
            CameraConfig("elev_moderate_20", 20.0, -10.0, 10.0, 1.2, 1.8),    # Moderate down
            CameraConfig("elev_steep_45", 45.0, -10.0, 10.0, 1.2, 1.8),       # Steep looking down
            CameraConfig("elev_extreme_60", 60.0, -10.0, 10.0, 1.2, 1.8),     # Very steep
            CameraConfig("elev_negative_10_up", -10.0, -10.0, 10.0, 1.2, 1.8), # Slight up

            # === DISTANCE ABLATION (camera distance) ===
            CameraConfig("dist_close_0.8", 26.0, -10.0, 10.0, 0.8, 1.8),      # Close
            CameraConfig("dist_near_1.0", 26.0, -10.0, 10.0, 1.0, 1.8),       # Near
            CameraConfig("dist_far_1.5", 26.0, -10.0, 10.0, 1.5, 1.8),        # Far
            CameraConfig("dist_very_far_2.0", 26.0, -10.0, 10.0, 2.0, 1.8),   # Very far

            # === LOOK-AT HEIGHT ABLATION ===
            CameraConfig("lookat_low_0.5", 26.0, -10.0, 10.0, 1.2, 0.5),      # Low
            CameraConfig("lookat_medium_1.2", 26.0, -10.0, 10.0, 1.2, 1.2),   # Medium
            CameraConfig("lookat_high_2.2", 26.0, -10.0, 10.0, 1.2, 2.2),     # High

            # === EXTREME COMBINATIONS ===
            CameraConfig("extreme_combo_1", 60.0, -45.0, 45.0, 0.8, 0.5),     # Extreme: very steep, wide angle, close, low look-at
            CameraConfig("extreme_combo_2", -10.0, -30.0, 30.0, 2.0, 2.5),    # Extreme: looking up, wide, far, high look-at
            CameraConfig("extreme_combo_3", 45.0, -20.0, 20.0, 1.5, 1.0),     # Moderate extreme
        ]

    def get_extreme_configs(self) -> List[CameraConfig]:
        """Extreme variations: pushing boundaries."""
        return [
            # Extreme angles
            CameraConfig("extreme_straight_down_89", 89.0, -10.0, 10.0, 1.2, 1.8),
            CameraConfig("extreme_looking_up_45", -45.0, -10.0, 10.0, 1.2, 1.8),

            # Extreme horizontal angles
            CameraConfig("extreme_side_90_left", 26.0, -90.0, -90.0, 1.2, 1.8),
            CameraConfig("extreme_side_90_right", 26.0, 90.0, 90.0, 1.2, 1.8),

            # Extreme distances
            CameraConfig("extreme_distance_very_close_0.3", 26.0, -10.0, 10.0, 0.3, 1.8),
            CameraConfig("extreme_distance_far_3.0", 26.0, -10.0, 10.0, 3.0, 1.8),

            # Extreme look-at heights
            CameraConfig("extreme_lookat_ground_0.0", 26.0, -10.0, 10.0, 1.2, 0.0),
            CameraConfig("extreme_lookat_top_3.0", 26.0, -10.0, 10.0, 1.2, 3.0),
        ]

    def create_config_file(self, camera_cfg: CameraConfig) -> Path:
        """Create a YAML config file for a camera configuration."""
        config = self.base_config.copy()
        config["stage2"]["camera_elevation"] = camera_cfg.camera_elevation
        config["stage2"]["camera_azim_min"] = camera_cfg.camera_azim_min
        config["stage2"]["camera_azim_max"] = camera_cfg.camera_azim_max
        config["stage2"]["camera_dist"] = camera_cfg.camera_dist
        config["stage2"]["camera_look_at_height"] = camera_cfg.camera_look_at_height

        # Modify output directories to be specific to this ablation
        output_base = self.results_dir / camera_cfg.name
        output_base.mkdir(parents=True, exist_ok=True)

        config["depth_maps_dir"] = str((output_base / "depth_maps").resolve())
        config["images_dir"] = str((output_base / "images").resolve())

        config_path = self.configs_dir / f"{camera_cfg.name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_path

    def create_sample_prompts_file(self) -> Path:
        """Create a small prompts file for quick testing."""
        prompts_path = self.output_dir / "ablation_prompts.txt"
        original_prompts_path = Path(self.base_config["prompts_file"])

        # Make path absolute if relative
        if not original_prompts_path.is_absolute():
            original_prompts_path = self.project_root / original_prompts_path

        if not original_prompts_path.exists():
            # Create default prompts if file doesn't exist
            sample_prompts = [
                "A red wooden chair",
                "A blue ceramic vase",
                "A golden metallic sphere",
            ]
        else:
            with open(original_prompts_path, "r") as f:
                all_prompts = [line.strip() for line in f if line.strip()]
            sample_prompts = all_prompts[:self.sample_size]

        with open(prompts_path, "w") as f:
            for prompt in sample_prompts:
                f.write(prompt + "\n")

        return prompts_path

    def run_ablation(self, camera_configs: List[CameraConfig], sample_only: bool = True) -> Dict:
        """
        Run the full pipeline for each camera configuration.

        Args:
            camera_configs: List of camera configurations to test
            sample_only: If True, run on sample prompts only

        Returns:
            Dictionary of results for each camera configuration
        """
        # Create sample prompts if needed
        if sample_only:
            prompts_file = self.create_sample_prompts_file()
        else:
            prompts_file = Path(self.base_config["prompts_file"])

        results = {}
        start_time = time.time()

        print("\n" + "=" * 80)
        print("CAMERA PLACEMENT ABLATION STUDY")
        print("=" * 80)
        print(f"Total configurations: {len(camera_configs)}")
        print(f"Sample mode: {sample_only}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80 + "\n")

        for i, camera_cfg in enumerate(camera_configs, 1):
            print(f"\n[{i}/{len(camera_configs)}] {camera_cfg.description()}")
            print("-" * 80)

            try:
                # Create config file
                config_path = self.create_config_file(camera_cfg)

                # Update base config to use sample prompts
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                cfg["prompts_file"] = str(prompts_file)
                with open(config_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False)

                # Run generate_depth_maps.py
                print(f"  Stage 1+2: Generating depth maps...")
                depth_start = time.time()
                result_depth = subprocess.run(
                    ["python", "scripts/generate_depth_maps.py", "--config", str(config_path.resolve())],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout per config
                    cwd=str(self.project_root)
                )
                depth_time = time.time() - depth_start

                depth_maps_dir = Path(cfg["depth_maps_dir"]).resolve()
                depth_maps_count = len(list(depth_maps_dir.glob("depth_*.png"))) if depth_maps_dir.exists() else 0

                print(f"  ✓ Depth maps generated: {depth_maps_count} in {depth_time:.1f}s")

                if result_depth.returncode != 0:
                    print(f"  ✗ Error in Stage 1+2:")
                    print(f"    {result_depth.stderr[:500]}")

                # Run generate_images.py
                print(f"  Stage 3: Generating images...")
                image_start = time.time()
                result_image = subprocess.run(
                    ["python", "scripts/generate_images.py", "--config", str(config_path.resolve())],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=str(self.project_root)
                )
                image_time = time.time() - image_start

                images_dir = Path(cfg["images_dir"]).resolve()
                images_count = len(list(images_dir.glob("image_*.png"))) if images_dir.exists() else 0

                print(f"  ✓ Images generated: {images_count} in {image_time:.1f}s")

                if result_image.returncode != 0:
                    print(f"  ✗ Error in Stage 3:")
                    print(f"    {result_image.stderr[:500]}")

                # Collect metrics
                results[camera_cfg.name] = {
                    "config": asdict(camera_cfg),
                    "depth_maps_count": depth_maps_count,
                    "depth_time_sec": depth_time,
                    "images_count": images_count,
                    "image_time_sec": image_time,
                    "total_time_sec": depth_time + image_time,
                    "success": result_depth.returncode == 0 and result_image.returncode == 0,
                    "config_path": str(config_path),
                    "depth_maps_dir": str(depth_maps_dir),
                    "images_dir": str(images_dir),
                }

            except subprocess.TimeoutExpired:
                print(f"  ✗ Timeout after 10 minutes")
                results[camera_cfg.name] = {
                    "config": asdict(camera_cfg),
                    "success": False,
                    "error": "Timeout",
                }
            except Exception as e:
                print(f"  ✗ Exception: {e}")
                results[camera_cfg.name] = {
                    "config": asdict(camera_cfg),
                    "success": False,
                    "error": str(e),
                }

        total_time = time.time() - start_time

        # Save results
        self.results = results
        results_file = self.output_dir / "ablation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("\n" + "=" * 80)
        print("ABLATION STUDY COMPLETE")
        print("=" * 80)
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Results saved: {results_file}")
        print("=" * 80)

        return results

    def print_summary(self):
        """Print a summary of results."""
        if not self.results:
            print("No results to summarize. Run ablation first.")
            return

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        success_count = sum(1 for r in self.results.values() if r.get("success", False))
        print(f"Success rate: {success_count}/{len(self.results)} ({100*success_count/len(self.results):.1f}%)")

        # Depth map generation
        print("\nDepth Map Generation:")
        for name, result in self.results.items():
            if result.get("success"):
                count = result.get("depth_maps_count", 0)
                time_sec = result.get("depth_time_sec", 0)
                print(f"  {name:40s}: {count} maps in {time_sec:6.1f}s")

        # Image generation
        print("\nImage Generation:")
        for name, result in self.results.items():
            if result.get("success"):
                count = result.get("images_count", 0)
                time_sec = result.get("image_time_sec", 0)
                print(f"  {name:40s}: {count} images in {time_sec:6.1f}s")

        # Failures
        failures = {k: v for k, v in self.results.items() if not v.get("success", False)}
        if failures:
            print(f"\nFailed configurations ({len(failures)}):")
            for name, result in failures.items():
                error = result.get("error", "Unknown")
                print(f"  {name:40s}: {error}")

        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Camera placement sensitivity ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/inference.yaml",
        help="Base inference config file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="ablation/camera_sensitivity",
        help="Output directory for ablation results"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["standard", "extreme", "all"],
        default="standard",
        help="Ablation preset: standard (reasonable variations), extreme (edge cases), all (both)"
    )
    parser.add_argument(
        "--sample_only",
        action="store_true",
        help="Run on sample prompts only (faster testing)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=3,
        help="Number of prompts to sample"
    )

    args = parser.parse_args()

    # Create ablation study
    study = AblationStudy(args.config, args.output, sample_size=args.sample_size)

    # Select configurations
    if args.preset == "standard":
        configs = study.get_standard_configs()
    elif args.preset == "extreme":
        configs = study.get_extreme_configs()
    else:  # all
        configs = study.get_standard_configs() + study.get_extreme_configs()

    # Run ablation
    results = study.run_ablation(configs, sample_only=args.sample_only)

    # Print summary
    study.print_summary()


if __name__ == "__main__":
    main()
