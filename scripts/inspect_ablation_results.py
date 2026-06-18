#!/usr/bin/env python
"""
Quick inspection tool for ablation results.

View depth maps and generated images side-by-side for different camera configurations.
Useful for manual visual quality assessment.

Usage:
    python scripts/inspect_ablation_results.py ablation_results/camera_sensitivity
    python scripts/inspect_ablation_results.py ablation_results/camera_sensitivity -c baseline_default azim_frontal
    python scripts/inspect_ablation_results.py ablation_results/camera_sensitivity --compare_all
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_results(results_dir: Path) -> dict:
    """Load ablation results JSON."""
    results_file = results_dir / "ablation_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    with open(results_file, "r") as f:
        return json.load(f)


def get_sample_files(results_dir: Path, config_name: str, file_type: str = "depth") -> List[Path]:
    """
    Get all depth or image files for a configuration.

    Args:
        results_dir: Results directory
        config_name: Configuration name
        file_type: "depth" or "image"

    Returns:
        List of file paths
    """
    if file_type == "depth":
        dir_path = results_dir / "results" / config_name / "depth_maps"
        pattern = "depth_*.png"
    elif file_type == "image":
        dir_path = results_dir / "results" / config_name / "images"
        pattern = "image_*.png"
    else:
        raise ValueError(f"Unknown file type: {file_type}")

    if not dir_path.exists():
        return []

    return sorted(dir_path.glob(pattern))


def list_configurations(results_dir: Path) -> List[str]:
    """List all available configurations."""
    results = load_results(results_dir)
    return sorted(results.keys())


def visualize_single_config(
    results_dir: Path,
    config_name: str,
    show_both: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    """
    Visualize all samples for a single configuration.

    Shows depth maps and corresponding generated images.
    """
    depth_files = get_sample_files(results_dir, config_name, "depth")
    image_files = get_sample_files(results_dir, config_name, "image")

    if not depth_files and not image_files:
        print(f"No results found for {config_name}")
        return

    n_samples = max(len(depth_files), len(image_files))
    n_cols = 2 if show_both and depth_files and image_files else 1
    n_rows = n_samples

    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    for i in range(n_samples):
        # Depth map
        if i < len(depth_files):
            ax = fig.add_subplot(gs[i, 0])
            depth_img = Image.open(depth_files[i])
            ax.imshow(depth_img)
            ax.set_title(f"Depth Map {i+1:03d}", fontsize=10)
            ax.axis("off")

        # Generated image
        if show_both and i < len(image_files):
            ax = fig.add_subplot(gs[i, 1])
            gen_img = Image.open(image_files[i])
            ax.imshow(gen_img)
            ax.set_title(f"Generated Image {i+1:03d}", fontsize=10)
            ax.axis("off")

    fig.suptitle(f"Configuration: {config_name}", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def compare_configurations(
    results_dir: Path,
    config_names: List[str],
    sample_index: int = 0,
    file_type: str = "depth",
    save_path: Optional[Path] = None,
) -> None:
    """
    Compare the same sample across multiple configurations.

    Shows how different camera settings affect the same prompt.
    """
    results = load_results(results_dir)

    # Filter to valid configs
    valid_configs = [c for c in config_names if c in results]
    if not valid_configs:
        print(f"No valid configurations found among {config_names}")
        return

    images = []
    for config_name in valid_configs:
        files = get_sample_files(results_dir, config_name, file_type)
        if sample_index < len(files):
            img = Image.open(files[sample_index])
            images.append((config_name, img))

    if not images:
        print(f"No {file_type} files found at sample index {sample_index}")
        return

    # Create grid
    n_cols = min(4, len(images))
    n_rows = (len(images) + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)

    for idx, (config_name, img) in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(config_name.replace("_", " "), fontsize=10, wrap=True)
        ax.axis("off")

    fig.suptitle(
        f"Comparison: {file_type.capitalize()} (Sample {sample_index+1})",
        fontsize=14,
        fontweight="bold"
    )

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def compare_all_configurations(
    results_dir: Path,
    sample_index: int = 0,
    file_type: str = "depth",
    save_path: Optional[Path] = None,
) -> None:
    """Compare the same sample across ALL successful configurations."""
    results = load_results(results_dir)
    successful_configs = [k for k, v in results.items() if v.get("success", False)]

    print(f"Comparing {len(successful_configs)} successful configurations...")
    compare_configurations(results_dir, successful_configs, sample_index, file_type, save_path)


def print_configuration_stats(results_dir: Path) -> None:
    """Print statistics about available configurations."""
    results = load_results(results_dir)

    print("\n" + "=" * 80)
    print("AVAILABLE CONFIGURATIONS")
    print("=" * 80)

    successful = 0
    failed = 0

    for config_name, result in sorted(results.items()):
        status = "✓" if result.get("success", False) else "✗"
        depth_count = result.get("depth_maps_count", 0)
        image_count = result.get("images_count", 0)

        print(f"{status} {config_name:<45} | Depth: {depth_count} | Images: {image_count}")

        if result.get("success", False):
            successful += 1
        else:
            failed += 1

    print("-" * 80)
    print(f"Total: {successful} successful, {failed} failed ({len(results)} configurations)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and compare ablation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Ablation results directory"
    )
    parser.add_argument(
        "-c", "--configs",
        type=str,
        nargs="+",
        help="Specific configurations to compare"
    )
    parser.add_argument(
        "--compare_all",
        action="store_true",
        help="Compare all successful configurations"
    )
    parser.add_argument(
        "-s", "--sample",
        type=int,
        default=0,
        help="Sample index to inspect/compare (0-based)"
    )
    parser.add_argument(
        "-t", "--type",
        choices=["depth", "image"],
        default="depth",
        help="File type to show (depth or image)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Save to file instead of showing"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all configurations and exit"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # List mode
    if args.list:
        print_configuration_stats(results_dir)
        return

    # Comparison mode
    if args.compare_all:
        output = Path(args.output) if args.output else None
        compare_all_configurations(results_dir, args.sample, args.type, output)

    elif args.configs:
        output = Path(args.output) if args.output else None
        compare_configurations(results_dir, args.configs, args.sample, args.type, output)

    else:
        # Default: list configurations
        print_configuration_stats(results_dir)
        print("\nUsage examples:")
        print(f"  python {parser.prog} {args.results_dir} --list")
        print(f"  python {parser.prog} {args.results_dir} -c baseline_default azim_frontal")
        print(f"  python {parser.prog} {args.results_dir} --compare_all --type image")
        print(f"  python {parser.prog} {args.results_dir} --compare_all -o comparison.png")


if __name__ == "__main__":
    main()
