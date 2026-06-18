#!/usr/bin/env python
"""
Analyze and visualize camera sensitivity ablation results.

Generates:
1. Summary statistics and comparisons
2. Side-by-side depth map comparisons
3. Performance metrics (speed, success rate)
4. Sensitivity analysis (which parameters matter most)

Usage:
    python scripts/analyze_ablation.py ablation_results/camera_sensitivity
    python scripts/analyze_ablation.py ablation_results/camera_sensitivity --plot_depth_maps
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_results(results_dir: Path) -> Dict:
    """Load ablation results from JSON."""
    results_file = results_dir / "ablation_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r") as f:
        return json.load(f)


def print_performance_summary(results: Dict) -> None:
    """Print performance metrics for all configurations."""
    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY")
    print("=" * 100)

    # Sort by total time
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("total_time_sec", float("inf"))
    )

    print(f"\n{'Configuration':<45} {'Depth Time':<12} {'Image Time':<12} {'Total':<10} {'Status'}")
    print("-" * 100)

    for name, result in sorted_results:
        if result.get("success", False):
            depth_t = result.get("depth_time_sec", 0)
            image_t = result.get("image_time_sec", 0)
            total_t = depth_t + image_t
            status = "✓ Success"
        else:
            depth_t = image_t = total_t = 0
            status = f"✗ Failed ({result.get('error', 'Unknown')})"

        print(
            f"{name:<45} {depth_t:>8.1f}s    {image_t:>8.1f}s    {total_t:>7.1f}s   {status}"
        )

    # Statistics
    successful = [r for r in results.values() if r.get("success", False)]
    if successful:
        times = [r.get("depth_time_sec", 0) + r.get("image_time_sec", 0) for r in successful]
        print("\n" + "-" * 100)
        print(f"Success rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")
        print(f"Avg time: {np.mean(times):.1f}s ± {np.std(times):.1f}s")
        print(f"Min time: {np.min(times):.1f}s, Max time: {np.max(times):.1f}s")
        print("=" * 100 + "\n")


def analyze_parameter_impact(results: Dict) -> None:
    """Analyze which parameters have the most impact on performance."""
    print("\n" + "=" * 100)
    print("PARAMETER IMPACT ANALYSIS")
    print("=" * 100)

    successful = {k: v for k, v in results.items() if v.get("success", False)}
    if not successful:
        print("No successful results to analyze.")
        return

    # Extract parameters and times
    params = ["camera_elevation", "camera_azim_min", "camera_azim_max", "camera_dist", "camera_look_at_height"]

    for param in params:
        values = []
        times = []

        for name, result in successful.items():
            cfg = result.get("config", {})
            val = cfg.get(param)
            time_sec = result.get("depth_time_sec", 0) + result.get("image_time_sec", 0)

            if val is not None:
                values.append(val)
                times.append(time_sec)

        if values:
            corr = np.corrcoef(values, times)[0, 1]
            std_dev = np.std(times)
            mean_time = np.mean(times)

            print(
                f"\n{param:<30} | Correlation with time: {corr:>7.3f} | "
                f"Mean time: {mean_time:>7.1f}s | Std dev: {std_dev:>6.1f}s"
            )

            # Show range
            print(f"  Value range: {np.min(values):.1f} to {np.max(values):.1f}")

    print("=" * 100 + "\n")


def visualize_depth_maps(results_dir: Path, max_configs: int = 6) -> None:
    """Create side-by-side comparisons of depth maps for different camera settings."""
    print("\nGenerating depth map visualizations...")

    results = load_results(results_dir)
    successful = [
        (k, v) for k, v in results.items()
        if v.get("success", False) and v.get("depth_maps_count", 0) > 0
    ][:max_configs]

    if not successful:
        print("No successful depth maps to visualize.")
        return

    # Load first depth map from each configuration
    depth_images = []
    for name, result in successful:
        depth_dir = Path(result.get("depth_maps_dir", ""))
        depth_files = sorted(depth_dir.glob("depth_*.png"))

        if depth_files:
            img = Image.open(depth_files[0]).convert("RGB")
            depth_images.append((name, img))

    if not depth_images:
        print("No depth map files found.")
        return

    # Create grid visualization
    n_cols = min(3, len(depth_images))
    n_rows = (len(depth_images) + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    for idx, (name, img) in enumerate(depth_images):
        ax = fig.add_subplot(gs[idx])
        ax.imshow(img)
        ax.set_title(name, fontsize=10, wrap=True)
        ax.axis("off")

    # Save
    output_path = results_dir / "depth_map_comparison.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def visualize_parameter_sensitivity(results_dir: Path) -> None:
    """Create charts showing parameter sensitivity."""
    print("\nGenerating parameter sensitivity charts...")

    results = load_results(results_dir)
    successful = {k: v for k, v in results.items() if v.get("success", False)}

    if not successful:
        print("No successful results to visualize.")
        return

    # Parameters to analyze
    params = [
        ("camera_elevation", "Camera Elevation (degrees)"),
        ("camera_dist", "Camera Distance Multiplier"),
        ("camera_look_at_height", "Look-At Height (meters)"),
    ]

    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4))
    if len(params) == 1:
        axes = [axes]

    for ax, (param, label) in zip(axes, params):
        values = []
        times = []
        names = []

        for name, result in successful.items():
            cfg = result.get("config", {})
            val = cfg.get(param)
            time_sec = result.get("depth_time_sec", 0) + result.get("image_time_sec", 0)

            if val is not None:
                values.append(val)
                times.append(time_sec)
                names.append(name.replace("_", " "))

        if values:
            # Sort by parameter value
            sorted_data = sorted(zip(values, times, names))
            values, times, names = zip(*sorted_data)

            ax.scatter(values, times, s=100, alpha=0.6, edgecolors="black")
            ax.plot(values, times, alpha=0.3)
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel("Total Time (seconds)", fontsize=11)
            ax.set_title(f"Sensitivity to {label}", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = results_dir / "parameter_sensitivity.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def generate_comparison_table(results_dir: Path, output_file: Path) -> None:
    """Generate a detailed CSV comparison table."""
    print(f"\nGenerating comparison table...")

    results = load_results(results_dir)

    with open(output_file, "w") as f:
        # Header
        f.write("Configuration,")
        f.write("Elevation,Azim_Min,Azim_Max,Distance,LookAt_Height,")
        f.write("Depth_Maps,Depth_Time_s,Images,Image_Time_s,Total_Time_s,Success\n")

        # Rows
        for name, result in sorted(results.items()):
            cfg = result.get("config", {})
            success = "✓" if result.get("success", False) else "✗"

            f.write(
                f"{name},"
                f"{cfg.get('camera_elevation', '')},"
                f"{cfg.get('camera_azim_min', '')},"
                f"{cfg.get('camera_azim_max', '')},"
                f"{cfg.get('camera_dist', '')},"
                f"{cfg.get('camera_look_at_height', '')},"
                f"{result.get('depth_maps_count', '')},"
                f"{result.get('depth_time_sec', ''):.1f},"
                f"{result.get('images_count', '')},"
                f"{result.get('image_time_sec', ''):.1f},"
                f"{result.get('total_time_sec', ''):.1f},"
                f"{success}\n"
            )

    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze camera sensitivity ablation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to ablation results directory"
    )
    parser.add_argument(
        "--plot_depth_maps",
        action="store_true",
        help="Generate depth map comparison visualizations"
    )
    parser.add_argument(
        "--plot_sensitivity",
        action="store_true",
        help="Generate parameter sensitivity charts"
    )
    parser.add_argument(
        "--all_plots",
        action="store_true",
        help="Generate all plots"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Load results
    try:
        results = load_results(results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Print summaries
    print_performance_summary(results)
    analyze_parameter_impact(results)

    # Generate visualizations if requested
    if args.plot_depth_maps or args.all_plots:
        visualize_depth_maps(results_dir)

    if args.plot_sensitivity or args.all_plots:
        visualize_parameter_sensitivity(results_dir)

    # Generate comparison table
    csv_file = results_dir / "ablation_comparison.csv"
    generate_comparison_table(results_dir, csv_file)

    print(f"\n{'='*100}")
    print("Analysis complete. Check the results directory for detailed outputs.")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
