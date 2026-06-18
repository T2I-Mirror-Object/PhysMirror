import argparse
import os
import csv
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    import pyiqa
except ImportError:
    raise ImportError("Please install pyiqa: pip install pyiqa")

AVAILABLE_METRICS = ["clipiqa", "maniqa", "musiq"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute no-reference IQA scores (CLIP-IQA, MANIQA, MUSIQ) "
                    "for images in model subdirectories."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Root directory containing model subdirectories with images "
             "(e.g., inference_result/infer-images/infer-images)."
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", default=AVAILABLE_METRICS,
        choices=AVAILABLE_METRICS,
        help=f"IQA metrics to compute. Default: all ({', '.join(AVAILABLE_METRICS)})."
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Path to save per-image results CSV. "
             "A summary CSV is also saved alongside it."
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)."
    )
    parser.add_argument(
        "--model_dirs", type=str, nargs="+", default=None,
        help="Optional: specify which subdirectory names to evaluate. "
             "Default: all subdirectories in input_dir."
    )
    return parser.parse_args()


def get_image_files(directory: Path):
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return sorted(f.name for f in directory.iterdir()
                  if f.suffix.lower() in valid_ext)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist.")
        return

    # Discover model subdirectories
    if args.model_dirs:
        model_dirs = [input_dir / name for name in args.model_dirs]
    else:
        model_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())

    if not model_dirs:
        print(f"No subdirectories found in {input_dir}")
        return

    print(f"Models found ({len(model_dirs)}): {[d.name for d in model_dirs]}")
    print(f"Metrics       : {args.metrics}")
    print(f"Device        : {args.device}\n")

    # Load IQA metric models once
    print("Loading IQA models...")
    metric_fns = {}
    for metric_name in args.metrics:
        print(f"  [{metric_name}] loading...")
        metric_fns[metric_name] = pyiqa.create_metric(metric_name, device=args.device)
    print()

    # Evaluate
    all_rows = []
    # model_name -> metric_name -> list of scores
    model_stats: dict[str, dict[str, list[float]]] = {}

    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"Warning: {model_dir} does not exist, skipping.")
            continue

        model_name = model_dir.name
        image_files = get_image_files(model_dir)

        if not image_files:
            print(f"No images found in {model_dir}, skipping.")
            continue

        model_stats[model_name] = {m: [] for m in args.metrics}
        print(f"Processing [{model_name}] — {len(image_files)} images")

        for fname in tqdm(image_files, desc=model_name, ncols=80):
            img_path = str(model_dir / fname)
            row = {"model": model_name, "image": fname}

            for metric_name, metric_fn in metric_fns.items():
                with torch.no_grad():
                    score = metric_fn(img_path)
                score_val = float(score.item() if hasattr(score, "item") else score)
                row[metric_name] = round(score_val, 6)
                model_stats[model_name][metric_name].append(score_val)

            all_rows.append(row)

    if not all_rows:
        print("No results collected.")
        return

    df = pd.DataFrame(all_rows)

    # Summary table
    print("\n" + "=" * 65)
    print("SUMMARY — Average No-Reference IQA Scores per Model")
    print("=" * 65)

    summary_rows = []
    for model_name, stats in model_stats.items():
        row = {"Model": model_name}
        for metric in args.metrics:
            scores = stats[metric]
            row[metric.upper()] = f"{sum(scores) / len(scores):.4f}" if scores else "N/A"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print()

    # Save CSVs
    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_path, index=False)
        print(f"Per-image results saved to : {out_path}")

        summary_path = out_path.with_name(out_path.stem + "_summary" + out_path.suffix)
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary results saved to   : {summary_path}")


if __name__ == "__main__":
    main()
