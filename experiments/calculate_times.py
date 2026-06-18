import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate average stage times from benchmark JSON data."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output", type=str, default="average_times_results.json",
        help="Path to save the output JSON averages. Default: average_times_results.json"
    )
    return parser.parse_args()

def calculate_averages(items):
    """Calculates the average times for a given list of JSON objects."""
    if not items:
        return {
            "count": 0,
            "stage1_time": 0.0,
            "stage2_time": 0.0,
            "stage3_time": 0.0,
            "total_time": 0.0
        }
    
    count = len(items)
    return {
        "count": count,
        "stage1_time": sum(item.get("stage1_time", 0) for item in items) / count,
        "stage2_time": sum(item.get("stage2_time", 0) for item in items) / count,
        "stage3_time": sum(item.get("stage3_time", 0) for item in items) / count,
        "total_time": sum(item.get("total_time", 0) for item in items) / count
    }

def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return

    # Load the JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Filter out the object with prompt_idx == 0
    valid_data = [d for d in data if d.get("prompt_idx", -1) != 0]

    # 2. Categorize the data into the requested index ranges
    cat_all = valid_data
    cat_0_to_159 = [d for d in valid_data if 1 <= d.get("prompt_idx", -1) <= 159]
    cat_160_to_259 = [d for d in valid_data if 160 <= d.get("prompt_idx", -1) <= 259]
    cat_260_to_359 = [d for d in valid_data if 260 <= d.get("prompt_idx", -1) <= 359]

    # 3. Compute averages for each category
    results = {
        "all_objects": calculate_averages(cat_all),
        "idx_0_to_159": calculate_averages(cat_0_to_159),
        "idx_160_to_259": calculate_averages(cat_160_to_259),
        "idx_260_to_359": calculate_averages(cat_260_to_359)
    }

    # 4. Print a clean summary to the console
    print("\n" + "=" * 50)
    print("AVERAGE TIMES SUMMARY")
    print("=" * 50)
    
    for category, stats in results.items():
        print(f"\n[{category}] - Processed {stats['count']} items")
        if stats['count'] > 0:
            print(f"  Stage 1 Avg : {stats['stage1_time']:.4f}s")
            print(f"  Stage 2 Avg : {stats['stage2_time']:.4f}s")
            print(f"  Stage 3 Avg : {stats['stage3_time']:.4f}s")
            print(f"  Total Avg   : {stats['total_time']:.4f}s")
        else:
            print("  No items found in this range.")

    # 5. Save the results to the specified output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "-" * 50)
    print(f"Calculated averages successfully saved to: {output_path.absolute()}")

if __name__ == "__main__":
    main()