import argparse
import json
import os
import pandas as pd
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Map images to prompts from a CSV file based on filename index.")
    
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to the prompts.csv file."
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing the generated images."
    )
    parser.add_argument(
        "--output_json", type=str, default="prompts_map.json",
        help="Path to save the output JSON file."
    )
    parser.add_argument(
        "--prompt_col", type=str, default="prompt",
        help="The name of the column in the CSV that contains the text prompt. Defaults to 'prompt'."
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load the CSV
    try:
        df = pd.read_csv(args.csv_path, sep=';')
        print(f"Loaded CSV with {len(df)} rows.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

    # Check if the prompt column exists
    if args.prompt_col not in df.columns:
        print(f"Error: Column '{args.prompt_col}' not found in CSV.")
        print(f"Available columns: {list(df.columns)}")
        print("Please specify the correct column name using --prompt_col")
        sys.exit(1)

    # 2. Scan Image Directory
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' does not exist.")
        sys.exit(1)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print("No image files found in the directory.")
        sys.exit(1)

    prompt_map = {}
    matched_count = 0
    skipped_count = 0

    print("Mapping images to prompts...")

    # 3. Match Logic
    for filename in image_files:
        try:
            # Split filename by underscore to get the index (e.g., "0_filename..." -> "0")
            index_str = filename.split('_')[0]
            
            # Convert to integer index
            row_idx = int(index_str)

            # Check if index is within valid CSV range
            if 0 <= row_idx < len(df):
                # Extract prompt from the specific row and column
                prompt_text = str(df.iloc[row_idx][args.prompt_col])
                prompt_map[filename] = prompt_text
                matched_count += 1
            else:
                print(f"Warning: Image '{filename}' has index {row_idx}, which is out of bounds for the CSV (rows 0-{len(df)-1}).")
                skipped_count += 1
                
        except ValueError:
            print(f"Warning: Could not extract a valid number index from filename '{filename}'. Skipping.")
            skipped_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped_count += 1

    # 4. Save JSON
    with open(args.output_json, 'w') as f:
        json.dump(prompt_map, f, indent=4)

    print("-" * 30)
    print(f"Mapping complete.")
    print(f"Successfully mapped: {matched_count} images")
    print(f"Skipped: {skipped_count} images")
    print(f"Output saved to: {args.output_json}")

if __name__ == "__main__":
    main()