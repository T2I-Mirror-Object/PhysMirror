import sys
import os
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.stage1_mesh.objects_extractors import get_objects_extractor
    from src.stage1_mesh.shap_e_model import ShapEModel
except ImportError as e:
    print("\n[ERROR] Could not import modules.")
    print("Make sure your folder structure is:")
    print("  src/")
    print("    stage1_mesh/")
    print("      shap_e_model.py")
    print("      extractors/")
    print(f"Error details: {e}\n")
    sys.exit(1)

def debug_stage1(prompt: str, extractor_method: str, device: str):
    print("="*60)
    print(f"DEBUGGING STAGE 1: Text-to-3D")
    print(f"Full Prompt:     '{prompt}'")
    print(f"Extractor Method: {extractor_method}")
    print("="*60)

    print(f"\n[1/2] Testing Object Extraction...")
    
    try:
        extractor = get_objects_extractor(extractor_method)
        object_name_list = extractor.extract(prompt)
        
        print(f" > SUCCESS: Extracted object name: '{object_name_list}'")
    except Exception as e:
        print(f" > FAILED: Extractor crashed. Error: {e}")
        return

    print(f"\n[2/2] Testing Shap-E Generation...")
    
    start_time = time.time()
    
    try:
        model = ShapEModel(
            device=device,
            output_dir="outputs/debug_meshes",
            guidance=15.0,
            karras_steps=64 
        )
        
        for object_name in object_name_list:
            mesh_path = model.generate(prompt=object_name, object_name=object_name)
            
            elapsed = time.time() - start_time
            print(f" > SUCCESS: Mesh generated in {elapsed:.2f} seconds")
            print(f" > Saved at: {mesh_path}")

    except Exception as e:
        print(f" > FAILED: Shap-E crashed. Error: {e}")

    print("\n" + "="*60)
    print("DEBUG COMPLETE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Stage 1 (Text -> Mesh)")
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="a classic wooden chair in front of the mirror",
        help="The complex prompt to test"
    )
    parser.add_argument(
        "--extractor", 
        type=str, 
        default="simple", 
        choices=["simple", "simple2", "spacy"],
        help="Which extraction algorithm to use"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="cuda or cpu"
    )

    args = parser.parse_args()
    
    debug_stage1(args.prompt, args.extractor, args.device)