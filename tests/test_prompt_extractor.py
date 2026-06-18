import sys
import os

# Ensure the src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage1_mesh.objects_extractors.heuristic import SimpleSplitObjectsExtractor2

def run_tests():
    extractor = SimpleSplitObjectsExtractor2()

    test_cases = [
        "a red chair in front of the mirror, in a cozy modern living room",
        "an apple in front of the mirror, with warm sunlight shining through the window",
        "a yellow dog, an old clock in front of the mirror, in a minimalist apartment setting",
        "a wooden table, a beautiful vase in front of the mirror, in a calm and elegant home interior",
        "an elegant sofa, a TV in front of the mirror, with soft shadows on the wooden floor",
        # Test case with the reflection wording handled by the extractor
        "a potted plant in front of the mirror, in a small room with plain white walls and a wooden floor, both the potted plant and its reflection are visible"
    ]

    print("=== Testing SimpleSplitObjectsExtractor2 ===\n")
    for i, prompt in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"  Prompt:   '{prompt}'")
        extracted = extractor.extract(prompt)
        print(f"  Result:   {extracted}")
        print("-" * 50)

if __name__ == "__main__":
    run_tests()
