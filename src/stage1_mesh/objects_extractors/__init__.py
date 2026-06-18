from .heuristic import SimpleSplitObjectsExtractor, SimpleSplitObjectsExtractor2

def get_objects_extractor(method_name: str):
    """Factory function to pick the algorithm."""
    if method_name == "simple":
        return SimpleSplitObjectsExtractor()
    elif method_name == "simple2":
        return SimpleSplitObjectsExtractor2()
    else:
        raise ValueError(f"Unknown objects extractor method: {method_name}")