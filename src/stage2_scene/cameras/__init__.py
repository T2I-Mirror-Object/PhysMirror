from .strategies import RandomSideStrategy, FixedFrontalStrategy

# 1. The Registry
CAMERA_STRATEGIES = {
    "random_side": RandomSideStrategy,
    "fixed_front": FixedFrontalStrategy,
    # "orbit": OrbitCameraStrategy (Add this later!)
}

# 2. The Factory
def get_camera_strategy(name: str):
    if name not in CAMERA_STRATEGIES:
        raise ValueError(f"Unknown camera strategy: '{name}'. Available: {list(CAMERA_STRATEGIES.keys())}")
    return CAMERA_STRATEGIES[name]