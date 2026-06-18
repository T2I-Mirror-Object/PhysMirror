from .models.flux_omini import FluxOminiWrapper
from .models.seg2any_wrapper import Seg2AnyWrapper
from .models.flux_depth import FluxDepthWrapper
from .models.multi_condition import MirrorMultiConditionWrapper

MODEL_REGISTRY = {
    "flux_omini": FluxOminiWrapper,
    "seg2any": Seg2AnyWrapper,
    "flux_depth": FluxDepthWrapper,
    "multi_cond": MirrorMultiConditionWrapper,
}

def get_t2i_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown T2I model: {name}")
    return MODEL_REGISTRY[name]()