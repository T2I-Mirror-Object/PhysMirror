"""Factory for Stage 1 Text-to-3D models."""

from .base import BaseTextTo3D

# Registry of available models
MODEL_REGISTRY = {
    "shap_e": "ShapEModel",
    "trellis": "TrellisModel",
}


def get_text_to_3d_model(name: str, **kwargs) -> BaseTextTo3D:
    """
    Factory function to get a Text-to-3D model by name.

    Args:
        name: Model name. Options: "shap_e", "trellis"
        **kwargs: Arguments passed to the model constructor

    Returns:
        An instance of BaseTextTo3D

    Example:
        >>> model = get_text_to_3d_model("trellis", device="cuda", model_name="microsoft/TRELLIS-text-large")
        >>> model.generate("a red chair", "outputs/chair.glb")
    """
    name_lower = name.lower()

    if name_lower not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    # Lazy import to avoid loading unnecessary dependencies
    if name_lower == "shap_e":
        from .shap_e_model import ShapEModel
        return ShapEModel(**kwargs)
    elif name_lower == "trellis":
        from .trellis_model import TrellisModel
        return TrellisModel(**kwargs)
    else:
        raise ValueError(f"Model '{name}' is registered but not implemented")


def list_available_models():
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())
