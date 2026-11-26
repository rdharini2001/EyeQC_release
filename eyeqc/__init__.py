"""EyeQC: interpretable quality assessment for colour fundus photographs.

The model used in the paper is loaded with ``eyeqc.model.load_model``.
"""

from .core import ACTION_RULES, DISPLAY_NAMES, FEATURES, EyeQCModel, extract_features
from .maps import MAP_KEYS, overlay, quality_maps, tile_grid

__all__ = [
    "ACTION_RULES",
    "DISPLAY_NAMES",
    "FEATURES",
    "EyeQCModel",
    "extract_features",
    "MAP_KEYS",
    "overlay",
    "quality_maps",
    "tile_grid",
]

__version__ = "3.0.0"
