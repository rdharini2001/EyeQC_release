"""Optional FLAIR image-text comparison (not part of EyeQC and not used by its score)."""
from .engine import CATEGORY_PROMPTS, FlairUnavailableError, benchmark_image, load_flair

__all__ = ["CATEGORY_PROMPTS", "FlairUnavailableError", "benchmark_image", "load_flair"]
