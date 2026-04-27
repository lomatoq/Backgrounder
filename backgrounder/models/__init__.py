from .base import BaseSegmenter
from .birefnet import BiRefNetSegmenter
from .ben2 import BEN2Segmenter
from .depth_anything import DepthAnythingV2Small
from .inspyrenet import InSPyReNetSegmenter

__all__ = [
    "BaseSegmenter",
    "BiRefNetSegmenter",
    "BEN2Segmenter",
    "DepthAnythingV2Small",
    "InSPyReNetSegmenter",
]
