from __future__ import annotations
from abc import ABC, abstractmethod

from PIL import Image

from backgrounder.result import SegmentationOutput


class BaseSegmenter(ABC):
    """Common interface for all segmentation / matting models."""

    _loaded: bool = False

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _load(self) -> None: ...

    @abstractmethod
    def _predict(self, image: Image.Image) -> SegmentationOutput: ...

    def load(self) -> "BaseSegmenter":
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def predict(self, image: Image.Image) -> SegmentationOutput:
        if not self._loaded:
            self.load()
        return self._predict(image)

    def unload(self) -> None:
        """Release model weights from device memory."""
        pass
