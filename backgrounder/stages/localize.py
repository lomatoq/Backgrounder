from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from backgrounder.models.sam2 import SAM2Segmenter
    from backgrounder.models.sam3 import SAM3Segmenter
    from backgrounder.models.owlv2 import OWLv2Localizer


def sam2_refine(
    image: Image.Image,
    alpha: np.ndarray,
    sam2: "SAM2Segmenter",
    owlv2: Optional["OWLv2Localizer"] = None,
    subject_type: str = "generic",
) -> np.ndarray:
    """
    Stage G: SAM 2.1 mask refinement.

    If OWLv2 is supplied, first localise subjects as bounding boxes
    (better for multi-object / complex scenes), then prompt SAM 2.1
    with those boxes.  Falls back to auto-derived point prompts when
    OWLv2 returns no detections or is not loaded.
    """
    boxes: Optional[List[List[float]]] = None
    if owlv2 is not None:
        detected = owlv2.detect(image, subject_type=subject_type)
        if detected:
            boxes = detected

    return sam2.refine(image, alpha, boxes=boxes)


def sam3_refine(
    image: Image.Image,
    alpha: np.ndarray,
    sam3: "SAM3Segmenter",
    owlv2: Optional["OWLv2Localizer"] = None,
    subject_type: str = "generic",
) -> tuple[np.ndarray, dict]:
    """
    Stage G: SAM 3.1 concept/box refinement.

    SAM 3 is strongest when it can use a semantic prompt plus a coarse box.
    OWLv2 boxes are preferred when available; otherwise SAM 3 receives boxes
    derived from the current alpha.
    """
    boxes: Optional[List[List[float]]] = None
    if owlv2 is not None:
        detected = owlv2.detect(image, subject_type=subject_type)
        if detected:
            boxes = detected

    return sam3.refine(image, alpha, subject_type=subject_type, boxes=boxes)
