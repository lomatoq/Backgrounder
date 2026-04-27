from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from backgrounder.models.base import BaseSegmenter
from backgrounder.result import SegmentationOutput


def _run_one(model: BaseSegmenter, image: Image.Image) -> SegmentationOutput:
    return model.predict(image)


def ensemble_predict(
    models: List[BaseSegmenter],
    image: Image.Image,
    weights: Optional[List[float]] = None,
    parallel: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[SegmentationOutput]]:
    """
    Run multiple segmenters and return a weighted-average alpha.

    Returns
    -------
    alpha_ensemble : float32 [0,1], H×W
    uncertainty    : float32 [0,1], H×W — per-pixel std across models (normalised)
    outputs        : individual SegmentationOutput per model
    """
    if weights is None:
        weights = [1.0] * len(models)
    assert len(weights) == len(models)

    outputs: List[SegmentationOutput] = []

    if parallel and len(models) > 1:
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {pool.submit(_run_one, m, image): i for i, m in enumerate(models)}
            results: Dict[int, SegmentationOutput] = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        outputs = [results[i] for i in range(len(models))]
    else:
        outputs = [m.predict(image) for m in models]

    alphas = np.stack([o.alpha for o in outputs], axis=0)  # (N, H, W)
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()

    alpha_ensemble = (alphas * w[:, None, None]).sum(axis=0)

    # Uncertainty: std of individual alphas normalised to [0, 1]
    uncertainty = alphas.std(axis=0)
    if uncertainty.max() > 0:
        uncertainty = uncertainty / uncertainty.max()

    return alpha_ensemble.astype(np.float32), uncertainty.astype(np.float32), outputs
