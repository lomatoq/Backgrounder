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
    tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[SegmentationOutput]]:
    """
    Run multiple segmenters and return a weighted-average alpha.

    Parameters
    ----------
    tta : bool
        Test-Time Augmentation. When True, each model is also run on a
        horizontally flipped copy of the image; the flipped alpha is mirrored
        back and averaged with the original. Doubles Stage-B inference time
        but significantly improves quality on low-contrast or asymmetric
        scenes where a single forward pass is biased by image orientation.

    Returns
    -------
    alpha_ensemble : float32 [0,1], H×W
    uncertainty    : float32 [0,1], H×W — per-pixel std across models (normalised)
    outputs        : individual SegmentationOutput per model (original orientation)
    """
    if weights is None:
        weights = [1.0] * len(models)
    assert len(weights) == len(models)

    # Original-orientation inference
    outputs: List[SegmentationOutput] = []
    if parallel and len(models) > 1:
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {pool.submit(_run_one, m, image): i for i, m in enumerate(models)}
            results: Dict[int, SegmentationOutput] = {}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        outputs = [results[i] for i in range(len(models))]
    else:
        outputs = [m.predict(image) for m in models]

    alphas = np.stack([o.alpha for o in outputs], axis=0)  # (N, H, W)

    if tta:
        # Horizontal flip augmentation: run all models on the flipped image,
        # then flip the alpha predictions back before averaging.
        image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        outputs_flip: List[SegmentationOutput] = []
        if parallel and len(models) > 1:
            with ThreadPoolExecutor(max_workers=len(models)) as pool:
                futures_f = {pool.submit(_run_one, m, image_flipped): i for i, m in enumerate(models)}
                results_f: Dict[int, SegmentationOutput] = {}
                for fut in as_completed(futures_f):
                    results_f[futures_f[fut]] = fut.result()
            outputs_flip = [results_f[i] for i in range(len(models))]
        else:
            outputs_flip = [m.predict(image_flipped) for m in models]

        # Un-flip each alpha so it aligns with the original image.
        alphas_flip = np.stack([np.fliplr(o.alpha) for o in outputs_flip], axis=0)
        # Concatenate: all N original + N flipped predictions with equal weight.
        alphas = np.concatenate([alphas, alphas_flip], axis=0)
        weights = list(weights) + list(weights)  # same weights for both orientations

    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()

    alpha_ensemble = (alphas * w[:, None, None]).sum(axis=0)

    # Uncertainty: normalised std across all predictions (incl. TTA flips)
    uncertainty = alphas.std(axis=0)
    if uncertainty.max() > 0:
        uncertainty = uncertainty / uncertainty.max()

    return alpha_ensemble.astype(np.float32), uncertainty.astype(np.float32), outputs
