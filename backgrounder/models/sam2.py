from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from backgrounder.utils import choose_dtype

# SAM 2.1 — Meta Research — Apache-2.0 licensed.
# Requires transformers >= 4.49.
_MODEL_SMALL    = "facebook/sam2.1-hiera-small"
_MODEL_BASE     = "facebook/sam2.1-hiera-base-plus"


class SAM2Segmenter:
    """
    SAM 2.1 — precise mask prediction via point / box prompts.
    Apache-2.0 licensed.  Requires transformers >= 4.49.

    Stage G: activated when ensemble quality is below sam2_quality_trigger
    or when the scene is classified as 'complex_multi'.
    Prompts are either bounding boxes supplied by OWLv2 or point prompts
    auto-derived from the coarse alpha.
    """

    def __init__(
        self,
        device: str = "cpu",
        fp16: bool = False,
        model_id: str = _MODEL_BASE,
    ) -> None:
        self._device = device
        self._dtype = choose_dtype(device, fp16)
        self._model_id = model_id
        self._model = None
        self._processor = None
        self._loaded = False

    def load(self) -> "SAM2Segmenter":
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def _load(self) -> None:
        try:
            from transformers import Sam2Processor, Sam2Model
        except ImportError as e:
            raise ImportError(
                "SAM 2.1 requires transformers >= 4.49. "
                "Run: pip install -U transformers"
            ) from e
        self._processor = Sam2Processor.from_pretrained(self._model_id)
        self._model = Sam2Model.from_pretrained(self._model_id)
        self._model = self._model.to(device=self._device, dtype=self._dtype)
        self._model.eval()

    # ------------------------------------------------------------------ #

    def refine(
        self,
        image: Image.Image,
        coarse_alpha: np.ndarray,
        boxes: Optional[List[List[float]]] = None,
    ) -> np.ndarray:
        """
        Refine coarse alpha using SAM 2.1 mask prediction.

        boxes : list of [x1, y1, x2, y2] pixel coords (from OWLv2).
                If None, point prompts are auto-derived from coarse_alpha.

        Returns refined alpha float32 [0, 1], same H×W as coarse_alpha.
        """
        if not self._loaded:
            self.load()

        rgb = image.convert("RGB")
        orig_wh = rgb.size  # (W, H)

        # ── build processor inputs ─────────────────────────────────────
        if boxes is not None and len(boxes) > 0:
            # Box prompts: one prompt per box
            input_boxes = [[[float(c) for c in b] for b in boxes]]
            raw = self._processor(
                images=rgb,
                input_boxes=input_boxes,
                return_tensors="pt",
            )
        else:
            # Point prompts: all fg + bg as a single multi-point prompt
            points, labels = _alpha_to_points(coarse_alpha, n_fg=8, n_bg=4)
            if len(points) == 0:
                return coarse_alpha
            input_points = [[points.tolist()]]   # (batch=1, 1 group, N, 2)
            input_labels = [[labels.tolist()]]   # (batch=1, 1 group, N)
            raw = self._processor(
                images=rgb,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            )

        # ── move to device (model keys only) ──────────────────────────
        _MODEL_KEYS = {"pixel_values", "input_points", "input_labels", "input_boxes"}
        model_inputs = {}
        for k, v in raw.items():
            if k not in _MODEL_KEYS:
                continue
            if isinstance(v, torch.Tensor):
                v = v.to(
                    device=self._device,
                    dtype=self._dtype if v.is_floating_point() else v.dtype,
                )
            model_inputs[k] = v

        # ── forward ───────────────────────────────────────────────────
        with torch.inference_mode():
            outputs = self._model(**model_inputs)

        # ── post-process masks (version-agnostic) ─────────────────────
        # pred_masks: (1, num_prompts, num_masks, H', W') — raw logits
        # Resize to original size via F.interpolate; no processor API needed.
        import torch.nn.functional as F

        H_orig, W_orig = coarse_alpha.shape
        iou_scores = outputs.iou_scores[0]   # (P, M)
        pred_masks = outputs.pred_masks[0]   # (P, M, H', W')

        P = pred_masks.shape[0]
        best: List[torch.Tensor] = []
        for p in range(P):
            best_idx = int(iou_scores[p].argmax().item())
            mask_logit = pred_masks[p, best_idx].unsqueeze(0).unsqueeze(0).float()  # (1,1,H',W')
            mask_up = F.interpolate(mask_logit, size=(H_orig, W_orig),
                                    mode="bilinear", align_corners=False)
            best.append((mask_up.squeeze() > 0.0))  # sigmoid(0) = 0.5 threshold
        if not best:
            return coarse_alpha

        sam_mask = torch.stack(best, dim=0).any(dim=0).float().numpy()  # (H, W)

        # ── fuse SAM mask into uncertain band ─────────────────────────
        uncertain = (coarse_alpha > 0.05) & (coarse_alpha < 0.95)
        refined = coarse_alpha.copy()
        refined[uncertain] = (
            0.35 * coarse_alpha[uncertain] + 0.65 * sam_mask[uncertain]
        )
        return np.clip(refined, 0.0, 1.0).astype(np.float32)

    def unload(self) -> None:
        del self._model, self._processor
        self._model = self._processor = None
        self._loaded = False
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()


# ── helpers ────────────────────────────────────────────────────────────────

def _alpha_to_points(
    alpha: np.ndarray,
    n_fg: int = 8,
    n_bg: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample stratified fg + bg points from coarse alpha, returned as (x, y)."""
    fg_yx = np.argwhere(alpha > 0.75)
    bg_yx = np.argwhere(alpha < 0.05)
    rng = np.random.default_rng(42)

    def _sample(pool: np.ndarray, n: int) -> np.ndarray:
        if len(pool) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return pool[idx][:, ::-1].astype(np.float32)  # (y,x) → (x,y)

    fg_pts = _sample(fg_yx, n_fg)
    bg_pts = _sample(bg_yx, n_bg)

    if len(fg_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.array([], dtype=np.int64)

    pts = np.concatenate([fg_pts, bg_pts], axis=0) if len(bg_pts) else fg_pts
    lbl = np.array([1] * len(fg_pts) + [0] * len(bg_pts), dtype=np.int64)
    return pts, lbl
