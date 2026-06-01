from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, find_objects, label

from backgrounder.utils import choose_dtype


_PROMPTS = {
    "portrait": "person",
    "animal_fur": "animal",
    "plant_thin": "plant",
    "product": "product",
    "product_opaque": "product",
    "product_glass": "transparent product",
    "transparent": "transparent object",
    "transparent_object": "transparent object",
    "vehicle": "vehicle",
    "anime": "cartoon character",
    "flat_cartoon": "cartoon character",
    "sticker_logo": "sticker",
    "complex_multi": "main objects",
    "busy_scene": "foreground objects",
    "generic": "main foreground object",
}

_SKIP_TYPES = {
    "text_logo",
    "text_glow",
    "document_screenshot",
    "solid_screen_keying",
}


class SAM3Segmenter:
    """
    SAM 3 / 3.1 image-mask backend.

    The official SAM 3 repo is not a normal lightweight dependency yet, so this
    wrapper imports it lazily and fails with a clear setup error. The pipeline
    catches that error and continues with the normal matting stack.
    """

    def __init__(
        self,
        device: str = "cpu",
        fp16: bool = False,
        model_version: str = "sam3.1",
        checkpoint_path: str | None = None,
        confidence_threshold: float = 0.30,
    ) -> None:
        self._device = device
        self._dtype = choose_dtype(device, fp16)
        self._model_version = model_version
        self._checkpoint_path = checkpoint_path
        self._confidence_threshold = confidence_threshold
        self._model = None
        self._processor = None
        self._loaded = False
        self._last_error: str | None = None

    def load(self) -> "SAM3Segmenter":
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def _load(self) -> None:
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
        except ImportError as exc:
            raise ImportError(
                "SAM 3.1 requires the official facebookresearch/sam3 package. "
                "Install it from https://github.com/facebookresearch/sam3 and make sure "
                "the facebook/sam3.1 checkpoint is accessible."
            ) from exc

        checkpoint_path = self._checkpoint_path
        if not checkpoint_path:
            checkpoint_path = str(download_ckpt_from_hf(version=self._model_version))

        self._model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=self._device,
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        if hasattr(self._model, "to"):
            # SAM3's image processor expects mixed bf16 activations on CUDA. Casting
            # the full model to fp16 makes the official decoder hit dtype mismatches.
            self._model = self._model.to(device=self._device)
        if hasattr(self._model, "eval"):
            self._model.eval()

        try:
            self._processor = Sam3Processor(
                self._model,
                device=self._device,
                confidence_threshold=self._confidence_threshold,
            )
        except TypeError:
            self._processor = Sam3Processor(self._model)

    def refine(
        self,
        image: Image.Image,
        coarse_alpha: np.ndarray,
        subject_type: str = "generic",
        boxes: Optional[list[list[float]]] = None,
    ) -> tuple[np.ndarray, dict]:
        if subject_type in _SKIP_TYPES:
            return coarse_alpha, {"sam3_status": "skipped_subject"}
        if not self._loaded:
            self.load()

        rgb = image.convert("RGB")
        w, h = rgb.size
        prompt = _PROMPTS.get(subject_type, _PROMPTS["generic"])
        meta: dict = {
            "sam3_status": "started",
            "sam3_prompt": prompt,
            "sam3_model_version": self._model_version,
        }
        self._last_error = None

        masks = self._predict_with_text(rgb, prompt)
        source = "text"

        if masks is None or masks.size == 0:
            alpha_boxes = boxes or _alpha_boxes(coarse_alpha, max_boxes=6)
            meta["sam3_box_prompts"] = len(alpha_boxes)
            masks = self._predict_with_boxes(rgb, alpha_boxes)
            source = "box"

        if masks is None or masks.size == 0:
            meta["sam3_status"] = "skipped_no_masks"
            if self._last_error:
                meta["sam3_error"] = self._last_error[:240]
            return coarse_alpha, meta

        sam_mask = _union_masks(masks, (h, w))
        refined, fuse_meta = _fuse_sam3_mask(coarse_alpha, sam_mask)
        meta.update(fuse_meta)
        meta["sam3_source"] = source
        return refined, meta

    def _predict_with_text(self, image: Image.Image, prompt: str) -> np.ndarray | None:
        if self._processor is None:
            return None
        try:
            with self._autocast():
                state = self._processor.set_image(image)
                output = self._processor.set_text_prompt(state=state, prompt=prompt)
        except TypeError:
            try:
                with self._autocast():
                    state = self._processor.set_image(image=image)
                    output = self._processor.set_text_prompt(state=state, prompt=prompt)
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                return None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return None
        return _extract_masks(output, self._confidence_threshold)

    def _predict_with_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
    ) -> np.ndarray | None:
        if self._processor is None or not boxes:
            return None
        try:
            with self._autocast():
                state = self._processor.set_image(image)
        except TypeError:
            with self._autocast():
                state = self._processor.set_image(image=image)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return None

        w, h = image.size
        output: Any = None
        for box in boxes[:6]:
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) * 0.5) / max(1, w)
            cy = ((y1 + y2) * 0.5) / max(1, h)
            bw = max(1.0, x2 - x1) / max(1, w)
            bh = max(1.0, y2 - y1) / max(1, h)
            norm_box = [float(cx), float(cy), float(bw), float(bh)]
            try:
                with self._autocast():
                    output = self._processor.add_geometric_prompt(
                        state=state,
                        box=norm_box,
                        label=True,
                    )
                if isinstance(output, dict) and "state" in output:
                    state = output["state"]
            except TypeError:
                try:
                    with self._autocast():
                        output = self._processor.add_geometric_prompt(
                            state,
                            box=norm_box,
                            label=True,
                        )
                except Exception as exc:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    continue
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                continue

        return _extract_masks(output, self._confidence_threshold)

    def _autocast(self):
        if self._device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def unload(self) -> None:
        del self._model, self._processor
        self._model = self._processor = None
        self._loaded = False
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()


def _extract_masks(output: Any, confidence_threshold: float) -> np.ndarray | None:
    if output is None:
        return None
    if not isinstance(output, dict):
        output = getattr(output, "__dict__", {})

    masks = output.get("masks")
    if masks is None:
        masks = output.get("pred_masks")
    if masks is None:
        return None

    scores = output.get("scores")
    if scores is None:
        scores = output.get("object_scores")
    masks_np = _to_numpy(masks)
    if masks_np is None:
        return None

    masks_np = np.asarray(masks_np)
    while masks_np.ndim > 3 and 1 in masks_np.shape:
        masks_np = np.squeeze(masks_np, axis=int(np.where(np.asarray(masks_np.shape) == 1)[0][0]))
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    if masks_np.ndim != 3:
        return None

    scores_np = _to_numpy(scores)
    if scores_np is not None:
        scores_np = np.asarray(scores_np).reshape(-1)
        if scores_np.size == masks_np.shape[0]:
            keep = scores_np >= confidence_threshold
            if keep.any():
                masks_np = masks_np[keep]

    if masks_np.size == 0:
        return None
    if masks_np.dtype != bool:
        if float(np.nanmax(masks_np)) > 1.0 or float(np.nanmin(masks_np)) < 0.0:
            masks_np = 1.0 / (1.0 + np.exp(-masks_np))
        masks_np = masks_np > 0.5
    return masks_np.astype(bool)


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def _union_masks(masks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    union = np.zeros((h, w), dtype=bool)
    for mask in masks:
        m = np.asarray(mask).astype(bool)
        if m.shape != (h, w):
            pil = Image.fromarray((m.astype(np.uint8) * 255), mode="L")
            pil = pil.resize((w, h), Image.NEAREST)
            m = np.asarray(pil) > 127
        union |= m
    return union


def _fuse_sam3_mask(alpha: np.ndarray, sam_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    coarse = alpha > 0.20
    if not coarse.any() or not sam_mask.any():
        return alpha, {"sam3_status": "skipped_empty_mask"}

    sam_mask = binary_fill_holes(sam_mask)
    coarse_area = float(coarse.mean())
    sam_area = float(sam_mask.mean())
    area_ratio = sam_area / max(coarse_area, 1e-6)
    inter = float((coarse & sam_mask).sum())
    union = float((coarse | sam_mask).sum())
    iou = inter / max(union, 1.0)

    meta = {
        "sam3_area_ratio": round(area_ratio, 4),
        "sam3_iou_with_alpha": round(iou, 4),
    }
    if area_ratio < 0.12 or area_ratio > 3.25 or iou < 0.16:
        meta["sam3_status"] = "rejected_geometry"
        return alpha, meta

    sam_dil = binary_dilation(sam_mask, structure=np.ones((3, 3), dtype=bool), iterations=2)
    sam_core = binary_erosion(sam_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    # Deep interior of a confident SAM segmentation is solid foreground. The
    # silhouette / soft-edge band (hair, fur) lives within a few px of the mask
    # boundary, so an eroded interior never touches it.
    sam_interior = binary_erosion(sam_mask, structure=np.ones((3, 3), dtype=bool), iterations=4)

    refined = alpha.copy()
    remove = (alpha > 0.025) & (alpha < 0.94) & ~sam_dil
    refined[remove] = 0.0

    fill = sam_core & (alpha > 0.08) & (alpha < 0.70)
    refined[fill] = np.maximum(refined[fill], 0.76)

    # Solidify the confident interior regardless of how low the base alpha is.
    # This removes sub-threshold salt speckle (alpha < 0.08) *inside* the body —
    # the dark-suit-on-dark-background failure — which the fill above skips at the
    # silhouette. Edges are protected by the 4-px erosion.
    interior_speckle = sam_interior & (refined < 0.94)
    refined[sam_interior] = np.maximum(refined[sam_interior], 0.985)

    refined = np.where(refined < 0.025, 0.0, refined)

    meta["sam3_status"] = "applied"
    meta["sam3_removed_ratio"] = round(float(remove.mean()), 5)
    meta["sam3_filled_ratio"] = round(float(fill.mean()), 5)
    meta["sam3_interior_solidified_ratio"] = round(float(interior_speckle.mean()), 5)

    _debug_dump_fuse(alpha, sam_mask, sam_interior, refined)
    return np.clip(refined, 0.0, 1.0).astype(np.float32), meta


def _debug_dump_fuse(alpha, sam_mask, sam_interior, refined) -> None:
    """Save intermediate masks as PNGs when BACKGROUNDER_DEBUG_DIR is set."""
    import os
    out = os.environ.get("BACKGROUNDER_DEBUG_DIR")
    if not out:
        return
    try:
        os.makedirs(out, exist_ok=True)
        def _save(arr, name):
            a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
            Image.fromarray((a * 255).astype(np.uint8), mode="L").save(os.path.join(out, name))
        _save(alpha, "01_alpha_in.png")
        _save(sam_mask.astype(np.float32), "02_sam_mask.png")
        _save(sam_interior.astype(np.float32), "03_sam_interior.png")
        _save(refined, "04_refined.png")
    except Exception:
        pass


def _alpha_boxes(alpha: np.ndarray, max_boxes: int = 6) -> list[list[float]]:
    mask = alpha > 0.35
    if not mask.any():
        return []
    labeled, count = label(mask)
    objects = find_objects(labeled)
    boxes: list[tuple[int, list[float]]] = []
    h, w = alpha.shape
    pad = max(4, min(h, w) // 80)
    for idx, slc in enumerate(objects, 1):
        if slc is None:
            continue
        ys, xs = slc
        area = int((labeled[slc] == idx).sum())
        if area < max(32, int(alpha.size * 0.001)):
            continue
        x1 = max(0, xs.start - pad)
        y1 = max(0, ys.start - pad)
        x2 = min(w, xs.stop + pad)
        y2 = min(h, ys.stop + pad)
        boxes.append((area, [float(x1), float(y1), float(x2), float(y2)]))
    boxes.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in boxes[:max_boxes]]
