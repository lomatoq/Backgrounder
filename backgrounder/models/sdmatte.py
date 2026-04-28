from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from backgrounder.utils import resize_alpha_to


PromptMode = Literal["bbox", "mask", "trimap", "point"]


class SDMatteRefiner:
    """
    Adapter for vivoCameraResearch/SDMatte or LiteSDMatte.

    The official SDMatte code is a research repo, not a packaged library. This
    adapter loads the repo from a local checkout and feeds our coarse alpha as
    a visual prompt. It refines only the requested unknown band so the pipeline
    can keep deterministic coarse foreground/background decisions.
    """

    _MODEL_REPOS = {
        "sdmatte": "LongfeiHuang/SDMatte",
        "lite": "LongfeiHuang/LiteSDMatte",
    }

    def __init__(
        self,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        *,
        variant: str = "lite",
        device: str = "cuda",
        pretrained_model_name_or_path: Optional[str] = None,
        prompt_mode: PromptMode = "bbox",
        input_size: int = 1024,
        is_transparent: bool = False,
    ) -> None:
        if device != "cuda":
            raise RuntimeError(
                "SDMatte's official implementation hardcodes CUDA in forward(). "
                "Use device='cuda' or keep use_sdmatte=False."
            )

        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.variant = variant.lower()
        self.device = device
        self.pretrained_model_name_or_path = (
            pretrained_model_name_or_path or self._MODEL_REPOS.get(self.variant, self._MODEL_REPOS["lite"])
        )
        self.prompt_mode = prompt_mode
        self.input_size = input_size
        self.is_transparent = is_transparent
        self._model = None
        self._loaded = False

    def load(self) -> "SDMatteRefiner":
        if self._loaded:
            return self

        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"SDMatte repo not found: {self.repo_path}. "
                "Clone https://github.com/vivoCameraResearch/SDMatte and set sdmatte_repo_path."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SDMatte checkpoint not found: {self.checkpoint_path}. "
                "Download LiteSDMatte.pth or SDMatte.pth and set sdmatte_checkpoint_path."
            )

        sys.path.insert(0, str(self.repo_path))
        try:
            class_name = "LiteSDMatte" if self.variant in {"lite", "litesdmatte"} else "SDMatte"
            module = importlib.import_module(f"modeling.{class_name}")
            ModelClass = getattr(module, class_name)
        except Exception as exc:
            raise ImportError(
                "Could not import SDMatte model classes from the local repo. "
                "Make sure sdmatte_repo_path points to vivoCameraResearch/SDMatte."
            ) from exc

        # load_weight=True  → download backbone from HuggingFace Hub (model_name_or_path is a HF repo ID)
        # load_weight=False → read architecture config from local directory (must have subfolders
        #                     text_encoder/, vae/, unet/, tokenizer/ with config.json files)
        model_path = self.pretrained_model_name_or_path
        load_weight = not Path(model_path).is_dir()
        if load_weight and not Path(model_path).exists():
            # HF model ID — will download automatically
            pass
        elif not load_weight:
            # Validate expected subdirs exist
            for sub in ("vae", "unet", "tokenizer"):
                if not (Path(model_path) / sub).exists():
                    raise FileNotFoundError(
                        f"SDMatte model dir missing '{sub}/' subfolder: {model_path}. "
                        "Run: huggingface-cli download LongfeiHuang/LiteSDMatte --local-dir <path>"
                    )

        self._model = ModelClass(
            pretrained_model_name_or_path=model_path,
            load_weight=load_weight,
            conv_scale=3,
            num_inference_steps=1,
            aux_input=self._aux_input_name,
            add_noise=False,
            use_dis_loss=True,
            use_aux_input=True,
            use_coor_input=True,
            use_attention_mask=True,
            residual_connection=False,
            use_encoder_hidden_states=True,
            use_attention_mask_list=[True, True, True],
            use_encoder_hidden_states_list=[False, True, False],
        ).to(self.device)

        try:
            from detectron2.checkpoint import DetectionCheckpointer
        except ImportError as exc:
            raise ImportError(
                "SDMatte checkpoint loading requires detectron2. "
                "Install it in the same environment as the SDMatte repo."
            ) from exc

        DetectionCheckpointer(self._model).load(str(self.checkpoint_path))
        self._model.eval()
        self._loaded = True
        return self

    @property
    def _aux_input_name(self) -> str:
        return {
            "bbox": "bbox_mask",
            "mask": "mask",
            "trimap": "trimap",
            "point": "point_mask",
        }.get(self.prompt_mode, "bbox_mask")

    def refine(
        self,
        image: Image.Image,
        coarse_alpha: np.ndarray,
        trimap: np.ndarray,
    ) -> np.ndarray:
        if not self._loaded:
            self.load()

        orig_wh = image.size
        image_t, prompt_t, coords_t, point_coords_t = self._prepare_inputs(image, coarse_alpha, trimap)
        data = {
            "image": image_t,
            "alpha": torch.zeros((1, 1, self.input_size, self.input_size), dtype=torch.float32),
            "is_trans": torch.tensor([1 if self.is_transparent else 0], dtype=torch.long),
            "caption": [""],
            self._aux_input_name: prompt_t,
            self._coord_name: coords_t,
        }
        if self.prompt_mode == "point":
            data["point_coords"] = point_coords_t

        with torch.inference_mode():
            pred = self._model(data)

        alpha_1024 = pred.squeeze().detach().float().cpu().numpy()
        alpha_full = resize_alpha_to(alpha_1024, orig_wh)

        unknown = (trimap == 128).astype(np.float32)
        result = coarse_alpha * (1.0 - unknown) + alpha_full * unknown
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    @property
    def _coord_name(self) -> str:
        if self.prompt_mode == "point":
            return "point_coords"
        if self.prompt_mode == "mask":
            return "mask_coords"
        if self.prompt_mode == "trimap":
            return "trimap_coords"
        return "bbox_coords"

    def _prepare_inputs(
        self,
        image: Image.Image,
        coarse_alpha: np.ndarray,
        trimap: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        size = (self.input_size, self.input_size)

        rgb = image.convert("RGB").resize(size, Image.BILINEAR)
        image_np = np.asarray(rgb).astype(np.float32) / 255.0
        image_np = image_np * 2.0 - 1.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()

        alpha_small = resize_alpha_to(coarse_alpha, size)
        trimap_small = np.asarray(
            Image.fromarray(trimap, mode="L").resize(size, Image.NEAREST)
        ).astype(np.float32)
        trimap_small[trimap_small == 128] = 0.5
        trimap_small[trimap_small == 255] = 1.0
        trimap_small[trimap_small > 1.0] /= 255.0

        if self.prompt_mode == "trimap":
            prompt = trimap_small
            coords = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        elif self.prompt_mode == "mask":
            prompt = (alpha_small > 0.5).astype(np.float32)
            coords = self._mask_coords(prompt)
        elif self.prompt_mode == "point":
            prompt, point_coords = self._point_prompt(alpha_small)
            coords = np.array([point_coords], dtype=np.float32)
        else:
            prompt = np.zeros_like(alpha_small, dtype=np.float32)
            x1, y1, x2, y2 = self._mask_coords(alpha_small > 0.05)[0]
            h, w = prompt.shape
            prompt[int(y1 * h) : max(int(y2 * h), int(y1 * h) + 1), int(x1 * w) : max(int(x2 * w), int(x1 * w) + 1)] = 1.0
            coords = np.array([[x1, y1, x2, y2]], dtype=np.float32)

        prompt = prompt.astype(np.float32) * 2.0 - 1.0
        prompt_t = torch.from_numpy(prompt).unsqueeze(0).unsqueeze(0).float()
        coords_t = torch.from_numpy(coords).float()
        if self.prompt_mode != "point":
            point_coords_t = torch.zeros((1, 20), dtype=torch.float32)
        else:
            point_coords_t = coords_t
        return image_t, prompt_t, coords_t, point_coords_t

    @staticmethod
    def _mask_coords(mask: np.ndarray) -> np.ndarray:
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        h, w = mask.shape
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        return np.array([[x1 / w, y1 / h, (x2 + 1) / w, (y2 + 1) / h]], dtype=np.float32)

    @staticmethod
    def _point_prompt(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fg = np.argwhere(alpha > 0.75)
        prompt = np.zeros_like(alpha, dtype=np.float32)
        if len(fg) == 0:
            return prompt, np.zeros(20, dtype=np.float32)

        h, w = alpha.shape
        rng = np.random.default_rng(42)
        picks = fg[rng.choice(len(fg), size=min(10, len(fg)), replace=False)]
        coords: list[float] = []
        for y, x in picks:
            prompt[int(y), int(x)] = 1.0
            coords.extend([float(x) / w, float(y) / h])
        prompt_np = gaussian_filter(prompt, sigma=20.0)
        if prompt_np.max() > 0:
            prompt = prompt_np / prompt_np.max()
        coords.extend([0.0] * (20 - len(coords)))
        return prompt.astype(np.float32), np.array(coords[:20], dtype=np.float32)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        torch.cuda.empty_cache()
