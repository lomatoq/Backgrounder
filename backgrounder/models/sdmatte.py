from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from backgrounder.utils import resize_alpha_to


PromptMode = Literal["bbox", "mask", "trimap", "point"]

# Pre-trained checkpoints from 1038lab/SDMatte (safetensors, MIT license).
SDMATTE_CHECKPOINTS = {
    "sdmatte": "https://huggingface.co/1038lab/SDMatte/resolve/main/SDMatte.safetensors",
    "sdmatte_plus": "https://huggingface.co/1038lab/SDMatte/resolve/main/SDMatte_plus.safetensors",
}

# SD 2.1 base architecture configs (no weights) from Manojb/stable-diffusion-2-1-base.
# We only need the JSON configs since the safetensors checkpoint carries the weights.
SD21_CONFIG_FILES = {
    "model_index.json",
    "text_encoder/config.json",
    "vae/config.json",
    "unet/config.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/merges.txt",
    "tokenizer/vocab.json",
    "tokenizer/special_tokens_map.json",
    "feature_extractor/preprocessor_config.json",
}
SD21_CONFIG_BASE_URL = "https://huggingface.co/Manojb/stable-diffusion-2-1-base/resolve/main"


class SDMatteRefiner:
    """
    SDMatte diffusion-based matting refiner.

    Uses bundled modeling code (backgrounder.sdmatte_vendor) so no external
    SDMatte repo or detectron2 install is needed. Loads weights from the
    safetensors checkpoint published by 1038lab on Hugging Face.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        device: str = "cuda",
        variant: str = "sdmatte",
        prompt_mode: PromptMode = "trimap",
        input_size: int = 1024,
        is_transparent: bool = False,
    ) -> None:
        if device != "cuda":
            raise RuntimeError(
                "SDMatte hardcodes CUDA in its forward(). Use device='cuda' or keep use_sdmatte=False."
            )

        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.device = device
        self.variant = variant.lower()
        if self.variant not in SDMATTE_CHECKPOINTS:
            raise ValueError(f"Unknown SDMatte variant: {self.variant}. Choose from {list(SDMATTE_CHECKPOINTS)}.")
        self.prompt_mode = prompt_mode
        self.input_size = input_size
        self.is_transparent = is_transparent
        self._model = None
        self._loaded = False

    @property
    def _aux_input_name(self) -> str:
        return {
            "bbox": "bbox_mask",
            "mask": "mask",
            "trimap": "trimap",
            "point": "point_mask",
        }.get(self.prompt_mode, "trimap")

    @property
    def _coord_name(self) -> str:
        return {
            "bbox": "bbox_coords",
            "mask": "mask_coords",
            "trimap": "trimap_coords",
            "point": "point_coords",
        }[self.prompt_mode]

    def load(self) -> "SDMatteRefiner":
        if self._loaded:
            return self

        sd21_dir = self._ensure_sd21_configs()
        ckpt_path = self._ensure_checkpoint()

        from backgrounder.sdmatte_vendor.modeling.SDMatte import SDMatte as SDMatteCore

        self._model = SDMatteCore(
            pretrained_model_name_or_path=str(sd21_dir),
            load_weight=False,
            use_aux_input=True,
            aux_input=self._aux_input_name,
            aux_input_list=["point_mask", "bbox_mask", "mask", "trimap"],
            attn_mask_aux_input=["point_mask", "bbox_mask", "mask", "trimap"],
            use_encoder_hidden_states=True,
            use_attention_mask=True,
            add_noise=False,
        )

        state_dict = load_file(str(ckpt_path), device="cpu")
        # Some checkpoints wrap the state_dict in {'state_dict': ...} or similar.
        for key in ("state_dict", "model_state_dict", "model", "module"):
            inner = state_dict.get(key)
            if isinstance(inner, dict):
                state_dict = inner
                break
        self._model.load_state_dict(state_dict, strict=False)
        self._model.eval()
        # fp16 halves VRAM usage and ~2× inference speed on CUDA.
        self._model.half().to(self.device)

        self._loaded = True
        return self

    def _ensure_sd21_configs(self) -> Path:
        sd21_dir = self.cache_dir / "stable-diffusion-2-1-base"
        sd21_dir.mkdir(parents=True, exist_ok=True)

        for rel in SD21_CONFIG_FILES:
            target = sd21_dir / rel
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"{SD21_CONFIG_BASE_URL}/{rel}"
            print(f"[SDMatte] Downloading config: {rel}")
            self._download(url, target)
        return sd21_dir

    def _ensure_checkpoint(self) -> Path:
        url = SDMATTE_CHECKPOINTS[self.variant]
        ckpt_path = self.cache_dir / f"{self.variant}.safetensors"
        if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
            return ckpt_path
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[SDMatte] Downloading {self.variant} weights (~5 GB) from {url}")
        self._download(url, ckpt_path)
        return ckpt_path

    @staticmethod
    def _download(url: str, target: Path) -> None:
        from tqdm import tqdm
        tmp = target.with_suffix(target.suffix + ".tmp")
        with urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            bar = tqdm(
                total=total or None,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc=target.name, leave=True,
            )
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
            bar.close()
        os.replace(tmp, target)

    def refine(
        self,
        image: Image.Image,
        coarse_alpha: np.ndarray,
        trimap: np.ndarray,
    ) -> np.ndarray:
        if not self._loaded:
            self.load()

        orig_w, orig_h = image.size
        size = (self.input_size, self.input_size)

        # Image: resize, normalize to [-1, 1].
        rgb = image.convert("RGB").resize(size, Image.BILINEAR)
        img_np = np.asarray(rgb).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        img_t = (img_t - 0.5) / 0.5
        img_t = img_t.to(self.device, dtype=torch.float32)

        # Aux input prompt: scale to [-1, 1].
        prompt_t, coords_t, point_coords_t = self._prepare_prompt(coarse_alpha, trimap, size)
        prompt_t = prompt_t.to(self.device)
        coords_t = coords_t.to(self.device)
        if point_coords_t is not None:
            point_coords_t = point_coords_t.to(self.device)

        is_trans = torch.tensor([1 if self.is_transparent else 0], dtype=torch.long, device=self.device)
        data = {
            "image": img_t,
            "is_trans": is_trans,
            "caption": [""],
            self._aux_input_name: prompt_t,
            self._coord_name: coords_t,
        }
        if self.prompt_mode == "point" and point_coords_t is not None:
            data["point_coords"] = point_coords_t

        # Model is fp16; cast all tensor inputs accordingly.
        data = {
            k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in data.items()
        }
        with torch.no_grad():
            pred = self._model(data)

        # pred is [B, 1, H, W] in [0, 1]
        alpha_pred = pred.squeeze().detach().float().cpu().numpy()
        alpha_full = resize_alpha_to(alpha_pred, (orig_w, orig_h))

        # Composite: refine only the trimap unknown band; keep coarse fg/bg intact.
        unknown = (trimap == 128).astype(np.float32)
        result = coarse_alpha * (1.0 - unknown) + alpha_full * unknown
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def _prepare_prompt(
        self,
        coarse_alpha: np.ndarray,
        trimap: np.ndarray,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h_in, w_in = size
        alpha_small = resize_alpha_to(coarse_alpha, size)
        trimap_small = np.asarray(
            Image.fromarray(trimap, mode="L").resize(size, Image.NEAREST)
        ).astype(np.float32)
        # Trimap encoding: 0 = bg, 0.5 = unknown, 1 = fg
        trimap_norm = np.where(trimap_small == 128, 0.5, np.where(trimap_small >= 200, 1.0, 0.0)).astype(np.float32)

        if self.prompt_mode == "trimap":
            prompt = trimap_norm
            coords = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
            point_coords = None
        elif self.prompt_mode == "mask":
            prompt = (alpha_small > 0.5).astype(np.float32)
            coords = self._mask_bbox(prompt)
            point_coords = None
        elif self.prompt_mode == "bbox":
            prompt = np.zeros_like(alpha_small, dtype=np.float32)
            x1, y1, x2, y2 = self._mask_bbox(alpha_small > 0.05)[0]
            prompt[int(y1 * h_in) : max(int(y2 * h_in), int(y1 * h_in) + 1),
                   int(x1 * w_in) : max(int(x2 * w_in), int(x1 * w_in) + 1)] = 1.0
            coords = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            point_coords = None
        else:  # point
            prompt, pc = self._point_prompt(alpha_small)
            coords = np.array([pc[:4]], dtype=np.float32) if pc.size >= 4 else np.array([[0, 0, 1, 1]], dtype=np.float32)
            point_coords = torch.from_numpy(pc).unsqueeze(0).float()

        prompt = prompt * 2.0 - 1.0
        prompt_t = torch.from_numpy(prompt).unsqueeze(0).unsqueeze(0).float()
        coords_t = torch.from_numpy(coords).float()
        return prompt_t, coords_t, point_coords

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> np.ndarray:
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        h, w = mask.shape
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        return np.array([[x1 / w, y1 / h, (x2 + 1) / w, (y2 + 1) / h]], dtype=np.float32)

    @staticmethod
    def _point_prompt(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from scipy.ndimage import gaussian_filter
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
        prompt_smooth = gaussian_filter(prompt, sigma=20.0)
        if prompt_smooth.max() > 0:
            prompt = prompt_smooth / prompt_smooth.max()
        coords.extend([0.0] * (20 - len(coords)))
        return prompt.astype(np.float32), np.asarray(coords[:20], dtype=np.float32)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
