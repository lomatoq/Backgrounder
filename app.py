"""
Quick Gradio demo for the Backgrounder pipeline.

  pip install gradio
  python app.py
"""
from __future__ import annotations
import json
import os

import numpy as np
from PIL import Image

# ── lazy pipeline singleton ────────────────────────────────────────────────
_pipeline = None
_pipeline_cfg: dict = {}


def _sdmatte_ui_defaults() -> dict:
    cache_dir = os.environ.get("BACKGROUNDER_SDMATTE_CACHE", "~/.cache/backgrounder/sdmatte")
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if cuda_available:
        status = "SDMatte ready (CUDA detected). Weights auto-download on first use (~5 GB)."
    else:
        status = "SDMatte requires CUDA — disabled on this device."
    return {
        "enabled": False,           # off by default — user opts in
        "cache_dir": cache_dir,
        "status": status,
    }


def _get_pipeline(
    device: str,
    segmenters: str,
    use_depth: bool,
    use_classifier: bool,
    use_closed_form: bool,
    use_uncertainty_sharpen: bool,
    use_sdmatte: bool,
    sdmatte_cache_dir: str,
    sdmatte_variant: str,
    sdmatte_prompt_mode: str,
    use_sam2: bool,
    use_owlv2: bool,
):
    global _pipeline, _pipeline_cfg

    cfg_key = (
        device, segmenters, use_depth, use_classifier,
        use_closed_form, use_uncertainty_sharpen,
        use_sdmatte, sdmatte_cache_dir, sdmatte_variant, sdmatte_prompt_mode,
        use_sam2, use_owlv2,
    )
    if _pipeline is not None and _pipeline_cfg.get("key") == cfg_key:
        return _pipeline

    from backgrounder import BackgroundRemovalPipeline, PipelineConfig, SegmenterID, Device

    seg_ids = [SegmenterID(s.strip()) for s in segmenters.split(",")]
    config = PipelineConfig(
        segmenters=seg_ids,
        device=Device(device),
        use_depth=use_depth,
        use_classifier=use_classifier,
        use_closed_form_refine=use_closed_form,
        use_uncertainty_sharpen=use_uncertainty_sharpen,
        use_sdmatte=use_sdmatte,
        sdmatte_cache_dir=sdmatte_cache_dir or "~/.cache/backgrounder/sdmatte",
        sdmatte_variant=sdmatte_variant,
        sdmatte_prompt_mode=sdmatte_prompt_mode,
        use_sam2=use_sam2,
        use_owlv2=use_owlv2 and use_sam2,
        fp16=(device == "cuda"),
    )
    _pipeline = BackgroundRemovalPipeline(config).load()
    _pipeline_cfg = {"key": cfg_key}
    return _pipeline


# ── inference function ─────────────────────────────────────────────────────

def remove_background(
    image: Image.Image,
    device: str,
    segmenters: str,
    use_depth: bool,
    use_classifier: bool,
    use_closed_form: bool,
    use_uncertainty_sharpen: bool,
    use_sdmatte: bool,
    sdmatte_cache_dir: str,
    sdmatte_variant: str,
    sdmatte_prompt_mode: str,
    use_sam2: bool,
    use_owlv2: bool,
    subject_override: str,
    checkerboard: bool,
) -> tuple[Image.Image, Image.Image, str]:
    """
    Returns: (result_rgba, preview_on_checker, info_text)
    """
    if image is None:
        return None, None, "Upload an image first."

    pipeline = _get_pipeline(
        device, segmenters, use_depth, use_classifier,
        use_closed_form, use_uncertainty_sharpen,
        use_sdmatte, sdmatte_cache_dir, sdmatte_variant, sdmatte_prompt_mode,
        use_sam2, use_owlv2,
    )

    # Subject type override
    pipeline.config.subject_type_override = subject_override if subject_override != "auto" else None

    result = pipeline.process(image)

    # Compose over checkerboard for visual preview
    preview = _compose_checker(result.rgba) if checkerboard else result.rgba

    info = {
        "quality_score": round(result.quality_score, 3),
        "subject_type": result.metadata.get("subject_type", "n/a"),
        "expert_used": result.metadata.get("expert", "n/a"),
        "sdmatte_used": result.metadata.get("sdmatte_used", False),
        "sam2_used": result.metadata.get("sam2_used", False),
        "timings_ms": result.metadata.get("timings_ms", {}),
        "quality_detail": result.metadata.get("quality", ""),
    }
    return result.rgba, preview, json.dumps(info, indent=2)


def _compose_checker(rgba: Image.Image, square: int = 16) -> Image.Image:
    """Compose RGBA over a grey checkerboard background (vectorised)."""
    w, h = rgba.size
    yy, xx = np.indices((h, w))
    light = ((xx // square) + (yy // square)) % 2 == 0
    bg = np.where(light[..., None], 220, 180).astype(np.uint8)
    bg = np.broadcast_to(bg, (h, w, 3)).copy()
    checker = Image.fromarray(bg, mode="RGB")
    checker.paste(rgba, mask=rgba.split()[3])
    return checker


# ── build UI ───────────────────────────────────────────────────────────────

def build_ui():
    import gradio as gr
    sdmatte_defaults = _sdmatte_ui_defaults()

    subject_choices = [
        "auto", "portrait", "animal_fur", "product",
        "plant_thin", "transparent", "vehicle", "anime",
        "complex_multi", "generic",
    ]

    with gr.Blocks(title="Backgrounder", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## Backgrounder — SOTA background removal\n"
            "BiRefNet HR · BEN2 · Depth Anything V2 · CLIP classifier · "
            "SAM 2.1 · OWLv2"
        )

        with gr.Row():
            # ── left column: input + settings ──
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Input image")

                with gr.Accordion("Settings", open=False):
                    device = gr.Radio(
                        ["auto", "cuda", "mps", "cpu"],
                        value="auto", label="Device",
                    )
                    segmenters = gr.Textbox(
                        value="birefnet_hr,ben2",
                        label="Segmenters (comma-separated)",
                        info="birefnet_hr · ben2 · inspyrenet",
                    )
                    use_depth = gr.Checkbox(value=True, label="Depth Anything V2-Small")
                    use_classifier = gr.Checkbox(value=True, label="CLIP subject classifier")
                    use_closed_form = gr.Checkbox(value=True, label="Closed-form matting refine")
                    use_uncertainty_sharpen = gr.Checkbox(value=True, label="Uncertainty-gated sharpening")
                    subject_override = gr.Dropdown(
                        subject_choices, value="auto", label="Subject type override",
                    )

                    gr.Markdown("**Phase 3** — activate for harder images")
                    gr.Markdown(sdmatte_defaults["status"])
                    use_sdmatte = gr.Checkbox(
                        value=sdmatte_defaults["enabled"],
                        label="SDMatte diffusion refiner (auto-downloads ~5 GB on first use)",
                        info="Triggers when quality < 0.72. Requires CUDA + diffusers.",
                    )
                    sdmatte_cache_dir = gr.Textbox(
                        value=sdmatte_defaults["cache_dir"],
                        label="SDMatte cache dir (weights + SD2.1 configs)",
                        placeholder="~/.cache/backgrounder/sdmatte",
                    )
                    sdmatte_variant = gr.Radio(
                        ["sdmatte", "sdmatte_plus"], value="sdmatte", label="SDMatte variant",
                    )
                    sdmatte_prompt_mode = gr.Radio(
                        ["trimap", "bbox", "mask", "point"],
                        value="trimap", label="SDMatte prompt",
                    )
                    use_sam2 = gr.Checkbox(
                        value=False,
                        label="SAM 2.1 refiner  (triggers when quality < 0.60 or complex_multi)",
                        info="Requires transformers ≥ 4.49 · downloads ~400 MB on first use",
                    )
                    use_owlv2 = gr.Checkbox(
                        value=False,
                        label="OWLv2 localizer  (box prompts for SAM 2.1)",
                        info="Only active when SAM 2.1 is enabled · ~300 MB on first use",
                    )

                    checkerboard = gr.Checkbox(value=True, label="Preview on checkerboard")

                btn = gr.Button("Remove background", variant="primary")

            # ── right column: outputs ──
            with gr.Column(scale=1):
                out_rgba = gr.Image(type="pil", label="Result (RGBA)", image_mode="RGBA")
                out_preview = gr.Image(type="pil", label="Preview on checker")
                out_info = gr.Code(label="Diagnostics (JSON)", language="json")

        _inputs = [
            inp, device, segmenters, use_depth, use_classifier,
            use_closed_form, use_uncertainty_sharpen,
            use_sdmatte, sdmatte_cache_dir, sdmatte_variant, sdmatte_prompt_mode,
            use_sam2, use_owlv2, subject_override, checkerboard,
        ]

        btn.click(fn=remove_background, inputs=_inputs,
                  outputs=[out_rgba, out_preview, out_info])
        inp.upload(fn=remove_background, inputs=_inputs,
                   outputs=[out_rgba, out_preview, out_info])

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share, inbrowser=True)
