"""
Quick Gradio demo for the Backgrounder pipeline.

  pip install gradio
  python app.py
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ── lazy pipeline singleton ────────────────────────────────────────────────
_pipeline = None
_pipeline_cfg: dict = {}


def _get_pipeline(device: str, segmenters: str, use_depth: bool, use_classifier: bool):
    global _pipeline, _pipeline_cfg

    cfg_key = (device, segmenters, use_depth, use_classifier)
    if _pipeline is not None and _pipeline_cfg.get("key") == cfg_key:
        return _pipeline

    from backgrounder import BackgroundRemovalPipeline, PipelineConfig, SegmenterID, Device

    seg_ids = [SegmenterID(s.strip()) for s in segmenters.split(",")]
    config = PipelineConfig(
        segmenters=seg_ids,
        device=Device(device),
        use_depth=use_depth,
        use_classifier=use_classifier,
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
    subject_override: str,
    checkerboard: bool,
) -> tuple[Image.Image, Image.Image, str]:
    """
    Returns: (result_rgba, preview_on_checker, info_text)
    """
    if image is None:
        return None, None, "Upload an image first."

    pipeline = _get_pipeline(device, segmenters, use_depth, use_classifier)

    # Subject type override
    pipeline.config.subject_type_override = subject_override if subject_override != "auto" else None

    result = pipeline.process(image)

    # Compose over checkerboard for visual preview
    preview = _compose_checker(result.rgba) if checkerboard else result.rgba

    info = {
        "quality_score": round(result.quality_score, 3),
        "subject_type": result.metadata.get("subject_type", "n/a"),
        "expert_used": result.metadata.get("expert", "n/a"),
        "timings_ms": result.metadata.get("timings_ms", {}),
        "quality_detail": result.metadata.get("quality", ""),
    }
    return result.rgba, preview, json.dumps(info, indent=2)


def _compose_checker(rgba: Image.Image, square: int = 16) -> Image.Image:
    """Compose RGBA over a grey checkerboard background."""
    w, h = rgba.size
    checker = Image.new("RGB", (w, h))
    pix = checker.load()
    for y in range(h):
        for x in range(w):
            light = ((x // square) + (y // square)) % 2 == 0
            v = 220 if light else 180
            pix[x, y] = (v, v, v)
    checker.paste(rgba, mask=rgba.split()[3])
    return checker


# ── build UI ───────────────────────────────────────────────────────────────

def build_ui():
    import gradio as gr

    subject_choices = [
        "auto", "portrait", "animal_fur", "product",
        "plant_thin", "transparent", "vehicle", "anime",
        "complex_multi", "generic",
    ]

    with gr.Blocks(title="Backgrounder", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Backgrounder — SOTA background removal\n"
                    "BiRefNet HR + BEN2 + Depth Anything V2 + CLIP classifier")

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
                    subject_override = gr.Dropdown(
                        subject_choices, value="auto", label="Subject type override",
                    )
                    checkerboard = gr.Checkbox(value=True, label="Preview on checkerboard")

                btn = gr.Button("Remove background", variant="primary")

            # ── right column: outputs ──
            with gr.Column(scale=1):
                out_rgba = gr.Image(type="pil", label="Result (RGBA)", image_mode="RGBA")
                out_preview = gr.Image(type="pil", label="Preview on checker")
                out_info = gr.Code(label="Diagnostics (JSON)", language="json")

        btn.click(
            fn=remove_background,
            inputs=[inp, device, segmenters, use_depth, use_classifier,
                    subject_override, checkerboard],
            outputs=[out_rgba, out_preview, out_info],
        )

        # Also trigger on image upload for quick feedback.
        inp.upload(
            fn=remove_background,
            inputs=[inp, device, segmenters, use_depth, use_classifier,
                    subject_override, checkerboard],
            outputs=[out_rgba, out_preview, out_info],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share, inbrowser=True)
