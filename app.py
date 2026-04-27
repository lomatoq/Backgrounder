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


def _get_pipeline(
    device: str,
    segmenters: str,
    use_depth: bool,
    use_classifier: bool,
    use_sam2: bool,
    use_owlv2: bool,
):
    global _pipeline, _pipeline_cfg

    cfg_key = (device, segmenters, use_depth, use_classifier, use_sam2, use_owlv2)
    if _pipeline is not None and _pipeline_cfg.get("key") == cfg_key:
        return _pipeline

    from backgrounder import BackgroundRemovalPipeline, PipelineConfig, SegmenterID, Device

    seg_ids = [SegmenterID(s.strip()) for s in segmenters.split(",")]
    config = PipelineConfig(
        segmenters=seg_ids,
        device=Device(device),
        use_depth=use_depth,
        use_classifier=use_classifier,
        use_sam2=use_sam2,
        use_owlv2=use_owlv2 and use_sam2,  # OWLv2 only meaningful with SAM2
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

    pipeline = _get_pipeline(device, segmenters, use_depth, use_classifier, use_sam2, use_owlv2)

    # Subject type override
    pipeline.config.subject_type_override = subject_override if subject_override != "auto" else None

    result = pipeline.process(image)

    # Compose over checkerboard for visual preview
    preview = _compose_checker(result.rgba) if checkerboard else result.rgba

    info = {
        "quality_score": round(result.quality_score, 3),
        "subject_type": result.metadata.get("subject_type", "n/a"),
        "expert_used": result.metadata.get("expert", "n/a"),
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

    subject_choices = [
        "auto", "portrait", "animal_fur", "product",
        "plant_thin", "transparent", "vehicle", "anime",
        "complex_multi", "generic",
    ]

    with gr.Blocks(title="Backgrounder") as demo:
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
                    subject_override = gr.Dropdown(
                        subject_choices, value="auto", label="Subject type override",
                    )

                    gr.Markdown("**Phase 3** — activate for harder images")
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
    import gradio as gr
    ui.launch(server_port=args.port, share=args.share, inbrowser=True, theme=gr.themes.Soft())
