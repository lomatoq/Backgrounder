"""
Quick Gradio demo for the Backgrounder pipeline.

  pip install gradio
  python app.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image

# ── lazy pipeline singleton ────────────────────────────────────────────────
_pipeline = None
_pipeline_cfg: dict = {}


PRESETS = {
    "Smart Auto": {
        "segmenters": "birefnet_hr,ben2",
        "use_depth": False,
        "use_classifier": True,
        "use_vitmatte": False,
        "use_closed_form": False,
        "use_uncertainty_sharpen": True,
        "use_solid_background_cleanup": True,
        "use_tta": False,
        "use_sdmatte": False,
        "force_sdmatte": False,
        "use_sam2": False,
        "use_owlv2": False,
    },
    "Fast": {
        "segmenters": "birefnet_hr",
        "use_depth": False,
        "use_classifier": True,
        "use_vitmatte": False,
        "use_closed_form": False,
        "use_uncertainty_sharpen": True,
        "use_solid_background_cleanup": True,
        "use_tta": False,
        "use_sdmatte": False,
        "force_sdmatte": False,
        "use_sam2": False,
        "use_owlv2": False,
    },
    "Max Quality": {
        "segmenters": "birefnet_hr,ben2",
        "use_depth": False,
        "use_classifier": True,
        "use_vitmatte": True,
        "use_closed_form": True,
        "use_uncertainty_sharpen": True,
        "use_solid_background_cleanup": True,
        "use_tta": True,
        "use_sdmatte": True,
        "force_sdmatte": False,
        "use_sam2": True,
        "use_owlv2": True,
    },
}


def _preset_values(name: str) -> list:
    preset = PRESETS.get(name, PRESETS["Smart Auto"])
    return [
        preset["segmenters"],
        preset["use_depth"],
        preset["use_classifier"],
        preset["use_vitmatte"],
        preset["use_closed_form"],
        preset["use_uncertainty_sharpen"],
        preset["use_solid_background_cleanup"],
        preset["use_tta"],
        preset["use_sdmatte"],
        preset["force_sdmatte"],
        preset["use_sam2"],
        preset["use_owlv2"],
    ]


def _restart_soon(delay_s: float = 1.5) -> None:
    def _restart() -> None:
        time.sleep(delay_s)
        os.execv(sys.executable, [sys.executable, *sys.argv])

    threading.Thread(target=_restart, daemon=True).start()


def update_from_git() -> str:
    """Pull the latest code without restarting the current process."""
    root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        proc = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Update timed out after 120s. The app was not restarted."
    except Exception as exc:
        return f"Update failed before git could run: {exc}"

    output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    if proc.returncode != 0:
        return (
            "Update failed. The app was not restarted.\n\n"
            f"{output or 'git pull returned a non-zero exit code.'}"
        )

    return "Update complete. Click Restart app to load the new code.\n\n" + (
        output or "Already up to date."
    )


def restart_app() -> str:
    """Restart this Gradio app process."""
    if os.environ.get("BACKGROUNDER_NO_AUTO_RESTART") == "1":
        return "Restart is disabled by BACKGROUNDER_NO_AUTO_RESTART=1."

    _restart_soon()
    return "Restarting the app now; the browser tab will reconnect shortly."


def _sdmatte_ui_defaults() -> dict:
    cache_dir = os.environ.get("BACKGROUNDER_SDMATTE_CACHE", "~/.cache/backgrounder/sdmatte")
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if cuda_available:
        status = "SDMatte available (CUDA detected), but off by default. Enable it only for difficult mattes."
    else:
        status = "SDMatte requires CUDA — disabled on this device."
    return {
        "enabled": False,
        "force": False,
        "cache_dir": cache_dir,
        "status": status,
    }


def _get_pipeline(
    device: str,
    segmenters: str,
    use_depth: bool,
    use_classifier: bool,
    use_vitmatte: bool,
    use_closed_form: bool,
    use_uncertainty_sharpen: bool,
    use_solid_background_cleanup: bool,
    use_tta: bool,
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
        use_vitmatte, use_closed_form, use_uncertainty_sharpen,
        use_solid_background_cleanup, use_tta,
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
        use_vitmatte=use_vitmatte,
        vitmatte_allow_nc=use_vitmatte,
        use_closed_form_refine=use_closed_form,
        use_uncertainty_sharpen=use_uncertainty_sharpen,
        use_solid_background_cleanup=use_solid_background_cleanup,
        use_tta=use_tta,
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
    mode: str,
    device: str,
    segmenters: str,
    use_depth: bool,
    use_classifier: bool,
    use_vitmatte: bool,
    use_closed_form: bool,
    use_uncertainty_sharpen: bool,
    use_solid_background_cleanup: bool,
    use_tta: bool,
    use_sdmatte: bool,
    force_sdmatte: bool,
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

    if image.mode == "RGBA":
        rgba_arr = np.array(image)
        orig_alpha = rgba_arr[..., 3].astype(np.float32) / 255.0
        partial = (orig_alpha > 0.05) & (orig_alpha < 0.95)
        transparent = orig_alpha < 0.99
        if transparent.sum() > 0.001 * orig_alpha.size or partial.sum() > 0:
            preview = _compose_checker(image) if checkerboard else image
            info = {
                "mode": mode,
                "quality_score": 1.0,
                "subject_type": "rgba_input",
                "expert_used": "passthrough",
                "passthrough": "existing_alpha",
                "tta": False,
                "sdmatte_used": False,
                "sam2_used": False,
                "solid_bg_spill_cleanup": "skipped_rgba_input",
                "timings_ms": {"total": 0.0},
            }
            return image, preview, json.dumps(info, indent=2)

    pipeline = _get_pipeline(
        device, segmenters, use_depth, use_classifier,
        use_vitmatte, use_closed_form, use_uncertainty_sharpen,
        use_solid_background_cleanup, use_tta,
        use_sdmatte, sdmatte_cache_dir, sdmatte_variant, sdmatte_prompt_mode,
        use_sam2, use_owlv2,
    )

    # Per-request settings (don't require model reload)
    pipeline.config.subject_type_override = subject_override if subject_override != "auto" else None
    pipeline.config.force_sdmatte = force_sdmatte and use_sdmatte

    result = pipeline.process(image)

    # Compose over checkerboard for visual preview
    preview = _compose_checker(result.rgba) if checkerboard else result.rgba

    info = {
        "mode": mode,
        "quality_score": round(result.quality_score, 3),
        "subject_type": result.metadata.get("subject_type", "n/a"),
        "expert_used": result.metadata.get("expert", "n/a"),
        "neural_advisor_top": result.metadata.get("neural_advisor_top", []),
        "route_id": result.metadata.get("route_id", None),
        "route_image_family": result.metadata.get("route_image_family", None),
        "route_material_hint": result.metadata.get("route_material_hint", None),
        "route_background": result.metadata.get("route_background", None),
        "route_keyable": result.metadata.get("route_keyable", None),
        "route_notes": result.metadata.get("route_notes", []),
        "cg_clean_border": result.metadata.get("cg_clean_border", result.metadata.get("route_clean_border", None)),
        "cg_bg_rgb": result.metadata.get("cg_bg_rgb", result.metadata.get("route_bg_rgb", None)),
        "cg_border_p95": result.metadata.get("cg_border_p95", result.metadata.get("route_border_p95", None)),
        "cg_border_p99": result.metadata.get("cg_border_p99", result.metadata.get("route_border_p99", None)),
        "cg_uncertain_band_ratio": result.metadata.get("cg_uncertain_band_ratio", result.metadata.get("route_uncertain_band_ratio", None)),
        "cg_alpha_hole_ratio": result.metadata.get("cg_alpha_hole_ratio", result.metadata.get("route_alpha_hole_ratio", None)),
        "cg_alpha_halo_ratio": result.metadata.get("cg_alpha_halo_ratio", result.metadata.get("route_alpha_halo_ratio", None)),
        "tta": result.metadata.get("tta", False),
        "sdmatte_used": result.metadata.get("sdmatte_used", False),
        "sdmatte_forced": result.metadata.get("sdmatte_forced", False),
        "sdmatte_skipped": result.metadata.get("sdmatte_skipped", None),
        "sam2_used": result.metadata.get("sam2_used", False),
        "solid_bg_cleanup": result.metadata.get("solid_bg_cleanup", None),
        "solid_bg_cleanup_bg_rgb": result.metadata.get("bg_rgb", None),
        "solid_bg_spill_cleanup": result.metadata.get("solid_bg_spill_cleanup", None),
        "solid_bg_spill_bg_rgb": result.metadata.get("solid_bg_bg_rgb", None),
        "checkerboard_cleanup": result.metadata.get("checkerboard_cleanup", None),
        "graphic_border_residue_cleanup": result.metadata.get("graphic_border_residue_cleanup", None),
        "portrait_lower_surface_cleanup": result.metadata.get("portrait_lower_surface_cleanup", None),
        "visual_bg_colored_residue_ratio": result.metadata.get("visual_bg_colored_residue_ratio", None),
        "busy_graphic_despill": result.metadata.get("busy_graphic_despill", False),
        "foreground_decontam": result.metadata.get("foreground_decontam", None),
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
        "auto",
        "portrait", "animal_fur", "plant_thin",
        "product", "product_opaque", "product_glass",
        "transparent", "transparent_object",
        "vehicle",
        "anime", "flat_cartoon",
        "complex_multi",
        "text_logo", "sticker_logo", "text_glow",
        "document_screenshot", "solid_screen_keying", "busy_scene",
        "generic",
    ]

    with gr.Blocks(title="Backgrounder", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## Backgrounder - background removal\n"
            "Smart Auto runs the fast BiRefNet HR + BEN2 cutout path. "
            "Depth is experimental and stays off unless you enable it in developer settings."
        )

        with gr.Row():
            # ── left column: input + settings ──
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Input image")
                mode = gr.Radio(
                    ["Smart Auto", "Fast", "Max Quality"],
                    value="Smart Auto",
                    label="Mode",
                )
                subject_override = gr.Dropdown(
                    subject_choices, value="auto", label="Subject",
                )
                checkerboard = gr.Checkbox(value=True, label="Preview on checkerboard")

                with gr.Accordion("Developer settings", open=False):
                    device = gr.Radio(
                        ["auto", "cuda", "mps", "cpu"],
                        value="auto", label="Device",
                    )
                    segmenters = gr.Textbox(
                        value="birefnet_hr,ben2",
                        label="Segmenters (comma-separated)",
                        info="birefnet_hr · ben2 · inspyrenet",
                    )
                    use_depth = gr.Checkbox(value=False, label="Depth Anything V2-Small (experimental)")
                    use_classifier = gr.Checkbox(value=True, label="CLIP subject classifier")
                    use_vitmatte = gr.Checkbox(
                        value=False,
                        label="ViTMatte refiner — best for hair/fur (NC weights: non-commercial only)",
                        info="hustvl/vitmatte-small-composition-1k — Adobe Composition-1K licence",
                    )
                    use_closed_form = gr.Checkbox(value=False, label="Closed-form matting (experimental)")
                    use_uncertainty_sharpen = gr.Checkbox(value=True, label="Uncertainty-gated sharpening")
                    use_solid_background_cleanup = gr.Checkbox(
                        value=True,
                        label="Remove solid background color spill",
                        info="Final export cleanup for leftover blue/green rims on flat backgrounds.",
                    )
                    use_tta = gr.Checkbox(
                        value=False,
                        label="Test-time augmentation — TTA  (2× slower Stage B, better for dark/low-contrast)",
                        info="Runs each segmenter on original + horizontal flip, averages the results. "
                             "Helps Spider-Man-on-dark-background type images.",
                    )
                    gr.Markdown("**Phase 3** — activate for harder images")
                    gr.Markdown(sdmatte_defaults["status"])
                    use_sdmatte = gr.Checkbox(
                        value=sdmatte_defaults["enabled"],
                        label="SDMatte diffusion refiner (auto-downloads ~5 GB on first use)",
                        info="Triggers when quality < 0.72. Requires CUDA + diffusers.",
                    )
                    force_sdmatte = gr.Checkbox(
                        value=sdmatte_defaults["force"],
                        label="Force SDMatte (ignore quality gate)",
                        info="Run SDMatte on every image regardless of quality score. Only active when SDMatte is enabled.",
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

                btn = gr.Button("Remove background", variant="primary")
                with gr.Row():
                    update_btn = gr.Button("Update from GitHub")
                    restart_btn = gr.Button("Restart app")
                update_status = gr.Textbox(label="Update status", interactive=False)

            # ── right column: outputs ──
            with gr.Column(scale=1):
                out_preview = gr.Image(type="pil", label="Preview on checker")
                out_rgba = gr.Image(type="pil", label="Download RGBA (transparent)", image_mode="RGBA")
                out_info = gr.Code(label="Diagnostics (JSON)", language="json")

        _inputs = [
            inp, mode, device, segmenters, use_depth, use_classifier,
            use_vitmatte, use_closed_form, use_uncertainty_sharpen,
            use_solid_background_cleanup,
            use_tta,
            use_sdmatte, force_sdmatte, sdmatte_cache_dir, sdmatte_variant, sdmatte_prompt_mode,
            use_sam2, use_owlv2, subject_override, checkerboard,
        ]

        btn.click(fn=remove_background, inputs=_inputs,
                  outputs=[out_rgba, out_preview, out_info])
        inp.upload(fn=remove_background, inputs=_inputs,
                   outputs=[out_rgba, out_preview, out_info])
        mode.change(
            fn=lambda name: _preset_values(name),
            inputs=[mode],
            outputs=[
                segmenters, use_depth, use_classifier, use_vitmatte,
                use_closed_form, use_uncertainty_sharpen,
                use_solid_background_cleanup, use_tta,
                use_sdmatte, force_sdmatte, use_sam2, use_owlv2,
            ],
        )
        update_btn.click(
            fn=update_from_git,
            inputs=[],
            outputs=[update_status],
        )
        restart_btn.click(
            fn=restart_app,
            inputs=[],
            outputs=[update_status],
            js="() => { setTimeout(() => window.location.reload(), 5000); return []; }",
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
