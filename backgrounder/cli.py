from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from backgrounder.config import PipelineConfig, SegmenterID, Device


# Presets mirror the Gradio app's Fast / Smart Auto / Max Quality modes so the
# CLI and UI behave identically. Individual flags below override the preset.
_MODE_PRESETS = {
    "fast": {
        "segmenters": "birefnet_hr",
        "use_tta": False, "use_vitmatte": False, "use_closed_form_refine": False,
        "use_sdmatte": False, "use_sam3": False, "use_sam2": False, "use_owlv2": False,
        "route_mode": "fast",
    },
    "smart": {
        "segmenters": "birefnet_hr,ben2",
        "use_tta": False, "use_vitmatte": False, "use_closed_form_refine": False,
        "use_sdmatte": False, "use_sam3": False, "use_sam2": False, "use_owlv2": False,
        "route_mode": "smart",
    },
    "max": {
        "segmenters": "birefnet_hr,ben2",
        "use_tta": True, "use_vitmatte": True, "use_closed_form_refine": True,
        "use_sdmatte": True, "use_sam3": True, "use_sam2": True, "use_owlv2": True,
        "route_mode": "max",
    },
}


@click.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", default=None,
              help="Output file (single input) or directory (batch).")
@click.option("--output-dir", default=None, type=click.Path(),
              help="Output directory for batch processing.")
@click.option("--mode", default="smart", show_default=True,
              type=click.Choice(["fast", "smart", "max"]),
              help="Quality preset (matches the Gradio app). Individual flags override it.")
@click.option("--device", default="auto", show_default=True,
              type=click.Choice(["auto", "cuda", "mps", "cpu"]),
              help="Compute device.")
@click.option("--segmenters", default=None,
              help="Comma-separated segmenter IDs (overrides preset): birefnet_hr,ben2,inspyrenet")
@click.option("--depth", is_flag=True, default=False,
              help="Enable Depth Anything V2-Small experimental refinement.")
@click.option("--fp32", is_flag=True, default=False,
              help="Force float32 (useful on MPS or for debugging).")
@click.option("--trimap-dilation", default=10, show_default=True, type=int,
              help="Trimap unknown-band half-width in pixels.")
@click.option("--premultiplied", is_flag=True, default=False,
              help="Save premultiplied alpha PNG (avoids fringing on composition).")
@click.option("--quality-threshold", default=0.55, show_default=True, type=float,
              help="Re-run refinement if quality score is below this value.")
@click.option("--no-closed-form/--closed-form", "closed_form", default=None,
              help="Toggle closed-form matting refinement (default follows --mode).")
@click.option("--closed-form-max-pixels", default=65536, show_default=True, type=int,
              help="Maximum solve resolution for closed-form matting.")
@click.option("--no-uncertainty-sharpen", is_flag=True, default=False,
              help="Disable ensemble-agreement alpha sharpening.")
@click.option("--no-classifier", is_flag=True, default=False,
              help="Disable CLIP subject classifier.")
@click.option("--subject-type", default=None,
              help="Override subject type, e.g. portrait,animal_fur,product,transparent,anime,generic")
@click.option("--vitmatte/--no-vitmatte", default=None,
              help="ViTMatte hair/fur refiner (NC weights). Default follows --mode.")
@click.option("--sdmatte/--no-sdmatte", default=None,
              help="SDMatte diffusion refiner (CUDA, ~5 GB). Default follows --mode.")
@click.option("--force-sdmatte", is_flag=True, default=False,
              help="Run SDMatte on every image regardless of quality/router gate.")
@click.option("--sdmatte-cache-dir", default="~/.cache/backgrounder/sdmatte", show_default=True,
              help="Where SDMatte weights + SD2.1 configs are cached.")
@click.option("--sdmatte-variant", default="sdmatte", show_default=True,
              type=click.Choice(["sdmatte", "sdmatte_plus"]),
              help="Which SDMatte checkpoint to load.")
@click.option("--sdmatte-prompt-mode", default="trimap", show_default=True,
              type=click.Choice(["bbox", "mask", "trimap", "point"]),
              help="Visual prompt passed to SDMatte from the coarse alpha.")
@click.option("--sdmatte-quality-trigger", default=0.72, show_default=True, type=float,
              help="Run SDMatte only when the internal quality score is below this value.")
@click.option("--sam2/--no-sam2", default=None,
              help="SAM 2.1 boundary refiner. Default follows --mode.")
@click.option("--sam3/--no-sam3", default=None,
              help="SAM 3.1 concept/box refiner (CUDA). Default follows --mode.")
@click.option("--owlv2/--no-owlv2", default=None,
              help="OWLv2 box-prompt localizer for SAM. Default follows --mode.")
@click.option("--route-mode", default=None,
              type=click.Choice(["fast", "smart", "max"]),
              help="Router lambda (overrides preset): fast=cheap only, max=spend freely.")
@click.option("--no-region-router", is_flag=True, default=False,
              help="Disable the Phase-1 failure-map router (use legacy triggers).")
@click.option("--no-decontam-unmix", is_flag=True, default=False,
              help="Disable Phase-0 closed-form foreground unmixing.")
@click.option("--tta", is_flag=True, default=False,
              help="Force test-time augmentation on (default follows --mode).")
@click.option("--tile-size", default=0, show_default=True, type=int,
              help="Tile size for 4K+ images (0 = disabled, e.g. 1024).")
@click.option("--debug", is_flag=True, default=False,
              help="Print quality report, route decisions and per-stage timings.")
def main(
    inputs: tuple[str, ...],
    output_path: Optional[str],
    output_dir: Optional[str],
    mode: str,
    device: str,
    segmenters: Optional[str],
    depth: bool,
    fp32: bool,
    trimap_dilation: int,
    premultiplied: bool,
    quality_threshold: float,
    closed_form: Optional[bool],
    closed_form_max_pixels: int,
    no_uncertainty_sharpen: bool,
    no_classifier: bool,
    subject_type: Optional[str],
    vitmatte: Optional[bool],
    sdmatte: Optional[bool],
    force_sdmatte: bool,
    sdmatte_cache_dir: str,
    sdmatte_variant: str,
    sdmatte_prompt_mode: str,
    sdmatte_quality_trigger: float,
    sam2: Optional[bool],
    sam3: Optional[bool],
    owlv2: Optional[bool],
    route_mode: Optional[str],
    no_region_router: bool,
    no_decontam_unmix: bool,
    tta: bool,
    tile_size: int,
    debug: bool,
) -> None:
    """Remove backgrounds from one or more images.

    \b
    Single image (Smart Auto):
        backgrounder photo.jpg result.png

    \b
    Max Quality with diagnostics:
        backgrounder photo.jpg out.png --mode max --debug

    \b
    Batch (saves to ./output/):
        backgrounder *.jpg --output-dir ./output/
    """
    from backgrounder.pipeline import BackgroundRemovalPipeline

    preset = _MODE_PRESETS[mode]

    # Resolve each setting: explicit flag wins, else preset, else config default.
    seg_str = segmenters or preset["segmenters"]
    seg_ids = [SegmenterID(s.strip()) for s in seg_str.split(",")]

    use_vitmatte = preset["use_vitmatte"] if vitmatte is None else vitmatte
    use_sdmatte = preset["use_sdmatte"] if sdmatte is None else sdmatte
    use_sam2 = preset["use_sam2"] if sam2 is None else sam2
    use_sam3 = preset["use_sam3"] if sam3 is None else sam3
    use_owlv2 = preset["use_owlv2"] if owlv2 is None else owlv2
    use_closed_form = preset["use_closed_form_refine"] if closed_form is None else closed_form
    use_tta = preset["use_tta"] or tta

    config = PipelineConfig(
        segmenters=seg_ids,
        device=Device(device),
        use_depth=depth,
        fp16=not fp32,
        trimap_dilation=trimap_dilation,
        premultiplied=premultiplied,
        quality_threshold=quality_threshold,
        use_closed_form_refine=use_closed_form,
        closed_form_max_pixels=closed_form_max_pixels,
        use_uncertainty_sharpen=not no_uncertainty_sharpen,
        use_classifier=not no_classifier,
        subject_type_override=subject_type,
        use_vitmatte=use_vitmatte,
        vitmatte_allow_nc=use_vitmatte,
        use_sdmatte=use_sdmatte,
        force_sdmatte=force_sdmatte and use_sdmatte,
        sdmatte_cache_dir=sdmatte_cache_dir,
        sdmatte_variant=sdmatte_variant,
        sdmatte_prompt_mode=sdmatte_prompt_mode,
        sdmatte_quality_trigger=sdmatte_quality_trigger,
        use_sam2=use_sam2,
        use_sam3=use_sam3,
        use_owlv2=use_owlv2 and (use_sam2 or use_sam3),
        use_tta=use_tta,
        use_region_router=not no_region_router,
        route_mode=route_mode or preset["route_mode"],
        use_decontam_unmix=not no_decontam_unmix,
        tile_size=tile_size,
    )

    click.echo(f"Loading models on {device} (mode={mode})…")
    pipeline = BackgroundRemovalPipeline(config)
    pipeline.load()
    click.echo("Models ready.")

    input_paths = [Path(p) for p in inputs]

    # Resolve output paths
    if len(input_paths) == 1 and output_path is not None:
        pairs = [(input_paths[0], Path(output_path))]
    else:
        out_dir = Path(output_dir) if output_dir else Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs = [(p, out_dir / (p.stem + "_nobg.png")) for p in input_paths]

    for inp, out in tqdm(pairs, desc="Processing", unit="img"):
        result = pipeline.process_path(inp, out)
        if debug:
            meta = result.metadata
            click.echo(f"\n{inp.name}  →  {out.name}")
            click.echo(f"  subject  : {meta.get('subject_type', 'n/a')}")
            click.echo(f"  quality  : {meta.get('quality', 'n/a')}")
            click.echo(f"  router   : mode={meta.get('route_mode')} "
                       f"experts={meta.get('route_experts_used')}")
            decisions = meta.get("route_decisions")
            if decisions:
                click.echo(f"  decisions: {json.dumps(decisions)}")
            click.echo(f"  timings  : {meta.get('timings_ms', {})}")


if __name__ == "__main__":
    main()
