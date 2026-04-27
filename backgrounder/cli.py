from __future__ import annotations
import os
import platform
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from backgrounder.config import PipelineConfig, SegmenterID, Device


@click.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", default=None,
              help="Output file (single input) or directory (batch).")
@click.option("--output-dir", default=None, type=click.Path(),
              help="Output directory for batch processing.")
@click.option("--device", default="auto", show_default=True,
              type=click.Choice(["auto", "cuda", "mps", "cpu"]),
              help="Compute device.")
@click.option("--segmenters", default="birefnet_hr,ben2", show_default=True,
              help="Comma-separated segmenter IDs: birefnet_hr,ben2,inspyrenet")
@click.option("--no-depth", is_flag=True, default=False,
              help="Disable Depth Anything V2-Small (faster, worse on low-contrast).")
@click.option("--fp32", is_flag=True, default=False,
              help="Force float32 (useful on MPS or for debugging).")
@click.option("--trimap-dilation", default=10, show_default=True, type=int,
              help="Trimap unknown-band half-width in pixels.")
@click.option("--premultiplied", is_flag=True, default=False,
              help="Save premultiplied alpha PNG (avoids fringing on composition).")
@click.option("--quality-threshold", default=0.55, show_default=True, type=float,
              help="Re-run refinement if quality score is below this value.")
@click.option("--no-closed-form", is_flag=True, default=False,
              help="Disable closed-form matting refinement in the trimap unknown band.")
@click.option("--closed-form-max-pixels", default=65536, show_default=True, type=int,
              help="Maximum solve resolution for closed-form matting.")
@click.option("--no-uncertainty-sharpen", is_flag=True, default=False,
              help="Disable ensemble-agreement alpha sharpening.")
@click.option("--no-classifier", is_flag=True, default=False,
              help="Disable CLIP subject classifier (Phase 2).")
@click.option("--subject-type", default=None,
              help="Override subject type: portrait,animal_fur,product,plant_thin,transparent,vehicle,anime,complex_multi,generic")
@click.option("--vitmatte", is_flag=True, default=False,
              help="Enable ViTMatte refiner for hair/fur. NOTE: weights are NC (Adobe Composition-1k).")
@click.option("--sdmatte", is_flag=True, default=False,
              help="Enable SDMatte/LiteSDMatte diffusion refiner. Requires CUDA, detectron2, repo path and checkpoint.")
@click.option("--sdmatte-auto", is_flag=True, default=False,
              help="Enable SDMatte automatically on Windows+CUDA when env paths are configured.")
@click.option("--sdmatte-repo-path", default=None, type=click.Path(exists=True, file_okay=False),
              help="Local checkout of vivoCameraResearch/SDMatte.")
@click.option("--sdmatte-checkpoint-path", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Path to LiteSDMatte.pth or SDMatte.pth.")
@click.option("--sdmatte-variant", default="lite", show_default=True,
              type=click.Choice(["lite", "sdmatte"]),
              help="Which SDMatte model class to load.")
@click.option("--sdmatte-prompt-mode", default="bbox", show_default=True,
              type=click.Choice(["bbox", "mask", "trimap", "point"]),
              help="Visual prompt passed to SDMatte from the coarse alpha.")
@click.option("--sdmatte-quality-trigger", default=0.72, show_default=True, type=float,
              help="Run SDMatte only when the internal quality score is below this value.")
@click.option("--tile-size", default=0, show_default=True, type=int,
              help="Tile size for 4K+ images (0 = disabled, e.g. 1024).")
@click.option("--debug", is_flag=True, default=False,
              help="Print quality report and timings per image.")
def main(
    inputs: tuple[str, ...],
    output_path: Optional[str],
    output_dir: Optional[str],
    device: str,
    segmenters: str,
    no_depth: bool,
    fp32: bool,
    trimap_dilation: int,
    premultiplied: bool,
    quality_threshold: float,
    no_closed_form: bool,
    closed_form_max_pixels: int,
    no_uncertainty_sharpen: bool,
    no_classifier: bool,
    subject_type: Optional[str],
    vitmatte: bool,
    sdmatte: bool,
    sdmatte_auto: bool,
    sdmatte_repo_path: Optional[str],
    sdmatte_checkpoint_path: Optional[str],
    sdmatte_variant: str,
    sdmatte_prompt_mode: str,
    sdmatte_quality_trigger: float,
    tile_size: int,
    debug: bool,
) -> None:
    """Remove backgrounds from one or more images.

    \b
    Single image:
        backgrounder photo.jpg result.png

    \b
    Batch (saves to ./output/):
        backgrounder *.jpg --output-dir ./output/
    """
    from backgrounder.pipeline import BackgroundRemovalPipeline

    if sdmatte_auto:
        sdmatte_repo_path = sdmatte_repo_path or os.environ.get("BACKGROUNDER_SDMATTE_REPO")
        sdmatte_checkpoint_path = (
            sdmatte_checkpoint_path
            or os.environ.get("BACKGROUNDER_SDMATTE_CHECKPOINT")
        )
        if platform.system().lower() == "windows" and sdmatte_repo_path and sdmatte_checkpoint_path:
            sdmatte = True

    seg_ids = [SegmenterID(s.strip()) for s in segmenters.split(",")]
    config = PipelineConfig(
        segmenters=seg_ids,
        device=Device(device),
        use_depth=not no_depth,
        fp16=not fp32,
        trimap_dilation=trimap_dilation,
        premultiplied=premultiplied,
        quality_threshold=quality_threshold,
        use_closed_form_refine=not no_closed_form,
        closed_form_max_pixels=closed_form_max_pixels,
        use_uncertainty_sharpen=not no_uncertainty_sharpen,
        use_classifier=not no_classifier,
        subject_type_override=subject_type,
        use_vitmatte=vitmatte,
        vitmatte_allow_nc=vitmatte,
        use_sdmatte=sdmatte,
        sdmatte_repo_path=sdmatte_repo_path,
        sdmatte_checkpoint_path=sdmatte_checkpoint_path,
        sdmatte_variant=sdmatte_variant,
        sdmatte_prompt_mode=sdmatte_prompt_mode,
        sdmatte_quality_trigger=sdmatte_quality_trigger,
        tile_size=tile_size,
    )

    click.echo(f"Loading models on {device}…")
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
            click.echo(f"\n{inp.name}  →  {out.name}")
            click.echo(f"  quality : {result.metadata.get('quality', 'n/a')}")
            click.echo(f"  timings : {result.metadata.get('timings_ms', {})}")
