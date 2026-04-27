from __future__ import annotations
from pathlib import Path
from typing import List, Optional

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
@click.option("--no-classifier", is_flag=True, default=False,
              help="Disable CLIP subject classifier (Phase 2).")
@click.option("--subject-type", default=None,
              help="Override subject type: portrait,animal_fur,product,plant_thin,transparent,vehicle,anime,complex_multi,generic")
@click.option("--vitmatte", is_flag=True, default=False,
              help="Enable ViTMatte refiner for hair/fur. NOTE: weights are NC (Adobe Composition-1k).")
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
    no_classifier: bool,
    subject_type: Optional[str],
    vitmatte: bool,
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

    seg_ids = [SegmenterID(s.strip()) for s in segmenters.split(",")]
    config = PipelineConfig(
        segmenters=seg_ids,
        device=Device(device),
        use_depth=not no_depth,
        fp16=not fp32,
        trimap_dilation=trimap_dilation,
        premultiplied=premultiplied,
        quality_threshold=quality_threshold,
        use_classifier=not no_classifier,
        subject_type_override=subject_type,
        use_vitmatte=vitmatte,
        vitmatte_allow_nc=vitmatte,
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
