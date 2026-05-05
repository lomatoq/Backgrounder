from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backgrounder import BackgroundRemovalPipeline, PipelineConfig, SegmenterID, Device


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Backgrounder over a folder and save RGBA/checker/diagnostics outputs."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--subject", default=None, help="Override subject type, or omit for neural advisor.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--segmenters", default="birefnet_hr,ben2")
    parser.add_argument("--sdmatte", action="store_true", help="Enable SDMatte, still route-gated.")
    parser.add_argument("--force-sdmatte", action="store_true", help="Force SDMatte on every image.")
    parser.add_argument("--sam3", action="store_true", help="Enable optional SAM 3.1 refinement.")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = [
        p for p in sorted(args.input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if args.limit > 0:
        images = images[: args.limit]

    config = PipelineConfig(
        segmenters=[SegmenterID(s.strip()) for s in args.segmenters.split(",") if s.strip()],
        device=Device(args.device),
        use_sdmatte=args.sdmatte,
        force_sdmatte=args.force_sdmatte and args.sdmatte,
        use_sam3=args.sam3,
        use_tta=args.tta,
        subject_type_override=args.subject,
    )
    pipeline = BackgroundRemovalPipeline(config).load()

    summary = []
    for idx, path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {path.name}", flush=True)
        image = Image.open(path)
        result = pipeline.process(image)

        stem_dir = args.output_dir / path.stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        result.rgba.save(stem_dir / "result_rgba.png")
        _checker(result.rgba).save(stem_dir / "preview_checker.png")
        (stem_dir / "diagnostics.json").write_text(
            json.dumps(result.metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        row = {
            "file": path.name,
            "quality_score": round(result.quality_score, 4),
            "subject_type": result.metadata.get("subject_type"),
            "route_id": result.metadata.get("route_id"),
            "route_background": result.metadata.get("route_background"),
            "route_image_family": result.metadata.get("route_image_family"),
            "route_material_hint": result.metadata.get("route_material_hint"),
            "solid_bg_spill_cleanup": result.metadata.get("solid_bg_spill_cleanup"),
            "checkerboard_cleanup": result.metadata.get("checkerboard_cleanup"),
            "graphic_border_residue_cleanup": result.metadata.get("graphic_border_residue_cleanup"),
            "portrait_lower_surface_cleanup": result.metadata.get("portrait_lower_surface_cleanup"),
            "visual_bg_colored_residue_ratio": result.metadata.get("visual_bg_colored_residue_ratio"),
            "busy_graphic_despill": result.metadata.get("busy_graphic_despill", False),
            "foreground_decontam": result.metadata.get("foreground_decontam"),
            "sdmatte_used": result.metadata.get("sdmatte_used", False),
            "sam3_used": result.metadata.get("sam3_used", False),
            "sam3_skipped": result.metadata.get("sam3_skipped"),
            "sam2_used": result.metadata.get("sam2_used", False),
            "total_ms": result.metadata.get("timings_ms", {}).get("total"),
        }
        summary.append(row)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"done: {args.output_dir / 'summary.json'}", flush=True)


def _checker(rgba: Image.Image, square: int = 16) -> Image.Image:
    import numpy as np

    rgba = rgba.convert("RGBA")
    w, h = rgba.size
    yy, xx = np.indices((h, w))
    light = ((xx // square) + (yy // square)) % 2 == 0
    bg = np.where(light[..., None], 220, 180).astype(np.uint8)
    bg = np.broadcast_to(bg, (h, w, 3)).copy()
    checker = Image.fromarray(bg, mode="RGB")
    checker.paste(rgba, mask=rgba.split()[3])
    return checker


if __name__ == "__main__":
    main()
