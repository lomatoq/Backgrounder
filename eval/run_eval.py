"""
Eval harness (spec §4): run the pipeline over eval/test_images and save outputs.

  python eval/run_eval.py --mode max

For each image it writes eval/results/<name>/:
  rgba.png   – final cutout (RGBA)
  alpha.png  – final alpha matte
  stages/    – per-stage alpha dumps (BACKGROUNDER_DEBUG_DIR)
  info.json  – metadata (route, experts, timings, quality)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

# Console may be cp1252 (Windows); never let a non-ASCII filename crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "test_images"
RESULTS = ROOT / "results"

_MODE = {
    "fast":  dict(segs="birefnet_hr",      tta=False, vit=False, cf=False, sd=False, s3=False, s2=False, owl=False),
    "smart": dict(segs="birefnet_hr,ben2", tta=False, vit=False, cf=False, sd=False, s3=False, s2=False, owl=False),
    "max":   dict(segs="birefnet_hr,ben2", tta=True,  vit=True,  cf=True,  sd=True,  s3=True,  s2=True,  owl=True),
}


def _slug(name: str) -> str:
    s = "".join(c if (c.isascii() and (c.isalnum() or c in "-_")) else "_" for c in name)
    return s.strip("_")[:40] or "img"


def build_pipeline(mode: str, device: str):
    from backgrounder.config import PipelineConfig, Device, SegmenterID
    from backgrounder.pipeline import BackgroundRemovalPipeline
    m = _MODE[mode]
    cfg = PipelineConfig(
        segmenters=[SegmenterID(s) for s in m["segs"].split(",")],
        device=Device(device), fp16=(device == "cuda"),
        use_classifier=True,
        use_tta=m["tta"], use_vitmatte=m["vit"], vitmatte_allow_nc=m["vit"],
        use_closed_form_refine=m["cf"],
        use_sdmatte=m["sd"], use_sam3=m["s3"], use_sam2=m["s2"], use_owlv2=m["owl"],
        route_mode=mode,
    )
    return BackgroundRemovalPipeline(cfg).load()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="max", choices=list(_MODE))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="substring filter on filenames")
    args = ap.parse_args()

    pipe = build_pipeline(args.mode, args.device)
    imgs = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if args.only:
        imgs = [p for p in imgs if args.only in p.name]

    for idx, p in enumerate(imgs):
        # Stable ASCII output dir so cyrillic/odd filenames never break paths.
        out = RESULTS / args.mode / f"{idx:02d}_{_slug(p.stem)}"
        (out / "stages").mkdir(parents=True, exist_ok=True)
        os.environ["BACKGROUNDER_DEBUG_DIR"] = str(out / "stages")
        try:
            res = pipe.process(Image.open(p))
            res.rgba.save(out / "rgba.png")
            Image.fromarray((res.alpha.clip(0, 1) * 255).astype("uint8"), "L").save(out / "alpha.png")
            (out / "info.json").write_text(
                json.dumps(res.metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            (out / "source.txt").write_text(p.name, encoding="utf-8")
            line = (f"{idx:02d} {p.name} -> q={res.quality_score:.3f} "
                    f"subject={res.metadata.get('subject_type')} "
                    f"experts={res.metadata.get('route_experts_used')} sam3={res.metadata.get('sam3_used')}")
        except Exception as exc:  # one bad image must not abort the batch
            line = f"{idx:02d} {p.name} -> FAILED: {type(exc).__name__}: {exc}"
        print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
