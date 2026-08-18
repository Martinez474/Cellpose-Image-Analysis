#!/usr/bin/env python3
"""Run Cellpose on a directory and export JSON plus ImageJ ROI archives."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from roifile import ImagejRoi, roiwrite

# When this file is run as ``python tools/batch_predict.py``, Python puts the
# tools directory (rather than the project root) on sys.path.
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from predict import run_model, write_predictions


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def export_rois(predictions: list[dict[str, object]], output_path: Path) -> None:
    rois = []
    for number, prediction in enumerate(predictions, start=1):
        points = prediction.get("polygon")
        if not isinstance(points, list) or len(points) < 3:
            continue
        rois.append(ImagejRoi.frompoints(points, name=f"Prediction-{number}"))
    if rois:
        roiwrite(output_path, rois)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    images = sorted(
        path for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for index, image_path in enumerate(images, start=1):
        stem = image_path.stem
        json_path = args.results_dir / f"{stem}.json"
        roi_path = args.results_dir / f"{stem}.zip"
        started = time.time()
        print(f"[{index}/{len(images)}] {image_path.name}", flush=True)
        try:
            predictions = run_model(
                image_path=image_path,
                model_name="cpsam_v2",
                use_gpu=True,
                cellprob_threshold=0.0,
                flow_threshold=0.4,
                min_size=15,
            )
            write_predictions(json_path, predictions)
            export_rois(predictions, roi_path)
            elapsed = time.time() - started
            summary.append({
                "image": str(image_path),
                "json": str(json_path),
                "roi_archive": str(roi_path),
                "predictions": len(predictions),
                "seconds": round(elapsed, 2),
                "status": "ok",
            })
            print(f"  {len(predictions)} predictions in {elapsed:.2f} seconds", flush=True)
        except Exception as error:  # keep the batch moving and report failures
            elapsed = time.time() - started
            summary.append({
                "image": str(image_path),
                "seconds": round(elapsed, 2),
                "status": "failed",
                "error": str(error),
            })
            print(f"  FAILED: {error}", file=sys.stderr, flush=True)

    summary_path = args.results_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failed = sum(item["status"] == "failed" for item in summary)
    print(f"Finished {len(images)} images: {len(images) - failed} succeeded, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
