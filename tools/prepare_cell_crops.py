#!/usr/bin/env python3
"""Create one image crop and one instance mask per human-annotated cell."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow direct execution as ``python tools/prepare_cell_crops.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.prepare_finetune_masks import annotation_to_instances, pair_files


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def split_for_source(index: int, validation_every: int, test_every: int) -> str:
    """Assign all cells from one source image to a single data split."""
    if index % test_every == 0:
        return "test"
    if index % validation_every == 0:
        return "validation"
    return "train"


def square_crop_bounds(
    x0: int, y0: int, x1: int, y1: int,
    image_width: int, image_height: int,
    padding: int, crop_size: int,
) -> tuple[int, int, int, int]:
    """Return a square crop centered on a cell and clipped to image bounds."""
    required_width = (x1 - x0) + 2 * padding
    required_height = (y1 - y0) + 2 * padding
    side = max(crop_size, required_width, required_height)
    side = min(side, image_width, image_height)
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    left = max(0, min(image_width - side, center_x - side // 2))
    top = max(0, min(image_height - side, center_y - side // 2))
    return left, top, left + side, top + side


def main() -> int:
    # Extract instance labels first, then write one image crop and binary mask
    # for every annotated cell.
    parser = argparse.ArgumentParser(
        description="Create one Cellpose training crop per human-annotated cell."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("cellpose_cell_crops"))
    parser.add_argument("--padding", type=int, default=20,
                        help="Context pixels around each cell")
    parser.add_argument("--crop-size", type=int, default=0,
                        help="Minimum square crop size; 0 uses the cell bounds plus padding")
    parser.add_argument("--minimum-area", type=float, default=80.0)
    parser.add_argument("--yellow-hue-min", type=int, default=20)
    parser.add_argument("--yellow-hue-max", type=int, default=45)
    parser.add_argument("--yellow-saturation-min", type=int, default=100)
    parser.add_argument("--yellow-value-min", type=int, default=120)
    parser.add_argument("--yellow-close-size", type=int, default=3)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--test-every", type=int, default=10)
    args = parser.parse_args()
    if args.padding < 0 or args.crop_size < 0:
        parser.error("padding and crop-size must be nonnegative")
    if not 0 <= args.yellow_hue_min <= args.yellow_hue_max <= 179:
        parser.error("yellow hue values must be between 0 and 179")
    if not 0 <= args.yellow_saturation_min <= 255 or not 0 <= args.yellow_value_min <= 255:
        parser.error("yellow saturation/value must be between 0 and 255")
    if args.yellow_close_size < 1 or args.yellow_close_size % 2 == 0:
        parser.error("yellow close size must be a positive odd number")

    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "cell_manifest.tsv"
    rows: list[dict[str, object]] = []
    pairs = pair_files(args.input_dir, args.annotation_dir)

    # Splitting by source image prevents near-duplicate cells from one
    # micrograph leaking into validation or test data.
    for source_index, (image_path, annotation_path, pair_score) in enumerate(pairs):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        annotation = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
        if image is None or annotation is None or image.shape[:2] != annotation.shape[:2]:
            rows.append({"status": "excluded", "image": str(image_path.resolve()),
                         "annotation": str(annotation_path.resolve()),
                         "reason": "dimension mismatch"})
            continue

        instances = annotation_to_instances(
            annotation_path, args.minimum_area, args.yellow_hue_min,
            args.yellow_hue_max, args.yellow_saturation_min,
            args.yellow_value_min, args.yellow_close_size,
        )
        labels = [label for label in np.unique(instances) if label != 0]
        split = split_for_source(source_index, args.validation_every, args.test_every)
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        height, width = image.shape[:2]
        source_stem = image_path.stem.replace(" ", "_")

        for cell_number, label in enumerate(labels, start=1):
            ys, xs = np.where(instances == label)
            if len(xs) == 0:
                continue
            left, top, right, bottom = square_crop_bounds(
                int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1,
                width, height, args.padding, args.crop_size,
            )
            image_crop = image[top:bottom, left:right]
            mask_crop = np.where(
                instances[top:bottom, left:right] == label, 1, 0
            ).astype(np.uint16)
            stem = f"{source_stem}_cell_{cell_number:04d}"
            image_output = split_dir / f"{stem}_img.png"
            mask_output = split_dir / f"{stem}_masks.tif"
            cv2.imwrite(str(image_output), image_crop)
            cv2.imwrite(str(mask_output), mask_crop)
            rows.append({
                "status": "ok", "split": split, "image": str(image_path.resolve()),
                "annotation": str(annotation_path.resolve()), "cell": cell_number,
                "label": int(label), "x": left, "y": top,
                "width": right - left, "height": bottom - top,
                "pair_score": round(pair_score, 4),
                "crop": str(image_output.resolve()),
                "mask": str(mask_output.resolve()),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["status", "split", "image", "annotation", "cell", "label",
                  "x", "y", "width", "height", "pair_score", "crop", "mask", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    included = sum(row["status"] == "ok" for row in rows)
    print(f"Created {included} individual cell crops.")
    print(f"Manifest: {manifest_path}")
    return 0 if included else 1


if __name__ == "__main__":
    raise SystemExit(main())
