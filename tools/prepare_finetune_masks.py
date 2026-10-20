#!/usr/bin/env python3
"""Create Cellpose instance-label masks from human yellow-outline annotations."""

from __future__ import annotations

import argparse
import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def normalized_name(path: Path) -> str:
    """Normalize filenames so source and annotation images can be matched."""
    value = path.stem.lower()
    for phrase in ("grain pic", "grain", "results", "jpg", "png", "pic"):
        value = value.replace(phrase, " ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def pair_files(input_dir: Path, annotation_dir: Path) -> list[tuple[Path, Path, float]]:
    """Pair each source image with its closest annotation filename."""
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    annotations = sorted(p for p in annotation_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images or not annotations:
        raise RuntimeError("Both input and annotation directories must contain images.")
    scores = np.array([
        [SequenceMatcher(None, normalized_name(a), normalized_name(i)).ratio()
         for i in images]
        for a in annotations
    ])
    annotation_indices, image_indices = linear_sum_assignment(-scores)
    pairs = [(images[i], annotations[a], float(scores[a, i]))
             for a, i in zip(annotation_indices, image_indices)]
    return sorted(pairs, key=lambda item: item[0].name)


def annotation_to_instances(
    annotation_path: Path,
    minimum_area: float,
    yellow_hue_min: int = 20,
    yellow_hue_max: int = 45,
    yellow_saturation_min: int = 100,
    yellow_value_min: int = 120,
    yellow_close_size: int = 3,
) -> np.ndarray:
    """Turn yellow human outlines into connected, integer-labeled instances."""
    image = cv2.imread(str(annotation_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read annotation: {annotation_path}")
    # HSV thresholds isolate yellow annotation lines while rejecting most
    # microscope-image colors; morphology closes small gaps in those lines.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(
        hsv,
        np.array([yellow_hue_min, yellow_saturation_min, yellow_value_min], np.uint8),
        np.array([yellow_hue_max, 255, 255], np.uint8),
    )
    kernel = np.ones((yellow_close_size, yellow_close_size), np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel)
    # Treat the yellow outlines as walls. Connected components of the inverse
    # outline then correspond to the enclosed human-labelled cells, including
    # cells that touch one another. A one-pixel dilation closes tiny gaps in
    # JPEG-rendered outlines without materially shrinking the interiors.
    walls = cv2.dilate(yellow, np.ones((3, 3), np.uint8), iterations=1)
    interior = cv2.bitwise_not(walls)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(interior, 8)
    instances = np.zeros(labels.shape, dtype=np.uint16)
    next_label = 1
    for component in range(1, component_count):
        x = stats[component, cv2.CC_STAT_LEFT]
        y = stats[component, cv2.CC_STAT_TOP]
        width = stats[component, cv2.CC_STAT_WIDTH]
        height = stats[component, cv2.CC_STAT_HEIGHT]
        touches_border = x <= 0 or y <= 0 or x + width >= labels.shape[1] or y + height >= labels.shape[0]
        if not touches_border and stats[component, cv2.CC_STAT_AREA] >= minimum_area:
            instances[labels == component] = next_label
            next_label += 1
    return instances


def main() -> int:
    # The manifest records accepted and excluded pairs so mismatches and
    # annotations with no closed contours can be reviewed later.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("cellpose_finetune"))
    parser.add_argument("--minimum-area", type=float, default=80.0)
    parser.add_argument("--yellow-hue-min", type=int, default=20)
    parser.add_argument("--yellow-hue-max", type=int, default=45)
    parser.add_argument("--yellow-saturation-min", type=int, default=100)
    parser.add_argument("--yellow-value-min", type=int, default=120)
    parser.add_argument("--yellow-close-size", type=int, default=3)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--test-every", type=int, default=10)
    args = parser.parse_args()
    if args.validation_every <= 0 or args.test_every <= 0:
        parser.error("split intervals must be positive")
    if not 0 <= args.yellow_hue_min <= args.yellow_hue_max <= 179:
        parser.error("yellow hue values must be between 0 and 179")
    if not 0 <= args.yellow_saturation_min <= 255 or not 0 <= args.yellow_value_min <= 255:
        parser.error("yellow saturation/value must be between 0 and 255")
    if args.yellow_close_size < 1 or args.yellow_close_size % 2 == 0:
        parser.error("yellow close size must be a positive odd number")

    args.output_dir = args.output_dir.expanduser().resolve()
    masks_dir = args.output_dir / "source_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "source_manifest.tsv"
    rows: list[dict[str, object]] = []
    for index, (image, annotation, score) in enumerate(
        pair_files(args.input_dir, args.annotation_dir)
    ):
        source = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        annotation_image = cv2.imread(str(annotation), cv2.IMREAD_GRAYSCALE)
        if source is None or annotation_image is None or source.shape != annotation_image.shape:
            rows.append({"status": "excluded", "image": str(image.resolve()),
                         "annotation": str(annotation.resolve()), "reason": "dimension mismatch"})
            continue
        instances = annotation_to_instances(
            annotation,
            args.minimum_area,
            args.yellow_hue_min,
            args.yellow_hue_max,
            args.yellow_saturation_min,
            args.yellow_value_min,
            args.yellow_close_size,
        )
        if int(instances.max()) == 0:
            rows.append({"status": "excluded", "image": str(image.resolve()),
                         "annotation": str(annotation.resolve()), "reason": "no closed yellow contours"})
            continue
        split = "test" if index % args.test_every == 0 else (
            "validation" if index % args.validation_every == 0 else "train"
        )
        mask_path = masks_dir / f"{image.stem}_masks.tif"
        cv2.imwrite(str(mask_path), instances)
        rows.append({"status": "ok", "split": split, "image": str(image.resolve()),
                     "annotation": str(annotation.resolve()), "mask": str(mask_path.resolve()),
                     "pair_score": round(score, 4), "instances": int(instances.max())})

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["status", "split", "image", "annotation", "mask", "pair_score", "instances", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    included = sum(row["status"] == "ok" for row in rows)
    print(f"Prepared {included} image/mask pairs; excluded {len(rows) - included}.")
    print(f"Manifest: {manifest_path}")
    return 0 if included else 1


if __name__ == "__main__":
    raise SystemExit(main())
