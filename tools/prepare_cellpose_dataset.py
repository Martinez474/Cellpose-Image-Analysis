#!/usr/bin/env python3
"""Validate paired images/masks and crop them into a Cellpose dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


VALID_SPLITS = {"train", "validation", "test"}
REQUIRED_COLUMNS = {"sample_id", "image_path", "mask_path", "split"}


class DatasetError(ValueError):
    """Raised when the source dataset is unsafe or internally inconsistent."""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    mask_path: Path
    split: str
    condition: str = ""


def safe_sample_id(value: str) -> str:
    """Return a filesystem-safe sample identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not cleaned:
        raise DatasetError("Every row must have a non-empty sample_id.")
    return cleaned


def resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def load_manifest(manifest_path: Path) -> list[Sample]:
    """Load and validate the source-level split manifest."""
    manifest_path = manifest_path.resolve()
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise DatasetError(
                "Manifest is missing columns: " + ", ".join(sorted(missing))
            )

        samples: list[Sample] = []
        seen_ids: set[str] = set()
        seen_images: set[Path] = set()

        for row_number, row in enumerate(reader, start=2):
            sample_id = safe_sample_id(row["sample_id"])
            split = row["split"].strip().lower()
            if split not in VALID_SPLITS:
                raise DatasetError(
                    f"Row {row_number}: split must be train, validation, or test."
                )
            if sample_id in seen_ids:
                raise DatasetError(f"Row {row_number}: duplicate sample_id {sample_id!r}.")

            image_path = resolve_path(row["image_path"], manifest_path.parent)
            mask_path = resolve_path(row["mask_path"], manifest_path.parent)
            if image_path in seen_images:
                raise DatasetError(
                    f"Row {row_number}: image appears more than once: {image_path}"
                )
            if not image_path.is_file():
                raise DatasetError(f"Row {row_number}: image does not exist: {image_path}")
            if not mask_path.is_file():
                raise DatasetError(f"Row {row_number}: mask does not exist: {mask_path}")

            seen_ids.add(sample_id)
            seen_images.add(image_path)
            samples.append(
                Sample(
                    sample_id=sample_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    split=split,
                    condition=(row.get("condition") or "").strip(),
                )
            )

    if not samples:
        raise DatasetError("Manifest contains no samples.")
    return samples


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_no_cross_split_duplicates(samples: Iterable[Sample]) -> None:
    """Reject identical source images assigned to different splits."""
    seen: dict[str, Sample] = {}
    for sample in samples:
        digest = file_digest(sample.image_path)
        previous = seen.get(digest)
        if previous is not None and previous.split != sample.split:
            raise DatasetError(
                "Identical images occur in different splits: "
                f"{previous.sample_id} ({previous.split}) and "
                f"{sample.sample_id} ({sample.split})."
            )
        seen[digest] = sample


def load_pair(sample: Sample) -> tuple[Image.Image, np.ndarray]:
    with Image.open(sample.image_path) as source_image:
        image = source_image.copy()
    with Image.open(sample.mask_path) as source_mask:
        mask_size = source_mask.size
        mask = np.array(source_mask)

    if mask.ndim != 2:
        raise DatasetError(
            f"{sample.sample_id}: mask must be a single-channel integer label image; "
            f"got shape {mask.shape}. An RGB annotation overlay is not a mask."
        )
    if not np.issubdtype(mask.dtype, np.integer):
        raise DatasetError(
            f"{sample.sample_id}: mask values must be integers; got {mask.dtype}."
        )
    if image.size != mask_size:
        raise DatasetError(
            f"{sample.sample_id}: image size {image.size} does not match "
            f"mask size {mask_size}."
        )
    if mask.min(initial=0) < 0:
        raise DatasetError(f"{sample.sample_id}: mask contains negative labels.")

    labels = np.unique(mask)
    foreground = labels[labels != 0]
    if foreground.size == 0:
        raise DatasetError(f"{sample.sample_id}: mask contains no labeled objects.")

    return image, mask


def tile_starts(length: int, tile_size: int) -> list[int]:
    """Cover an axis completely, using overlap only for the final tile."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_size))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def relabel_mask(mask: np.ndarray) -> np.ndarray:
    """Make positive labels consecutive within one crop."""
    output = np.zeros(mask.shape, dtype=np.uint16)
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if labels.size > np.iinfo(np.uint16).max:
        raise DatasetError("A crop contains more than 65,535 objects.")
    for new_label, old_label in enumerate(labels, start=1):
        output[mask == old_label] = new_label
    return output


def border_object_count(mask: np.ndarray) -> int:
    border_labels = np.unique(
        np.concatenate((mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]))
    )
    return int(np.count_nonzero(border_labels))


def crop_sample(
    sample: Sample, image: Image.Image, mask: np.ndarray, tile_size: int
) -> Iterable[tuple[Image.Image, np.ndarray, int, int]]:
    width, height = image.size
    if width < tile_size or height < tile_size:
        raise DatasetError(
            f"{sample.sample_id}: image size {image.size} is smaller than the "
            f"requested {tile_size}x{tile_size} crop."
        )

    for y in tile_starts(height, tile_size):
        for x in tile_starts(width, tile_size):
            box = (x, y, x + tile_size, y + tile_size)
            image_crop = image.crop(box)
            mask_crop = relabel_mask(mask[y : y + tile_size, x : x + tile_size])
            yield image_crop, mask_crop, x, y


def prepare_dataset(manifest: Path, output: Path, tile_size: int = 256) -> int:
    if tile_size <= 0:
        raise DatasetError("tile_size must be positive.")
    if output.exists() and any(output.iterdir()):
        raise DatasetError(f"Output directory is not empty: {output.resolve()}")

    samples = load_manifest(manifest)
    validate_no_cross_split_duplicates(samples)

    # Validate every pair before writing any output.
    loaded = [(sample, *load_pair(sample)) for sample in samples]

    output.mkdir(parents=True, exist_ok=True)
    for split in VALID_SPLITS:
        (output / split).mkdir(exist_ok=True)

    crop_rows: list[dict[str, object]] = []
    for sample, image, mask in loaded:
        split_dir = output / sample.split
        for index, (image_crop, mask_crop, x, y) in enumerate(
            crop_sample(sample, image, mask, tile_size)
        ):
            stem = f"{sample.sample_id}_crop_{index:03d}"
            image_name = f"{stem}_img.png"
            mask_name = f"{stem}_masks.tif"
            image_crop.save(split_dir / image_name)
            Image.fromarray(mask_crop).save(split_dir / mask_name)

            object_count = int(np.count_nonzero(np.unique(mask_crop)))
            crop_rows.append(
                {
                    "split": sample.split,
                    "condition": sample.condition,
                    "sample_id": sample.sample_id,
                    "source_image": str(sample.image_path),
                    "image_crop": str(Path(sample.split) / image_name),
                    "mask_crop": str(Path(sample.split) / mask_name),
                    "x": x,
                    "y": y,
                    "width": tile_size,
                    "height": tile_size,
                    "object_count": object_count,
                    "border_object_count": border_object_count(mask_crop),
                }
            )

    fieldnames = [
        "split",
        "condition",
        "sample_id",
        "source_image",
        "image_crop",
        "mask_crop",
        "x",
        "y",
        "width",
        "height",
        "object_count",
        "border_object_count",
    ]
    with (output / "crops.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(crop_rows)

    return len(crop_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tile-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        count = prepare_dataset(args.manifest, args.output, args.tile_size)
    except (DatasetError, OSError) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Created {count} paired crops in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
