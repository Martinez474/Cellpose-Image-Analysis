#!/usr/bin/env python3
"""Fine-tune Cellpose on the prepared image/mask tiles."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np


def load_tiles(directory: Path, limit: int | None = None):
    """Load matching image/mask pairs from one dataset split."""
    image_paths = sorted(directory.glob("*_img.png"))
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise RuntimeError(f"No *_img.png tiles found in {directory}")
    images, labels = [], []
    for image_path in image_paths:
        mask_path = image_path.with_name(image_path.name.replace("_img.png", "_masks.tif"))
        if not mask_path.is_file():
            raise RuntimeError(f"Missing matching mask: {mask_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None or mask.ndim != 2:
            raise RuntimeError(f"Could not read image/mask pair: {image_path}")
        # Cellpose expects RGB arrays while OpenCV reads color images as BGR.
        images.append(image[:, :, ::-1])
        labels.append(mask.astype(np.int32))
    return images, labels, image_paths


def main() -> int:
    # The dataset contains separate train/ and validation/ directories with
    # matching *_img.png and *_masks.tif files.
    parser = argparse.ArgumentParser(description="Fine-tune Cellpose on prepared tiles.")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Directory containing train/ and validation/ tile folders")
    parser.add_argument("--output", type=Path, default=Path("models/cellpose_finetuned"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-train-tiles", type=int, default=None,
                        help="Limit tiles for a timing/overfitting pilot")
    parser.add_argument("--max-validation-tiles", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    import torch
    from cellpose import models, train

    # --cpu overrides CUDA detection so CPU fallback is explicit.
    use_gpu = not args.cpu and torch.cuda.is_available()
    train_images, train_labels, train_paths = load_tiles(
        args.dataset / "train", args.max_train_tiles
    )
    validation_dir = args.dataset / "validation"
    test_data = test_labels = None
    validation_paths = []
    if validation_dir.is_dir() and list(validation_dir.glob("*_img.png")):
        test_data, test_labels, validation_paths = load_tiles(
            validation_dir, args.max_validation_tiles
        )

    print(f"Device: {'CUDA' if use_gpu else 'CPU'}")
    print(f"Training tiles: {len(train_images)}; validation tiles: {len(validation_paths)}")
    print(f"Epochs: {args.epochs}")
    started = time.time()
    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam_v2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Cellpose adds its own models directory; copy the result back to the
    # exact --output path below so callers have an unambiguous model location.
    save_root = args.output.parent.parent if args.output.parent.name == "models" else args.output.parent
    filename, _, _ = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        load_files=False,
        channel_axis=2,
        batch_size=1,
        learning_rate=1e-5,
        n_epochs=args.epochs,
        compute_flows=True,
        save_path=str(save_root),
        model_name=args.output.name,
        save_every=max(1, min(100, args.epochs)),
        min_train_masks=1,
    )
    # Cellpose always writes under <save_path>/models. Copy it to the exact
    # path requested by the caller so --output models/name is unambiguous.
    if Path(filename).resolve() != args.output.resolve():
        shutil.copy2(filename, args.output)
    print(f"Training finished in {time.time() - started:.2f} seconds")
    print(f"Model files saved under {args.output.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
