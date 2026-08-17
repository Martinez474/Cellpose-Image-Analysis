#!/usr/bin/env python3
"""Run Cellpose on one image and write ImageJ-compatible polygon JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable
import cv2

import numpy as np


DEFAULT_MODEL = "cpsam_v2"
MODEL_CONFIG = Path(__file__).resolve().parent / "trained_model.txt"


class PredictionError(RuntimeError):
    """Raised when inference input or output is invalid."""


def resolve_model(model_argument: str | None) -> str:
    """Use an explicit model, the notebook-trained model, or Cellpose's default."""
    if model_argument:
        return model_argument
    if MODEL_CONFIG.is_file():
        configured = MODEL_CONFIG.read_text(encoding="utf-8").strip()
        if configured:
            model_path = Path(configured).expanduser()
            if not model_path.is_absolute():
                model_path = MODEL_CONFIG.parent / model_path
            model_path = model_path.resolve()
            if not model_path.is_file():
                raise PredictionError(
                    f"Configured trained model does not exist: {model_path}"
                )
            return str(model_path)
    return DEFAULT_MODEL


def normalize_polygon(outline: np.ndarray) -> list[list[int]]:
    """Convert a Cellpose/OpenCV x,y outline into JSON-safe integer points."""
    points = np.asarray(outline)
    if points.ndim != 2 or points.shape[1] != 2:
        raise PredictionError(f"Invalid polygon shape: {points.shape}")

    polygon: list[list[int]] = []
    for x, y in points:
        point = [int(x), int(y)]
        if not polygon or polygon[-1] != point:
            polygon.append(point)

    # ImageJ closes PolygonRoi objects, so a repeated final point is unnecessary.
    if len(polygon) > 1 and polygon[0] == polygon[-1]:
        polygon.pop()
    return polygon


def masks_to_predictions(
    masks: np.ndarray,
    outline_extractor: Callable[[np.ndarray], Iterable[np.ndarray]],
) -> list[dict[str, object]]:
    """Convert a 2D Cellpose instance-label mask into polygon predictions."""
    masks = np.asarray(masks)
    if masks.ndim != 2:
        raise PredictionError(
            f"Only 2D images are supported in this phase; got mask shape {masks.shape}."
        )

    predictions: list[dict[str, object]] = []
    for outline in outline_extractor(masks):
        polygon = normalize_polygon(outline)
        if len(polygon) < 3:
            continue
        predictions.append(
            {
                "label": "Cell",
                # Cellpose does not expose a calibrated per-object confidence.
                "confidence": 1.0,
                "polygon": polygon,
            }
        )
    return predictions


def write_predictions(path: Path, predictions: list[dict[str, object]]) -> None:
    """Atomically write the JSON consumed by the Java ImageJ plugin."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"predictions": predictions}

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run_cellpose(
    image_path: Path,
    model_name: str,
    use_gpu: bool,
    cellprob_threshold: float,
    flow_threshold: float,
    min_size: int,
) -> list[dict[str, object]]:
    """Load Cellpose lazily, run one 2D image, and build predictions."""
    try:
        from cellpose import io, models, utils
    except ModuleNotFoundError as error:
        if error.name == "cellpose":
            raise PredictionError(
                "Cellpose is not installed. Create the AI environment described in "
                "README.md and install requirements-ai.txt."
            ) from error
        raise

    image = io.imread(str(image_path))
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_name)
    result = model.eval(
        image,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
    )
    masks = result[0]
    return masks_to_predictions(
        masks,
        lambda value: utils.outlines_list(value, multiprocessing=False),
    )


def run_model(
    image_path: Path,
    model_name: str,
    use_gpu: bool,
    cellprob_threshold: float,
    flow_threshold: float,
    min_size: int,
) -> list[dict[str, object]]:
    """Run the scratch model, or Cellpose only when explicitly requested."""
    model_path = Path(model_name).expanduser()
    if model_path.is_file() and model_path.suffix == ".pt":
        from tools.simple_model import binary_mask_to_outlines, predict_binary_mask

        binary_mask = predict_binary_mask(image_path, model_path.resolve())


        debug_path = Path("/tmp/simple_model_mask.png")
        cv2.imwrite(str(debug_path), binary_mask)

        print(
            f"Scratch mask: shape={binary_mask.shape}, "
            f"foreground={np.count_nonzero(binary_mask)}/{binary_mask.size}, "
            f"saved={debug_path}"
        )
        outlines = binary_mask_to_outlines(binary_mask, minimum_area=max(1, min_size))
        return masks_to_predictions(binary_mask, lambda unused: outlines)
    return run_cellpose(
        image_path, model_name, use_gpu, cellprob_threshold, flow_threshold, min_size
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Cellpose and export polygons for the ImageJ plugin."
    )
    parser.add_argument("image", type=Path, help="Input 2D image, such as image.tif")
    parser.add_argument("output", type=Path, help="Destination predictions.json")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Built-in model name or path to fine-tuned weights. By default, uses "
            "trained_model.txt when present, otherwise cpsam_v2."
        ),
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU/MPS when available")
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--min-size", type=int, default=15)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    image_path = args.image.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not image_path.is_file():
        print(f"Prediction failed: input image does not exist: {image_path}", file=sys.stderr)
        return 1
    if image_path == output_path:
        print("Prediction failed: input and output paths must differ.", file=sys.stderr)
        return 1
    if args.min_size < 0:
        print("Prediction failed: --min-size cannot be negative.", file=sys.stderr)
        return 1

    try:
        predictions = run_model(
            image_path=image_path,
            model_name=resolve_model(args.model),
            use_gpu=args.gpu,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
            min_size=args.min_size,
        )
        write_predictions(output_path, predictions)
    except (PredictionError, OSError, ValueError) as error:
        print(f"Prediction failed: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {len(predictions)} predictions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
