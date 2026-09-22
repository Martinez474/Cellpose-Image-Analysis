#!/usr/bin/env python3
"""Detect ellipse-shaped objects with OpenCV and export ImageJ polygons.

This is a deliberately simple, tunable baseline. It is not a trained neural
network: contours are extracted from both threshold polarities, fitted with
``cv2.fitEllipse``, and retained when their size, aspect ratio, and normalized
ellipse-fit error pass the requested thresholds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from predict import write_predictions


def ellipse_fit_error(contour: np.ndarray, ellipse: tuple) -> float:
    """Return mean normalized radial error for points on a fitted ellipse."""
    (cx, cy), (width, height), angle = ellipse
    major = max(width, height) / 2.0
    minor = min(width, height) / 2.0
    if major <= 0 or minor <= 0:
        return float("inf")

    # OpenCV's angle follows the first returned axis. Rotate points so x is
    # aligned with the major axis before measuring normalized radius.
    theta = np.deg2rad(float(angle))
    if width < height:
        theta += np.pi / 2.0
    points = contour.reshape(-1, 2).astype(np.float64)
    dx = points[:, 0] - float(cx)
    dy = points[:, 1] - float(cy)
    x_major = dx * np.cos(theta) + dy * np.sin(theta)
    y_minor = -dx * np.sin(theta) + dy * np.cos(theta)
    radius = np.sqrt((x_major / major) ** 2 + (y_minor / minor) ** 2)
    return float(np.mean(np.abs(radius - 1.0)))


def ellipse_polygon(ellipse: tuple) -> list[list[int]]:
    """Approximate a fitted ellipse with ImageJ-compatible polygon points."""
    (cx, cy), (width, height), angle = ellipse
    major = max(width, height) / 2.0
    minor = min(width, height) / 2.0
    draw_angle = float(angle) + (90.0 if width < height else 0.0)
    points = cv2.ellipse2Poly(
        (int(round(cx)), int(round(cy))),
        (max(1, int(round(major))), max(1, int(round(minor)))),
        int(round(draw_angle)), 0, 360, 5,
    )
    return [[int(x), int(y)] for x, y in points]


def detect_ellipses(
    image_path: Path,
    *,
    threshold: int | None = None,
    min_area: float = 40.0,
    max_area: float = 10.0,
    min_axis: float = 4.0,
    min_axis_ratio: float = 0.20,
    max_fit_error: float = 0.50,
    blur: int = 3,
) -> list[dict[str, object]]:
    """Find ellipse-like contours using both bright and dark threshold masks."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if blur < 0 or blur % 2 == 0:
        raise ValueError("--blur must be a positive odd number or zero")
    if blur:
        image = cv2.GaussianBlur(image, (blur, blur), 0)

    threshold_value = int(threshold) if threshold is not None else int(
        cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    )
    predictions: list[dict[str, object]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for polarity in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, binary = cv2.threshold(image, threshold_value, 255, polarity)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            # A contour touching the image edge is usually the thresholded
            # background, not an individual cell.
            x, y, width_box, height_box = cv2.boundingRect(contour)
            if x <= 0 or y <= 0 or x + width_box >= image.shape[1] or y + height_box >= image.shape[0]:
                continue
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area or len(contour) < 5:
                continue
            ellipse = cv2.fitEllipse(contour)
            (_, _), (width, height), _ = ellipse
            major = max(float(width), float(height))
            minor = min(float(width), float(height))
            if minor < min_axis or major <= 0 or minor / major < min_axis_ratio:
                continue
            error = ellipse_fit_error(contour, ellipse)
            if error > max_fit_error:
                continue
            polygon = ellipse_polygon(ellipse)
            if len(polygon) < 5:
                continue
            center = ellipse[0]
            key = (round(center[0] / 3), round(center[1] / 3), round(major / 3), round(minor / 3))
            if key in seen:
                continue
            seen.add(key)
            predictions.append({
                "label": "Ellipse",
                "confidence": round(max(0.0, 1.0 - error / max_fit_error), 4),
                "polygon": polygon,
            })
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect ellipse-shaped objects with OpenCV.")
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=int, default=None, help="Gray threshold; default is Otsu")
    parser.add_argument("--min-area", type=float, default=40.0)
    parser.add_argument("--max-area", type=float, default=1_000_000.0)
    parser.add_argument("--min-axis", type=float, default=4.0)
    parser.add_argument("--min-axis-ratio", type=float, default=0.20)
    parser.add_argument("--max-fit-error", type=float, default=0.20)
    parser.add_argument("--blur", type=int, default=3)
    args = parser.parse_args(argv)
    image_path = args.image.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not image_path.is_file():
        print(f"Ellipse prediction failed: input image does not exist: {image_path}", file=sys.stderr)
        return 1
    if image_path == output_path:
        print("Ellipse prediction failed: input and output paths must differ.", file=sys.stderr)
        return 1
    timer = time.time()
    try:
        predictions = detect_ellipses(
            image_path, threshold=args.threshold, min_area=args.min_area,
            max_area=args.max_area, min_axis=args.min_axis,
            min_axis_ratio=args.min_axis_ratio, max_fit_error=args.max_fit_error,
            blur=args.blur,
        )
        write_predictions(output_path, predictions)
    except (OSError, ValueError, cv2.error) as error:
        print(f"Ellipse prediction failed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {len(predictions)} ellipse predictions to {output_path}")
    print(f"Prediction time: {time.time() - timer:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
