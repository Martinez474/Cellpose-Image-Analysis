"""Small from-scratch segmentation model and pseudo-label preparation."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = PROJECT_ROOT.parent
INPUT_DIRECTORY = THESIS_ROOT / "Input Micrographs"
OUTPUT_DIRECTORY = THESIS_ROOT / "Output Micrographs"
DATASET_DIRECTORY = PROJECT_ROOT / "simple_dataset"
MODEL_DIRECTORY = PROJECT_ROOT / "models"
MODEL_CONFIG = PROJECT_ROOT / "trained_model.txt"
IMAGE_SIZE = 256


def normalized_name(path: Path) -> str:
    value = path.stem.lower()
    for phrase in ("grain pic", "grain", "results", "jpg", "png", "pic"):
        value = value.replace(phrase, " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def pair_micrographs() -> list[tuple[Path, Path, float]]:
    """Pair all input/output files one-to-one by normalized filename similarity."""
    inputs = sorted(path for path in INPUT_DIRECTORY.iterdir() if path.is_file())
    outputs = sorted(path for path in OUTPUT_DIRECTORY.iterdir() if path.is_file())
    if not inputs or len(inputs) != len(outputs):
        raise RuntimeError(
            f"Expected equally sized non-empty folders; found {len(inputs)} inputs "
            f"and {len(outputs)} outputs."
        )
    scores = np.array([
        [SequenceMatcher(None, normalized_name(output), normalized_name(image)).ratio()
         for image in inputs]
        for output in outputs
    ])
    output_indices, input_indices = linear_sum_assignment(-scores)
    pairs = [
        (inputs[input_index], outputs[output_index], float(scores[output_index, input_index]))
        for output_index, input_index in zip(output_indices, input_indices)
    ]
    return sorted(pairs, key=lambda pair: pair[0].name)


def extract_pseudo_mask(annotation_path: Path) -> np.ndarray:
    """Fill contours enclosed by the yellow outlines in an annotated preview."""
    annotation = cv2.imread(str(annotation_path), cv2.IMREAD_COLOR)
    if annotation is None:
        raise RuntimeError(f"Could not read annotation: {annotation_path}")
    hsv = cv2.cvtColor(annotation, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(
        hsv,
        np.array([20, 100, 120], dtype=np.uint8),
        np.array([45, 255, 255], dtype=np.uint8),
    )
    yellow = cv2.morphologyEx(
        yellow, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
    )
    contours, hierarchy = cv2.findContours(
        yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros(yellow.shape, dtype=np.uint8)
    if hierarchy is None:
        return mask
    maximum_area = mask.size * 0.20
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        parent = hierarchy[0][index][3]
        if parent >= 0 and 80 <= area <= maximum_area:
            cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def prepare_real_dataset(test_fraction: float = 0.2) -> dict[str, object]:
    """Create aligned 256px image/pseudo-mask pairs and a reproducible split."""
    pairs = pair_micrographs()
    accepted: list[tuple[Path, Path, float]] = []
    excluded: list[dict[str, object]] = []
    for image_path, annotation_path, score in pairs:
        with Image.open(image_path) as image, Image.open(annotation_path) as annotation:
            if image.size != annotation.size:
                excluded.append({
                    "image": str(image_path),
                    "annotation": str(annotation_path),
                    "image_size": list(image.size),
                    "annotation_size": list(annotation.size),
                    "reason": "dimension mismatch",
                })
                continue
        accepted.append((image_path, annotation_path, score))

    if len(accepted) < 5:
        raise RuntimeError("Too few aligned input/output pairs were found.")

    for split in ("train", "test"):
        (DATASET_DIRECTORY / split).mkdir(parents=True, exist_ok=True)

    # Deterministic every-fifth-image test split distributes filenames/conditions.
    records: list[dict[str, object]] = []
    for index, (image_path, annotation_path, score) in enumerate(accepted):
        split = "test" if index % max(2, round(1 / test_fraction)) == 0 else "train"
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        pseudo_mask = extract_pseudo_mask(annotation_path)
        pseudo_mask = cv2.resize(
            pseudo_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST
        )
        if np.count_nonzero(pseudo_mask) == 0:
            excluded.append({
                "image": str(image_path),
                "annotation": str(annotation_path),
                "reason": "no closed yellow contours",
            })
            continue
        stem = f"sample_{index:03d}"
        image_output = DATASET_DIRECTORY / split / f"{stem}_img.png"
        mask_output = DATASET_DIRECTORY / split / f"{stem}_mask.png"
        image.save(image_output)
        Image.fromarray(pseudo_mask).save(mask_output)
        records.append({
            "split": split,
            "image": str(image_path),
            "annotation": str(annotation_path),
            "filename_score": score,
            "prepared_image": str(image_output),
            "prepared_mask": str(mask_output),
        })

    manifest = {"records": records, "excluded": excluded}
    (DATASET_DIRECTORY / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


class TinySegmentationNet(nn.Module):
    """A compact encoder-decoder initialized entirely from random weights."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, image):
        return self.decoder(self.encoder(image))


class PreparedDataset(Dataset):
    def __init__(self, directory: Path):
        self.images = sorted(directory.glob("*_img.png"))
        if not self.images:
            raise RuntimeError(f"No prepared images in {directory}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = image_path.with_name(image_path.name.replace("_img.png", "_mask.png"))
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(mask_path) as source:
            mask = (np.asarray(source.convert("L")) > 0).astype(np.float32)
        return (
            torch.from_numpy(image.transpose(2, 0, 1)),
            torch.from_numpy(mask[None, :, :]),
        )


def dice_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    intersection = (predictions * targets).sum()
    return (2 * intersection + 1) / (predictions.sum() + targets.sum() + 1)


def train_simple_model(epochs: int = 5) -> tuple[Path, list[dict[str, float]]]:
    """Train TinySegmentationNet from scratch on the prepared pseudo-labels."""
    torch.manual_seed(7)
    prepare_real_dataset()
    train_loader = DataLoader(PreparedDataset(DATASET_DIRECTORY / "train"), batch_size=2, shuffle=True)
    test_loader = DataLoader(PreparedDataset(DATASET_DIRECTORY / "test"), batch_size=1)
    model = TinySegmentationNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    positive_weight = torch.tensor([2.0])
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(images), masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        model.eval()
        scores = []
        with torch.no_grad():
            for images, masks in test_loader:
                scores.append(float(dice_score(model(images), masks).item()))
        row = {
            "epoch": float(epoch + 1),
            "train_loss": total_loss / len(train_loader),
            "test_dice": sum(scores) / len(scores),
        }
        history.append(row)
        print(row)

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_path = (MODEL_DIRECTORY / "simple_scratch_model.pt").resolve()
    torch.save({
        "model_type": "tiny_binary_segmentation",
        "state_dict": model.state_dict(),
        "image_size": IMAGE_SIZE,
        "threshold": 0.5,
        "history": history,
    }, model_path)
    MODEL_CONFIG.write_text(str(model_path) + "\n", encoding="utf-8")
    (MODEL_DIRECTORY / "simple_training_metrics.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return model_path, history


def predict_binary_mask(image_path: Path, model_path: Path) -> np.ndarray:
    """Run the compact model and return a binary mask at original resolution."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "tiny_binary_segmentation":
        raise RuntimeError(f"Unsupported simple model: {model_path}")
    size = int(checkpoint["image_size"])
    model = TinySegmentationNet()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with Image.open(image_path) as source:
        original_size = source.size
        resized = source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1))[None, :, :, :]
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].numpy()
    binary = (probability >= float(checkpoint.get("threshold", 0.5))).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.resize(binary, original_size, interpolation=cv2.INTER_NEAREST)


def binary_mask_to_outlines(mask: np.ndarray, minimum_area: int = 40) -> list[np.ndarray]:
    """Split touching foreground regions and return one outline per object."""

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    binary = (mask > 0).astype(np.uint8)

    # Remove small isolated noise.
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    if np.count_nonzero(binary) == 0:
        return []

    # Distance from each foreground pixel to the nearest background pixel.
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Generate conservative foreground seeds.
    max_distance = float(distance.max())
    if max_distance <= 0:
        return []

    sure_foreground = (distance >= 0.35 * max_distance).astype(np.uint8)

    # Label the seed regions.
    component_count, markers = cv2.connectedComponents(sure_foreground)

    if component_count <= 1:
        contours, _ = cv2.findContours(
            binary * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    else:
        # Watershed requires background=1 and unknown=0.
        markers = markers + 1

        sure_background = cv2.dilate(
            binary,
            np.ones((3, 3), np.uint8),
            iterations=2,
        )

        unknown = sure_background - sure_foreground
        markers[unknown > 0] = 0

        watershed_image = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(watershed_image, markers.astype(np.int32))

        contours = []

        for label in np.unique(markers):
            if label <= 1:
                continue

            instance = np.zeros(binary.shape, dtype=np.uint8)
            instance[markers == label] = 255

            instance_contours, _ = cv2.findContours(
                instance,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            contours.extend(instance_contours)

    outlines = []

    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue

        epsilon = 0.002 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        ).reshape(-1, 2)

        if len(simplified) >= 3:
            outlines.append(simplified)

    return outlines
