import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.prepare_cellpose_dataset import DatasetError, prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_manifest(self, rows):
        path = self.root / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "image_path", "mask_path", "split", "condition"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def create_pair(self, name="sample", rgb_mask=False):
        image_path = self.root / f"{name}.png"
        mask_path = self.root / f"{name}_masks.png"

        image = np.full((500, 500, 3), 180, dtype=np.uint8)
        mask = np.zeros((500, 500), dtype=np.uint16)
        mask[50:150, 50:150] = 4
        mask[300:450, 300:450] = 9
        Image.fromarray(image).save(image_path)
        if rgb_mask:
            Image.fromarray(np.repeat(mask[:, :, None].astype(np.uint8), 3, axis=2)).save(
                mask_path
            )
        else:
            Image.fromarray(mask).save(mask_path)
        return image_path, mask_path

    def test_creates_four_matched_256_pixel_crops(self):
        image_path, mask_path = self.create_pair()
        manifest = self.write_manifest(
            [{
                "sample_id": "source 01",
                "image_path": image_path,
                "mask_path": mask_path,
                "split": "train",
                "condition": "control",
            }]
        )

        count = prepare_dataset(manifest, self.root / "dataset")

        self.assertEqual(count, 4)
        with (self.root / "dataset/crops.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({(row["x"], row["y"]) for row in rows}, {
            ("0", "0"), ("244", "0"), ("0", "244"), ("244", "244")
        })
        for row in rows:
            with Image.open(self.root / "dataset" / row["image_crop"]) as image:
                with Image.open(self.root / "dataset" / row["mask_crop"]) as mask:
                    self.assertEqual(image.size, (256, 256))
                    self.assertEqual(mask.size, image.size)

    def test_rejects_rgb_annotation_overlay_as_mask(self):
        image_path, mask_path = self.create_pair(rgb_mask=True)
        manifest = self.write_manifest(
            [{
                "sample_id": "source01",
                "image_path": image_path,
                "mask_path": mask_path,
                "split": "train",
            }]
        )

        with self.assertRaisesRegex(DatasetError, "RGB annotation overlay"):
            prepare_dataset(manifest, self.root / "dataset")

    def test_rejects_identical_images_in_different_splits(self):
        first_image, first_mask = self.create_pair("first")
        second_image, second_mask = self.create_pair("second")
        manifest = self.write_manifest([
            {
                "sample_id": "first",
                "image_path": first_image,
                "mask_path": first_mask,
                "split": "train",
            },
            {
                "sample_id": "second",
                "image_path": second_image,
                "mask_path": second_mask,
                "split": "test",
            },
        ])

        with self.assertRaisesRegex(DatasetError, "different splits"):
            prepare_dataset(manifest, self.root / "dataset")


if __name__ == "__main__":
    unittest.main()
