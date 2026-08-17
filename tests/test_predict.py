import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from predict import (
    DEFAULT_MODEL,
    PredictionError,
    masks_to_predictions,
    normalize_polygon,
    resolve_model,
    write_predictions,
)


class PredictTests(unittest.TestCase):
    def test_uses_builtin_model_when_no_trained_model_is_configured(self):
        self.assertEqual(resolve_model(None), DEFAULT_MODEL)

    def test_normalizes_points_and_removes_duplicate_closure(self):
        outline = np.array([[1, 2], [4, 2], [4, 8], [1, 2]])

        self.assertEqual(normalize_polygon(outline), [[1, 2], [4, 2], [4, 8]])

    def test_converts_outlines_to_java_json_contract(self):
        masks = np.array([[0, 1], [2, 2]], dtype=np.uint16)
        outlines = [
            np.array([[1, 2], [4, 2], [4, 8], [1, 8]]),
            np.array([[10, 11], [12, 11]]),
        ]

        predictions = masks_to_predictions(masks, lambda unused: outlines)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["label"], "Cell")
        self.assertEqual(predictions[0]["confidence"], 1.0)
        self.assertEqual(
            predictions[0]["polygon"], [[1, 2], [4, 2], [4, 8], [1, 8]]
        )

    def test_rejects_3d_masks(self):
        with self.assertRaisesRegex(PredictionError, "Only 2D images"):
            masks_to_predictions(np.zeros((2, 3, 4)), lambda unused: [])

    def test_writes_expected_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "predictions.json"
            predictions = [{
                "label": "Cell",
                "confidence": 1.0,
                "polygon": [[1, 2], [3, 4], [5, 6]],
            }]

            write_predictions(output, predictions)

            with output.open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"predictions": predictions})


if __name__ == "__main__":
    unittest.main()
