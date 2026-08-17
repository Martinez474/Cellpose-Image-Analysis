# Cellpose Image Analysis

This project is being developed as an instance-segmentation pipeline:

1. pair each source micrograph with a ground-truth instance mask;
2. split whole source images into training, validation, and test groups;
3. create matched image/mask crops;
4. fine-tune and evaluate Cellpose; and
5. import model predictions into ImageJ/Fiji as ROIs.

The Java code in `src/` runs the Python Cellpose pipeline on the exact file open
in ImageJ. After Python writes `predictions.json`, it converts every polygon into
an ImageJ `PolygonRoi`, adds the ROIs to ROI Manager, and displays them over the
image. The dataset tool in `tools/` implements the validation and cropping
portions of steps 1–3.

Before running the ImageJ plugin, the current image must be backed by a saved
file. The plugin uses that file path as the input to `predict.py`; this guarantees
that prediction coordinates correspond to the displayed image. Compile and run
`src/Run_AI_Detection.java`, then wait for the completion message. Python output
is also written to the ImageJ log.

## Predict one image

`predict.py` provides the Python-to-Java boundary:

```bash
python predict.py image.tif predictions.json
```

It runs the current built-in `cpsam_v2` Cellpose model, converts each instance
mask boundary into an `[x, y]` polygon, and writes the JSON format consumed by
the ImageJ plugin. The first run may download built-in model weights.

To use a fine-tuned model later:

```bash
python predict.py image.tif predictions.json --model /absolute/path/to/model
```

Useful inference controls include `--gpu`, `--cellprob-threshold`,
`--flow-threshold`, and `--min-size`. Cellpose does not provide a calibrated
confidence for each individual object, so `confidence` is currently written as
`1.0` and should not be interpreted as a probability.

The repository pins pyenv Python 3.11.15 in `.python-version`. Create and
activate its project-local virtual environment, then install the dependencies:

```bash
pyenv exec python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-torch-cpu.txt
python -m pip install -r requirements-data.txt -r requirements-ai.txt
```

The separate PyTorch step intentionally installs CPU-only wheels and avoids a
large, unnecessary CUDA runtime download. A CUDA-specific PyTorch installation
can replace it later on a compatible NVIDIA training machine.

The custom thesis model will replace `cpsam_v2` after true instance masks have
been exported, paired crops have been prepared, and training has been validated.

## Simple from-scratch training notebook

`notebooks/Simple_Scratch_Training.ipynb` trains a compact convolutional model
from random weights. It pairs `Input Micrographs` with `Output Micrographs` and
extracts rough pseudo-masks from their yellow outlines. Start it with:

```bash
source .venv/bin/activate
jupyter lab notebooks/Simple_Scratch_Training.ipynb
```

Run all cells in order. Once `trained_model.txt` exists, both `predict.py` and
the ImageJ plugin use that trained model automatically. Delete
`trained_model.txt` to return to built-in `cpsam_v2`.

This model is intentionally rough. The annotated output micrographs contain
numbers and display graphics rather than clean instance masks, so they produce
noisy training labels. Replace them with true integer masks before reporting
scientific accuracy.

## Required mask format

A mask must be a single-channel integer PNG or TIFF with the same dimensions as
its image. Pixel value `0` is background; values `1`, `2`, `3`, and so on identify
separate objects. An RGB image containing yellow outlines, object numbers, or the
original micrograph is an annotation preview, not a training mask.

Keep masks lossless. Do not save masks as JPEG.

## Source manifest

Copy `dataset_manifest.example.csv` and add one row per **original image**, not
one row per crop. Required columns are:

- `sample_id`: unique filesystem-safe identifier;
- `image_path`: input micrograph;
- `mask_path`: matching integer mask; and
- `split`: `train`, `validation`, or `test`.

`condition` is optional but recommended for auditing experimental coverage.
Paths may be absolute or relative to the manifest file.

Assign the split before cropping. All crops from one source image remain in the
same split. The tool also detects byte-identical images assigned to different
splits.

## Create crops

Install the small data-preparation dependencies:

```bash
python3 -m pip install -r requirements-data.txt
```

Then run:

```bash
python3 tools/prepare_cellpose_dataset.py \
  --manifest dataset_manifest.csv \
  --output prepared_dataset \
  --tile-size 256
```

The output directory must be empty. It receives `train/`, `validation/`, and
`test/` folders containing Cellpose-compatible `_img.png`/`_masks.tif` pairs.
It also receives `crops.csv`, which records source provenance, crop coordinates,
object counts, and how many objects touch each crop boundary.

For a 500x500 image and 256x256 crops, the final crops overlap by 12 pixels so
that no source pixels are discarded or artificially padded.

## Tests

```bash
python3 -m unittest discover -v
```
