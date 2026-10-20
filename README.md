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

See [docs/diagrams.md](docs/diagrams.md) for Mermaid and D2 diagrams of the
Cellpose and ellipse-detection pipelines.

Before running the ImageJ plugin, the current image must be backed by a saved
file. The plugin uses that file path as the input to `predict.py`; this guarantees
that prediction coordinates correspond to the displayed image. Compile and run
`src/Run_AI_Detection.java`, then wait for the completion message. Python output
is also written to the ImageJ log.

## Setup on a new computer

The repository can live anywhere; it does not need to be inside ImageJ/Fiji's
installation directory. The plugin locates the project by finding `predict.py`.

1. Install Git, Python 3.11, and ImageJ or Fiji.
2. Clone the repository and enter it:

   ```bash
   git clone <repository-url>
   cd Cellpose-Image-Analysis
   ```

3. Create the virtual environment and install the common dependencies:

   ```bash
   python3 -m venv .venv       # Windows: py -3.11 -m venv .venv
   source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements-ai.txt -r requirements-data.txt
   ```

   With an NVIDIA GPU, install the CUDA PyTorch packages instead of
   the CPU packages:

   ```bash
   python -m pip install -r requirements-torch-cuda.txt
   ```

   On a Mac, or on a computer without NVIDIA CUDA, install the platform's
   normal PyTorch build 

   ```bash
   python -m pip install torch torchvision
   ```

   Cellpose will use Apple MPS where supported and otherwise use the CPU.

4. In ImageJ/Fiji, choose **Plugins → Compile and Run…**, select
   `src/Run_AI_Detection.java`, and run it. Gson must be available to ImageJ;
   copy `lib/gson-2.14.0.jar` into ImageJ/Fiji's `jars/` folder if the compiler
   reports missing `com.google.gson` classes. The project itself does not need
   to be copied into the ImageJ `plugins` folder.

For a permanent menu entry, compile all Java files and place the resulting
`.class` files in ImageJ/Fiji's `plugins` folder, and place Gson in `jars/`.
Keep `cellpose-config.json` beside the installed plugin (or start ImageJ from
the project directory) if the project is stored somewhere unrelated to the
ImageJ installation.

## Project paths and optional configuration

The plugin automatically searches the current directory and its parent
directories for `predict.py`, so the project can be moved to another computer.
It also detects the platform-specific virtual-environment executable
(`.venv/bin/python` on Linux/macOS and `.venv/Scripts/python.exe` on Windows).
If ImageJ was launched from a different location and automatic detection cannot
find the project, the plugin asks you to select the project folder once.

For a custom layout, copy `cellpose-config.example.json` to
`cellpose-config.json` and edit its paths. Relative paths are resolved from the
configuration file's directory; absolute paths are also allowed. The
configuration file is ignored by Git so machine-specific paths stay local.
Set `cellposeModel` to a fine-tuned Cellpose model path when you want the
ImageJ Cellpose plugin to use it instead of `cpsam_v2`.

## Human masks and fine-tuning preparation

The human `Output Micrographs` use yellow outlines. The preparation utility
extracts those closed outlines, assigns each connected cell a unique integer
label, and pairs the result with its original image:

```bash
python tools/prepare_finetune_masks.py \
  --input-dir "../Input Micrographs" \
  --annotation-dir "../Output Micrographs" \
  --output-dir cellpose_finetune
```

It writes a source-level manifest and masks under `cellpose_finetune/source_masks`.
Pairs with mismatched dimensions are excluded for review. Create identical
image/mask tiles with GDAL using:

```bash
tools/tile_cellpose_dataset_gdal.sh \
  cellpose_finetune/source_manifest.tsv \
  cellpose_finetune/tiles_v2 256 16
```

The last two arguments are tile size and overlap in pixels. This uses
`gdal_translate`, not `gdaldem`: `gdaldem` is for terrain analysis. Splitting
is performed before tiling so crops from one original image cannot leak between
training, validation, and test sets.

### One-cell-per-sample crops

To create a dataset where every sample contains one human-annotated cell and
one matching mask, use:

```bash
python tools/prepare_cell_crops.py \
  --input-dir "../Input Micrographs" \
  --annotation-dir "../Output Micrographs" \
  --output-dir cellpose_cell_crops \
  --padding 8 \
  --crop-size 0
```

Each split contains matching files such as:

```text
sample_cell_0001_img.png
sample_cell_0001_masks.tif
```

The mask contains only that cell, with foreground value `1` and background
value `0`. `--crop-size 0` creates a tight square around each cell; increase
`--padding` for more surrounding context. The split is assigned by original
source image before individual cells are extracted.

To tune yellow-outline extraction, adjust the mask command without changing
the source code. For example, a stricter color selection is:

```bash
python tools/prepare_finetune_masks.py \
  --input-dir "../Input Micrographs" \
  --annotation-dir "../Output Micrographs" \
  --output-dir cellpose_finetune_tight \
  --yellow-saturation-min 140 \
  --yellow-value-min 120 \
  --yellow-close-size 3 \
  --minimum-area 80
```

The hue range is OpenCV's `0`–`179` scale. Increase
`--yellow-saturation-min` to reject more background colors; increase
`--yellow-close-size` only when outlines have small gaps. Use a new output
directory for each experiment so masks can be compared.

## Cellpose fine-tuning pilot

After reviewing the masks and tiles, use the fine-tuning script. A one-tile,
one-epoch run is useful for checking that training works and measuring speed,
but it will overfit and is not a meaningful model-quality test:

```bash
python tools/finetune_cellpose.py \
  --dataset cellpose_finetune/tiles_v2 \
  --output models/cellpose_pilot \
  --epochs 1 \
  --max-train-tiles 1 \
  --max-validation-tiles 1
```

For a real experiment, increase the epoch count and remove the tile limits:

```bash
python tools/finetune_cellpose.py \
  --dataset cellpose_finetune/tiles_v2 \
  --output models/cellpose_finetuned \
  --epochs 100
```

Training time depends on the GPU, tile count, and epochs. The pilot gives the
best estimate for the full run: approximately multiply its time by the number
of training tiles and by the full epoch count, allowing for validation overhead.

## Experimental ellipse detector

`predict_ellipse.py` is a simple OpenCV baseline separate from Cellpose. It
thresholds the image, fits ellipses to contours, and writes the same
`predictions`/`polygon` JSON structure used by the ROI importer. Try it from
the project directory with:

```bash
python predict_ellipse.py image.jpg ellipse_predictions.json \
  --min-area 40 --min-axis-ratio 0.20 --max-fit-error 0.20
```

The thresholds are intentionally command-line options so they can be tuned for
different images. In ImageJ, compile and run
`src/Run_AI_DetectionEllipse.java`; it uses the ellipse detector and loads its
results through the existing Gson/ROI workflow. Results are written to
`ellipse_predictions.json`, leaving the Cellpose `predictions.json` untouched.

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

The repository includes `.python-version` for pyenv users, but pyenv is
optional. The setup commands above create the project-local environment. Use
`requirements-torch-cpu.txt` only when a Linux CPU-only installation is needed;
use `requirements-torch-cuda.txt` for the supported NVIDIA CUDA installation.

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
