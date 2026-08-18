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

   On Linux with an NVIDIA GPU, install the CUDA PyTorch packages instead of
   the CPU packages:

   ```bash
   python -m pip install -r requirements-torch-cuda.txt
   ```

   On a Mac, or on a computer without NVIDIA CUDA, install the platform's
   normal PyTorch build (`python -m pip install torch torchvision`). Cellpose
   will use Apple MPS where supported and otherwise use the CPU.

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

For a custom layout, copy `cellpose-config.example.json` to
`cellpose-config.json` and edit its paths. Relative paths are resolved from the
configuration file's directory; absolute paths are also allowed. The
configuration file is ignored by Git so machine-specific paths stay local.

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
