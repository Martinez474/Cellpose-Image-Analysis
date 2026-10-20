#!/usr/bin/env bash
set -euo pipefail

# Extract matching image/mask windows from the validated manifest. The split
# supplied by the manifest is preserved so source images cannot cross splits.
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 source_manifest.tsv output_dir [tile_size] [overlap]" >&2
  exit 2
fi

MANIFEST=$1
OUTPUT_DIR=$2
TILE_SIZE=${3:-256}
OVERLAP=${4:-16}
# Overlap preserves context at tile boundaries; reject an invalid step size.
STEP=$((TILE_SIZE - OVERLAP))
[[ $STEP -gt 0 ]] || { echo "overlap must be smaller than tile size" >&2; exit 2; }
command -v gdal_translate >/dev/null || { echo "gdal_translate is required" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR/train" "$OUTPUT_DIR/validation" "$OUTPUT_DIR/test"
index=0
while IFS=$'\t' read -r status split image annotation mask score instances reason; do
  # Excluded manifest rows are retained for review but are never tiled.
  [[ "$status" == "ok" ]] || continue
  [[ -n "$mask" && -f "$image" && -f "$mask" ]] || {
    echo "Skipping missing pair: $image" >&2
    continue
  }
  size=$(gdalinfo "$image" | awk '/Size is/{gsub(/[(),]/, "", $3); gsub(/[(),]/, "", $4); print $3, $4; exit}')
  read -r width height <<< "$size"
  [[ -n "$width" && -n "$height" ]] || { echo "Could not read size: $image" >&2; exit 1; }
  stem=$(basename "$image"); stem=${stem%.*}; stem=$(echo "$stem" | tr -cs 'A-Za-z0-9_.-' '_')
  split_dir="$OUTPUT_DIR/$split"
  # Extract the same pixel window from the source image and its label mask.
  for ((y=0; y<height; y+=STEP)); do
    ch=$TILE_SIZE; (( y + ch > height )) && ch=$((height-y))
    for ((x=0; x<width; x+=STEP)); do
      cw=$TILE_SIZE; (( x + cw > width )) && cw=$((width-x))
      name=$(printf '%s_%04d_%04d' "$stem" "$x" "$y")
      gdal_translate -q -of PNG -srcwin "$x" "$y" "$cw" "$ch" "$image" "$split_dir/${name}_img.png"
      gdal_translate -q -of GTiff -ot UInt16 -srcwin "$x" "$y" "$cw" "$ch" "$mask" "$split_dir/${name}_masks.tif"
    done
  done
  index=$((index+1))
  echo "Tiled $split: $image"
done < <(tail -n +2 "$MANIFEST")
echo "Tiled $index source image/mask pairs into $OUTPUT_DIR"
