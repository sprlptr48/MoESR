#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$ROOT_DIR/dataset"
ZIP_DIR="$DATA_DIR/_zips"
EXT_DIR="$DATA_DIR/_extracted"

TRAIN_LR="$DATA_DIR/train/LR"
TRAIN_HR="$DATA_DIR/train/HR"
VAL_LR="$DATA_DIR/val/LR"
VAL_HR="$DATA_DIR/val/HR"

DIV2K_URLS=(
  "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
  "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
  "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"
  "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip"
)

log() {
  printf '\n[%s] %s\n' "$1" "$2"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

install_pkg() {
  if need_cmd apt-get; then
    sudo apt-get update
    sudo apt-get install -y "$@"
    return
  fi
  if need_cmd dnf; then
    sudo dnf install -y "$@"
    return
  fi
  if need_cmd yum; then
    sudo yum install -y "$@"
    return
  fi
  if need_cmd pacman; then
    sudo pacman -Sy --noconfirm "$@"
    return
  fi
  echo "No supported package manager found. Install manually: $*" >&2
  exit 1
}

ensure_tooling() {
  if ! need_cmd aria2c; then
    log "0/5" "Installing aria2c..."
    install_pkg aria2
  fi

  if ! need_cmd 7z && ! need_cmd unzip; then
    log "0/5" "Installing archive extraction tools..."
    if need_cmd apt-get; then
      install_pkg p7zip-full unzip
    elif need_cmd dnf || need_cmd yum; then
      install_pkg p7zip p7zip-plugins unzip
    elif need_cmd pacman; then
      install_pkg p7zip unzip
    fi
  fi

  if ! need_cmd python3; then
    echo "python3 is required but not installed." >&2
    exit 1
  fi
}

extract_best() {
  local zip_path="$1"
  local dest_dir="$2"
  mkdir -p "$dest_dir"
  if need_cmd 7z; then
    echo "  Extracting (7z): $zip_path"
    7z x "$zip_path" "-o$dest_dir" -y >/dev/null
  elif need_cmd unzip; then
    echo "  Extracting (unzip): $zip_path"
    unzip -q -o "$zip_path" -d "$dest_dir"
  else
    echo "No extractor available for $zip_path" >&2
    exit 1
  fi
}

mkdir -p "$TRAIN_LR" "$TRAIN_HR" "$VAL_LR" "$VAL_HR" "$ZIP_DIR" "$EXT_DIR"

ensure_tooling

log "1/5" "Downloading DIV2K..."
for url in "${DIV2K_URLS[@]}"; do
  echo "  -> $url"
  aria2c \
    --dir="$ZIP_DIR" \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=5M \
    --continue=true \
    --file-allocation=none \
    "$url"
done

log "2/5" "Extracting archives..."
extract_best "$ZIP_DIR/DIV2K_train_HR.zip" "$EXT_DIR/train_hr"
extract_best "$ZIP_DIR/DIV2K_valid_HR.zip" "$EXT_DIR/valid_hr"
extract_best "$ZIP_DIR/DIV2K_train_LR_bicubic_X4.zip" "$EXT_DIR/train_lr"
extract_best "$ZIP_DIR/DIV2K_valid_LR_bicubic_X4.zip" "$EXT_DIR/valid_lr"

log "3/5" "Organising into dataset/..."
find "$EXT_DIR/train_hr/DIV2K_train_HR" -maxdepth 1 -type f -name '*.png' -exec mv -t "$TRAIN_HR" {} +
find "$EXT_DIR/valid_hr/DIV2K_valid_HR" -maxdepth 1 -type f -name '*.png' -exec mv -t "$VAL_HR" {} +
find "$EXT_DIR/train_lr/DIV2K_train_LR_bicubic/X4" -maxdepth 1 -type f -name '*.png' -exec mv -t "$TRAIN_LR" {} +
find "$EXT_DIR/valid_lr/DIV2K_valid_LR_bicubic/X4" -maxdepth 1 -type f -name '*.png' -exec mv -t "$VAL_LR" {} +

log "4/5" "Game screenshots..."
SCREENSHOT_SRC="$ROOT_DIR/my_screenshots"
if [[ -d "$SCREENSHOT_SRC" ]]; then
  echo "  Found my_screenshots/ - generating LR pairs with Python..."
  SCREENSHOT_SRC="$SCREENSHOT_SRC" TRAIN_HR="$TRAIN_HR" TRAIN_LR="$TRAIN_LR" python3 <<'PY'
import os
import pathlib
from PIL import Image

src = pathlib.Path(os.environ["SCREENSHOT_SRC"])
hr = pathlib.Path(os.environ["TRAIN_HR"])
lr = pathlib.Path(os.environ["TRAIN_LR"])

images = sorted(list(src.glob("*.png")) + list(src.glob("*.jpg")) + list(src.glob("*.jpeg")))
print(f"  Found {len(images)} screenshots")
for image_path in images:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if width < 512 or height < 512:
        print(f"  Skipping {image_path.name} (too small: {width}x{height})")
        continue

    width = (width // 4) * 4
    height = (height // 4) * 4
    image = image.crop((0, 0, width, height))
    out_name = f"screenshot_{image_path.stem}.png"
    image.save(hr / out_name)
    image.resize((width // 4, height // 4), Image.BICUBIC).save(lr / out_name)
    print(f"  Processed {image_path.name} -> HR {width}x{height} / LR {width // 4}x{height // 4}")

print("  Done.")
PY
else
  echo "  No my_screenshots/ folder found - skipping."
  echo "  (Drop PNGs into codex/my_screenshots/ and re-run to add them)"
fi

log "5/5" "Cleaning up staging folders..."
rm -rf "$EXT_DIR"

train_hr_count=$(find "$TRAIN_HR" -maxdepth 1 -type f | wc -l | tr -d ' ')
train_lr_count=$(find "$TRAIN_LR" -maxdepth 1 -type f | wc -l | tr -d ' ')
val_hr_count=$(find "$VAL_HR" -maxdepth 1 -type f | wc -l | tr -d ' ')
val_lr_count=$(find "$VAL_LR" -maxdepth 1 -type f | wc -l | tr -d ' ')

printf '\nDone.\n'
printf '  train/HR : %s images\n' "$train_hr_count"
printf '  train/LR : %s images\n' "$train_lr_count"
printf '  val/HR   : %s images\n' "$val_hr_count"
printf '  val/LR   : %s images\n' "$val_lr_count"

if [[ "$train_hr_count" != "$train_lr_count" ]]; then
  printf '\nWARNING: HR/LR counts do not match - check for missing files.\n'
fi

printf '\nNext step:\n'
printf '  python -m moesr.train --config debug_small --train-lr dataset/train/LR --train-hr dataset/train/HR --val-lr dataset/val/LR --val-hr dataset/val/HR --output-dir runs/smoke --batch-size 1 --num-workers 0 --steps 10 --log-interval 1 --device cuda\n'
