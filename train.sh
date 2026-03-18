#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-debug_small}"
TRAIN_LR="${TRAIN_LR:-dataset/train/LR}"
TRAIN_HR="${TRAIN_HR:-dataset/train/HR}"
VAL_LR="${VAL_LR:-dataset/val/LR}"
VAL_HR="${VAL_HR:-dataset/val/HR}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/div2k_debug}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
STEPS="${STEPS:-3000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-200}"
VAL_MAX_IMAGES="${VAL_MAX_IMAGES:-2}"
VAL_TILE_SIZE="${VAL_TILE_SIZE:-48}"
VAL_TILE_OVERLAP="${VAL_TILE_OVERLAP:-16}"
DEVICE="${DEVICE:-cuda}"

python -m moesr.train \
  --config "$CONFIG" \
  --train-lr "$TRAIN_LR" \
  --train-hr "$TRAIN_HR" \
  --val-lr "$VAL_LR" \
  --val-hr "$VAL_HR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --steps "$STEPS" \
  --log-interval "$LOG_INTERVAL" \
  --val-interval "$VAL_INTERVAL" \
  --val-max-images "$VAL_MAX_IMAGES" \
  --val-tile-size "$VAL_TILE_SIZE" \
  --val-tile-overlap "$VAL_TILE_OVERLAP" \
  --device "$DEVICE" \
  "$@"
