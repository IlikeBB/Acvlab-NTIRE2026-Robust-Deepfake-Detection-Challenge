#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$BUNDLE_DIR/code"
CKPT="$BUNDLE_DIR/artifacts/checkpoint/best_auc.pt"
OUT_DIR="${OUT_DIR:-$BUNDLE_DIR/repro_outputs}"
DATA_ROOT="${DATA_ROOT:-${1:-}}"
CLIP_ROOT="${CLIP_ROOT:-${2:-}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ -z "$DATA_ROOT" || -z "$CLIP_ROOT" ]]; then
  echo "Usage: DATA_ROOT=/path/to/validation_data_final CLIP_ROOT=/path/to/clip-vit-large-patch14 [PYTHON_BIN=python] bash run_repro_submission.sh"
  exit 1
fi

mkdir -p "$OUT_DIR" "$CODE_DIR/cache"

(
  cd "$CODE_DIR"
  "$PYTHON_BIN" full_submission.py \
    --ckpt "$CKPT" \
    --clip_model "$CLIP_ROOT" \
    --txt_path "$DATA_ROOT" \
    --root_dir "$DATA_ROOT" \
    --out_csv "$OUT_DIR/repro_submission.txt" \
    --zip_submission \
    --zip_path "$OUT_DIR/repro_submission.zip" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --multi_expert_enable \
    --multi_expert_k 3 \
    --multi_expert_route quality_bin \
    --infer_quality_bin \
    --quality_cuts 4.719781875610352 5.441120147705078 \
    --tta_mode none \
    --tta_logit_agg mean \
    --on_error skip
)

echo "Saved:"
echo "  $OUT_DIR/repro_submission.txt"
echo "  $OUT_DIR/repro_submission.zip"
