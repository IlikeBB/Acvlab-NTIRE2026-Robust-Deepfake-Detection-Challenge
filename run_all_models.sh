#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./run_all_models.sh --input_dir /path/to/images [options] [-- extra full_submission args]

Optional:
  --batch_size   Batch size used for all 3 models (default: 32)
  --num_workers  DataLoader workers (default: 4)
  --clip_model   Local CLIP model directory
  --python       Python executable (default: python)
  --help         Show this help
EOF
}

INPUT_DIR=""
BATCH_SIZE="32"
NUM_WORKERS="4"
CLIP_MODEL=""
PYTHON_BIN="${PYTHON:-python}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_dir)
      INPUT_DIR="${2:-}"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="${2:-}"
      shift 2
      ;;
    --clip_model)
      CLIP_MODEL="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$INPUT_DIR" ]]; then
  usage
  exit 1
fi

for MODEL_ID in 1 2 3; do
  CMD=(
    "$ROOT_DIR/run_model.sh"
    --model "$MODEL_ID" \
    --input_dir "$INPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --python "$PYTHON_BIN"
  )
  if [[ -n "$CLIP_MODEL" ]]; then
    CMD+=(--clip_model "$CLIP_MODEL")
  fi
  CMD+=(--)
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi
  "${CMD[@]}"
done

echo "[done] generated 3 prediction CSVs under $ROOT_DIR/submissions"
