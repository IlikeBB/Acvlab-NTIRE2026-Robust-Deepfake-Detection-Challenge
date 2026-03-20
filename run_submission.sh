#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./run_submission.sh --input_dir /path/to/images [options] [-- extra full_submission args]

This runs:
  1. 3 single-model inference jobs
  2. Simple average ensemble: (pred1 + pred2 + pred3) / 3
  3. Codabench zip generation
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

CMD=(
  "$ROOT_DIR/run_all_models.sh"
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

"$PYTHON_BIN" "$ROOT_DIR/run_ensemble.py"
