#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CLIP_MODEL="/ssd2/ntire_RDFC_2026_cvpr/clip-vit-large-patch14"

usage() {
  cat <<'EOF'
Usage:
  ./run_model.sh --model {1|2|3} --input_dir /path/to/images [options] [-- extra full_submission args]

Required:
  --model        Model index: 1, 2, or 3
  --input_dir    Directory containing inference images

Optional:
  --output_csv   Output CSV path
  --batch_size   Batch size (default: 32)
  --num_workers  DataLoader workers (default: 4)
  --clip_model   Local CLIP model directory
  --python       Python executable (default: python)
  --help         Show this help

Examples:
  ./run_model.sh --model 2 --input_dir /data/test_images
  ./run_model.sh --model 1 --input_dir /data/test_images --batch_size 16 --output_csv submissions/m1.csv
EOF
}

MODEL_ID=""
INPUT_DIR=""
OUTPUT_CSV=""
BATCH_SIZE="32"
NUM_WORKERS="4"
CLIP_MODEL=""
PYTHON_BIN="${PYTHON:-python}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_ID="${2:-}"
      shift 2
      ;;
    --input_dir)
      INPUT_DIR="${2:-}"
      shift 2
      ;;
    --output_csv)
      OUTPUT_CSV="${2:-}"
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

if [[ -z "$MODEL_ID" || -z "$INPUT_DIR" ]]; then
  usage
  exit 1
fi

case "$MODEL_ID" in
  1)
    CKPT_PATH="$ROOT_DIR/checkpoints/model1_lb0.8568_best_auc.pt"
    DEFAULT_OUTPUT="$ROOT_DIR/submissions/pred_model1_lb0.8568.csv"
    ;;
  2)
    CKPT_PATH="$ROOT_DIR/checkpoints/model2_lb0.8572_best_auc.pt"
    DEFAULT_OUTPUT="$ROOT_DIR/submissions/pred_model2_lb0.8572.csv"
    ;;
  3)
    CKPT_PATH="$ROOT_DIR/checkpoints/model3_lb0.8568strict_best_auc.pt"
    DEFAULT_OUTPUT="$ROOT_DIR/submissions/pred_model3_lb0.8568strict.csv"
    ;;
  *)
    echo "[error] --model must be 1, 2, or 3"
    exit 1
    ;;
esac

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[error] input directory not found: $INPUT_DIR"
  exit 1
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[error] checkpoint not found: $CKPT_PATH"
  exit 1
fi

if [[ -z "$OUTPUT_CSV" ]]; then
  OUTPUT_CSV="$DEFAULT_OUTPUT"
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"

if [[ -z "$CLIP_MODEL" && -d "$DEFAULT_CLIP_MODEL" ]]; then
  CLIP_MODEL="$DEFAULT_CLIP_MODEL"
fi

CMD=(
  "$PYTHON_BIN"
  "$ROOT_DIR/code/full_submission.py"
  --ckpt "$CKPT_PATH"
  --txt_path "$INPUT_DIR"
  --out_csv "$OUTPUT_CSV"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
)

if [[ -n "$CLIP_MODEL" ]]; then
  CMD+=(--clip_model "$CLIP_MODEL")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[run] model=$MODEL_ID"
echo "[run] ckpt=$CKPT_PATH"
echo "[run] input_dir=$INPUT_DIR"
echo "[run] output_csv=$OUTPUT_CSV"
if [[ -n "$CLIP_MODEL" ]]; then
  echo "[run] clip_model=$CLIP_MODEL"
fi

"${CMD[@]}"
