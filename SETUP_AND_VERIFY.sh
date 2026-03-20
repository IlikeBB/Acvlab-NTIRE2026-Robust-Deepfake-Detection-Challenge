#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CLIP_MODEL="/ssd2/ntire_RDFC_2026_cvpr/clip-vit-large-patch14"
PYTHON_BIN="${PYTHON:-python}"
TEST_INPUT_DIR=""
RUN_SMOKE_TEST=0
CLIP_MODEL=""

usage() {
  cat <<'EOF'
Usage:
  ./SETUP_AND_VERIFY.sh [options]

Options:
  --test_input_dir  Directory used for an optional smoke inference run
  --clip_model      Local CLIP model directory
  --python          Python executable (default: python)
  --run_smoke_test  Actually run model 2 on --test_input_dir
  --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test_input_dir)
      TEST_INPUT_DIR="${2:-}"
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
    --run_smoke_test)
      RUN_SMOKE_TEST=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$CLIP_MODEL" && -d "$DEFAULT_CLIP_MODEL" ]]; then
  CLIP_MODEL="$DEFAULT_CLIP_MODEL"
fi

echo "============================================================"
echo "STEP 1: package structure"
echo "============================================================"

for dir in checkpoints code submissions docs; do
  [[ -d "$ROOT_DIR/$dir" ]] || { echo "[error] missing directory: $dir"; exit 1; }
  echo "[ok] $dir/"
done

for file in README.md requirements.txt run_model.sh run_all_models.sh run_submission.sh run_ensemble.py code/full_submission.py; do
  [[ -f "$ROOT_DIR/$file" ]] || { echo "[error] missing file: $file"; exit 1; }
  echo "[ok] $file"
done

echo
echo "============================================================"
echo "STEP 2: python environment"
echo "============================================================"

echo "[info] python=$("$PYTHON_BIN" --version 2>&1)"
"$PYTHON_BIN" -c "import torch; print(f'[ok] torch {torch.__version__}, cuda={torch.cuda.is_available()}')" || true
"$PYTHON_BIN" -c "import torchvision; print(f'[ok] torchvision {torchvision.__version__}')" || true
"$PYTHON_BIN" -c "import numpy; print(f'[ok] numpy {numpy.__version__}')" || true
"$PYTHON_BIN" -c "import PIL; print(f'[ok] pillow {PIL.__version__}')" || true
"$PYTHON_BIN" -c "import tqdm; print(f'[ok] tqdm {tqdm.__version__}')" || true
"$PYTHON_BIN" -c "import transformers; print(f'[ok] transformers {transformers.__version__}')" || true

echo
echo "============================================================"
echo "STEP 3: checkpoint files"
echo "============================================================"

for ckpt in "$ROOT_DIR"/checkpoints/*.pt; do
  [[ -f "$ckpt" ]] || { echo "[error] checkpoint missing: $ckpt"; exit 1; }
  echo "[ok] $(basename "$ckpt") size=$(du -h "$ckpt" | cut -f1)"
done

echo
echo "============================================================"
echo "STEP 4: wrapper help check"
echo "============================================================"

"$ROOT_DIR/run_model.sh" --help >/dev/null
"$ROOT_DIR/run_all_models.sh" --help >/dev/null
"$ROOT_DIR/run_submission.sh" --help >/dev/null
"$PYTHON_BIN" "$ROOT_DIR/run_ensemble.py" --help >/dev/null
echo "[ok] wrapper entrypoints are callable"

echo
echo "============================================================"
echo "STEP 5: optional smoke inference"
echo "============================================================"

if [[ "$RUN_SMOKE_TEST" -eq 1 ]]; then
  [[ -n "$TEST_INPUT_DIR" ]] || { echo "[error] --run_smoke_test requires --test_input_dir"; exit 1; }
  [[ -d "$TEST_INPUT_DIR" ]] || { echo "[error] test input directory not found: $TEST_INPUT_DIR"; exit 1; }
  SMOKE_OUT="$ROOT_DIR/submissions/smoke_test_model2.csv"
  CMD=(
    "$ROOT_DIR/run_model.sh"
    --model 2
    --input_dir "$TEST_INPUT_DIR"
    --output_csv "$SMOKE_OUT"
    --batch_size 1
    --num_workers 0
    --python "$PYTHON_BIN"
  )
  if [[ -n "$CLIP_MODEL" ]]; then
    CMD+=(--clip_model "$CLIP_MODEL")
  fi
  "${CMD[@]}"
  [[ -f "$SMOKE_OUT" ]] || { echo "[error] smoke output missing"; exit 1; }
  echo "[ok] smoke inference completed: $SMOKE_OUT"
else
  echo "[skip] smoke inference not requested"
fi

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "[next] single model: ./run_model.sh --model 2 --input_dir /path/to/images"
echo "[next] full pipeline: ./run_submission.sh --input_dir /path/to/images"
