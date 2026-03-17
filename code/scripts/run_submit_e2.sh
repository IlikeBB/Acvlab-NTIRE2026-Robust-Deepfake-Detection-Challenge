#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# E2 checkpoint submit (no AMP).
CKPT_DIR="${CKPT_DIR:-/ssd2/ntire_RDFC_2026_cvpr/fan_exp/New-Effort-AIGI-Detection/New-Effort-AIGI-Detection_groupDRO/checkpoints/E2_DF40_tuneoff-AUXLoss}"
CKPT_PATH="${CKPT_PATH:-${CKPT_DIR}/best_auc.pt}"

TXT_PATH="${TXT_PATH:-/ssd2/ntire_RDFC_2026_cvpr/challange_data/validation_data_final}"
ROOT_DATA_DIR="${ROOT_DATA_DIR:-/ssd2/ntire_RDFC_2026_cvpr/challange_data/validation_data_final}"

OUT_CSV="${OUT_CSV:-${CKPT_DIR}/submission_E2_best_auc.txt}"
CLIP_MODEL="${CLIP_MODEL:-/work/u1657859/ntire_RDFC_2026_cvpr/clip-vit-large-patch14/}"

IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Match E2 training/inference patch-pooling params.
PATCH_POOL_TAU="${PATCH_POOL_TAU:-1.8}"
PATCH_POOL_MODE="${PATCH_POOL_MODE:-lse}"
PATCH_TRIM_P="${PATCH_TRIM_P:-0.2}"
PATCH_QUALITY="${PATCH_QUALITY:-cos_norm}"
PATCH_POOL_GAMMA="${PATCH_POOL_GAMMA:-0.08}"

echo "[submit] ckpt=${CKPT_PATH}"
echo "[submit] txt_path=${TXT_PATH}"
echo "[submit] out=${OUT_CSV}"

python full_submission.py \
  --ckpt "${CKPT_PATH}" \
  --txt_path "${TXT_PATH}" \
  --root_dir "${ROOT_DATA_DIR}" \
  --out_csv "${OUT_CSV}" \
  --clip_model "${CLIP_MODEL}" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --patch_pool_tau "${PATCH_POOL_TAU}" \
  --patch_pool_mode "${PATCH_POOL_MODE}" \
  --patch_trim_p "${PATCH_TRIM_P}" \
  --patch_quality "${PATCH_QUALITY}" \
  --patch_pool_gamma "${PATCH_POOL_GAMMA}" \
  --on_error skip

echo "[done] submission saved: ${OUT_CSV}"
