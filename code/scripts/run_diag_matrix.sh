#!/usr/bin/env bash
set -euo pipefail

# 8-group diagnostic matrix for selected images.
# Usage:
#   bash scripts/run_diag_matrix.sh /path/to/hard_images.txt [out_base_dir]
#   bash scripts/run_diag_matrix.sh /path/to/image_dir [out_base_dir]
#
# Environment overrides (optional):
#   CKPT=... CLIP_MODEL=... TTA_K=9 AMP=1 REPO_ROOT=. SEED=418

INPUT_PATH="${1:-}"
OUT_BASE="${2:-checkpoints/Effort_aug_pooling_baseline_DF40_tuneoff-AUXLoss/diag_matrix}"

if [[ -z "${INPUT_PATH}" ]]; then
  echo "Usage: bash scripts/run_diag_matrix.sh /path/to/hard_images.txt|/path/to/image_dir [out_base_dir]" >&2
  exit 1
fi

INPUT_ARG=()
if [[ -d "${INPUT_PATH}" ]]; then
  INPUT_ARG=(--image_dir "${INPUT_PATH}")
elif [[ -f "${INPUT_PATH}" ]]; then
  INPUT_ARG=(--image_list "${INPUT_PATH}")
else
  echo "Input not found: ${INPUT_PATH}" >&2
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-.}"
CKPT="${CKPT:-checkpoints/Effort_aug_pooling_baseline_DF40_tuneoff-AUXLoss/best_auc.pt}"
CLIP_MODEL="${CLIP_MODEL:-/work/u1657859/ntire_RDFC_2026_cvpr/clip-vit-large-patch14/}"
TTA_K="${TTA_K:-9}"
AMP="${AMP:-1}"
SEED="${SEED:-418}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
PATCH_POOL_TAU="${PATCH_POOL_TAU:-1.8}"
PATCH_POOL_MODE="${PATCH_POOL_MODE:-lse}"
PATCH_TRIM_P="${PATCH_TRIM_P:-0.2}"
PATCH_QUALITY="${PATCH_QUALITY:-cos_norm}"
PATCH_POOL_GAMMA="${PATCH_POOL_GAMMA:-0.08}"

COMMON_ARGS=(
  --repo_root "${REPO_ROOT}"
  --ckpt "${CKPT}"
  --clip_model "${CLIP_MODEL}"
  "${INPUT_ARG[@]}"
  --image_size "${IMAGE_SIZE}"
  --patch_pool
  --patch_pool_tau "${PATCH_POOL_TAU}"
  --patch_pool_mode "${PATCH_POOL_MODE}"
  --patch_trim_p "${PATCH_TRIM_P}"
  --patch_quality "${PATCH_QUALITY}"
  --patch_pool_gamma "${PATCH_POOL_GAMMA}"
  --seed "${SEED}"
  --tta_k "${TTA_K}"
  --blur_sigmas 0 0.5 1 2 3.5 5 7
  --jpeg_qualities 95 80 65 50 35 25 15
)

if [[ "${AMP}" == "1" ]]; then
  COMMON_ARGS+=(--amp)
fi

mkdir -p "${OUT_BASE}"

run_case() {
  local name="$1"
  shift
  echo "[RUN] ${name}"
  python scripts/diag_comp_stability.py \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --outdir "${OUT_BASE}/${name}"
}

# Group A: Local stability
run_case "A1_hflip_only" \
  --tta_hflip_p 0.5 \
  --tta_disable_blur \
  --tta_disable_jpeg

run_case "A2_micro_blur_only" \
  --tta_hflip_p 0 \
  --tta_blur_sig 0 1.5 \
  --tta_disable_jpeg

run_case "A3_micro_jpeg_only" \
  --tta_hflip_p 0 \
  --tta_jpeg_q 60 95 \
  --tta_disable_blur

run_case "A4_micro_mix" \
  --tta_hflip_p 0.5 \
  --tta_blur_sig 0 1.5 \
  --tta_jpeg_q 60 95

# Group B: Medium stress
run_case "B1_medium_blur_only" \
  --tta_hflip_p 0 \
  --tta_blur_sig 0 3.0 \
  --tta_disable_jpeg

run_case "B2_medium_jpeg_only" \
  --tta_hflip_p 0 \
  --tta_jpeg_q 35 95 \
  --tta_disable_blur

run_case "B3_medium_mix" \
  --tta_hflip_p 0.5 \
  --tta_blur_sig 0 3.0 \
  --tta_jpeg_q 35 95

# Group C: Stress boundary
run_case "C1_extreme_mix" \
  --tta_hflip_p 0.5 \
  --tta_blur_sig 0 7.0 \
  --tta_jpeg_q 10 95

echo "[DONE] matrix results under: ${OUT_BASE}"
