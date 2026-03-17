#!/usr/bin/env bash
set -euo pipefail

# Goal:
# - Keep MEv1 strict recipe unchanged as base
# - Add external-data curriculum training (A external warmup -> B competition FT -> C optional consistency FT)
# - Write logs/ckpts into app paths used by existing score/submission workflows

PYTHON_BIN="${PYTHON_BIN:-/home/u1657859/miniconda3/envs/eamamba/bin/python}"
APP_REPO="${APP_REPO:-/work/u1657859/Jess/app}"
ISO_REPO="${ISO_REPO:-/work/u1657859/Jess/app-isolated-20260227-191252}"
STAGED_SCRIPT="${ISO_REPO}/scripts/run_staged_ext_consistency.py"
RUN_SCRIPT="${APP_REPO}/scripts/run_multiexpert_local.py"

BASE_CONFIG="${BASE_CONFIG:-/work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_strict_full_recheck_s2048_20260305/config.json}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/work/u1657859/Jess/app/local_checkpoints}"
SCORE_DIR="${SCORE_DIR:-/work/u1657859/Jess/app/score}"

SEED="${SEED:-2048}"
EXP_PREFIX="${EXP_PREFIX:-Exp_LB90_extv1_s${SEED}_0308}"
LOG_PATH="${LOG_PATH:-${SCORE_DIR}/${EXP_PREFIX}.live.log}"

# Stage lengths (total ~= A+B+C epochs)
STAGEA_NITER="${STAGEA_NITER:-2}"
STAGEB_NITER="${STAGEB_NITER:-4}"
STAGEC_NITER="${STAGEC_NITER:-1}"

# Keep strict core recipe where possible
TRAIN_FRAME_NUM="${TRAIN_FRAME_NUM:-8}"
VAL_FRAME_NUM="${VAL_FRAME_NUM:-1}"
MULTI_EXPERT_K="${MULTI_EXPERT_K:-3}"
MULTI_EXPERT_ROUTE="${MULTI_EXPERT_ROUTE:-quality_bin}"
NUM_THREADS="${NUM_THREADS:-8}"

# Batch fallback ladder to avoid OOM across stages
BATCH_SIZES="${BATCH_SIZES:-24 20 16}"

# External data controls
EXTERNAL_MAX_PER_CLASS="${EXTERNAL_MAX_PER_CLASS:-120000}"
COMP_VAL_RATIO="${COMP_VAL_RATIO:-0.20}"
STAGEC_COMP_RATIO="${STAGEC_COMP_RATIO:-1.0}"  # 1.0 => comp-only stage C
CUSTOM_EXTERNAL_ROOTS="${CUSTOM_EXTERNAL_ROOTS:-}"
EXTERNAL_MANIFEST_CSV="${EXTERNAL_MANIFEST_CSV:-/work/u1657859/Jess/app/local_checkpoints/manifests/ext_dfd_train_manifest.csv}"
USE_PREBUILT_MANIFEST="${USE_PREBUILT_MANIFEST:-1}"
SHARED_EXT_QCACHE="${SHARED_EXT_QCACHE:-/work/u1657859/Jess/app/local_checkpoints/manifests/ext_dfd_train_qcache.csv}"
SHARED_COMP_QCACHE="${SHARED_COMP_QCACHE:-/work/u1657859/Jess/app/local_checkpoints/manifests/comp_train1000_qcache.csv}"

# By default, turn OFF tri-consistency to stay closer to strict baseline behavior.
USE_STAGEC_TRI="${USE_STAGEC_TRI:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${SCORE_DIR}"

read -r -a BS_ARR <<< "${BATCH_SIZES}"
if [[ -n "${CUSTOM_EXTERNAL_ROOTS}" ]]; then
  IFS='|' read -r -a EXT_ROOTS <<< "${CUSTOM_EXTERNAL_ROOTS}"
else
  EXT_ROOTS=(
    "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/DF40_frames/train/DF40"
    "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/nitra_frames_face/train/Celeb-DF"
    "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/nitra_frames_face/train/DeeperForensics-1.0"
    "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/nitra_frames_face/train/DeeperForecsics++"
  )
fi

CMD=(
  "${PYTHON_BIN}" "${STAGED_SCRIPT}"
  --python_bin "${PYTHON_BIN}"
  --repo "${APP_REPO}"
  --run_script "${RUN_SCRIPT}"
  --base_config "${BASE_CONFIG}"
  --checkpoints_dir "${CHECKPOINTS_DIR}"
  --exp_prefix "${EXP_PREFIX}"
  --seed "${SEED}"
  --batch_sizes "${BS_ARR[@]}"
  --num_threads "${NUM_THREADS}"
  --train_frame_num "${TRAIN_FRAME_NUM}"
  --val_frame_num "${VAL_FRAME_NUM}"
  --multi_expert_k "${MULTI_EXPERT_K}"
  --multi_expert_route "${MULTI_EXPERT_ROUTE}"
  --stageA_niter "${STAGEA_NITER}"
  --stageB_niter "${STAGEB_NITER}"
  --stageC_niter "${STAGEC_NITER}"
  --comp_val_ratio "${COMP_VAL_RATIO}"
  --comp_split_seed "${SEED}"
  --stagec_comp_ratio "${STAGEC_COMP_RATIO}"
  --stagea_comp_like_aug
  --stagea_aug_strength strong
  --ext_quality_cache_path "${SHARED_EXT_QCACHE}"
  --comp_quality_cache_path "${SHARED_COMP_QCACHE}"
)

if [[ "${USE_PREBUILT_MANIFEST}" == "1" ]]; then
  CMD+=(
    --external_manifest_csv "${EXTERNAL_MANIFEST_CSV}"
    --skip_manifest_build
  )
else
  CMD+=(
    --external_roots "${EXT_ROOTS[@]}"
    --external_max_per_class "${EXTERNAL_MAX_PER_CLASS}"
  )
fi

if [[ "${USE_STAGEC_TRI}" == "1" ]]; then
  CMD+=(--stagec_use_tri_consistency)
else
  CMD+=(--no_stagec_use_tri_consistency)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry_run)
fi

{
  echo "[start] $(date -Is)"
  echo "[exp_prefix] ${EXP_PREFIX}"
  echo "[base_config] ${BASE_CONFIG}"
  echo "[batch_sizes] ${BATCH_SIZES}"
  echo "[stage_niter] A=${STAGEA_NITER} B=${STAGEB_NITER} C=${STAGEC_NITER}"
  echo "[external_max_per_class] ${EXTERNAL_MAX_PER_CLASS}"
  echo "[use_prebuilt_manifest] ${USE_PREBUILT_MANIFEST}"
  echo "[external_manifest_csv] ${EXTERNAL_MANIFEST_CSV}"
  echo "[shared_ext_qcache] ${SHARED_EXT_QCACHE}"
  echo "[shared_comp_qcache] ${SHARED_COMP_QCACHE}"
  echo "[use_stagec_tri] ${USE_STAGEC_TRI}"
  echo "[dry_run] ${DRY_RUN}"
  printf '[cmd]'; printf ' %q' "${CMD[@]}"; echo
} | tee "${LOG_PATH}"

"${CMD[@]}" 2>&1 | tee -a "${LOG_PATH}"

{
  echo "[done] $(date -Is)"
  echo "[log] ${LOG_PATH}"
  echo "[ckpt_stageA] ${CHECKPOINTS_DIR}/${EXP_PREFIX}_A_extwarm/best_auc.pt"
  echo "[ckpt_stageB] ${CHECKPOINTS_DIR}/${EXP_PREFIX}_B_compft/best_auc.pt"
  echo "[ckpt_stageC] ${CHECKPOINTS_DIR}/${EXP_PREFIX}_C_cons/best_auc.pt"
} | tee -a "${LOG_PATH}"
