#!/usr/bin/env bash
set -euo pipefail

# Backbone sweep for MEv1 recipe: 3 backbones x 2 freeze modes = 6 runs.
#
# Usage:
#   bash scripts/run_backbone_sixpack.sh
#
# Optional env overrides:
#   SEEDS="2048 418" NITER=8
#   BACKBONES="L14_local=/path/to/clip-vit-large-patch14/;L14_336=openai/clip-vit-large-patch14-336;H14=laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
#   FREEZE_MODES="1 0"
#   TRAIN_CSV="/abs/path/train.csv" VAL_DATA_ROOT="/abs/path/training_data_final"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_TAG="${BASE_TAG:-BB6_mev1}"
SEEDS="${SEEDS:-2048}"
NITER="${NITER:-6}"

TRAIN_CSV="${TRAIN_CSV:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/DF40_frames/image_frame_list_face.csv}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/training_data_final/}"
CACHE_ROOT="${CACHE_ROOT:-/home/u1657859/work/ntire_RDFC_2026_cvpr/fan_exp/New-Effort-AIGI-Detection_CVaRDRO/cache/e1_e6_matrix_qcache}"

NUM_THREADS="${NUM_THREADS:-4}"
USE_AMP="${USE_AMP:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/u1657859/miniconda3/envs/eamamba/bin/python}"
TRAIN_FRAME_NUM="${TRAIN_FRAME_NUM:-8}"
VAL_FRAME_NUM="${VAL_FRAME_NUM:-1}"
SVD_CACHE_DIR="${SVD_CACHE_DIR:-cache}"
EARLYSTOP_EPOCH="${EARLYSTOP_EPOCH:-0}"

BATCH_SIZE_FROZEN="${BATCH_SIZE_FROZEN:-24}"
BATCH_SIZE_UNFROZEN="${BATCH_SIZE_UNFROZEN:-12}"
BATCH_SIZE_FROZEN_336="${BATCH_SIZE_FROZEN_336:-${BATCH_SIZE_FROZEN}}"
BATCH_SIZE_UNFROZEN_336="${BATCH_SIZE_UNFROZEN_336:-${BATCH_SIZE_UNFROZEN}}"
LR_FROZEN="${LR_FROZEN:-0.0002}"
LR_UNFROZEN="${LR_UNFROZEN:-0.00007}"
RESIDUAL_LR_FROZEN="${RESIDUAL_LR_FROZEN:-None}"
RESIDUAL_LR_UNFROZEN="${RESIDUAL_LR_UNFROZEN:-0.00003}"

SVD_RANK_CAP_FROZEN="${SVD_RANK_CAP_FROZEN:-None}"
SVD_RANK_CAP_UNFROZEN="${SVD_RANK_CAP_UNFROZEN:-None}"
MULTI_EXPERT_DIV_LAMBDA="${MULTI_EXPERT_DIV_LAMBDA:-0.002}"
GROUPDRO_CVAR_LAMBDA="${GROUPDRO_CVAR_LAMBDA:-0.7}"
GROUPDRO_CVAR_MIN_K="${GROUPDRO_CVAR_MIN_K:-2}"
GROUPDRO_CVAR_WARMUP_STEPS="${GROUPDRO_CVAR_WARMUP_STEPS:-0.1}"
GROUPDRO_CVAR_RAMP_STEPS="${GROUPDRO_CVAR_RAMP_STEPS:-0.2}"

ENABLE_VAL_MICRO_TTA="${ENABLE_VAL_MICRO_TTA:-0}"
MICRO_TTA_K="${MICRO_TTA_K:-9}"
IMAGE_SIZE_DEFAULT="${IMAGE_SIZE_DEFAULT:-224}"
IMAGE_SIZE_FOR_336="${IMAGE_SIZE_FOR_336:-336}"

BACKBONES="${BACKBONES:-L14_local=/work/u1657859/ntire_RDFC_2026_cvpr/clip-vit-large-patch14/;L14_336=openai/clip-vit-large-patch14-336;H14=laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
FREEZE_MODES="${FREEZE_MODES:-1 0}"  # 1=frozen backbone, 0=unfrozen backbone

mkdir -p "${CACHE_ROOT}"

IFS=';' read -r -a BB_ENTRIES <<< "${BACKBONES}"
N_BB="${#BB_ENTRIES[@]}"
N_FZ="$(wc -w <<< "${FREEZE_MODES}" | awk '{print $1}')"
N_SEEDS="$(wc -w <<< "${SEEDS}" | awk '{print $1}')"
TOTAL_CASES=$((N_BB * N_FZ * N_SEEDS))
DONE_CASES=0

echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] total_cases=${TOTAL_CASES}"
echo "[INFO] backbones=${BACKBONES}"
echo "[INFO] freeze_modes=${FREEZE_MODES}"
echo "[INFO] seeds=${SEEDS}"
echo "[INFO] train_csv=${TRAIN_CSV}"
echo "[INFO] val_data_root=${VAL_DATA_ROOT}"
echo "[INFO] python_bin=${PYTHON_BIN}"
echo "[INFO] train_frame_num=${TRAIN_FRAME_NUM} val_frame_num=${VAL_FRAME_NUM}"
echo "[INFO] svd_cache_dir=${SVD_CACHE_DIR}"
echo "[INFO] earlystop_epoch=${EARLYSTOP_EPOCH} me_div_lambda=${MULTI_EXPERT_DIV_LAMBDA}"
echo "[INFO] cvar_defaults=lambda:${GROUPDRO_CVAR_LAMBDA} min_k:${GROUPDRO_CVAR_MIN_K} warmup:${GROUPDRO_CVAR_WARMUP_STEPS} ramp:${GROUPDRO_CVAR_RAMP_STEPS}"

for SEED in ${SEEDS}; do
  CACHE_TRAIN="${CACHE_ROOT}/q_train_s${SEED}.csv"
  CACHE_VAL="${CACHE_ROOT}/q_val_s${SEED}.csv"
  for ENTRY in "${BB_ENTRIES[@]}"; do
    ALIAS="${ENTRY%%=*}"
    CLIP_MODEL="${ENTRY#*=}"
    for FZ in ${FREEZE_MODES}; do
      DONE_CASES=$((DONE_CASES + 1))
      CASE_START_TS="$(date +%s)"

      if [[ "${FZ}" == "1" ]]; then
        FIX_BACKBONE=1
        BATCH_SIZE="${BATCH_SIZE_FROZEN}"
        LR="${LR_FROZEN}"
        RESIDUAL_LR="${RESIDUAL_LR_FROZEN}"
        SVD_CAP="${SVD_RANK_CAP_FROZEN}"
      else
        FIX_BACKBONE=0
        BATCH_SIZE="${BATCH_SIZE_UNFROZEN}"
        LR="${LR_UNFROZEN}"
        RESIDUAL_LR="${RESIDUAL_LR_UNFROZEN}"
        SVD_CAP="${SVD_RANK_CAP_UNFROZEN}"
      fi
      if [[ "${CLIP_MODEL}" == *"-336"* || "${ALIAS}" == *"_336"* ]]; then
        if [[ "${FZ}" == "1" ]]; then
          BATCH_SIZE="${BATCH_SIZE_FROZEN_336}"
        else
          BATCH_SIZE="${BATCH_SIZE_UNFROZEN_336}"
        fi
      fi

      RUN_NAME="Exp_${BASE_TAG}_${ALIAS}_fz${FZ}_s${SEED}"

      echo "[RUN ${DONE_CASES}/${TOTAL_CASES}] name=${RUN_NAME}"
      echo "[RUN ${DONE_CASES}/${TOTAL_CASES}] clip_model=${CLIP_MODEL}"
      echo "[RUN ${DONE_CASES}/${TOTAL_CASES}] fix_backbone=${FIX_BACKBONE} batch=${BATCH_SIZE} lr=${LR} residual_lr=${RESIDUAL_LR} svd_cap=${SVD_CAP}"

      PYTHONUNBUFFERED=1 "${PYTHON_BIN}" - <<PY &
import train

run_name = "${RUN_NAME}"
seed = int("${SEED}")
niter = int("${NITER}")
clip_model = "${CLIP_MODEL}"
train_csv = "${TRAIN_CSV}"
val_data_root = "${VAL_DATA_ROOT}"
cache_train = "${CACHE_TRAIN}"
cache_val = "${CACHE_VAL}"
fix_backbone = bool(int("${FIX_BACKBONE}"))
batch_size = int("${BATCH_SIZE}")
lr = float("${LR}")
residual_lr_raw = "${RESIDUAL_LR}"
svd_rank_cap_raw = "${SVD_CAP}"
num_threads = int("${NUM_THREADS}")
use_amp = bool(int("${USE_AMP}"))
enable_val_micro_tta = bool(int("${ENABLE_VAL_MICRO_TTA}"))
micro_tta_k = int("${MICRO_TTA_K}")
image_size_default = int("${IMAGE_SIZE_DEFAULT}")
image_size_for_336 = int("${IMAGE_SIZE_FOR_336}")
train_frame_num = int("${TRAIN_FRAME_NUM}")
val_frame_num = int("${VAL_FRAME_NUM}")
svd_cache_dir = "${SVD_CACHE_DIR}"
earlystop_epoch = int("${EARLYSTOP_EPOCH}")
multi_expert_div_lambda = float("${MULTI_EXPERT_DIV_LAMBDA}")
groupdro_cvar_lambda = float("${GROUPDRO_CVAR_LAMBDA}")
groupdro_cvar_min_k = int("${GROUPDRO_CVAR_MIN_K}")
groupdro_cvar_warmup_steps = float("${GROUPDRO_CVAR_WARMUP_STEPS}")
groupdro_cvar_ramp_steps = float("${GROUPDRO_CVAR_RAMP_STEPS}")

def set_opt(k, v):
    train.OPT_OVERRIDES[k] = v
    train.PARSER_DEFAULTS[k] = v

residual_lr = None if residual_lr_raw.lower() == "none" else float(residual_lr_raw)
svd_rank_cap = None if svd_rank_cap_raw.lower() == "none" else int(svd_rank_cap_raw)
image_size = image_size_for_336 if ("-336" in clip_model or "_336" in clip_model) else image_size_default

# Core
set_opt("name", run_name)
set_opt("seed", seed)
set_opt("niter", niter)
set_opt("train_csv", train_csv)
set_opt("val_data_root", val_data_root)
set_opt("clip_model", clip_model)
set_opt("fix_backbone", fix_backbone)
set_opt("batch_size", batch_size)
set_opt("num_threads", num_threads)
set_opt("use_amp", use_amp)
set_opt("lr", lr)
set_opt("residual_lr", residual_lr)
set_opt("svd_rank_cap", svd_rank_cap)
set_opt("svd_cache_dir", svd_cache_dir)
set_opt("image_size", image_size)
set_opt("train_frame_num", train_frame_num)
set_opt("val_frame_num", val_frame_num)
set_opt("earlystop_epoch", earlystop_epoch)

# Keep MEv1 winning recipe knobs fixed; only sweep backbone + freeze mode.
set_opt("quality_balance", True)
set_opt("val_quality_diag", True)
set_opt("use_groupdro", True)
set_opt("groupdro_group_key", "quality_bin")
set_opt("groupdro_class_balance", True)
set_opt("groupdro_eta", 0.05)
set_opt("groupdro_warmup_steps", 0.10)
set_opt("groupdro_update_every", 5)
set_opt("groupdro_q_temp", 2.0)
set_opt("groupdro_q_mix", 0.2)
set_opt("groupdro_use_cvar", False)
set_opt("groupdro_cvar_lambda", groupdro_cvar_lambda)
set_opt("groupdro_cvar_min_k", groupdro_cvar_min_k)
set_opt("groupdro_cvar_warmup_steps", groupdro_cvar_warmup_steps)
set_opt("groupdro_cvar_ramp_steps", groupdro_cvar_ramp_steps)

set_opt("multi_expert_enable", True)
set_opt("multi_expert_k", 3)
set_opt("multi_expert_route", "quality_bin")
set_opt("multi_expert_div_lambda", multi_expert_div_lambda)

set_opt("patch_pool_tau", 1.8)
set_opt("patch_pool_mode", "lse")
set_opt("patch_trim_p", 0.2)
set_opt("patch_quality", "cos_norm")
set_opt("patch_pool_gamma", 0.08)
set_opt("patch_warmup_steps", 0.10)
set_opt("patch_ramp_steps", 0.30)

set_opt("quality_cache_path", None)
set_opt("quality_cache_train_path", cache_train)
set_opt("quality_cache_val_path", cache_val)

set_opt("tri_consistency_enable", False)
set_opt("tri_consistency_apply_prob", 0.0)
set_opt("tri_consistency_align_weight", 0.0)
set_opt("tri_consistency_ce_weight", 0.0)

set_opt("val_micro_tta", enable_val_micro_tta)
set_opt("val_micro_tta_k", micro_tta_k)
set_opt("val_micro_tta_seed", 0)
set_opt("val_micro_tta_blur_max_sig", 1.5)
set_opt("val_micro_tta_jpeg_qual", [60, 95])
set_opt("val_micro_tta_contrast", [0.9, 1.1])
set_opt("val_micro_tta_gamma", [0.9, 1.1])
set_opt("val_micro_tta_resize_jitter", 8)

train.main()
PY
      PY_PID=$!

      while kill -0 "${PY_PID}" 2>/dev/null; do
        sleep 60
        if kill -0 "${PY_PID}" 2>/dev/null; then
          ELAPSED=$(( $(date +%s) - CASE_START_TS ))
          echo "[HEARTBEAT ${DONE_CASES}/${TOTAL_CASES}] name=${RUN_NAME} elapsed=${ELAPSED}s still running..."
        fi
      done
      wait "${PY_PID}"

      CASE_ELAPSED=$(( $(date +%s) - CASE_START_TS ))
      echo "[DONE ${DONE_CASES}/${TOTAL_CASES}] name=${RUN_NAME} elapsed=${CASE_ELAPSED}s"
    done
  done
done

echo "[ALL DONE] ${TOTAL_CASES} runs completed."
