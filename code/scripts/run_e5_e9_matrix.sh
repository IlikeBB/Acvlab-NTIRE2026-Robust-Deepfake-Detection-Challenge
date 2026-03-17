#!/usr/bin/env bash
set -euo pipefail

# E5~E9 experiment matrix runner (bash + inline python).
#
# Usage:
#   bash scripts/run_e5_e9_matrix.sh
#
# Optional env overrides:
#   BASE_TAG="DF40_tuneoff-AUXLoss" NITER=10
#   RUNS="E5a E5b E7 E8 E9A E9B"
#   SEEDS="418 777"
#   ENABLE_VAL_MICRO_TTA=1 MICRO_TTA_K=9
#   TRAIN_CSV="/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/DF40_frames/image_frame_list_face.csv"
#   VAL_DATA_ROOT="/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/training_data_final/"
#   CLIP_MODEL="/work/u1657859/ntire_RDFC_2026_cvpr/clip-vit-large-patch14/"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_TAG="${BASE_TAG:-DF40_tuneoff-AUXLoss}"
NITER="${NITER:-6}"
RUNS="${RUNS:-E5a E5b E6 E7 E8 E9A E9B}"
SEEDS="${SEEDS:-418}"

ENABLE_VAL_MICRO_TTA="${ENABLE_VAL_MICRO_TTA:-1}"
MICRO_TTA_K="${MICRO_TTA_K:-9}"
TRAIN_CSV="${TRAIN_CSV:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/DF40_frames/image_frame_list_face.csv}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/training_data_final/}"
CLIP_MODEL="${CLIP_MODEL:-/work/u1657859/ntire_RDFC_2026_cvpr/clip-vit-large-patch14/}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/cache}"
# Cache naming mode:
#   shared  -> one cache pair for all runs (q_train.csv / q_val.csv)
#   per_seed -> one cache pair per seed (q_train_s<seed>.csv / q_val_s<seed>.csv)
CACHE_SCOPE="${CACHE_SCOPE:-shared}"

mkdir -p "${CACHE_ROOT}"

echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] train_csv=${TRAIN_CSV}"
echo "[INFO] val_data_root=${VAL_DATA_ROOT}"
echo "[INFO] clip_model=${CLIP_MODEL}"
echo "[INFO] qcache_root=${CACHE_ROOT}"
echo "[INFO] qcache_scope=${CACHE_SCOPE}"

count_words() {
  # count tokens in a space-separated string
  set -- $1
  echo $#
}

N_RUNS="$(count_words "${RUNS}")"
N_SEEDS="$(count_words "${SEEDS}")"
TOTAL_CASES=$((N_RUNS * N_SEEDS))
DONE_CASES=0

echo "[INFO] seeds=${SEEDS} runs=${RUNS}"
echo "[INFO] total_cases=${TOTAL_CASES}"

for SEED in ${SEEDS}; do
  for CASE in ${RUNS}; do
    DONE_CASES=$((DONE_CASES + 1))
    RUN_NAME="Exp_${CASE}_${BASE_TAG}_s${SEED}"
    if [[ "${CACHE_SCOPE}" == "per_seed" ]]; then
      CACHE_TRAIN="${CACHE_ROOT}/q_train_s${SEED}.csv"
      CACHE_VAL="${CACHE_ROOT}/q_val_s${SEED}.csv"
    else
      CACHE_TRAIN="${CACHE_ROOT}/q_train.csv"
      CACHE_VAL="${CACHE_ROOT}/q_val.csv"
    fi
    CASE_START_TS="$(date +%s)"

    echo "[RUN ${DONE_CASES}/${TOTAL_CASES}] case=${CASE} seed=${SEED} name=${RUN_NAME} start=$(date '+%F %T')"
    echo "[RUN ${DONE_CASES}/${TOTAL_CASES}] qcache_train=${CACHE_TRAIN}"
    echo "[RUN ${DONE_CASES}/${TOTAL_CASES}] qcache_val=${CACHE_VAL}"
    [[ -f "${CACHE_TRAIN}" ]] && echo "[RUN ${DONE_CASES}/${TOTAL_CASES}] qcache_train HIT (reuse existing)"
    [[ -f "${CACHE_VAL}" ]] && echo "[RUN ${DONE_CASES}/${TOTAL_CASES}] qcache_val HIT (reuse existing)"

    PYTHONUNBUFFERED=1 python - <<PY &
import train

case = "${CASE}"
seed = int("${SEED}")
niter = int("${NITER}")
run_name = "${RUN_NAME}"
cache_train = "${CACHE_TRAIN}"
cache_val = "${CACHE_VAL}"
enable_val_micro_tta = bool(int("${ENABLE_VAL_MICRO_TTA}"))
micro_tta_k = int("${MICRO_TTA_K}")
train_csv = "${TRAIN_CSV}"
val_data_root = "${VAL_DATA_ROOT}"
clip_model = "${CLIP_MODEL}"

def set_opt(k, v):
    train.OPT_OVERRIDES[k] = v
    train.PARSER_DEFAULTS[k] = v

# ---------------- Common baseline ----------------
set_opt("name", run_name)
set_opt("seed", seed)
set_opt("niter", niter)
set_opt("train_csv", train_csv)
set_opt("val_data_root", val_data_root)
set_opt("clip_model", clip_model)
set_opt("quality_balance", True)
set_opt("use_groupdro", False)
set_opt("groupdro_class_balance", False)
set_opt("use_reweighted_erm", False)
set_opt("use_stratified_sampler", False)
set_opt("stratified_sampler_key", "stratum_id")
set_opt("groupdro_group_key", "quality_bin")
set_opt("groupdro_eta", 0.10)
set_opt("groupdro_warmup_steps", 0)
set_opt("quality_cuts", None)
set_opt("quality_cache_path", None)
set_opt("quality_cache_train_path", cache_train)
set_opt("quality_cache_val_path", cache_val)
set_opt("val_micro_tta", enable_val_micro_tta)
set_opt("val_micro_tta_k", micro_tta_k)
set_opt("val_micro_tta_seed", 0)
set_opt("val_micro_tta_blur_max_sig", 1.5)
set_opt("val_micro_tta_jpeg_qual", [60, 95])
set_opt("val_micro_tta_contrast", [0.9, 1.1])
set_opt("val_micro_tta_gamma", [0.9, 1.1])
set_opt("val_micro_tta_resize_jitter", 8)

# ---------------- Case-specific knobs ----------------
if case == "E5a":
    # quality-aware logging baseline
    pass
elif case == "E5b":
    # quality-balanced training via reweighted ERM
    set_opt("use_reweighted_erm", True)
elif case == "E6":
    # GroupDRO (no class-balanced aggregation)
    set_opt("use_groupdro", True)
    set_opt("groupdro_class_balance", False)
elif case == "E7":
    # GroupDRO + class-balanced aggregation
    set_opt("use_groupdro", True)
    set_opt("groupdro_class_balance", True)
elif case == "E8":
    # E7 + label-free quality-bin stratified sampler
    set_opt("use_groupdro", True)
    set_opt("groupdro_class_balance", True)
    set_opt("use_stratified_sampler", True)
    set_opt("stratified_sampler_key", "quality_bin")
elif case == "E9A":
    # warmup=10%, eta=0.05
    set_opt("use_groupdro", True)
    set_opt("groupdro_class_balance", True)
    set_opt("groupdro_warmup_steps", 0.10)
    set_opt("groupdro_eta", 0.05)
elif case == "E9B":
    # warmup=5%, eta=0.15
    set_opt("use_groupdro", True)
    set_opt("groupdro_class_balance", True)
    set_opt("groupdro_warmup_steps", 0.05)
    set_opt("groupdro_eta", 0.15)
else:
    raise ValueError(f"Unknown case: {case}")

train.main()
PY
    PY_PID=$!

    # Heartbeat every 60s so you know it's still running.
    while kill -0 "${PY_PID}" 2>/dev/null; do
      sleep 60
      if kill -0 "${PY_PID}" 2>/dev/null; then
        ELAPSED=$(( $(date +%s) - CASE_START_TS ))
        echo "[HEARTBEAT ${DONE_CASES}/${TOTAL_CASES}] case=${CASE} seed=${SEED} elapsed=${ELAPSED}s still running..."
      fi
    done
    wait "${PY_PID}"

    CASE_ELAPSED=$(( $(date +%s) - CASE_START_TS ))
    echo "[DONE ${DONE_CASES}/${TOTAL_CASES}] case=${CASE} seed=${SEED} elapsed=${CASE_ELAPSED}s end=$(date '+%F %T')"
  done
done

echo "[DONE] all runs finished"
