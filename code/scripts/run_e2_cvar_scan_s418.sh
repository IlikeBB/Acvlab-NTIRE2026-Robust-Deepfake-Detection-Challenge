#!/usr/bin/env bash
set -euo pipefail

# E2 CVaR ablation scan runner (BASE + A/B/C/D groups)
#
# Usage:
#   bash scripts/run_e2_cvar_scan_s418.sh
#
# Optional env overrides:
#   NITER=2
#   SEEDS="418 2048"
#   RUNS="E2_BASE E2_A3 E2_B1"
#   BASE_TAG="DF40_tuneoff-AUXLoss"
#   TRAIN_CSV="..."
#   VAL_DATA_ROOT="..."
#   CLIP_MODEL="..."

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_TAG="${BASE_TAG:-DF40_tuneoff-AUXLoss}"
NITER="${NITER:-3}"
SEEDS="${SEEDS:-418}"
RUNS="${RUNS:-E2_BASE E2_A1 E2_A2 E2_A3 E2_B1 E2_B2 E2_C1 E2_C2 E2_D1}"

TRAIN_CSV="${TRAIN_CSV:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/DF40_frames/image_frame_list_face.csv}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/training_data_final/}"
CLIP_MODEL="${CLIP_MODEL:-/work/u1657859/ntire_RDFC_2026_cvpr/clip-vit-large-patch14/}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/cache/e1_e6_matrix_qcache}"
CACHE_SCOPE="${CACHE_SCOPE:-per_seed}" # shared | per_seed

mkdir -p "${CACHE_ROOT}"

echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] runs=${RUNS}"
echo "[INFO] seeds=${SEEDS}"
echo "[INFO] niter=${NITER}"
echo "[INFO] train_csv=${TRAIN_CSV}"
echo "[INFO] val_data_root=${VAL_DATA_ROOT}"
echo "[INFO] clip_model=${CLIP_MODEL}"
echo "[INFO] qcache_root=${CACHE_ROOT}"
echo "[INFO] qcache_scope=${CACHE_SCOPE}"

TOTAL_CASES=0
for CASE in ${RUNS}; do
  for _SEED in ${SEEDS}; do
    TOTAL_CASES=$((TOTAL_CASES + 1))
  done
done
DONE_CASES=0

for CASE in ${RUNS}; do
  for SEED in ${SEEDS}; do
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

    PYTHONUNBUFFERED=1 python - <<PY &
import train

case = "${CASE}"
seed = int("${SEED}")
niter = int("${NITER}")
run_name = "${RUN_NAME}"
cache_train = "${CACHE_TRAIN}"
cache_val = "${CACHE_VAL}"
train_csv = "${TRAIN_CSV}"
val_data_root = "${VAL_DATA_ROOT}"
clip_model = "${CLIP_MODEL}"

def set_opt(k, v):
    train.OPT_OVERRIDES[k] = v
    train.PARSER_DEFAULTS[k] = v

# ---------------- Base E2-BASE config ----------------
set_opt("name", run_name)
set_opt("seed", seed)
set_opt("niter", niter)
set_opt("batch_size", 48)
set_opt("num_threads", 4)
set_opt("train_csv", train_csv)
set_opt("val_data_root", val_data_root)
set_opt("clip_model", clip_model)
set_opt("quality_balance", True)
set_opt("quality_bins", 3)
set_opt("quality_cache_path", None)
set_opt("quality_cache_train_path", cache_train)
set_opt("quality_cache_val_path", cache_val)
set_opt("quality_cuts", None)

set_opt("use_groupdro", True)
set_opt("groupdro_group_key", "quality_bin")
set_opt("groupdro_eta", 0.05)
set_opt("groupdro_update_every", 5)
set_opt("groupdro_q_temp", 2.0)
set_opt("groupdro_q_mix", 0.2)

set_opt("groupdro_use_cvar", True)
set_opt("groupdro_cvar_alpha", 0.30)
set_opt("groupdro_cvar_lambda", 0.70)
set_opt("groupdro_cvar_warmup_steps", 0.10)
set_opt("groupdro_cvar_ramp_steps", 0.20)
set_opt("groupdro_cvar_cap_p", 0.99)
set_opt("groupdro_cvar_trim", False)
set_opt("groupdro_cvar_min_k", 2)

# ---------------- Case-specific: change one thing only ----------------
if case == "E2_BASE":
    pass

# A group: lambda schedule / strength
elif case == "E2_A1":
    set_opt("groupdro_cvar_lambda", 0.70)
    set_opt("groupdro_cvar_warmup_steps", 0.20)
    set_opt("groupdro_cvar_ramp_steps", 0.40)
elif case == "E2_A2":
    set_opt("groupdro_cvar_lambda", 0.50)
    set_opt("groupdro_cvar_warmup_steps", 0.10)
    set_opt("groupdro_cvar_ramp_steps", 0.20)
elif case == "E2_A3":
    set_opt("groupdro_cvar_lambda", 0.50)
    set_opt("groupdro_cvar_warmup_steps", 0.20)
    set_opt("groupdro_cvar_ramp_steps", 0.40)

# B group: tail cap / trim
elif case == "E2_B1":
    set_opt("groupdro_cvar_cap_p", 0.98)
    set_opt("groupdro_cvar_trim", False)
elif case == "E2_B2":
    set_opt("groupdro_cvar_cap_p", 0.99)
    set_opt("groupdro_cvar_trim", True)

# C group: alpha softness
elif case == "E2_C1":
    set_opt("groupdro_cvar_alpha", 0.50)
elif case == "E2_C2":
    set_opt("groupdro_cvar_alpha", 0.20)

# D group: flatter q
elif case == "E2_D1":
    set_opt("groupdro_q_temp", 3.0)

else:
    raise ValueError(f"Unknown case: {case}")

train.main()
PY
    PY_PID=$!

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
