#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/home/u1657859/miniconda3/envs/eamamba/bin/python}"
ROOT="${ROOT:-/work/u1657859/Jess/app-isolated-20260227-191252}"
BASE_CFG="${BASE_CFG:-/work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_strict_full_recheck_s2048_20260305/config.json}"
CKPT_ROOT="${CKPT_ROOT:-/work/u1657859/Jess/app/local_checkpoints}"
SUB_DIR="${SUB_DIR:-/work/u1657859/Jess/app/local_checkpoints/submissions}"
VAL_DIR="${VAL_DIR:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final}"
SCORE_DIR="${SCORE_DIR:-/work/u1657859/Jess/app/score}"
SUMMARY_CSV="${SUMMARY_CSV:-${SCORE_DIR}/mev1_softsvd_peft_loop_$(date +%Y%m%d_%H%M%S).csv}"
MAX_STEPS="${MAX_STEPS:-600}"

mkdir -p "${SUB_DIR}" "${SCORE_DIR}"
cat > "${SUMMARY_CSV}" << CSV
run_name,seed,adapter_last_n_layers,svd_residual_mode,svd_residual_gate_init,peft_unfreeze_last_n_blocks,val_auc_epoch0,ckpt_path,sub_zip
CSV

# Format:
# seed,last_n_layers,svd_mode,gate_init,peft_last_n_blocks
GRID=(
  "2048,0,add,0.0,0"
  "2048,2,add,0.0,0"
  "2048,2,gated,-1.5,0"
  "2048,4,gated,-2.0,0"
  "2048,4,gated,-2.0,1"
  "2048,6,gated,-2.0,1"
)

for ITEM in "${GRID[@]}"; do
  IFS=',' read -r SEED LASTN MODE GINIT PEFTN <<< "${ITEM}"
  TAG="s${SEED}_ln${LASTN}_${MODE}_g${GINIT}_p${PEFTN}"
  RUN_NAME="Exp_M1soft_${TAG}_$(date +%m%d)"
  RUN_DIR="${CKPT_ROOT}/${RUN_NAME}"

  echo "[run] ${RUN_NAME}"
  "${PY}" - << PY
import json
import train

base_cfg = "${BASE_CFG}"
run_name = "${RUN_NAME}"
seed = int("${SEED}")
lastn = int("${LASTN}")
mode = "${MODE}"
ginit = float("${GINIT}")
peftn = int("${PEFTN}")

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = json.load(f)

for k, v in cfg.items():
    train.OPT_OVERRIDES[k] = v
    train.PARSER_DEFAULTS[k] = v

train.OPT_OVERRIDES["name"] = run_name
train.PARSER_DEFAULTS["name"] = run_name
train.OPT_OVERRIDES["seed"] = seed
train.PARSER_DEFAULTS["seed"] = seed
train.OPT_OVERRIDES["niter"] = 1
train.PARSER_DEFAULTS["niter"] = 1
train.OPT_OVERRIDES["max_train_steps_per_epoch"] = int("${MAX_STEPS}")
train.PARSER_DEFAULTS["max_train_steps_per_epoch"] = int("${MAX_STEPS}")
train.OPT_OVERRIDES["resume_path"] = None
train.PARSER_DEFAULTS["resume_path"] = None

train.OPT_OVERRIDES["adapter_last_n_layers"] = lastn
train.PARSER_DEFAULTS["adapter_last_n_layers"] = lastn
train.OPT_OVERRIDES["svd_residual_mode"] = mode
train.PARSER_DEFAULTS["svd_residual_mode"] = mode
train.OPT_OVERRIDES["svd_residual_gate_init"] = ginit
train.PARSER_DEFAULTS["svd_residual_gate_init"] = ginit
train.OPT_OVERRIDES["peft_unfreeze_last_n_blocks"] = peftn
train.PARSER_DEFAULTS["peft_unfreeze_last_n_blocks"] = peftn
train.OPT_OVERRIDES["peft_unfreeze_layernorm"] = True
train.PARSER_DEFAULTS["peft_unfreeze_layernorm"] = True
train.OPT_OVERRIDES["peft_unfreeze_bias"] = False
train.PARSER_DEFAULTS["peft_unfreeze_bias"] = False

train.main()
PY

  LOG_PATH="$(ls -1t "${RUN_DIR}"/train_*.log | head -n 1)"
  CKPT_PATH="${RUN_DIR}/best_auc.pt"
  VAL_AUC="$("${PY}" - << PY
import re
log_path = "${LOG_PATH}"
pat = re.compile(r"epoch\\s+0\\s+val .*?auc=([0-9]*\\.?[0-9]+)")
auc = ""
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            auc = m.group(1)
            break
print(auc)
PY
)"

  SUB_TXT="${SUB_DIR}/sub_${RUN_NAME}_e0.txt"
  SUB_ZIP="${SUB_DIR}/sub_${RUN_NAME}_e0.zip"
  "${PY}" "${ROOT}/full_submission.py" \
    --ckpt "${CKPT_PATH}" \
    --txt_path "${VAL_DIR}" \
    --root_dir "${VAL_DIR}" \
    --out_csv "${SUB_TXT}" \
    --zip_submission \
    --zip_path "${SUB_ZIP}" \
    --batch_size 64 \
    --num_workers 4 \
    --strict \
    --tta_mode none

  echo "${RUN_NAME},${SEED},${LASTN},${MODE},${GINIT},${PEFTN},${VAL_AUC},${CKPT_PATH},${SUB_ZIP}" >> "${SUMMARY_CSV}"
  echo "[done] ${RUN_NAME} val_auc=${VAL_AUC} zip=${SUB_ZIP}"
done

echo "[all_done] ${SUMMARY_CSV}"
