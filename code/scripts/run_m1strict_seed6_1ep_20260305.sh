#!/usr/bin/env bash
set -euo pipefail

PY=/home/u1657859/miniconda3/envs/eamamba/bin/python
ROOT=/work/u1657859/Jess/app-isolated-20260227-191252
BASE_CFG=/work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_strict_full_recheck_s2048_20260305/config.json
CKPT_ROOT=/work/u1657859/Jess/app/local_checkpoints
SUB_DIR=/work/u1657859/Jess/app/local_checkpoints/submissions
VAL_DIR=/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final
SCORE_DIR=/work/u1657859/Jess/app/score
SUMMARY_CSV=${SCORE_DIR}/m1strict_seed6_1ep_summary_20260305.csv

mkdir -p "${SUB_DIR}" "${SCORE_DIR}"

# 6 different seeds (including baseline seed 2048)
SEEDS=(2048 3407 42 777 2026 1234)

# summary header
cat > "${SUMMARY_CSV}" << CSV
seed,run_name,ckpt_path,val_auc_epoch0,log_path,sub_txt,sub_zip
CSV

for SEED in "${SEEDS[@]}"; do
  RUN_NAME="Exp_M1strict_s${SEED}_1ep_0305"
  RUN_DIR="${CKPT_ROOT}/${RUN_NAME}"
  LOG_PATH=""

  echo "[run] seed=${SEED} run=${RUN_NAME} start=$(date '+%F %T')"

  "${PY}" - << PY
import json
import train

base_cfg = "${BASE_CFG}"
seed = int("${SEED}")
run_name = "${RUN_NAME}"

with open(base_cfg, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

# Load strict base config, then override only run-identifying fields.
for k, v in cfg.items():
    train.OPT_OVERRIDES[k] = v
    train.PARSER_DEFAULTS[k] = v

train.OPT_OVERRIDES['name'] = run_name
train.PARSER_DEFAULTS['name'] = run_name

train.OPT_OVERRIDES['seed'] = seed
train.PARSER_DEFAULTS['seed'] = seed

# user request: each version = 1 epoch
train.OPT_OVERRIDES['niter'] = 1
train.PARSER_DEFAULTS['niter'] = 1

# keep strict from-scratch behavior
train.OPT_OVERRIDES['resume_path'] = None
train.PARSER_DEFAULTS['resume_path'] = None

train.main()
PY

  LOG_PATH=$(ls -1t "${RUN_DIR}"/train_*.log | head -n 1)
  CKPT_PATH="${RUN_DIR}/best_auc.pt"

  VAL_AUC=$(
    "${PY}" - << PY
import re
log_path = "${LOG_PATH}"
auc = ""
pat = re.compile(r"epoch\\s+0\\s+val .*?auc=([0-9]*\\.?[0-9]+)")
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = pat.search(line)
        if m:
            auc = m.group(1)
            break
print(auc)
PY
  )

  SUB_TXT="${SUB_DIR}/sub_m1strict_s${SEED}_e0_0305.txt"
  SUB_ZIP="${SUB_DIR}/sub_m1strict_s${SEED}_e0_0305.zip"

  echo "[infer] seed=${SEED} ckpt=${CKPT_PATH}"
  "${PY}" "${ROOT}/full_submission.py" \
    --ckpt "${CKPT_PATH}" \
    --txt_path "${VAL_DIR}" \
    --root_dir "${VAL_DIR}" \
    --out_csv "${SUB_TXT}" \
    --zip_submission \
    --zip_path "${SUB_ZIP}" \
    --batch_size 64 \
    --num_workers 4 \
    --image_size 224 \
    --multi_expert_enable \
    --multi_expert_k 3 \
    --multi_expert_route quality_bin \
    --infer_quality_bin \
    --quality_cuts 4.719781875610352 5.441120147705078 \
    --tta_mode none \
    --on_error skip \
    --strict

  "${PY}" - << PY
from pathlib import Path
import zipfile
z = Path("${SUB_ZIP}")
if not z.exists():
    raise SystemExit(f"zip missing: {z}")
with zipfile.ZipFile(z, 'r') as zf:
    names = zf.namelist()
    if names != ['submission.txt']:
        raise SystemExit(f"invalid zip members: {names}")
    lines = zf.read('submission.txt').decode('utf-8').strip().splitlines()
    if len(lines) != 100:
        raise SystemExit(f"expected 100 lines, got {len(lines)}")
print('zip ok', z)
PY

  echo "${SEED},${RUN_NAME},${CKPT_PATH},${VAL_AUC},${LOG_PATH},${SUB_TXT},${SUB_ZIP}" >> "${SUMMARY_CSV}"
  echo "[done] seed=${SEED} val_auc_epoch0=${VAL_AUC} zip=${SUB_ZIP}"

done

echo "[all_done] summary=${SUMMARY_CSV}"
