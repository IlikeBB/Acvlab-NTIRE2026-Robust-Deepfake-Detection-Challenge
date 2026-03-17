#!/usr/bin/env bash
set -euo pipefail

PY=/home/u1657859/miniconda3/envs/eamamba/bin/python
ROOT=/work/u1657859/Jess/app-isolated-20260227-191252
BASE_CFG=/work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_strict_full_recheck_s2048_20260305/config.json
CKPT_ROOT=/work/u1657859/Jess/app/local_checkpoints
SUB_DIR=/work/u1657859/Jess/app/local_checkpoints/submissions
VAL_DIR=/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final
SCORE_DIR=/work/u1657859/Jess/app/score
SEED="${SEED:-2048}"
NITER="${NITER:-1}"
TAG="${TAG:-fm256scan_0308}"
SUMMARY_CSV="${SCORE_DIR}/${TAG}_summary.csv"
CASES_ONLY="${CASES_ONLY:-}"
TRAIN_FRAME_NUM_OVERRIDE="${TRAIN_FRAME_NUM_OVERRIDE:-}"
VAL_FRAME_NUM_OVERRIDE="${VAL_FRAME_NUM_OVERRIDE:-}"

mkdir -p "${SUB_DIR}" "${SCORE_DIR}"

cat > "${SUMMARY_CSV}" << CSV
alias,run_name,clip_model,image_size,batch_size,normalize_mean,normalize_std,val_auc_epoch0,log_path,ckpt_path,sub_zip
CSV

# alias|clip_model|image_size|batch_size|normalize_mean_csv|normalize_std_csv
CASES=(
  "clipl14|/work/u1657859/ntire_RDFC_2026_cvpr/clip-vit-large-patch14/|224|24|0.48145466,0.4578275,0.40821073|0.26862954,0.26130258,0.27577711"
  "eva02l|timm:eva02_large_patch14_clip_224.merged2b|224|12|0.48145466,0.4578275,0.40821073|0.26862954,0.26130258,0.27577711"
  "dinov2l|timm:vit_large_patch14_dinov2.lvd142m|224|12|0.485,0.456,0.406|0.229,0.224,0.225"
  "siglipl256|timm:vit_large_patch16_siglip_256.v2_webli|256|10|0.5,0.5,0.5|0.5,0.5,0.5"
)

for CASE in "${CASES[@]}"; do
  IFS='|' read -r ALIAS CLIP_MODEL IMAGE_SIZE BATCH_SIZE NMEAN NSTD <<< "${CASE}"
  if [[ -n "${CASES_ONLY}" ]]; then
    if [[ ",${CASES_ONLY}," != *",${ALIAS},"* ]]; then
      continue
    fi
  fi
  RUN_NAME="Exp_FM256_${ALIAS}_s${SEED}_e${NITER}_0308"
  RUN_DIR="${CKPT_ROOT}/${RUN_NAME}"

  echo "[run] ${RUN_NAME}"
  echo "  model=${CLIP_MODEL}"
  echo "  image_size=${IMAGE_SIZE} batch_size=${BATCH_SIZE}"
  echo "  norm_mean=${NMEAN} norm_std=${NSTD}"

  "${PY}" - << PY
import json
import train

base_cfg = "${BASE_CFG}"
seed = int("${SEED}")
niter = int("${NITER}")
run_name = "${RUN_NAME}"
clip_model = "${CLIP_MODEL}"
image_size = int("${IMAGE_SIZE}")
batch_size = int("${BATCH_SIZE}")
norm_mean = [float(v) for v in "${NMEAN}".split(",")]
norm_std = [float(v) for v in "${NSTD}".split(",")]

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = json.load(f)

for k, v in cfg.items():
    train.OPT_OVERRIDES[k] = v
    train.PARSER_DEFAULTS[k] = v

def set_opt(k, v):
    train.OPT_OVERRIDES[k] = v
    train.PARSER_DEFAULTS[k] = v

set_opt("name", run_name)
set_opt("seed", seed)
set_opt("niter", niter)
set_opt("resume_path", None)
set_opt("clip_model", clip_model)
set_opt("image_size", image_size)
set_opt("batch_size", batch_size)
set_opt("normalize_mean", norm_mean)
set_opt("normalize_std", norm_std)
if "${TRAIN_FRAME_NUM_OVERRIDE}" != "":
    set_opt("train_frame_num", int("${TRAIN_FRAME_NUM_OVERRIDE}"))
if "${VAL_FRAME_NUM_OVERRIDE}" != "":
    set_opt("val_frame_num", int("${VAL_FRAME_NUM_OVERRIDE}"))

train.main()
PY

  LOG_PATH="$(ls -1t "${RUN_DIR}"/train_*.log | head -n 1)"
  CKPT_PATH="${RUN_DIR}/best_auc.pt"
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[warn] missing ckpt: ${CKPT_PATH}"
    continue
  fi

  VAL_AUC=$(
    "${PY}" - << PY
import re
log_path = "${LOG_PATH}"
auc = ""
pat = re.compile(r"epoch\\s+0\\s+val .*?auc=([0-9]*\\.?[0-9]+)")
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            auc = m.group(1)
            break
print(auc)
PY
  )

  SUB_TXT="${SUB_DIR}/sub_fm_${ALIAS}_s${SEED}_e0_0308.txt"
  SUB_ZIP="${SUB_DIR}/sub_fm_${ALIAS}_s${SEED}_e0_0308.zip"

  "${PY}" "${ROOT}/full_submission.py" \
    --ckpt "${CKPT_PATH}" \
    --txt_path "${VAL_DIR}" \
    --root_dir "${VAL_DIR}" \
    --out_csv "${SUB_TXT}" \
    --zip_submission \
    --zip_path "${SUB_ZIP}" \
    --batch_size 64 \
    --num_workers 4 \
    --image_size "${IMAGE_SIZE}" \
    --normalize_mean ${NMEAN//,/ } \
    --normalize_std ${NSTD//,/ } \
    --tta_mode none \
    --on_error skip \
    --strict

  "${PY}" - << PY
from pathlib import Path
import zipfile
z = Path("${SUB_ZIP}")
if not z.exists():
    raise SystemExit(f"zip missing: {z}")
with zipfile.ZipFile(z, "r") as zf:
    names = zf.namelist()
    if names != ["submission.txt"]:
        raise SystemExit(f"invalid zip members: {names}")
    lines = zf.read("submission.txt").decode("utf-8").strip().splitlines()
    if len(lines) != 100:
        raise SystemExit(f"expected 100 lines, got {len(lines)}")
print("zip ok", z)
PY

  echo "${ALIAS},${RUN_NAME},${CLIP_MODEL},${IMAGE_SIZE},${BATCH_SIZE},${NMEAN},${NSTD},${VAL_AUC},${LOG_PATH},${CKPT_PATH},${SUB_ZIP}" >> "${SUMMARY_CSV}"
  echo "[done] ${RUN_NAME} val_auc_epoch0=${VAL_AUC} sub=${SUB_ZIP}"
done

echo "[all_done] summary=${SUMMARY_CSV}"
