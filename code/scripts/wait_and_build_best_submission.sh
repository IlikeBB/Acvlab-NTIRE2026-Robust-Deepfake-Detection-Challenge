#!/usr/bin/env bash
set -euo pipefail

EXP_PREFIX="${1:-Exp_MEv2_s3daug_v1}"
REPO_DIR="${2:-/work/u1657859/Jess/app}"
PY_BIN="${3:-/home/u1657859/miniconda3/envs/eamamba/bin/python}"

SUB_DIR="${REPO_DIR}/local_checkpoints/submissions"
OUT_TXT="${SUB_DIR}/sub_mev2_s3daug_v1_best.txt"
OUT_ZIP="${SUB_DIR}/sub_mev2_s3daug_v1_best.zip"

VAL_DIR="/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final"

mkdir -p "${SUB_DIR}"

wait_for_training_end() {
  while true; do
    pid="$(
      ps -eo pid,args | awk -v expname="${EXP_PREFIX}" '
        /run_staged_ext_consistency.py/ && $0 ~ ("--exp_prefix[[:space:]]+" expname) {print $1; exit}
      '
    )"
    if [[ -z "${pid}" ]]; then
      break
    fi
    echo "[wait] ${EXP_PREFIX} still running (pid=${pid})"
    sleep 60
  done
}

pick_best_ckpt() {
  local c="${REPO_DIR}/local_checkpoints/${EXP_PREFIX}_C_cons/best_auc.pt"
  local b="${REPO_DIR}/local_checkpoints/${EXP_PREFIX}_B_compft/best_auc.pt"
  local a="${REPO_DIR}/local_checkpoints/${EXP_PREFIX}_A_extwarm/best_auc.pt"

  if [[ -f "${c}" ]]; then
    echo "${c}"
    return 0
  fi
  if [[ -f "${b}" ]]; then
    echo "${b}"
    return 0
  fi
  if [[ -f "${a}" ]]; then
    echo "${a}"
    return 0
  fi
  return 1
}

wait_for_training_end

CKPT="$(pick_best_ckpt)"
if [[ -z "${CKPT}" ]]; then
  echo "[error] no checkpoint found under ${REPO_DIR}/local_checkpoints/${EXP_PREFIX}_*"
  exit 2
fi

echo "[submit] ckpt=${CKPT}"
"${PY_BIN}" "${REPO_DIR}/full_submission.py" \
  --ckpt "${CKPT}" \
  --txt_path "${VAL_DIR}" \
  --root_dir "${VAL_DIR}" \
  --out_csv "${OUT_TXT}" \
  --zip_submission \
  --zip_path "${OUT_ZIP}" \
  --batch_size 64 \
  --num_workers 8 \
  --multi_expert_enable \
  --multi_expert_k 3 \
  --multi_expert_route quality_bin \
  --infer_quality_bin \
  --tta_mode none \
  --on_error skip

"${PY_BIN}" - << 'PY'
from pathlib import Path
import zipfile

zip_path = Path("/work/u1657859/Jess/app/local_checkpoints/submissions/sub_mev2_s3daug_v1_best.zip")
if not zip_path.exists():
    raise SystemExit("[check] zip missing")

with zipfile.ZipFile(zip_path, "r") as zf:
    names = zf.namelist()
    if names != ["submission.txt"]:
        raise SystemExit(f"[check] invalid zip members: {names}")
    lines = zf.read("submission.txt").decode("utf-8").strip().splitlines()
    if len(lines) != 100:
        raise SystemExit(f"[check] expected 100 lines, got {len(lines)}")
    vals = [float(x.strip()) for x in lines]
    bad = [i + 1 for i, v in enumerate(vals) if (v < 0.0 or v > 1.0)]
    if bad:
        raise SystemExit(f"[check] out-of-range scores at lines: {bad[:10]}")
print("[check] submission format ok: 100 lines in [0,1], single submission.txt")
PY

echo "[done] best submission ready: ${OUT_ZIP}"
