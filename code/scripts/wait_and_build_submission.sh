#!/usr/bin/env bash
set -euo pipefail

EXP_PREFIX="${1:-Exp_MEv3_df40_drouter_rst_v3}"
REPO_DIR="${2:-/work/u1657859/Jess/app}"
PY_BIN="${3:-/home/u1657859/miniconda3/envs/eamamba/bin/python}"
ROUTE_MODE="${4:-degrade_router}"
EXPERT_K="${5:-3}"

SUB_DIR="${REPO_DIR}/local_checkpoints/submissions"
OUT_TXT="${SUB_DIR}/${EXP_PREFIX}_best.txt"
OUT_ZIP="${SUB_DIR}/${EXP_PREFIX}_best.zip"

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
  echo "[error] no checkpoint found for ${EXP_PREFIX}"
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
  --num_workers 4 \
  --multi_expert_enable \
  --multi_expert_k "${EXPERT_K}" \
  --multi_expert_route "${ROUTE_MODE}" \
  --tta_mode none \
  --on_error skip

"${PY_BIN}" - << PY
from pathlib import Path
import zipfile

zip_path = Path("${OUT_ZIP}")
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
print("[check] submission format ok")
print("[done] zip -> ${OUT_ZIP}")
PY

