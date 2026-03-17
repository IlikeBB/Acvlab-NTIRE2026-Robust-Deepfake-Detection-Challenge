#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PY_BIN="${PY_BIN:-/home/u1657859/miniconda3/envs/eamamba/bin/python}"
VAL_DIR="${VAL_DIR:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT_DIR}/checkpoints}"
OUT_DIR="${OUT_DIR:-/work/u1657859/Jess/app/local_checkpoints/submissions}"
BASELINE_AUC="${BASELINE_AUC:-0.8694}"
POLL_SEC="${POLL_SEC:-120}"
TARGET_EPOCH="${TARGET_EPOCH:-5}"
SUB_BATCH_SIZE="${SUB_BATCH_SIZE:-64}"
SUB_NUM_WORKERS="${SUB_NUM_WORKERS:-4}"
RUNS="${RUNS:-Exp_BB6_clean_backbone_only_L14_local_fz1_s2048;Exp_BB6_clean_backbone_only_L14_336_fz1_s2048}"

mkdir -p "${OUT_DIR}"

IFS=';' read -r -a RUN_LIST <<< "${RUNS}"

wait_one_run() {
  local run_name="$1"
  local run_dir="${CHECKPOINTS_DIR}/${run_name}"
  local target_ckpt="${run_dir}/epoch_${TARGET_EPOCH}.pt"

  echo "[wait] run=${run_name} target=${target_ckpt}"
  while true; do
    if [[ -f "${target_ckpt}" ]]; then
      echo "[wait] run=${run_name} reached epoch_${TARGET_EPOCH}.pt"
      break
    fi
    sleep "${POLL_SEC}"
  done
}

infer_image_size() {
  local run_dir="$1"
  "${PY_BIN}" - <<PY
import json
from pathlib import Path
p = Path("${run_dir}") / "config.json"
img = 224
if p.exists():
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        img = int(cfg.get("image_size", 224))
    except Exception:
        img = 224
print(img)
PY
}

build_one_submission() {
  local run_name="$1"
  local ckpt_kind="$2"   # best_auc or last
  local run_dir="${CHECKPOINTS_DIR}/${run_name}"
  local ckpt_path="${run_dir}/${ckpt_kind}.pt"
  local out_txt="${OUT_DIR}/${run_name}_${ckpt_kind}.txt"
  local out_zip="${OUT_DIR}/${run_name}_${ckpt_kind}.zip"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[warn] missing ckpt: ${ckpt_path}"
    return 1
  fi

  local image_size
  image_size="$(infer_image_size "${run_dir}")"

  echo "[submit] run=${run_name} kind=${ckpt_kind} image_size=${image_size}"
  "${PY_BIN}" "${ROOT_DIR}/full_submission.py" \
    --ckpt "${ckpt_path}" \
    --txt_path "${VAL_DIR}" \
    --root_dir "${VAL_DIR}" \
    --out_csv "${out_txt}" \
    --zip_submission \
    --zip_path "${out_zip}" \
    --batch_size "${SUB_BATCH_SIZE}" \
    --num_workers "${SUB_NUM_WORKERS}" \
    --image_size "${image_size}" \
    --tta_mode none \
    --on_error skip

  "${PY_BIN}" - <<PY
from pathlib import Path
import zipfile

z = Path("${out_zip}")
if not z.exists():
    raise SystemExit(f"[check] zip missing: {z}")
with zipfile.ZipFile(z, "r") as zf:
    names = zf.namelist()
    if names != ["submission.txt"]:
        raise SystemExit(f"[check] invalid zip members: {names}")
    lines = zf.read("submission.txt").decode("utf-8").strip().splitlines()
    if len(lines) != 100:
        raise SystemExit(f"[check] expected 100 lines, got {len(lines)}")
    vals = [float(x.strip().split(',')[0]) for x in lines]
    bad = [i + 1 for i, v in enumerate(vals) if not (0.0 <= v <= 1.0)]
    if bad:
        raise SystemExit(f"[check] out-of-range values at lines: {bad[:10]}")
print(f"[check] ok zip={z}")
PY
}

collect_auc_metrics() {
  local run_name="$1"
  local run_dir="${CHECKPOINTS_DIR}/${run_name}"
  "${PY_BIN}" - <<PY
import re
from pathlib import Path

run_dir = Path("${run_dir}")
logs = sorted(run_dir.glob("train_*.log"))
best = None
last = None
if logs:
    pat = re.compile(r"epoch\\s+(\\d+)\\s+val .*?auc=([0-9]*\\.?[0-9]+)")
    for line in logs[-1].read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if not m:
            continue
        auc = float(m.group(2))
        last = auc
        if best is None or auc > best:
            best = auc
if best is None:
    best = float("nan")
if last is None:
    last = float("nan")
print(f"{best},{last}")
PY
}

SUMMARY_FILE="${OUT_DIR}/summary_backbone_runs.txt"
{
  echo "baseline_auc=${BASELINE_AUC}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
} > "${SUMMARY_FILE}"

for run_name in "${RUN_LIST[@]}"; do
  wait_one_run "${run_name}"
  build_one_submission "${run_name}" "best_auc"
  build_one_submission "${run_name}" "last"

  auc_pair="$(collect_auc_metrics "${run_name}")"
  best_auc="${auc_pair%%,*}"
  last_auc="${auc_pair##*,}"

  delta_best="$("${PY_BIN}" - <<PY
base=float("${BASELINE_AUC}")
v=float("${best_auc}")
print("nan" if v != v else f"{(v-base):+.4f}")
PY
)"
  delta_last="$("${PY_BIN}" - <<PY
base=float("${BASELINE_AUC}")
v=float("${last_auc}")
print("nan" if v != v else f"{(v-base):+.4f}")
PY
)"

  {
    echo "run=${run_name}"
    echo "best_auc=${best_auc} delta_vs_baseline=${delta_best}"
    echo "last_auc=${last_auc} delta_vs_baseline=${delta_last}"
    echo "best_zip=${OUT_DIR}/${run_name}_best_auc.zip"
    echo "last_zip=${OUT_DIR}/${run_name}_last.zip"
    echo "---"
  } >> "${SUMMARY_FILE}"

  echo "[summary] run=${run_name} best_auc=${best_auc} delta=${delta_best} last_auc=${last_auc} delta_last=${delta_last}"
done

echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "${SUMMARY_FILE}"
echo "[done] summary -> ${SUMMARY_FILE}"
