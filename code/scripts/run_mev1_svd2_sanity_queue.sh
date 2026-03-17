#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/home/u1657859/miniconda3/envs/eamamba/bin/python}"
ROOT="${ROOT:-/work/u1657859/Jess/app-isolated-20260227-191252}"
BASE_CFG="${BASE_CFG:-/work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_strict_full_recheck_s2048_20260305/config.json}"
CKPT_ROOT="${CKPT_ROOT:-/work/u1657859/Jess/app/local_checkpoints}"
SUB_DIR="${SUB_DIR:-/work/u1657859/Jess/app/local_checkpoints/submissions}"
TEST_DIR="${TEST_DIR:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final}"
SCORE_DIR="${SCORE_DIR:-/work/u1657859/Jess/app/score}"

RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M)}"
SEED="${SEED:-2048}"
CASE_FILTER="${CASE_FILTER:-}"

# stage-1 sanity: short run for quick rejection
SANITY_NITER="${SANITY_NITER:-1}"
SANITY_MAX_STEPS="${SANITY_MAX_STEPS:-1200}"

# stage-2 full: strict full epoch (not truncated)
FULL_NITER="${FULL_NITER:-1}"
FULL_MAX_STEPS="${FULL_MAX_STEPS:-0}"
MAX_FULL_CASES="${MAX_FULL_CASES:-6}"

# sanity gates
MIN_SANITY_VAL="${MIN_SANITY_VAL:-0.846}"
MIN_SANITY_PROXY="${MIN_SANITY_PROXY:-0.852}"
MAX_SANITY_P99="${MAX_SANITY_P99:-0.995}"
MIN_SANITY_STD="${MIN_SANITY_STD:-0.20}"
MAX_SANITY_STD="${MAX_SANITY_STD:-0.45}"
REQUIRE_PROXY="${REQUIRE_PROXY:-1}"

PROXY_JSON="${PROXY_JSON:-${SCORE_DIR}/lb_proxy_linear_v2.json}"
SUMMARY_CSV="${SUMMARY_CSV:-${SCORE_DIR}/mev1_svd2_sanity_queue_${RUN_STAMP}.csv}"

mkdir -p "${SUB_DIR}" "${SCORE_DIR}"
cat > "${SUMMARY_CSV}" << CSV
ts,stage,case_id,run_name,seed,niter,max_steps,adapter_last_n,svd_mode,gate_init,train_mode,peft_last_n,val_auc_epoch0,val_auc_best,max_step_logged,pred_lb_best,p01,p50,p99,std,best_zip,last_zip,status
CSV

echo "[proxy] fitting LB proxy -> ${PROXY_JSON}"
"${PY}" "${ROOT}/scripts/lb_proxy_recalibrate.py" fit \
  --registry "${SCORE_DIR}/submission_lb_registry.csv" \
  --val_map "${SCORE_DIR}/merged_lb_table_app_app4.csv" \
  --out_json "${PROXY_JSON}" \
  >/dev/null || true

float_ge() {
  local a="$1"
  local b="$2"
  if [[ -z "${a}" ]]; then
    echo 0
    return
  fi
  awk -v a="${a}" -v b="${b}" 'BEGIN{print (a+0>=b+0)?1:0}'
}

float_le() {
  local a="$1"
  local b="$2"
  if [[ -z "${a}" ]]; then
    echo 0
    return
  fi
  awk -v a="${a}" -v b="${b}" 'BEGIN{print (a+0<=b+0)?1:0}'
}

parse_auc() {
  local log_path="$1"
  "${PY}" - "${log_path}" << 'PY'
import re,sys
log_path=sys.argv[1]
pat=re.compile(r"epoch\s+(\d+)\s+val\b.*?\bauc=([0-9]*\.?[0-9]+)")
epoch0=""
best=""
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m=pat.search(line)
        if not m:
            continue
        ep=int(m.group(1))
        auc=float(m.group(2))
        if ep==0 and epoch0=="":
            epoch0=f"{auc:.6f}"
        if best=="" or auc>float(best):
            best=f"{auc:.6f}"
print(epoch0)
print(best)
PY
}

parse_step_max() {
  local log_path="$1"
  "${PY}" - "${log_path}" << 'PY'
import re,sys
log_path=sys.argv[1]
pat=re.compile(r"\[train\]\s+step\s+(\d+)")
mx=0
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m=pat.search(line)
        if m:
            mx=max(mx,int(m.group(1)))
print(mx)
PY
}

predict_lb() {
  local sub_path="$1"
  local val_auc="$2"
  if [[ ! -f "${PROXY_JSON}" ]]; then
    echo ""
    return
  fi
  local raw
  raw="$("${PY}" "${ROOT}/scripts/lb_proxy_recalibrate.py" predict \
    --model_json "${PROXY_JSON}" \
    --submission "${sub_path}" \
    --val_auc "${val_auc}" \
    2>/dev/null | tail -n 1 || true)"
  if [[ -z "${raw}" ]]; then
    echo ""
    return
  fi
  "${PY}" -c 'import json,sys
raw=sys.argv[1]
try:
    obj=json.loads(raw)
    v=obj.get("pred_lb")
    print("" if v is None else f"{float(v):.6f}")
except Exception:
    print("")' "${raw}"
}

zip_stats() {
  local zip_path="$1"
  "${PY}" - "${zip_path}" << 'PY'
import zipfile, numpy as np, sys
p=sys.argv[1]
try:
    with zipfile.ZipFile(p, "r") as z:
        txt=[n for n in z.namelist() if n.endswith(".txt")]
        if not txt:
            print(",,,")
            raise SystemExit
        vals=[float(x.strip()) for x in z.read(txt[0]).decode("utf-8", errors="ignore").splitlines() if x.strip()]
    if len(vals)==0:
        print(",,,")
        raise SystemExit
    a=np.asarray(vals, dtype=np.float64)
    print(f"{np.quantile(a,0.01):.6f},{np.quantile(a,0.50):.6f},{np.quantile(a,0.99):.6f},{a.std():.6f}")
except Exception:
    print(",,,")
PY
}

train_case() {
  local stage="$1"
  local case_id="$2"
  local adapter_n="$3"
  local svd_mode="$4"
  local gate_init="$5"
  local train_mode="$6"
  local peft_n="$7"
  local niter="$8"
  local max_steps="$9"

  local run_name="M1Q_${case_id}_s${SEED}_${RUN_STAMP}_${stage}"
  local run_dir="${CKPT_ROOT}/${run_name}"
  local best_ckpt="${run_dir}/best_auc.pt"
  local last_ckpt="${run_dir}/last.pt"
  local best_txt="${SUB_DIR}/sub_${run_name}_best.txt"
  local best_zip="${SUB_DIR}/sub_${run_name}_best.zip"
  local last_txt="${SUB_DIR}/sub_${run_name}_last.txt"
  local last_zip="${SUB_DIR}/sub_${run_name}_last.zip"
  local status="train_done"
  local val_auc_e0=""
  local val_auc_best=""
  local step_max=""
  local pred_lb_best=""
  local p01=""
  local p50=""
  local p99=""
  local sstd=""

  echo "[run:${stage}] ${run_name}"
  if ! "${PY}" - << PY
import json
import train

with open("${BASE_CFG}", "r", encoding="utf-8") as f:
    cfg=json.load(f)

for k,v in cfg.items():
    train.OPT_OVERRIDES[k]=v
    train.PARSER_DEFAULTS[k]=v

def set_opt(k,v):
    train.OPT_OVERRIDES[k]=v
    train.PARSER_DEFAULTS[k]=v

set_opt("name", "${run_name}")
set_opt("seed", int("${SEED}"))
set_opt("niter", int("${niter}"))
set_opt("max_train_steps_per_epoch", int("${max_steps}"))
set_opt("resume_path", None)

# Keep strict MEV1 recipe; only vary this ablation axis.
set_opt("adapter_last_n_layers", int("${adapter_n}"))
set_opt("svd_residual_mode", "${svd_mode}")
set_opt("svd_residual_gate_init", float("${gate_init}"))
set_opt("svd_residual_train_mode", "${train_mode}")
set_opt("peft_unfreeze_last_n_blocks", int("${peft_n}"))
set_opt("peft_unfreeze_layernorm", True)
set_opt("peft_unfreeze_bias", False)

train.main()
PY
  then
    status="train_failed"
  fi

  local log_path
  log_path="$(ls -1t "${run_dir}"/train_*.log 2>/dev/null | head -n 1 || true)"
  if [[ -n "${log_path}" ]]; then
    readarray -t aucs < <(parse_auc "${log_path}")
    val_auc_e0="${aucs[0]:-}"
    val_auc_best="${aucs[1]:-}"
    step_max="$(parse_step_max "${log_path}")"
  fi

  if [[ "${status}" == "train_done" ]]; then
    if [[ -f "${best_ckpt}" ]]; then
      "${PY}" "${ROOT}/full_submission.py" \
        --ckpt "${best_ckpt}" \
        --txt_path "${TEST_DIR}" \
        --root_dir "${TEST_DIR}" \
        --out_csv "${best_txt}" \
        --zip_submission \
        --zip_path "${best_zip}" \
        --batch_size 64 \
        --num_workers 4 \
        --strict \
        --tta_mode none \
        --on_error skip || status="best_submit_failed"
      if [[ -f "${best_zip}" ]]; then
        pred_lb_best="$(predict_lb "${best_zip}" "${val_auc_best}")"
        stats="$(zip_stats "${best_zip}")"
        p01="$(echo "${stats}" | cut -d',' -f1)"
        p50="$(echo "${stats}" | cut -d',' -f2)"
        p99="$(echo "${stats}" | cut -d',' -f3)"
        sstd="$(echo "${stats}" | cut -d',' -f4)"
      fi
    else
      status="best_ckpt_missing"
    fi

    if [[ -f "${last_ckpt}" ]]; then
      "${PY}" "${ROOT}/full_submission.py" \
        --ckpt "${last_ckpt}" \
        --txt_path "${TEST_DIR}" \
        --root_dir "${TEST_DIR}" \
        --out_csv "${last_txt}" \
        --zip_submission \
        --zip_path "${last_zip}" \
        --batch_size 64 \
        --num_workers 4 \
        --strict \
        --tta_mode none \
        --on_error skip || true
    fi
  fi

  echo "$(date '+%F %T'),${stage},${case_id},${run_name},${SEED},${niter},${max_steps},${adapter_n},${svd_mode},${gate_init},${train_mode},${peft_n},${val_auc_e0},${val_auc_best},${step_max},${pred_lb_best},${p01},${p50},${p99},${sstd},${best_zip},${last_zip},${status}" >> "${SUMMARY_CSV}"
  echo "[done:${stage}] ${run_name} val=${val_auc_best:-NA} step_max=${step_max:-NA} pred=${pred_lb_best:-NA} p99=${p99:-NA} std=${sstd:-NA}"
}

# case format:
# case_id|adapter_n|svd_mode|gate_init|train_mode|peft_last_n
CASES=(
  "b0|0|add|0.0|full|0"
  "a1|1|add|0.0|full|0"
  "a2|2|add|0.0|full|0"
  "a3|3|add|0.0|full|0"
  "a4|4|add|0.0|full|0"
  "a2p1|2|add|0.0|full|1"
  "a4p1|4|add|0.0|full|1"
  "s2|2|softplus|-2.0|sigma_only|0"
  "s4|4|softplus|-2.0|sigma_only|0"
  "u2|2|softplus|-1.5|uv_only|0"
  "u4|4|softplus|-1.5|uv_only|0"
  "g4|4|gated|-1.5|full|0"
)

FULL_STARTED=0

for item in "${CASES[@]}"; do
  IFS='|' read -r CASE_ID ADAPTER_N SVD_MODE GATE_INIT TRAIN_MODE PEFT_N <<< "${item}"
  if [[ -n "${CASE_FILTER}" && ",${CASE_FILTER}," != *",${CASE_ID},"* ]]; then
    continue
  fi

  train_case "sanity" "${CASE_ID}" "${ADAPTER_N}" "${SVD_MODE}" "${GATE_INIT}" "${TRAIN_MODE}" "${PEFT_N}" "${SANITY_NITER}" "${SANITY_MAX_STEPS}"

  row="$("${PY}" -c 'import csv,sys
path,case_id=sys.argv[1],sys.argv[2]
last=None
with open(path,"r",encoding="utf-8",newline="") as f:
    for r in csv.DictReader(f):
        if r.get("stage")=="sanity" and r.get("case_id")==case_id:
            last=r
if last is None:
    print("||||||")
else:
    print("{}|{}|{}|{}|{}|{}".format(
        last.get("val_auc_best",""),
        last.get("pred_lb_best",""),
        last.get("p99",""),
        last.get("std",""),
        last.get("status",""),
        last.get("max_step_logged",""),
    ))' "${SUMMARY_CSV}" "${CASE_ID}")"

  SANITY_VAL="$(echo "${row}" | cut -d'|' -f1)"
  SANITY_PRED="$(echo "${row}" | cut -d'|' -f2)"
  SANITY_P99="$(echo "${row}" | cut -d'|' -f3)"
  SANITY_STD="$(echo "${row}" | cut -d'|' -f4)"
  SANITY_STATUS="$(echo "${row}" | cut -d'|' -f5)"
  SANITY_STEPS="$(echo "${row}" | cut -d'|' -f6)"

  if [[ "${SANITY_STATUS}" != "train_done" && "${SANITY_STATUS}" != "best_submit_failed" ]]; then
    echo "[skip:full] ${CASE_ID} sanity_status=${SANITY_STATUS:-NA}"
    continue
  fi

  val_ok="$(float_ge "${SANITY_VAL}" "${MIN_SANITY_VAL}")"
  p99_ok="$(float_le "${SANITY_P99}" "${MAX_SANITY_P99}")"
  std_lo_ok="$(float_ge "${SANITY_STD}" "${MIN_SANITY_STD}")"
  std_hi_ok="$(float_le "${SANITY_STD}" "${MAX_SANITY_STD}")"
  step_ok="$(float_ge "${SANITY_STEPS}" "${SANITY_MAX_STEPS}")"

  pred_ok=1
  if [[ "${REQUIRE_PROXY}" == "1" ]]; then
    pred_ok="$(float_ge "${SANITY_PRED}" "${MIN_SANITY_PROXY}")"
  fi

  if [[ "${FULL_STARTED}" -ge "${MAX_FULL_CASES}" ]]; then
    echo "[skip:full] ${CASE_ID} reached MAX_FULL_CASES=${MAX_FULL_CASES}"
    continue
  fi

  if [[ "${val_ok}" == "1" && "${pred_ok}" == "1" && "${p99_ok}" == "1" && "${std_lo_ok}" == "1" && "${std_hi_ok}" == "1" && "${step_ok}" == "1" ]]; then
    train_case "full1ep" "${CASE_ID}" "${ADAPTER_N}" "${SVD_MODE}" "${GATE_INIT}" "${TRAIN_MODE}" "${PEFT_N}" "${FULL_NITER}" "${FULL_MAX_STEPS}"
    FULL_STARTED=$((FULL_STARTED+1))
  else
    echo "[skip:full] ${CASE_ID} val=${SANITY_VAL:-NA} pred=${SANITY_PRED:-NA} p99=${SANITY_P99:-NA} std=${SANITY_STD:-NA} steps=${SANITY_STEPS:-NA}"
  fi
done

echo "[all_done] ${SUMMARY_CSV}"
