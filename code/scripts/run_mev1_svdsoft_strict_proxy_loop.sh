#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/home/u1657859/miniconda3/envs/eamamba/bin/python}"
ROOT="${ROOT:-/work/u1657859/Jess/app-isolated-20260227-191252}"
BASE_CFG="${BASE_CFG:-/work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_strict_full_recheck_s2048_20260305/config.json}"
CKPT_ROOT="${CKPT_ROOT:-/work/u1657859/Jess/app/local_checkpoints}"
SUB_DIR="${SUB_DIR:-/work/u1657859/Jess/app/local_checkpoints/submissions}"
TEST_DIR="${TEST_DIR:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final}"
SCORE_DIR="${SCORE_DIR:-/work/u1657859/Jess/app/score}"

RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
SEED="${SEED:-2048}"
CASE_FILTER="${CASE_FILTER:-}"

# Probe stage (quick screen)
PROBE_NITER="${PROBE_NITER:-1}"
PROBE_MAX_STEPS="${PROBE_MAX_STEPS:-800}"
MIN_VAL_FOR_FULL="${MIN_VAL_FOR_FULL:-0.84}"
MIN_PROXY_FOR_FULL="${MIN_PROXY_FOR_FULL:-0.845}"
RUN_FULL_IF_PRED="${RUN_FULL_IF_PRED:-1}"

# Full stage
FULL_NITER="${FULL_NITER:-}"
FULL_MAX_STEPS="${FULL_MAX_STEPS:-}"
if [[ -z "${FULL_NITER}" ]]; then
  FULL_NITER="$("${PY}" -c 'import json,sys;print(int(json.load(open(sys.argv[1],"r",encoding="utf-8")).get("niter",6)))' "${BASE_CFG}")"
fi
if [[ -z "${FULL_MAX_STEPS}" ]]; then
  FULL_MAX_STEPS="$("${PY}" -c 'import json,sys;print(int(json.load(open(sys.argv[1],"r",encoding="utf-8")).get("max_train_steps_per_epoch",0) or 0))' "${BASE_CFG}")"
fi

PROXY_JSON="${PROXY_JSON:-${SCORE_DIR}/lb_proxy_linear_v2.json}"
SUMMARY_CSV="${SUMMARY_CSV:-${SCORE_DIR}/mev1_svdsoft_strict_proxy_loop_${RUN_STAMP}.csv}"

mkdir -p "${SUB_DIR}" "${SCORE_DIR}"
cat > "${SUMMARY_CSV}" << CSV
ts,stage,case_id,run_name,seed,niter,max_steps,adapter_last_n,svd_mode,gate_init,train_mode,peft_last_n,val_auc_epoch0,val_auc_best,pred_lb_best,pred_lb_last,best_zip,last_zip,status
CSV

echo "[proxy] fitting LB proxy -> ${PROXY_JSON}"
"${PY}" "${ROOT}/scripts/lb_proxy_recalibrate.py" fit \
  --registry "${SCORE_DIR}/submission_lb_registry.csv" \
  --val_map "${SCORE_DIR}/merged_lb_table_app_app4.csv" \
  --out_json "${PROXY_JSON}" \
  >/dev/null || true

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

float_ge() {
  local a="$1"
  local b="$2"
  if [[ -z "${a}" ]]; then
    echo 0
    return
  fi
  awk -v a="${a}" -v b="${b}" 'BEGIN{print (a+0>=b+0)?1:0}'
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

  local run_name="Exp_M1svd2_${case_id}_s${SEED}_${RUN_STAMP}_${stage}"
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
  local pred_lb_best=""
  local pred_lb_last=""

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

# Strict MEV1 parity: only change these knobs.
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
      if [[ -f "${last_zip}" ]]; then
        pred_lb_last="$(predict_lb "${last_zip}" "${val_auc_best}")"
      fi
    fi
  fi

  echo "$(date '+%F %T'),${stage},${case_id},${run_name},${SEED},${niter},${max_steps},${adapter_n},${svd_mode},${gate_init},${train_mode},${peft_n},${val_auc_e0},${val_auc_best},${pred_lb_best},${pred_lb_last},${best_zip},${last_zip},${status}" >> "${SUMMARY_CSV}"
  echo "[done:${stage}] ${run_name} val_e0=${val_auc_e0:-NA} val_best=${val_auc_best:-NA} pred_best=${pred_lb_best:-NA}"
}

# case format:
# case_id|adapter_n|svd_mode|gate_init|train_mode|peft_last_n
CASES=(
  "strict_base|0|add|0.0|full|0"
  "ln2_add_full|2|add|0.0|full|0"
  "ln4_add_full|4|add|0.0|full|0"
  "ln2_soft_sigma|2|softplus|-2.0|sigma_only|0"
  "ln4_soft_sigma|4|softplus|-2.0|sigma_only|0"
  "ln4_soft_uv|4|softplus|-1.5|uv_only|0"
  "ln4_gated_full|4|gated|0.0|full|0"
  "ln4_soft_sigma_p1|4|softplus|-2.0|sigma_only|1"
)

for item in "${CASES[@]}"; do
  IFS='|' read -r CASE_ID ADAPTER_N SVD_MODE GATE_INIT TRAIN_MODE PEFT_N <<< "${item}"
  if [[ -n "${CASE_FILTER}" && ",${CASE_FILTER}," != *",${CASE_ID},"* ]]; then
    continue
  fi

  train_case "probe" "${CASE_ID}" "${ADAPTER_N}" "${SVD_MODE}" "${GATE_INIT}" "${TRAIN_MODE}" "${PEFT_N}" "${PROBE_NITER}" "${PROBE_MAX_STEPS}"

  if [[ "${RUN_FULL_IF_PRED}" != "1" ]]; then
    continue
  fi

  # Read latest probe row for this case.
  row="$("${PY}" -c 'import csv,sys
path,case_id=sys.argv[1],sys.argv[2]
last=None
with open(path,"r",encoding="utf-8",newline="") as f:
    for r in csv.DictReader(f):
        if r.get("stage")=="probe" and r.get("case_id")==case_id:
            last=r
if last is None:
    print("|||")
else:
    print("{}|{}|{}".format(last.get("val_auc_best",""), last.get("pred_lb_best",""), last.get("status","")))' "${SUMMARY_CSV}" "${CASE_ID}")"
  PROBE_VAL="$(echo "${row}" | cut -d'|' -f1)"
  PROBE_PRED="$(echo "${row}" | cut -d'|' -f2)"
  PROBE_STATUS="$(echo "${row}" | cut -d'|' -f3)"
  if [[ "${PROBE_STATUS}" != "train_done" && "${PROBE_STATUS}" != "best_submit_failed" ]]; then
    continue
  fi

  val_ok="$(float_ge "${PROBE_VAL}" "${MIN_VAL_FOR_FULL}")"
  pred_ok=0
  if [[ -n "${PROBE_PRED}" ]]; then
    pred_ok="$(float_ge "${PROBE_PRED}" "${MIN_PROXY_FOR_FULL}")"
  fi
  if [[ "${val_ok}" == "1" && "${pred_ok}" == "1" ]]; then
    train_case "full" "${CASE_ID}" "${ADAPTER_N}" "${SVD_MODE}" "${GATE_INIT}" "${TRAIN_MODE}" "${PEFT_N}" "${FULL_NITER}" "${FULL_MAX_STEPS}"
  elif [[ "${val_ok}" == "1" && -z "${PROBE_PRED}" ]]; then
    train_case "full" "${CASE_ID}" "${ADAPTER_N}" "${SVD_MODE}" "${GATE_INIT}" "${TRAIN_MODE}" "${PEFT_N}" "${FULL_NITER}" "${FULL_MAX_STEPS}"
  else
    echo "[skip:full] ${CASE_ID} probe_val=${PROBE_VAL:-NA} probe_pred=${PROBE_PRED:-NA}"
  fi
done

echo "[all_done] ${SUMMARY_CSV}"
