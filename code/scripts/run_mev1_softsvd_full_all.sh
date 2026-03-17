#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/home/u1657859/miniconda3/envs/eamamba/bin/python}"
ROOT="${ROOT:-/work/u1657859/Jess/app-isolated-20260227-191252}"
BASE_CFG="${BASE_CFG:-/work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_strict_full_recheck_s2048_20260305/config.json}"
CKPT_ROOT="${CKPT_ROOT:-/work/u1657859/Jess/app/local_checkpoints}"
SUB_DIR="${SUB_DIR:-/work/u1657859/Jess/app/local_checkpoints/submissions}"
VAL_DIR="${VAL_DIR:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final}"
TEST_DIR="${TEST_DIR:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final}"
SCORE_DIR="${SCORE_DIR:-/work/u1657859/Jess/app/score}"
SUMMARY_CSV="${SUMMARY_CSV:-${SCORE_DIR}/mev1_softsvd_full_all_$(date +%Y%m%d_%H%M%S).csv}"
BASE_CFG_NITER="$("${PY}" - "${BASE_CFG}" << 'PY'
import json, sys
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(int(cfg.get("niter", 6)))
PY
)"
BASE_CFG_MAX_STEPS="$("${PY}" - "${BASE_CFG}" << 'PY'
import json, sys
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(int(cfg.get("max_train_steps_per_epoch", 0) or 0))
PY
)"
MAX_STEPS="${MAX_TRAIN_STEPS_PER_EPOCH:-${BASE_CFG_MAX_STEPS}}"
NITER="${NITER:-${BASE_CFG_NITER}}"
SEED="${SEED:-2048}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NAMES="${NAMES:-}"
MIN_AUC_FOR_SUBMIT="${MIN_AUC_FOR_SUBMIT:-0.84}"
STOP_IF_AUC_BELOW="${STOP_IF_AUC_BELOW:-0}"
MAX_AUC_SKIP_COUNT="${MAX_AUC_SKIP_COUNT:-99999}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
MEV1_FORCE_NO_OSD_REGS="${MEV1_FORCE_NO_OSD_REGS:-0}"
SYNC_CONTEXT_HUB="${SYNC_CONTEXT_HUB:-1}"
CONTEXT_HUB_CONTENT_ROOT="${CONTEXT_HUB_CONTENT_ROOT:-/work/u1657859/Jess/app/context-hub-content}"

mkdir -p "${SUB_DIR}" "${SCORE_DIR}"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
cat > "${SUMMARY_CSV}" << CSV
run_name,seed,adapter_last_n_layers,svd_residual_mode,svd_residual_gate_init,peft_unfreeze_last_n_blocks,niter,max_train_steps_per_epoch,val_auc_epoch0,best_ckpt,last_ckpt,best_sub,best_txt,last_sub,last_txt,submission_status
CSV
fi

# format:
# LASTN,MODE,GINIT,PEFTN
GRID=(
  "0,add,0.0,0"
  "2,add,0.0,0"
  "2,gated,-1.5,0"
  "4,gated,-2.0,0"
  "4,gated,-2.0,1"
  "6,gated,-2.0,1"
)

float_ge() {
  local a="$1"
  local b="$2"
  if [[ -z "${a}" ]]; then
    echo 0
    return
  fi
  awk -v a="${a}" -v b="${b}" 'BEGIN{print (a+0>=b+0)?1:0}'
}

float_lt() {
  local a="$1"
  local b="$2"
  if [[ -z "${a}" ]]; then
    echo 1
    return
  fi
  awk -v a="${a}" -v b="${b}" 'BEGIN{print (a+0<b+0)?1:0}'
}

parse_auc() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    echo ""
    return
  fi
  "${PY}" - "${log_path}" << 'PY'
import re, sys, os
log_path = sys.argv[1]
patterns = [
    re.compile(r"epoch\\s+0\\s+val\\b.*?\\bauc=([0-9]*\\.?[0-9]+)"),
    re.compile(r"\\bval_auc\\s*[:=]\\s*([0-9]*\\.?[0-9]+)"),
    re.compile(r"\\bval\\b[^\\n]*?\\bauc\\s*[:=]\\s*([0-9]*\\.?[0-9]+)", re.IGNORECASE),
    re.compile(r"\\bauc\\s*[:=]\\s*([0-9]*\\.?[0-9]+)"),
]
auc = ""
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if not auc:
            for pat in patterns:
                m = pat.search(line)
                if m:
                    auc = m.group(1)
                    break
if not auc:
    csv_path = os.path.splitext(log_path)[0] + ".csv"
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as cf:
            for line in cf:
                if line.startswith("epoch") and ",val_auc" in line:
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) > 2:
                        try:
                            float(parts[2])
                            auc = parts[2]
                            break
                        except ValueError:
                            continue
print(auc)
PY
}

build_submission() {
  local ckpt="$1"
  local txt_out="$2"
  local zip_out="$3"
  "${PY}" "${ROOT}/full_submission.py" \
    --ckpt "${ckpt}" \
    --txt_path "${TEST_DIR}" \
    --root_dir "${TEST_DIR}" \
    --out_csv "${txt_out}" \
    --zip_submission \
    --zip_path "${zip_out}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --strict \
    --tta_mode none \
    --on_error skip
}

submit_and_report() {
  local ckpt="$1"
  local txt_out="$2"
  local zip_out="$3"
  if [[ ! -f "${ckpt}" ]]; then
    echo ""
    return 1
  fi
  if ! build_submission "${ckpt}" "${txt_out}" "${zip_out}"; then
    echo ""
    return 1
  fi
  echo "${zip_out}"
}

sync_context_hub_tracker() {
  if [[ "${SYNC_CONTEXT_HUB}" != "1" ]]; then
    return
  fi
  "${PY}" "${ROOT}/scripts/update_context_hub_tracker.py" \
    --score-dir "${SCORE_DIR}" \
    --content-root "${CONTEXT_HUB_CONTENT_ROOT}" \
    >/dev/null 2>&1 || true
}

SKIP_COUNT=0

for idx in "${!GRID[@]}"; do
  ITEM="${GRID[$idx]}"
  IFS=',' read -r LASTN MODE GINIT PEFTN <<< "${ITEM}"
  TAG="s${SEED}_ln${LASTN}_${MODE}_g${GINIT}_p${PEFTN}"
  RUN_NAME="Exp_M1softFULL_${TAG}_${RUN_STAMP}_r${idx}"

  if [[ -n "${NAMES}" ]]; then
    case ",${NAMES}," in
      *",${RUN_NAME},"* ) ;;
      *) continue ;;
    esac
  fi

  RUN_DIR="${CKPT_ROOT}/${RUN_NAME}"
  LOG_PATH="${RUN_DIR}/train_$(date +%Y%m%d-%H%M%S).log"
  BEST_AUC_PATH="${RUN_DIR}/best_auc.pt"
  LAST_AUC_PATH="${RUN_DIR}/last.pt"
  BEST_TXT="${SUB_DIR}/sub_${RUN_NAME}_best.txt"
  BEST_ZIP="${SUB_DIR}/sub_${RUN_NAME}_best.zip"
  LAST_TXT="${SUB_DIR}/sub_${RUN_NAME}_last.txt"
  LAST_ZIP="${SUB_DIR}/sub_${RUN_NAME}_last.zip"

  echo "[run] ${RUN_NAME}"
  if ! "${PY}" - << PY
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
train.OPT_OVERRIDES["niter"] = int("${NITER}")
train.PARSER_DEFAULTS["niter"] = int("${NITER}")
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
if "${TRAIN_BATCH_SIZE}" != "":
    train.OPT_OVERRIDES["batch_size"] = int("${TRAIN_BATCH_SIZE}")
    train.PARSER_DEFAULTS["batch_size"] = int("${TRAIN_BATCH_SIZE}")

if "${MEV1_FORCE_NO_OSD_REGS}" == "1":
    train.OPT_OVERRIDES["no_osd_regs"] = True
    train.PARSER_DEFAULTS["no_osd_regs"] = True

train.main()
PY
  then
    TRAIN_STATUS="train_failed"
  else
    TRAIN_STATUS="train_done"
  fi

  WAIT_LOG="$(ls -1t "${RUN_DIR}"/train_*.log 2>/dev/null | head -n 1 || true)"
  if [[ -z "${WAIT_LOG:-}" ]]; then
    echo "[warn] ${RUN_NAME} no training log found."
    VAL_AUC=""
  fi
  if [[ "${TRAIN_STATUS}" == "train_failed" ]]; then
    echo "[fail] ${RUN_NAME} training failed; skip auc/infer and continue."
    SUBMIT_STATUS="train_failed"
    echo "${RUN_NAME},${SEED},${LASTN},${MODE},${GINIT},${PEFTN},${NITER},${MAX_STEPS},,${BEST_AUC_PATH},${LAST_AUC_PATH},,,,${SUBMIT_STATUS}" >> "${SUMMARY_CSV}"
    sync_context_hub_tracker
    if (( SKIP_COUNT < 999999999 )); then
      SKIP_COUNT=$((SKIP_COUNT+1))
    fi
    continue
  fi
  VAL_AUC="$(parse_auc "${WAIT_LOG}")"

  SUBMIT_STATUS="skipped_auc"
  BEST_ZIP_OUT=""
  BEST_TXT_OUT=""
  LAST_ZIP_OUT=""
  LAST_TXT_OUT=""

  if [[ "$(float_ge "${VAL_AUC}" "${MIN_AUC_FOR_SUBMIT}")" == "1" ]]; then
    if [[ -f "${BEST_AUC_PATH}" ]]; then
      BEST_SUB="$(submit_and_report "${BEST_AUC_PATH}" "${BEST_TXT}" "${BEST_ZIP}")"
      if [[ -n "${BEST_SUB}" ]]; then
        SUBMIT_STATUS="best_generated"
        BEST_ZIP_OUT="${BEST_ZIP}"
        BEST_TXT_OUT="${BEST_TXT}"
      else
        SUBMIT_STATUS="best_submit_failed"
      fi
    else
      SUBMIT_STATUS="best_ckpt_missing"
    fi

    if [[ -f "${LAST_AUC_PATH}" ]]; then
      LAST_SUB="$(submit_and_report "${LAST_AUC_PATH}" "${LAST_TXT}" "${LAST_ZIP}")"
      if [[ -n "${LAST_SUB}" ]]; then
        LAST_ZIP_OUT="${LAST_ZIP}"
        LAST_TXT_OUT="${LAST_TXT}"
        if [[ "${SUBMIT_STATUS}" == "skipped_auc" ]]; then
          SUBMIT_STATUS="last_generated"
        elif [[ "${SUBMIT_STATUS}" == "best_generated" ]]; then
          SUBMIT_STATUS="best_last_generated"
        elif [[ "${SUBMIT_STATUS}" == "best_submit_failed" ]]; then
          SUBMIT_STATUS="best_failed_last_generated"
        fi
      elif [[ "${SUBMIT_STATUS}" == "best_generated" ]]; then
        SUBMIT_STATUS="best_generated_only"
      elif [[ "${SUBMIT_STATUS}" == "best_submit_failed" ]]; then
        SUBMIT_STATUS="best_and_last_submit_failed"
      elif [[ "${SUBMIT_STATUS}" == "skipped_auc" ]]; then
        SUBMIT_STATUS="last_submit_failed"
      fi
    fi
  else
    echo "[skip] ${RUN_NAME} val_auc=${VAL_AUC} < ${MIN_AUC_FOR_SUBMIT}. no submission generated."
  fi

  echo "${RUN_NAME},${SEED},${LASTN},${MODE},${GINIT},${PEFTN},${NITER},${MAX_STEPS},${VAL_AUC},${BEST_AUC_PATH},${LAST_AUC_PATH},${BEST_ZIP_OUT},${BEST_TXT_OUT},${LAST_ZIP_OUT},${LAST_TXT_OUT},${SUBMIT_STATUS}" >> "${SUMMARY_CSV}"
  echo "[done] ${RUN_NAME} val_auc=${VAL_AUC} status=${SUBMIT_STATUS} best=${BEST_ZIP_OUT:-_} last=${LAST_ZIP_OUT:-_}"
  echo "[log]  ${WAIT_LOG}"
  sync_context_hub_tracker

  if [[ "${VAL_AUC}" == "" ]]; then
    SKIP_COUNT=$((SKIP_COUNT+1))
  elif [[ "$(float_lt "${VAL_AUC}" "${MIN_AUC_FOR_SUBMIT}")" == "1" ]]; then
    SKIP_COUNT=$((SKIP_COUNT+1))
  else
    SKIP_COUNT=0
  fi

  if [[ "${STOP_IF_AUC_BELOW}" == "1" && "$(float_lt "${VAL_AUC}" "${MIN_AUC_FOR_SUBMIT}")" == "1" ]]; then
    echo "[abort] stop all runs because val_auc ${VAL_AUC} < ${MIN_AUC_FOR_SUBMIT}"
    break
  fi

  if (( SKIP_COUNT >= MAX_AUC_SKIP_COUNT )); then
    echo "[stop] skip_count=${SKIP_COUNT} reached cap=${MAX_AUC_SKIP_COUNT}"
    break
  fi

done

sync_context_hub_tracker
echo "[all_done] ${SUMMARY_CSV}"
