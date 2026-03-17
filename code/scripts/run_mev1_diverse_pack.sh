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
SUMMARY_CSV="${SUMMARY_CSV:-${SCORE_DIR}/mev1_diverse_pack_$(date +%Y%m%d_%H%M%S).csv}"

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
BASE_CFG_BATCH_SIZE="$("${PY}" - "${BASE_CFG}" << 'PY'
import json, sys
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(int(cfg.get("batch_size", 24)))
PY
)"

SEED="${SEED:-2048}"
DIVERSE_PROFILE="${DIVERSE_PROFILE:-full}" # full | quick
if [[ "${DIVERSE_PROFILE}" == "quick" ]]; then
  NITER="${NITER:-1}"
  MAX_STEPS="${MAX_TRAIN_STEPS_PER_EPOCH:-200}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
else
  NITER="${NITER:-${BASE_CFG_NITER}}"
  MAX_STEPS="${MAX_TRAIN_STEPS_PER_EPOCH:-${BASE_CFG_MAX_STEPS}}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${BASE_CFG_BATCH_SIZE}}"
fi
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MIN_AUC_FOR_SUBMIT="${MIN_AUC_FOR_SUBMIT:-0.84}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
CASE_FILTER="${CASE_FILTER:-}"
SYNC_CONTEXT_HUB="${SYNC_CONTEXT_HUB:-1}"
CONTEXT_HUB_CONTENT_ROOT="${CONTEXT_HUB_CONTENT_ROOT:-/work/u1657859/Jess/app/context-hub-content}"

mkdir -p "${SUB_DIR}" "${SCORE_DIR}"
cat > "${SUMMARY_CSV}" << CSV
run_name,case_id,aug_profile,seed,niter,max_train_steps_per_epoch,batch_size,val_auc_epoch0,best_ckpt,last_ckpt,best_sub,last_sub,submission_status
CSV

# case format:
# id|adapter_last_n|svd_mode|gate_init|peft_blocks|rank_en|rank_lambda|rank_margin|k|route|div_lambda|cvar_en|cvar_lambda|q_temp|q_mix|patch_tau|patch_gamma|aug_profile
CASES=(
  "base|0|add|0.0|0|0|0.0|0.2|3|quality_bin|0.002|0|0.7|2.0|0.2|1.8|0.08|base"
  "ranklite|0|add|0.0|0|1|0.05|0.2|3|quality_bin|0.002|0|0.7|2.0|0.2|1.8|0.08|base"
  "qsoft|0|add|0.0|0|0|0.0|0.2|3|quality_soft|0.002|0|0.7|1.5|0.1|1.8|0.08|base"
  "softsvd_ln4|4|gated|-2.0|1|0|0.0|0.2|3|quality_bin|0.002|0|0.7|2.0|0.2|1.8|0.08|base"
  "aug_strong|0|add|0.0|0|0|0.0|0.2|3|quality_bin|0.002|0|0.7|2.0|0.2|1.8|0.08|strong"
  "aug_motion|0|add|0.0|0|0|0.0|0.2|3|quality_bin|0.002|0|0.7|2.0|0.2|1.8|0.08|motion"
  "aug_grainy|0|add|0.0|0|0|0.0|0.2|3|quality_bin|0.002|0|0.7|2.0|0.2|1.8|0.08|grainy"
  "aug_clean|0|add|0.0|0|0|0.0|0.2|3|quality_bin|0.002|0|0.7|2.0|0.2|1.8|0.08|clean"
  "softsvd_ln4_strong|4|gated|-2.0|1|0|0.0|0.2|3|quality_bin|0.002|0|0.7|2.0|0.2|1.8|0.08|strong"
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

parse_auc() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    echo ""
    return
  fi
  "${PY}" - "${log_path}" << 'PY'
import re, sys
log_path = sys.argv[1]
patterns = [
    re.compile(r"epoch\s+0\s+val\b.*?\bauc=([0-9]*\.?[0-9]+)"),
    re.compile(r"\bval_auc\s*[:=]\s*([0-9]*\.?[0-9]+)"),
    re.compile(r"\bauc\s*[:=]\s*([0-9]*\.?[0-9]+)"),
]
auc = ""
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        for pat in patterns:
            m = pat.search(line)
            if m:
                auc = m.group(1)
                break
        if auc:
            break
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
    --batch_size "${INFER_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --strict \
    --tta_mode none \
    --on_error skip
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

for item in "${CASES[@]}"; do
  IFS='|' read -r CASE_ID ADAPTER_N SVD_MODE GATE_INIT PEFT_N \
    RANK_EN RANK_LAMBDA RANK_MARGIN K ROUTE DIV_LAMBDA CVAR_EN CVAR_LAMBDA \
    Q_TEMP Q_MIX PATCH_TAU PATCH_GAMMA AUG_PROFILE <<< "${item}"

  if [[ -n "${CASE_FILTER}" && ",${CASE_FILTER}," != *",${CASE_ID},"* ]]; then
    continue
  fi

  RUN_NAME="Exp_M1div_${CASE_ID}_s${SEED}_${RUN_STAMP}"
  RUN_DIR="${CKPT_ROOT}/${RUN_NAME}"
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

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = json.load(f)

for k, v in cfg.items():
    train.OPT_OVERRIDES[k] = v
    train.PARSER_DEFAULTS[k] = v

def set_opt(k, v):
    train.OPT_OVERRIDES[k] = v
    train.PARSER_DEFAULTS[k] = v

set_opt("name", run_name)
set_opt("seed", int("${SEED}"))
set_opt("niter", int("${NITER}"))
set_opt("max_train_steps_per_epoch", int("${MAX_STEPS}"))
set_opt("resume_path", None)
set_opt("batch_size", int("${TRAIN_BATCH_SIZE}"))

set_opt("adapter_last_n_layers", int("${ADAPTER_N}"))
set_opt("svd_residual_mode", "${SVD_MODE}")
set_opt("svd_residual_gate_init", float("${GATE_INIT}"))
set_opt("peft_unfreeze_last_n_blocks", int("${PEFT_N}"))
set_opt("peft_unfreeze_layernorm", True)
set_opt("peft_unfreeze_bias", False)

set_opt("rank_loss_enable", bool(int("${RANK_EN}")))
set_opt("rank_loss_lambda", float("${RANK_LAMBDA}"))
set_opt("rank_loss_margin", float("${RANK_MARGIN}"))

set_opt("multi_expert_k", int("${K}"))
set_opt("multi_expert_route", "${ROUTE}")
set_opt("multi_expert_div_lambda", float("${DIV_LAMBDA}"))
set_opt("multi_expert_route_temp", float("${Q_TEMP}"))

set_opt("groupdro_use_cvar", bool(int("${CVAR_EN}")))
set_opt("groupdro_cvar_lambda", float("${CVAR_LAMBDA}"))
set_opt("groupdro_q_temp", float("${Q_TEMP}"))
set_opt("groupdro_q_mix", float("${Q_MIX}"))

set_opt("patch_pool_tau", float("${PATCH_TAU}"))
set_opt("patch_pool_gamma", float("${PATCH_GAMMA}"))

aug_profile = str("${AUG_PROFILE}").lower()
if aug_profile == "clean":
    # Minimal synthetic degradation: keep only resize/normalize path to test robustness gap.
    set_opt("hflip_prob", 0.0)
    set_opt("blur_prob", 0.0)
    set_opt("jpg_prob", 0.0)
    set_opt("noise_prob", 0.0)
    set_opt("iso_noise_prob", 0.0)
    set_opt("motion_blur_prob", 0.0)
    set_opt("lowres_prob", 0.0)
    set_opt("grainy_group_prob", 0.0)
    set_opt("mono_group_prob", 0.0)
    set_opt("motion_grainy_group_prob", 0.0)
    set_opt("lowlight_group_prob", 0.0)
    set_opt("max_aug_per_image", 1)
elif aug_profile == "strong":
    set_opt("hflip_prob", 0.2)
    set_opt("blur_prob", 0.35)
    set_opt("blur_sig", [0.0, 2.8])
    set_opt("jpg_prob", 0.35)
    set_opt("jpg_qual", [20, 95])
    set_opt("noise_prob", 0.20)
    set_opt("noise_std", [3.0, 22.0])
    set_opt("iso_noise_prob", 0.20)
    set_opt("iso_shot", [1.2, 5.0])
    set_opt("motion_blur_prob", 0.15)
    set_opt("motion_blur_ksize", [7, 15])
    set_opt("lowres_prob", 0.20)
    set_opt("lowres_scale", [0.18, 0.45])
    set_opt("grainy_group_prob", 0.08)
    set_opt("mono_group_prob", 0.08)
    set_opt("motion_grainy_group_prob", 0.08)
    set_opt("lowlight_group_prob", 0.08)
    set_opt("max_aug_per_image", 2)
elif aug_profile == "motion":
    set_opt("hflip_prob", 0.2)
    set_opt("blur_prob", 0.25)
    set_opt("blur_sig", [0.0, 2.2])
    set_opt("jpg_prob", 0.25)
    set_opt("jpg_qual", [24, 95])
    set_opt("motion_blur_prob", 0.30)
    set_opt("motion_blur_ksize", [9, 21])
    set_opt("lowres_prob", 0.20)
    set_opt("lowres_scale", [0.18, 0.40])
    set_opt("noise_prob", 0.10)
    set_opt("iso_noise_prob", 0.12)
    set_opt("grainy_group_prob", 0.06)
    set_opt("mono_group_prob", 0.06)
    set_opt("motion_grainy_group_prob", 0.25)
    set_opt("lowlight_group_prob", 0.05)
    set_opt("max_aug_per_image", 2)
elif aug_profile == "grainy":
    set_opt("hflip_prob", 0.2)
    set_opt("blur_prob", 0.2)
    set_opt("jpg_prob", 0.2)
    set_opt("noise_prob", 0.18)
    set_opt("noise_std", [4.0, 25.0])
    set_opt("iso_noise_prob", 0.25)
    set_opt("iso_shot", [1.0, 4.5])
    set_opt("motion_blur_prob", 0.12)
    set_opt("lowres_prob", 0.12)
    set_opt("grainy_group_prob", 0.25)
    set_opt("mono_group_prob", 0.20)
    set_opt("motion_grainy_group_prob", 0.20)
    set_opt("lowlight_group_prob", 0.20)
    set_opt("max_aug_per_image", 2)
# "base": keep baseline config untouched.

train.main()
PY
  then
    TRAIN_STATUS="train_failed"
  else
    TRAIN_STATUS="train_done"
  fi

  WAIT_LOG="$(ls -1t "${RUN_DIR}"/train_*.log 2>/dev/null | head -n 1 || true)"
  VAL_AUC="$(parse_auc "${WAIT_LOG}")"
  STATUS="skipped_auc"
  BEST_ZIP_OUT=""
  LAST_ZIP_OUT=""

  if [[ "${TRAIN_STATUS}" == "train_failed" ]]; then
    STATUS="train_failed"
  elif [[ "$(float_ge "${VAL_AUC}" "${MIN_AUC_FOR_SUBMIT}")" == "1" ]]; then
    if [[ -f "${BEST_AUC_PATH}" ]]; then
      if build_submission "${BEST_AUC_PATH}" "${BEST_TXT}" "${BEST_ZIP}"; then
        BEST_ZIP_OUT="${BEST_ZIP}"
        STATUS="best_generated"
      else
        STATUS="best_submit_failed"
      fi
    fi
    if [[ -f "${LAST_AUC_PATH}" ]]; then
      if build_submission "${LAST_AUC_PATH}" "${LAST_TXT}" "${LAST_ZIP}"; then
        LAST_ZIP_OUT="${LAST_ZIP}"
        if [[ "${STATUS}" == "best_generated" ]]; then
          STATUS="best_last_generated"
        elif [[ "${STATUS}" == "skipped_auc" ]]; then
          STATUS="last_generated"
        fi
      fi
    fi
  else
    echo "[skip] ${RUN_NAME} val_auc=${VAL_AUC} < ${MIN_AUC_FOR_SUBMIT}"
  fi

  echo "${RUN_NAME},${CASE_ID},${AUG_PROFILE},${SEED},${NITER},${MAX_STEPS},${TRAIN_BATCH_SIZE},${VAL_AUC},${BEST_AUC_PATH},${LAST_AUC_PATH},${BEST_ZIP_OUT},${LAST_ZIP_OUT},${STATUS}" >> "${SUMMARY_CSV}"
  echo "[done] ${RUN_NAME} case=${CASE_ID} aug=${AUG_PROFILE} val_auc=${VAL_AUC} status=${STATUS}"
  sync_context_hub_tracker
done

sync_context_hub_tracker
echo "[all_done] ${SUMMARY_CSV}"
