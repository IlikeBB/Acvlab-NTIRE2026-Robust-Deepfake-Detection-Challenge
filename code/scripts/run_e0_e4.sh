#!/usr/bin/env bash
set -euo pipefail

# Run E0~E4 experiments (4 epochs each) based on tuneoff-AUXLoss baseline settings.
# Usage:
#   bash scripts/run_e0_e4.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NITER="${NITER:-4}"
BASE_TAG="${BASE_TAG:-DF40_tuneoff-AUXLoss}"
LOSS_FREQ="${LOSS_FREQ:-200}"
AUDIT_WINDOW_BATCHES="${AUDIT_WINDOW_BATCHES:-200}"

run_case() {
  local exp_name="$1"
  local quality_balance="$2"
  local use_stratified="$3"
  local use_reweight="$4"
  local aug_before_resize="$5"

  echo "[RUN] ${exp_name}"
  python - <<PY
import train

exp_name = "${exp_name}"
niter = int("${NITER}")
loss_freq = int("${LOSS_FREQ}")
audit_window_batches = int("${AUDIT_WINDOW_BATCHES}")
quality_balance = bool(int("${quality_balance}"))
use_stratified = bool(int("${use_stratified}"))
use_reweight = bool(int("${use_reweight}"))
aug_before_resize = bool(int("${aug_before_resize}"))

train.OPT_OVERRIDES["name"] = exp_name
train.OPT_OVERRIDES["niter"] = niter
train.OPT_OVERRIDES["loss_freq"] = loss_freq
train.OPT_OVERRIDES["audit_window_batches"] = audit_window_batches
train.OPT_OVERRIDES["quality_balance"] = quality_balance
train.OPT_OVERRIDES["use_stratified_sampler"] = use_stratified
train.OPT_OVERRIDES["use_reweighted_erm"] = use_reweight
train.OPT_OVERRIDES["reweight_use_sqrt"] = False
train.OPT_OVERRIDES["reweight_clip_min"] = 0.2
train.OPT_OVERRIDES["reweight_clip_max"] = 5.0
train.OPT_OVERRIDES["val_quality_diag"] = True
train.OPT_OVERRIDES["quality_bins"] = 3
train.OPT_OVERRIDES["quality_max_side"] = 256
train.OPT_OVERRIDES["quality_cache_train_path"] = "cache/q_train.csv"
train.OPT_OVERRIDES["quality_cache_val_path"] = "cache/q_val.csv"
train.OPT_OVERRIDES["aug_before_resize"] = aug_before_resize

train.PARSER_DEFAULTS["quality_balance"] = quality_balance
train.PARSER_DEFAULTS["loss_freq"] = loss_freq
train.PARSER_DEFAULTS["audit_window_batches"] = audit_window_batches
train.PARSER_DEFAULTS["use_stratified_sampler"] = use_stratified
train.PARSER_DEFAULTS["use_reweighted_erm"] = use_reweight
train.PARSER_DEFAULTS["reweight_use_sqrt"] = False
train.PARSER_DEFAULTS["reweight_clip_min"] = 0.2
train.PARSER_DEFAULTS["reweight_clip_max"] = 5.0
train.PARSER_DEFAULTS["val_quality_diag"] = True
train.PARSER_DEFAULTS["quality_bins"] = 3
train.PARSER_DEFAULTS["quality_max_side"] = 256
train.PARSER_DEFAULTS["quality_cache_train_path"] = "cache/q_train.csv"
train.PARSER_DEFAULTS["quality_cache_val_path"] = "cache/q_val.csv"
train.PARSER_DEFAULTS["aug_before_resize"] = aug_before_resize

train.main()
PY
}

# E0 Baseline
run_case "E0_${BASE_TAG}" 0 0 0 0
# E1 Stratified Sampler
run_case "E1_${BASE_TAG}" 1 1 0 0
# E2 Reweighted ERM
run_case "E2_${BASE_TAG}" 1 0 1 0
# E3 Aug-before-resize only
run_case "E3_${BASE_TAG}" 0 0 0 1
# E4 E1 + E3
run_case "E4_${BASE_TAG}" 1 1 0 1

echo "[DONE] E0~E4 finished"
