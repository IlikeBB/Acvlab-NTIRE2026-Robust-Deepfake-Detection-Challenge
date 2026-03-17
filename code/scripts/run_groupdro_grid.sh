#!/usr/bin/env bash
set -euo pipefail

# Batch runner for groupDRO-quality experiments.
# This script does NOT edit Config.py on disk. It overrides train module
# globals in-memory before calling train.main() for each run.
#
# Usage:
#   bash scripts/run_groupdro_grid.sh
#
# Optional env overrides:
#   NITER=6 QUALITY_BINS_LIST="3 4" QUALITY_BALANCE_LIST="0 1" BASE_TAG="DF40_tuneoff-AUXLoss"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_TAG="${BASE_TAG:-DF40_tuneoff-AUXLoss}"
NITER="${NITER:-10}"
QUALITY_BINS_LIST="${QUALITY_BINS_LIST:-3 4 5}"
QUALITY_BALANCE_LIST="${QUALITY_BALANCE_LIST:-0 1}"

for QB in ${QUALITY_BALANCE_LIST}; do
  for BINS in ${QUALITY_BINS_LIST}; do
    if [[ "${QB}" == "0" ]]; then
      RUN_NAME="Effort_${BASE_TAG}_qb0_bins${BINS}"
      CACHE_PATH="/tmp/qcache_${RUN_NAME}.csv"
    else
      RUN_NAME="Effort_${BASE_TAG}_qb1_bins${BINS}"
      CACHE_PATH="/tmp/qcache_${RUN_NAME}.csv"
    fi

    echo "[RUN] ${RUN_NAME}"
    python - <<PY
import train

run_name = "${RUN_NAME}"
qb = bool(int("${QB}"))
bins = int("${BINS}")
niter = int("${NITER}")
cache_path = "${CACHE_PATH}"

# Keep tuneoff-AUXLoss baseline as base, then override groupDRO knobs.
train.OPT_OVERRIDES["name"] = run_name
train.OPT_OVERRIDES["niter"] = niter
train.OPT_OVERRIDES["quality_balance"] = qb
train.OPT_OVERRIDES["val_quality_diag"] = True
train.OPT_OVERRIDES["quality_bins"] = bins
train.OPT_OVERRIDES["quality_max_side"] = 256
train.OPT_OVERRIDES["quality_cache_path"] = cache_path

# Keep parser defaults coherent with overrides for logging/config dump.
train.PARSER_DEFAULTS["quality_balance"] = qb
train.PARSER_DEFAULTS["val_quality_diag"] = True
train.PARSER_DEFAULTS["quality_bins"] = bins
train.PARSER_DEFAULTS["quality_max_side"] = 256
train.PARSER_DEFAULTS["quality_cache_path"] = cache_path

train.main()
PY
  done
done

echo "[DONE] all runs finished"
