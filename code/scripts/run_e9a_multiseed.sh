#!/usr/bin/env bash
set -euo pipefail

# Run only E9A with multiple seeds, and keep q-cache per seed.
#
# Usage:
#   bash scripts/run_e9a_multiseed.sh
#
# Optional env overrides:
#   SEEDS="418 777 1024 2048"
#   NITER=10
#   BASE_TAG="DF40_tuneoff-AUXLoss"
#   CACHE_ROOT="/path/to/cache"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEEDS="${SEEDS:-418 777 1024}"
NITER="${NITER:-6}"
BASE_TAG="${BASE_TAG:-DF40_tuneoff-AUXLoss}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/cache/e9a_multiseed_qcache}"

mkdir -p "${CACHE_ROOT}"

echo "[INFO] run=E9A only"
echo "[INFO] seeds=${SEEDS}"
echo "[INFO] cache_root=${CACHE_ROOT}"
echo "[INFO] cache_scope=per_seed"

RUNS="E9A" \
SEEDS="${SEEDS}" \
NITER="${NITER}" \
BASE_TAG="${BASE_TAG}" \
CACHE_SCOPE="per_seed" \
CACHE_ROOT="${CACHE_ROOT}" \
bash "${ROOT_DIR}/scripts/run_e5_e9_matrix.sh"

