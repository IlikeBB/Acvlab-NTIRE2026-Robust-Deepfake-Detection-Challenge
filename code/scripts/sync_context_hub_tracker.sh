#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/work/u1657859/Jess/app-isolated-20260227-191252}"
PY="${PY:-/home/u1657859/miniconda3/envs/eamamba/bin/python}"
CONTENT_ROOT="${CONTENT_ROOT:-/work/u1657859/Jess/app/context-hub-content}"
SCORE_DIR="${SCORE_DIR:-/work/u1657859/Jess/app/score}"
DO_BUILD="${DO_BUILD:-1}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

CMD=(
  "${PY}" "${ROOT}/scripts/update_context_hub_tracker.py"
  --score-dir "${SCORE_DIR}"
  --content-root "${CONTENT_ROOT}"
)

if [[ "${DO_BUILD}" == "1" ]]; then
  CMD+=(--build)
fi
if [[ "${VALIDATE_ONLY}" == "1" ]]; then
  CMD+=(--validate-only)
fi

"${CMD[@]}"

echo "[done] context-hub tracker synced at ${CONTENT_ROOT}"
