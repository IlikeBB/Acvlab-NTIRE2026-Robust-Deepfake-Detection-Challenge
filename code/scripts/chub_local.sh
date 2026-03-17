#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/work/u1657859/Jess/app-isolated-20260227-191252}"
CHUB_HOME="${CHUB_HOME:-${ROOT}/.chub-home}"
CONTEXT_HUB_DIST="${CONTEXT_HUB_DIST:-/work/u1657859/Jess/app/context-hub-content/dist}"
NPM_CACHE="${NPM_CACHE:-${CHUB_HOME}/.npm-cache}"
CHUB_SOURCE_POLICY="${CHUB_SOURCE_POLICY:-maintainer}"

mkdir -p "${CHUB_HOME}/.chub"
mkdir -p "${NPM_CACHE}"
cat > "${CHUB_HOME}/.chub/config.yaml" << EOF
sources:
  - name: local
    path: ${CONTEXT_HUB_DIST}
source: "${CHUB_SOURCE_POLICY}"
refresh_interval: 86400
telemetry: false
EOF

if [[ $# -eq 0 ]]; then
  set -- search
fi

HOME="${CHUB_HOME}" npm_config_cache="${NPM_CACHE}" npx -y @aisuite/chub "$@"
