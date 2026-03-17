#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$BUNDLE_DIR/repro_outputs}"
ORIG_TXT="$BUNDLE_DIR/artifacts/original_submission/sub_m1strictfull_e0_0305.txt"
ORIG_ZIP="$BUNDLE_DIR/artifacts/original_submission/sub_m1strictfull_e0_0305.zip"
REPRO_TXT="$OUT_DIR/repro_submission.txt"
REPRO_ZIP="$OUT_DIR/repro_submission.zip"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$REPRO_TXT" || ! -f "$REPRO_ZIP" ]]; then
  echo "Missing reproduction outputs under $OUT_DIR"
  exit 1
fi

"$PYTHON_BIN" - <<PY
import hashlib
import json
import zipfile
from pathlib import Path

orig_txt = Path(r"$ORIG_TXT")
orig_zip = Path(r"$ORIG_ZIP")
repro_txt = Path(r"$REPRO_TXT")
repro_zip = Path(r"$REPRO_ZIP")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

with zipfile.ZipFile(orig_zip) as z:
    orig_inner = z.read(z.namelist()[0])
with zipfile.ZipFile(repro_zip) as z:
    repro_inner = z.read(z.namelist()[0])

print(json.dumps({
    "txt_exact_match": orig_txt.read_bytes() == repro_txt.read_bytes(),
    "txt_sha256_original": sha(orig_txt),
    "txt_sha256_reproduced": sha(repro_txt),
    "zip_inner_exact_match": orig_inner == repro_inner,
    "zip_sha256_original": sha(orig_zip),
    "zip_sha256_reproduced": sha(repro_zip),
}, ensure_ascii=False, indent=2))
PY
