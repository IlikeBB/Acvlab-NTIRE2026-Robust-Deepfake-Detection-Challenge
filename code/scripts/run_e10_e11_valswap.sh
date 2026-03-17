#!/usr/bin/env bash
set -euo pipefail

# E10/E11 val-only runner:
# - E10: quality_proxy swap-test on the SAME checkpoint (blur / brightness / jpeg-ish)
# - E11: micro-TTA stability (tensor-space)
#
# Required:
#   CKPT_PATH=/path/to/best_auc.pt bash scripts/run_e10_e11_valswap.sh
#
# Optional:
#   CONFIG_PATH=/path/to/config.json
#   VAL_DATA_ROOT=/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/training_data_final/
#   PROXIES="blur brightness jpegish"
#   OUT_CSV=/path/to/e10_e11_valswap.csv
#   BATCH_SIZE=64 NUM_WORKERS=8
#   ENABLE_VAL_MICRO_TTA=1 MICRO_TTA_K=9
#   BRIGHTNESS_CUTS="-0.2,0.1"
#   JPEGISH_CUTS="-0.4,0.2"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CKPT_PATH="${CKPT_PATH:-}"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "[ERROR] CKPT_PATH is required"
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-$(dirname "${CKPT_PATH}")/config.json}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/training_data_final/}"
PROXIES="${PROXIES:-blur brightness jpegish}"
OUT_CSV="${OUT_CSV:-$(dirname "${CKPT_PATH}")/e10_e11_valswap.csv}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ENABLE_VAL_MICRO_TTA="${ENABLE_VAL_MICRO_TTA:-1}"
MICRO_TTA_K="${MICRO_TTA_K:-9}"
BRIGHTNESS_CUTS="${BRIGHTNESS_CUTS:-}"
JPEGISH_CUTS="${JPEGISH_CUTS:-}"

echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] ckpt=${CKPT_PATH}"
echo "[INFO] config=${CONFIG_PATH}"
echo "[INFO] val_data_root=${VAL_DATA_ROOT}"
echo "[INFO] proxies=${PROXIES}"
echo "[INFO] out_csv=${OUT_CSV}"

PYTHONUNBUFFERED=1 python - <<PY
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from models import get_model
from effort.utils import load_ckpt
from effort.data import MixedFolderDataset, collate_skip_none
from validate import validate

ckpt_path = Path("${CKPT_PATH}")
config_path = Path("${CONFIG_PATH}")
val_root = "${VAL_DATA_ROOT}"
proxies = "${PROXIES}".split()
out_csv = Path("${OUT_CSV}")
batch_size = int("${BATCH_SIZE}")
num_workers = int("${NUM_WORKERS}")
enable_micro = bool(int("${ENABLE_VAL_MICRO_TTA}"))
micro_k = int("${MICRO_TTA_K}")
brightness_cuts_raw = "${BRIGHTNESS_CUTS}".strip()
jpegish_cuts_raw = "${JPEGISH_CUTS}".strip()

if not ckpt_path.exists():
    raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
if not config_path.exists():
    raise FileNotFoundError(f"config not found: {config_path}")

with config_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

opt = SimpleNamespace(**cfg)
if not hasattr(opt, "on_error"):
    opt.on_error = "skip"
if not hasattr(opt, "image_size"):
    opt.image_size = 224
if not hasattr(opt, "quality_bins"):
    opt.quality_bins = 3
if not hasattr(opt, "quality_max_side"):
    opt.quality_max_side = 256
if not hasattr(opt, "quality_cache_val_path"):
    opt.quality_cache_val_path = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(opt).to(device)
missing, unexpected = load_ckpt(str(ckpt_path), model, strict=False, weights_only=False)
print(f"[INFO] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")

# Build a single val loader (blur-based metadata from fixed train cuts if config has quality_cuts)
val_ds = MixedFolderDataset(
    [val_root],
    image_size=int(opt.image_size),
    on_error=getattr(opt, "on_error", "skip"),
    augment=False,
    opt=opt,
    quality_balance=True,
    quality_bins=int(opt.quality_bins),
    quality_max_side=int(opt.quality_max_side),
    quality_cache_path=getattr(opt, "quality_cache_val_path", None),
    return_path=False,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_skip_none,
    persistent_workers=num_workers > 0,
)

blur_cuts = getattr(opt, "quality_cuts", None)
if blur_cuts is not None:
    blur_cuts = [float(x) for x in blur_cuts]

def parse_cuts(raw):
    if not raw:
        return None
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(x) for x in vals]

brightness_cuts = parse_cuts(brightness_cuts_raw)
jpegish_cuts = parse_cuts(jpegish_cuts_raw)

rows = []
for proxy in proxies:
    p = proxy.lower()
    if p == "blur":
        swap_proxy = None
        swap_cuts = blur_cuts
    elif p == "brightness":
        swap_proxy = "brightness"
        swap_cuts = brightness_cuts
        if swap_cuts is None:
            print("[WARN] BRIGHTNESS_CUTS not provided; fallback to val quantile bins inside validate().")
    elif p in {"jpeg", "jpegish", "blockiness"}:
        swap_proxy = "jpegish"
        swap_cuts = jpegish_cuts
        if swap_cuts is None:
            print("[WARN] JPEGISH_CUTS not provided; fallback to val quantile bins inside validate().")
    else:
        print(f"[WARN] skip unknown proxy: {proxy}")
        continue

    print(f"[RUN] proxy={proxy} cuts={swap_cuts}")
    metrics = validate(
        model,
        val_loader,
        amp=bool(getattr(opt, "use_amp", False)),
        seed=int(getattr(opt, "seed", 42)) + 999,
        device=device,
        tta_micro=enable_micro,
        tta_k=micro_k,
        tta_seed=0,
        tta_blur_max_sig=float(getattr(opt, "val_micro_tta_blur_max_sig", 1.5)),
        tta_jpeg_qual=getattr(opt, "val_micro_tta_jpeg_qual", [60, 95]),
        tta_contrast=getattr(opt, "val_micro_tta_contrast", [0.9, 1.1]),
        tta_gamma=getattr(opt, "val_micro_tta_gamma", [0.9, 1.1]),
        tta_resize_jitter=int(getattr(opt, "val_micro_tta_resize_jitter", 8)),
        swap_quality_proxy=swap_proxy,
        swap_quality_cuts=swap_cuts,
    )
    row = {
        "proxy": proxy,
        "auc": metrics.get("auc", float("nan")),
        "ap": metrics.get("ap", float("nan")),
        "tpr_at_fpr1": metrics.get("tpr_at_fpr1", float("nan")),
        "tpr_at_fpr0.5": metrics.get("tpr_at_fpr0.5", float("nan")),
        "blur_spearman_real_only": metrics.get("blur_spearman_real_only", float("nan")),
        "blur_spearman_all": metrics.get("blur_spearman_all", float("nan")),
        "worstbin_real_fpr1": metrics.get("worstbin_real_fpr1", float("nan")),
        "gap_real_fpr1": metrics.get("gap_real_fpr1", float("nan")),
        "group_gap_auc": metrics.get("group_gap_auc", float("nan")),
        "tta_range_micro_mean": metrics.get("tta_range_micro_mean", float("nan")),
        "tta_range_micro_p90": metrics.get("tta_range_micro_p90", float("nan")),
        "nll": metrics.get("nll", float("nan")),
        "brier": metrics.get("brier", float("nan")),
        "ece15": metrics.get("ece15", float("nan")),
    }
    rows.append(row)
    print("[METRIC]", row)

out_csv.parent.mkdir(parents=True, exist_ok=True)
fields = list(rows[0].keys()) if rows else ["proxy"]
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"[DONE] wrote {len(rows)} rows -> {out_csv}")
PY
