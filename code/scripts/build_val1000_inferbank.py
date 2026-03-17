#!/usr/bin/env python3
"""
Build a cross-run val1000 inference bank.

Input:
- model table CSV with columns: model_id, ckpt_path

Output (under out_dir):
- <model_id>.txt : raw score list from full_submission.py
- <model_id>.csv : path,label,score
- inferbank_summary.csv : status + metrics per model
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def _infer_label(path: Path) -> int:
    n = path.name.lower()
    if "fake" in n:
        return 1
    if "real" in n:
        return 0
    raise ValueError(f"Cannot infer label from filename: {path.name}")


def build_val_index(val_root: Path) -> pd.DataFrame:
    paths = [p for p in val_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    paths.sort(key=_sort_key)
    rows = [{"path": str(p), "label": _infer_label(p)} for p in paths]
    if not rows:
        raise RuntimeError(f"No images found under {val_root}")
    return pd.DataFrame(rows)


def run_one(
    py_exec: str,
    full_submission_py: Path,
    ckpt_path: Path,
    val_root: Path,
    out_txt: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> Tuple[bool, str]:
    cmd = [
        py_exec,
        str(full_submission_py),
        "--ckpt",
        str(ckpt_path),
        "--txt_path",
        str(val_root),
        "--root_dir",
        str(val_root),
        "--out_csv",
        str(out_txt),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--image_size",
        str(image_size),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    msg = proc.stdout if ok else (proc.stdout + "\n" + proc.stderr)
    return ok, msg[-2000:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build val1000 inference bank from multiple checkpoints.")
    p.add_argument(
        "--model_csv",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/global_val1000_infer_models_20260306.csv"),
    )
    p.add_argument(
        "--val_root",
        type=Path,
        default=Path("/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/training_data_final"),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/val1000_inferbank_20260306"),
    )
    p.add_argument(
        "--py_exec",
        type=str,
        default="/home/u1657859/miniconda3/envs/eamamba/bin/python",
    )
    p.add_argument(
        "--full_submission_py",
        type=Path,
        default=Path("/work/u1657859/Jess/app/full_submission.py"),
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    models = pd.read_csv(args.model_csv)
    if "model_id" not in models.columns or "ckpt_path" not in models.columns:
        raise ValueError("model_csv must contain model_id, ckpt_path columns")

    val_df = build_val_index(args.val_root)
    y_true = val_df["label"].astype(int).values

    rows: List[dict] = []
    for i, r in models.iterrows():
        model_id = str(r["model_id"])
        ckpt = Path(str(r["ckpt_path"]))
        out_txt = args.out_dir / f"{model_id}.txt"
        out_csv = args.out_dir / f"{model_id}.csv"

        rec = {
            "model_id": model_id,
            "ckpt_path": str(ckpt),
            "exists": ckpt.exists(),
            "status": "skipped",
            "auc": np.nan,
            "acc": np.nan,
            "logloss": np.nan,
            "message": "",
            "out_txt": str(out_txt),
            "out_csv": str(out_csv),
        }

        if not ckpt.exists():
            rec["status"] = "missing_ckpt"
            rows.append(rec)
            print(f"[{i+1}/{len(models)}] {model_id}: missing ckpt")
            continue

        need_run = bool(args.force) or (not out_txt.exists())
        if need_run:
            print(f"[{i+1}/{len(models)}] {model_id}: infer...")
            ok, msg = run_one(
                py_exec=args.py_exec,
                full_submission_py=args.full_submission_py,
                ckpt_path=ckpt,
                val_root=args.val_root,
                out_txt=out_txt,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                image_size=int(args.image_size),
            )
            rec["message"] = msg
            if not ok:
                rec["status"] = "infer_failed"
                rows.append(rec)
                print(f"[{i+1}/{len(models)}] {model_id}: FAILED")
                continue

        try:
            scores = np.loadtxt(out_txt, dtype=float)
            if scores.ndim == 0:
                scores = np.array([float(scores)])
            if len(scores) != len(val_df):
                rec["status"] = "length_mismatch"
                rec["message"] = f"len(scores)={len(scores)} len(val)={len(val_df)}"
                rows.append(rec)
                print(f"[{i+1}/{len(models)}] {model_id}: length mismatch")
                continue

            rec["auc"] = float(roc_auc_score(y_true, scores))
            rec["acc"] = float(accuracy_score(y_true, (scores >= 0.5).astype(int)))
            rec["logloss"] = float(log_loss(y_true, np.clip(scores, 1e-6, 1 - 1e-6)))
            rec["status"] = "ok"

            out_df = val_df.copy()
            out_df["score"] = scores
            out_df.to_csv(out_csv, index=False)
            print(f"[{i+1}/{len(models)}] {model_id}: ok auc={rec['auc']:.4f}")
        except Exception as e:
            rec["status"] = "parse_failed"
            rec["message"] = str(e)
            print(f"[{i+1}/{len(models)}] {model_id}: parse failed: {e}")

        rows.append(rec)

    summary = pd.DataFrame(rows)
    summary_path = args.out_dir / "inferbank_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
