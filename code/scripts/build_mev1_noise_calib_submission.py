#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def _list_images(root: Path):
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    paths.sort(key=_sort_key)
    return paths


def _label_from_name(path: Path):
    name = path.name.lower()
    if "fake" in name:
        return 1
    if "real" in name:
        return 0
    return -1


def _laplacian_var_log(path: Path, max_side: int = 256):
    with Image.open(path) as img:
        img = img.convert("L")
        w, h = img.size
        m = max(w, h)
        if m > int(max_side):
            scale = float(max_side) / float(m)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            img = img.resize((nw, nh), resample=Image.BILINEAR)
        g = np.asarray(img, dtype=np.float32)
    lap = (
        -4.0 * g
        + np.roll(g, 1, axis=0)
        + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1)
        + np.roll(g, -1, axis=1)
    )
    v = float(np.var(lap))
    return float(np.log(v + 1e-6))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _read_probs(path: Path):
    vals = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        vals.append(float(s.split(",")[0].strip()))
    return np.asarray(vals, dtype=np.float64)


def _write_submission_txt(path: Path, probs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in probs:
            vv = float(v)
            if vv < 0.0:
                vv = 0.0
            if vv > 1.0:
                vv = 1.0
            f.write(f"{vv:.6f}\n")


def _zip_submission(txt_path: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname="submission.txt")


def _auc_tieaware(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and ss[j] == ss[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    auc = (ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def _qbin_ids(q, cuts):
    q = np.asarray(q, dtype=np.float64)
    cuts = list(cuts)
    if not cuts:
        return np.zeros_like(q, dtype=np.int64)
    return np.searchsorted(np.asarray(cuts, dtype=np.float64), q, side="right").astype(np.int64)


def _fit_qbin_logit_shift(logits, y, qbin, n_bins):
    shifts = np.zeros((int(n_bins),), dtype=np.float64)
    for b in range(int(n_bins)):
        m = (qbin == b)
        if int(m.sum()) < 8:
            shifts[b] = 0.0
            continue
        mb0 = m & (y == 0)
        mb1 = m & (y == 1)
        if int(mb0.sum()) < 4 or int(mb1.sum()) < 4:
            shifts[b] = 0.0
            continue
        m0 = float(logits[mb0].mean())
        m1 = float(logits[mb1].mean())
        shifts[b] = -0.5 * (m0 + m1)
    return shifts


def _apply_qbin_shift(logits, qbin, shifts):
    return logits + shifts[qbin]


def _stratified_folds(y, k, seed):
    y = np.asarray(y, dtype=np.int64)
    idx0 = [i for i, v in enumerate(y.tolist()) if v == 0]
    idx1 = [i for i, v in enumerate(y.tolist()) if v == 1]
    rng = random.Random(int(seed))
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    folds = [[] for _ in range(int(k))]
    for i, ix in enumerate(idx0):
        folds[i % int(k)].append(ix)
    for i, ix in enumerate(idx1):
        folds[i % int(k)].append(ix)
    return [np.asarray(sorted(f), dtype=np.int64) for f in folds]


def _run_full_submission(
    python_bin: str,
    full_submission_py: str,
    ckpt: Path,
    data_root: Path,
    out_csv: Path,
    batch_size: int,
    num_workers: int,
):
    cmd = [
        str(python_bin),
        str(full_submission_py),
        "--ckpt",
        str(ckpt),
        "--txt_path",
        str(data_root),
        "--root_dir",
        str(data_root),
        "--out_csv",
        str(out_csv),
        "--batch_size",
        str(int(batch_size)),
        "--num_workers",
        str(int(num_workers)),
        "--multi_expert_enable",
        "--multi_expert_k",
        "3",
        "--multi_expert_route",
        "quality_bin",
        "--infer_quality_bin",
        "--on_error",
        "skip",
    ]
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--python_bin", type=str, default="/home/u1657859/miniconda3/envs/eamamba/bin/python")
    p.add_argument("--full_submission_py", type=str, default="/work/u1657859/Jess/app/full_submission.py")
    p.add_argument(
        "--base_ckpt",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_full6_8f_s2048_local/best_auc.pt",
    )
    p.add_argument(
        "--train_root",
        type=str,
        default="/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/training_data_final",
    )
    p.add_argument(
        "--test_root",
        type=str,
        default="/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final",
    )
    p.add_argument("--quality_max_side", type=int, default=256)
    p.add_argument("--quality_bins", type=int, default=3)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--cv_seed", type=int, default=2048)
    p.add_argument("--alpha_grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--tmp_dir",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/submissions/tmp_noise_calib",
    )
    p.add_argument(
        "--out_prefix",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/submissions/sub_mev1_ncal",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.base_ckpt).resolve()
    train_root = Path(args.train_root).resolve()
    test_root = Path(args.test_root).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    out_prefix = Path(args.out_prefix).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    train_pred_txt = tmp_dir / "train1000_pred.txt"
    test_pred_txt = tmp_dir / "test100_pred.txt"

    print("[run] base model on train1000...")
    _run_full_submission(
        python_bin=args.python_bin,
        full_submission_py=args.full_submission_py,
        ckpt=ckpt,
        data_root=train_root,
        out_csv=train_pred_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print("[run] base model on test100...")
    _run_full_submission(
        python_bin=args.python_bin,
        full_submission_py=args.full_submission_py,
        ckpt=ckpt,
        data_root=test_root,
        out_csv=test_pred_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_probs = _read_probs(train_pred_txt)
    test_probs = _read_probs(test_pred_txt)

    train_paths = _list_images(train_root)
    test_paths = _list_images(test_root)
    if len(train_paths) != len(train_probs):
        raise RuntimeError(f"train size mismatch paths={len(train_paths)} probs={len(train_probs)}")
    if len(test_paths) != len(test_probs):
        raise RuntimeError(f"test size mismatch paths={len(test_paths)} probs={len(test_probs)}")

    y = np.asarray([_label_from_name(p) for p in train_paths], dtype=np.int64)
    if int((y < 0).sum()) > 0:
        raise RuntimeError("Found unlabeled files in train_root; expected real/fake in filename.")

    print("[prep] computing q_log for train/test...")
    train_q = np.asarray(
        [_laplacian_var_log(p, max_side=int(args.quality_max_side)) for p in train_paths],
        dtype=np.float64,
    )
    test_q = np.asarray(
        [_laplacian_var_log(p, max_side=int(args.quality_max_side)) for p in test_paths],
        dtype=np.float64,
    )

    q_cuts = []
    if int(args.quality_bins) > 1:
        for j in range(1, int(args.quality_bins)):
            q_cuts.append(float(np.quantile(train_q, j / float(args.quality_bins))))

    train_bin = _qbin_ids(train_q, q_cuts)
    test_bin = _qbin_ids(test_q, q_cuts)
    n_bins = int(max(train_bin.max(), test_bin.max()) + 1)

    train_logit = _logit(train_probs)
    test_logit = _logit(test_probs)

    # CV for blend alpha selection (1000 used only as calibration domain, no model update).
    folds = _stratified_folds(y, int(args.cv_folds), int(args.cv_seed))
    alpha_grid = [float(a) for a in list(args.alpha_grid)]
    alpha_cv_auc = {a: [] for a in alpha_grid}
    for fi in range(len(folds)):
        val_idx = folds[fi]
        tr_mask = np.ones((len(y),), dtype=bool)
        tr_mask[val_idx] = False
        tr_idx = np.where(tr_mask)[0]
        shifts = _fit_qbin_logit_shift(
            logits=train_logit[tr_idx],
            y=y[tr_idx],
            qbin=train_bin[tr_idx],
            n_bins=n_bins,
        )
        l_cal = _apply_qbin_shift(train_logit[val_idx], train_bin[val_idx], shifts)
        l_raw = train_logit[val_idx]
        for a in alpha_grid:
            l_mix = (1.0 - a) * l_raw + a * l_cal
            p_mix = _sigmoid(l_mix)
            auc = _auc_tieaware(y[val_idx], p_mix)
            alpha_cv_auc[a].append(float(auc))

    alpha_mean_auc = {a: float(np.mean(v)) for a, v in alpha_cv_auc.items()}
    best_alpha = sorted(alpha_mean_auc.items(), key=lambda x: (-x[1], x[0]))[0][0]

    # Fit on full 1000
    full_shifts = _fit_qbin_logit_shift(train_logit, y, train_bin, n_bins=n_bins)
    train_l_cal = _apply_qbin_shift(train_logit, train_bin, full_shifts)
    test_l_cal = _apply_qbin_shift(test_logit, test_bin, full_shifts)

    # Three outputs for ablation: raw / full-cal / cv-best blend
    p_raw = train_probs
    p_full = _sigmoid(train_l_cal)
    p_best = _sigmoid((1.0 - best_alpha) * train_logit + best_alpha * train_l_cal)
    auc_raw = _auc_tieaware(y, p_raw)
    auc_full = _auc_tieaware(y, p_full)
    auc_best = _auc_tieaware(y, p_best)

    test_p_raw = test_probs
    test_p_full = _sigmoid(test_l_cal)
    test_p_best = _sigmoid((1.0 - best_alpha) * test_logit + best_alpha * test_l_cal)

    out_raw_txt = out_prefix.with_name(out_prefix.name + "_raw.txt")
    out_raw_zip = out_prefix.with_name(out_prefix.name + "_raw.zip")
    out_full_txt = out_prefix.with_name(out_prefix.name + "_qcal.txt")
    out_full_zip = out_prefix.with_name(out_prefix.name + "_qcal.zip")
    out_best_txt = out_prefix.with_name(out_prefix.name + "_cvbest.txt")
    out_best_zip = out_prefix.with_name(out_prefix.name + "_cvbest.zip")
    out_diag = out_prefix.with_name(out_prefix.name + "_diag.json")

    _write_submission_txt(out_raw_txt, test_p_raw)
    _zip_submission(out_raw_txt, out_raw_zip)
    _write_submission_txt(out_full_txt, test_p_full)
    _zip_submission(out_full_txt, out_full_zip)
    _write_submission_txt(out_best_txt, test_p_best)
    _zip_submission(out_best_txt, out_best_zip)

    diag = {
        "base_ckpt": str(ckpt),
        "train_root": str(train_root),
        "test_root": str(test_root),
        "n_train": int(len(train_probs)),
        "n_test": int(len(test_probs)),
        "quality_bins": int(args.quality_bins),
        "q_cuts": [float(x) for x in q_cuts],
        "full_shifts": [float(x) for x in full_shifts.tolist()],
        "cv_folds": int(args.cv_folds),
        "alpha_grid": alpha_grid,
        "alpha_mean_auc": {str(k): float(v) for k, v in alpha_mean_auc.items()},
        "best_alpha": float(best_alpha),
        "train_auc_raw": float(auc_raw),
        "train_auc_qcal": float(auc_full),
        "train_auc_cvbest": float(auc_best),
        "outputs": {
            "raw_zip": str(out_raw_zip),
            "qcal_zip": str(out_full_zip),
            "cvbest_zip": str(out_best_zip),
        },
    }
    out_diag.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print("[done] train auc raw=%.6f qcal=%.6f cvbest=%.6f" % (auc_raw, auc_full, auc_best))
    print("[done] best alpha =", best_alpha)
    print("[done] raw zip   ->", out_raw_zip)
    print("[done] qcal zip  ->", out_full_zip)
    print("[done] cvbest zip->", out_best_zip)
    print("[done] diag      ->", out_diag)


if __name__ == "__main__":
    main()
