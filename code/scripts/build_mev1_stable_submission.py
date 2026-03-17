#!/usr/bin/env python3
import argparse
import math
import subprocess
import zipfile
from pathlib import Path

import numpy as np


def _read_probs(path: Path) -> np.ndarray:
    vals = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = raw.strip()
        if not s:
            raise ValueError(f"Empty line at {i} in {path}")
        v = float(s.split(",")[0].strip())
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Out-of-range prob at line {i} in {path}: {v}")
        vals.append(v)
    if not vals:
        raise ValueError(f"No values in {path}")
    return np.asarray(vals, dtype=np.float64)


def _trimmed_mean(x: np.ndarray, trim_ratio: float) -> np.ndarray:
    n = int(x.shape[0])
    t = int(math.floor(float(trim_ratio) * n))
    if t <= 0 or (2 * t) >= n:
        return x.mean(axis=0)
    s = np.sort(x, axis=0)
    return s[t : n - t].mean(axis=0)


def _aggregate(preds: np.ndarray, mode: str, trim_ratio: float) -> np.ndarray:
    mode = str(mode).lower()
    eps = 1e-6

    if mode.startswith("logit_"):
        x = np.clip(preds, eps, 1.0 - eps)
        x = np.log(x / (1.0 - x))
        if mode == "logit_mean":
            y = x.mean(axis=0)
        elif mode == "logit_median":
            y = np.median(x, axis=0)
        elif mode == "logit_trimmed_mean":
            y = _trimmed_mean(x, trim_ratio=trim_ratio)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return 1.0 / (1.0 + np.exp(-y))

    if mode == "prob_mean":
        return preds.mean(axis=0)
    if mode == "prob_median":
        return np.median(preds, axis=0)
    if mode == "prob_trimmed_mean":
        return _trimmed_mean(preds, trim_ratio=trim_ratio)
    raise ValueError(f"Unknown mode: {mode}")


def _write_submission_txt(path: Path, probs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in probs:
            vv = float(v)
            if vv < 0.0 or vv > 1.0:
                raise ValueError(f"Out-of-range prob: {vv}")
            f.write(f"{vv:.6f}\n")


def _zip_submission(txt_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname="submission.txt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--python_bin", type=str, default="/home/u1657859/miniconda3/envs/eamamba/bin/python")
    p.add_argument("--full_submission_py", type=str, default="/work/u1657859/Jess/app/full_submission.py")
    p.add_argument(
        "--ckpts",
        type=str,
        nargs="+",
        default=[
            "/work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_full6_8f_s2048_local/best_auc.pt",
        ],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[2048, 4096, 8192])
    p.add_argument(
        "--txt_path",
        type=str,
        default="/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final",
    )
    p.add_argument(
        "--root_dir",
        type=str,
        default="/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--multi_expert_k", type=int, default=3)
    p.add_argument(
        "--multi_expert_route",
        type=str,
        default="quality_bin",
        choices=["quality_bin", "quality_soft", "uniform", "router", "hybrid"],
    )
    p.add_argument("--tri_view", action="store_true", default=False)
    p.add_argument("--tri_blur_sig", type=float, nargs="+", default=[0.2, 1.2])
    p.add_argument("--tri_jpeg_qual", type=int, nargs="+", default=[70, 95])
    p.add_argument("--tri_resize_jitter", type=int, default=4)
    p.add_argument("--tri_unsharp_amount", type=float, default=0.35)
    p.add_argument("--tri_restore_mix", type=float, default=0.25)
    p.add_argument(
        "--tta_mode",
        type=str,
        default="hflip",
        choices=["none", "hflip", "hflip_resize"],
    )
    p.add_argument("--tta_scales", type=float, nargs="+", default=[0.94, 1.06])
    p.add_argument("--tta_logit_agg", type=str, default="mean", choices=["mean", "median"])
    p.add_argument(
        "--agg_mode",
        type=str,
        default="logit_mean",
        choices=[
            "logit_mean",
            "logit_median",
            "logit_trimmed_mean",
            "prob_mean",
            "prob_median",
            "prob_trimmed_mean",
        ],
    )
    p.add_argument("--trim_ratio", type=float, default=0.15)
    p.add_argument(
        "--tmp_dir",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/submissions/tmp_stable_mev1",
    )
    p.add_argument(
        "--out_txt",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/submissions/submission_MEv1_qbin_s3_hflip_tta.txt",
    )
    p.add_argument(
        "--out_zip",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/submissions/submission_MEv1_qbin_s3_hflip_tta_codabench.zip",
    )
    p.add_argument("--keep_intermediate", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    tmp_dir = Path(args.tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for ckpt in args.ckpts:
        for s in args.seeds:
            plan.append((Path(ckpt).resolve(), int(s)))
    if len(plan) < 3:
        raise ValueError("Need at least 3 members in ensemble (ckpts x seeds)")

    all_preds = []
    for i, (ckpt, seed) in enumerate(plan):
        out_i = tmp_dir / f"pred_{i:02d}_{ckpt.stem}_s{seed}.txt"
        cmd = [
            str(args.python_bin),
            str(Path(args.full_submission_py).resolve()),
            "--ckpt",
            str(ckpt),
            "--txt_path",
            str(args.txt_path),
            "--root_dir",
            str(args.root_dir),
            "--out_csv",
            str(out_i),
            "--batch_size",
            str(int(args.batch_size)),
            "--num_workers",
            str(int(args.num_workers)),
            "--seed",
            str(int(seed)),
            "--multi_expert_enable",
            "--multi_expert_k",
            str(int(args.multi_expert_k)),
            "--multi_expert_route",
            str(args.multi_expert_route),
            "--infer_quality_bin",
        ]
        if bool(args.tri_view):
            cmd += [
                "--tri_view",
                "--tri_blur_sig",
                *[str(v) for v in list(args.tri_blur_sig)],
                "--tri_jpeg_qual",
                *[str(v) for v in list(args.tri_jpeg_qual)],
                "--tri_resize_jitter",
                str(int(args.tri_resize_jitter)),
                "--tri_unsharp_amount",
                str(float(args.tri_unsharp_amount)),
                "--tri_restore_mix",
                str(float(args.tri_restore_mix)),
            ]
        if str(args.tta_mode).lower() != "none":
            cmd += [
                "--tta_mode",
                str(args.tta_mode),
                "--tta_logit_agg",
                str(args.tta_logit_agg),
            ]
            if str(args.tta_mode).lower() == "hflip_resize":
                cmd += [
                    "--tta_scales",
                    *[str(v) for v in list(args.tta_scales)],
                ]

        print(f"[run {i+1}/{len(plan)}] ckpt={ckpt.name} seed={seed}")
        subprocess.run(cmd, check=True)
        all_preds.append(_read_probs(out_i))

    mat = np.stack(all_preds, axis=0)
    probs = _aggregate(mat, mode=args.agg_mode, trim_ratio=float(args.trim_ratio))

    out_txt = Path(args.out_txt).resolve()
    out_zip = Path(args.out_zip).resolve()
    _write_submission_txt(out_txt, probs)
    _zip_submission(out_txt, out_zip)

    print(
        "[done] ensemble built:",
        f"members={mat.shape[0]} files={mat.shape[1]}",
        f"agg={args.agg_mode}",
        f"trim_ratio={float(args.trim_ratio):.3f}",
    )
    print(f"[done] txt -> {out_txt}")
    print(f"[done] zip -> {out_zip}")

    if not bool(args.keep_intermediate):
        for p in tmp_dir.glob("pred_*.txt"):
            p.unlink(missing_ok=True)
        print(f"[cleanup] removed intermediates in {tmp_dir}")


if __name__ == "__main__":
    main()
