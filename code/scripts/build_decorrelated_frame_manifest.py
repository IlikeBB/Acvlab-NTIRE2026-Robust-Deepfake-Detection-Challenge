#!/usr/bin/env python3
"""
Build per-video decorrelated frame manifest (path,label) from a video-level CSV.

Input video CSV should contain at least:
  - frames_dir
  - label

Output:
  - path,label CSV for ManifestDataset
  - optional diagnostics report
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg"}


def _sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def _laplacian_qlog(gray: np.ndarray, eps: float = 1e-6) -> float:
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    v = float(np.var(lap))
    return float(np.log(max(v, eps)))


def _frame_descriptor(img: Image.Image, side: int = 32) -> Tuple[np.ndarray, float]:
    g = img.convert("L").resize((side, side), resample=Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32) / 255.0
    q = _laplacian_qlog(arr)
    desc = arr.reshape(-1).astype(np.float32)
    return desc, q


def _farthest_with_quality(
    descs: np.ndarray,
    qvals: np.ndarray,
    k: int,
    quality_weight: float = 0.15,
) -> List[int]:
    n = int(descs.shape[0])
    if n <= k:
        return list(range(n))
    if k <= 1:
        return [int(np.argmax(qvals))]

    # pairwise squared distances
    sq = np.sum(descs * descs, axis=1, keepdims=True)
    D = np.clip(sq + sq.T - 2.0 * (descs @ descs.T), a_min=0.0, a_max=None)

    # first frame: high quality with mild centrality penalty
    center = np.mean(D, axis=1)
    qz = (qvals - np.mean(qvals)) / (np.std(qvals) + 1e-6)
    cz = (center - np.mean(center)) / (np.std(center) + 1e-6)
    seed_score = 0.80 * qz - 0.20 * cz
    sel = [int(np.argmax(seed_score))]

    min_d = D[sel[0]].copy()
    for _ in range(1, k):
        mask = np.ones(n, dtype=bool)
        mask[sel] = False
        if not np.any(mask):
            break
        qz_now = (qvals - np.mean(qvals)) / (np.std(qvals) + 1e-6)
        score = min_d + float(quality_weight) * qz_now
        score[~mask] = -1e9
        nxt = int(np.argmax(score))
        sel.append(nxt)
        min_d = np.minimum(min_d, D[nxt])

    return sorted(sel)


def _uniform_indices(n: int, k: int) -> List[int]:
    if n <= k:
        return list(range(n))
    idx = np.linspace(0, n - 1, num=k, dtype=int)
    return sorted([int(i) for i in idx.tolist()])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build decorrelated per-video frame manifest")
    p.add_argument(
        "--video_csv",
        type=Path,
        required=True,
        help="CSV with frames_dir,label rows",
    )
    p.add_argument(
        "--out_csv",
        type=Path,
        required=True,
        help="Output manifest CSV with path,label",
    )
    p.add_argument("--frames_per_video", type=int, default=8)
    p.add_argument("--desc_side", type=int, default=32)
    p.add_argument("--quality_weight", type=float, default=0.15)
    p.add_argument("--report_md", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.report_md is not None:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)

    vdf = pd.read_csv(args.video_csv)
    required = {"frames_dir", "label"}
    miss = [c for c in required if c not in vdf.columns]
    if miss:
        raise ValueError(f"video_csv missing columns: {miss}")

    rows = []
    n_video_ok = 0
    n_video_miss = 0
    gains = []

    for _, r in vdf.iterrows():
        frames_dir = Path(str(r["frames_dir"]))
        label = int(r["label"])
        if not frames_dir.is_dir():
            n_video_miss += 1
            continue
        fs = sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS], key=_sort_key)
        if not fs:
            n_video_miss += 1
            continue

        descs = []
        qvals = []
        kept = []
        for p in fs:
            try:
                img = Image.open(p).convert("RGB")
                d, q = _frame_descriptor(img, side=int(args.desc_side))
                descs.append(d)
                qvals.append(q)
                kept.append(p)
            except Exception:
                continue
        if not kept:
            n_video_miss += 1
            continue
        n = len(kept)
        k = int(min(max(1, args.frames_per_video), n))

        X = np.stack(descs, axis=0).astype(np.float32)
        q = np.asarray(qvals, dtype=np.float32)
        idx_dec = _farthest_with_quality(X, q, k=k, quality_weight=float(args.quality_weight))
        idx_uni = _uniform_indices(n, k=k)

        # diversity gain: mean pairwise L2 distance selected vs uniform
        def _mean_pairwise(idx: Sequence[int]) -> float:
            if len(idx) <= 1:
                return 0.0
            sub = X[np.asarray(list(idx), dtype=int)]
            sq = np.sum(sub * sub, axis=1, keepdims=True)
            d2 = np.clip(sq + sq.T - 2.0 * (sub @ sub.T), 0.0, None)
            iu = np.triu_indices(len(idx), k=1)
            if len(iu[0]) == 0:
                return 0.0
            return float(np.mean(np.sqrt(d2[iu])))

        md = _mean_pairwise(idx_dec)
        mu = _mean_pairwise(idx_uni)
        gains.append(md - mu)

        for i in idx_dec:
            rows.append({"path": str(kept[int(i)]), "label": label})
        n_video_ok += 1

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)

    if args.report_md is not None:
        lines = []
        lines.append("# Decorrelated Frame Manifest Report")
        lines.append("")
        lines.append(f"- input_video_csv: `{args.video_csv}`")
        lines.append(f"- out_csv: `{args.out_csv}`")
        lines.append(f"- frames_per_video: `{int(args.frames_per_video)}`")
        lines.append(f"- videos_ok: `{n_video_ok}`")
        lines.append(f"- videos_missing_or_empty: `{n_video_miss}`")
        lines.append(f"- output_rows: `{len(out_df)}`")
        if len(out_df) > 0:
            vc = out_df["label"].value_counts().sort_index().to_dict()
            lines.append(f"- label_counts: `{vc}`")
        if gains:
            g = np.asarray(gains, dtype=float)
            lines.append(
                f"- diversity_gain(decorr - uniform): mean=`{float(g.mean()):.6f}` p50=`{float(np.quantile(g,0.5)):.6f}` p90=`{float(np.quantile(g,0.9)):.6f}`"
            )
        args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] out -> {args.out_csv}")
    print(f"[ok] rows={len(out_df)} videos_ok={n_video_ok} videos_miss={n_video_miss}")


if __name__ == "__main__":
    main()

