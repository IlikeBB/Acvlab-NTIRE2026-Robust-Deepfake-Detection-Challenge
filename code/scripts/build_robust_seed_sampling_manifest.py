#!/usr/bin/env python3
"""
Build robust training manifests to reduce seed variance.

Design:
- Select DF40 videos with a robustness-aware score:
  LB-proxy alignment + method diversity bonus - quality-extreme penalty.
- Keep class balance at video level.
- Expand each selected video into fixed number of frame paths (path,label CSV).
- Add competition anchor samples from historical records:
  helpful top200 + anti-transfer + top hard samples.
- Produce a mixed manifest (path,label) ready for --train_csv.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


IMG_EXTS = {".png", ".jpg", ".jpeg"}


def _zscore(x: pd.Series) -> pd.Series:
    std = float(x.std(ddof=0))
    if std < 1e-12:
        return pd.Series(np.zeros(len(x), dtype=np.float32), index=x.index)
    return (x - float(x.mean())) / std


def _sample_evenly(paths: List[Path], n: int) -> List[Path]:
    if n <= 0:
        return []
    if len(paths) <= n:
        return paths
    # Deterministic uniform pick across sorted frame list.
    idx = np.linspace(0, len(paths) - 1, num=n, dtype=int)
    return [paths[int(i)] for i in idx]


def _read_top_csv(path: Path, n: int, sort_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sort_col not in df.columns:
        raise ValueError(f"{path} missing sort col: {sort_col}")
    return df.sort_values(sort_col, ascending=False).head(n).copy()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class BuildOutputs:
    df40_video_csv: Path
    df40_path_csv: Path
    comp_anchor_csv: Path
    mix_csv: Path
    summary_csv: Path
    summary_md: Path


def select_df40_videos(
    df40_scores_csv: Path,
    target_per_class: int,
    fake_cap_ratio: float,
    quality_penalty: float,
    method_bonus_weight: float,
) -> pd.DataFrame:
    df = pd.read_csv(df40_scores_csv)
    need_cols = {
        "video_abs_path",
        "label",
        "method",
        "proxy_top20_mean",
        "q_log_mean",
        "q_log_std",
    }
    missing = sorted(list(need_cols - set(df.columns)))
    if missing:
        raise ValueError(f"df40 score file missing columns: {missing}")

    df = df.copy()
    df["video_abs_path"] = df["video_abs_path"].astype(str)
    df = df.drop_duplicates("video_abs_path").reset_index(drop=True)

    real_df = df[df["label"] == 0].copy()
    fake_df = df[df["label"] == 1].copy()
    n_real = min(target_per_class, len(real_df))
    n_fake = min(target_per_class, len(fake_df))

    real_df["robust_score"] = (
        _zscore(real_df["proxy_top20_mean"])
        - quality_penalty * _zscore(real_df["q_log_mean"]).abs()
    )
    real_sel = real_df.sort_values("robust_score", ascending=False).head(n_real).copy()

    fake_df["z_proxy"] = _zscore(fake_df["proxy_top20_mean"])
    fake_df["z_qabs"] = _zscore(fake_df["q_log_mean"]).abs()
    freq = fake_df["method"].value_counts()
    freq_map = fake_df["method"].map(freq).astype(float)
    # Prefer rarer methods for diversity.
    fake_df["method_bonus"] = np.log((freq.max() + 1.0) / (freq_map + 1.0))
    fake_df["robust_score"] = (
        fake_df["z_proxy"]
        - quality_penalty * fake_df["z_qabs"]
        + method_bonus_weight * fake_df["method_bonus"]
    )
    fake_sorted = fake_df.sort_values("robust_score", ascending=False).copy()

    cap = max(20, int(np.ceil(n_fake * fake_cap_ratio)))
    method_used: Dict[str, int] = {}
    fake_rows: List[pd.Series] = []

    for _, row in fake_sorted.iterrows():
        m = str(row["method"])
        used = method_used.get(m, 0)
        if used >= cap:
            continue
        fake_rows.append(row)
        method_used[m] = used + 1
        if len(fake_rows) >= n_fake:
            break

    # Fill remaining slots if caps are too strict.
    if len(fake_rows) < n_fake:
        used_paths = {str(r["video_abs_path"]) for r in fake_rows}
        for _, row in fake_sorted.iterrows():
            p = str(row["video_abs_path"])
            if p in used_paths:
                continue
            fake_rows.append(row)
            used_paths.add(p)
            if len(fake_rows) >= n_fake:
                break

    fake_sel = pd.DataFrame(fake_rows).copy()
    keep_cols = [
        "video_abs_path",
        "label",
        "method",
        "proxy_top20_mean",
        "q_log_mean",
        "q_log_std",
        "robust_score",
    ]
    out = pd.concat([real_sel[keep_cols], fake_sel[keep_cols]], axis=0, ignore_index=True)
    out["source_tag"] = "df40_robust_video"
    out = out.sort_values(["label", "robust_score"], ascending=[True, False]).reset_index(drop=True)
    return out


def expand_df40_to_paths(df40_video_df: pd.DataFrame, frames_per_video: int) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    miss_dirs = 0
    miss_frames = 0
    for _, row in df40_video_df.iterrows():
        vpath = Path(str(row["video_abs_path"]))
        if not vpath.is_dir():
            miss_dirs += 1
            continue
        frames = sorted([p for p in vpath.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()])
        if not frames:
            miss_frames += 1
            continue
        chosen = _sample_evenly(frames, frames_per_video)
        for p in chosen:
            records.append(
                {
                    "path": str(p),
                    "label": int(row["label"]),
                    "source": "df40_robust",
                    "video_abs_path": str(vpath),
                    "method": str(row["method"]),
                    "robust_score": float(row["robust_score"]),
                }
            )
    out = pd.DataFrame(records)
    out.attrs["missing_video_dirs"] = miss_dirs
    out.attrs["missing_frame_dirs"] = miss_frames
    return out


def build_comp_anchor(
    helpful_top_csv: Path,
    anti_transfer_csv: Path,
    hard_full_csv: Path,
    hard_top_k: int,
    rep_helpful: int,
    rep_anti: int,
    rep_hard: int,
) -> pd.DataFrame:
    path2rec: Dict[str, Dict[str, object]] = {}

    def _upsert(path: str, label: int, tag: str, rep: int) -> None:
        p = str(path)
        if p not in path2rec:
            path2rec[p] = {
                "path": p,
                "label": int(label),
                "repeat": int(rep),
                "tags": {tag},
            }
            return
        path2rec[p]["repeat"] = max(int(path2rec[p]["repeat"]), int(rep))
        path2rec[p]["tags"].add(tag)

    helpful = pd.read_csv(helpful_top_csv)
    for _, r in helpful.iterrows():
        _upsert(str(r["path"]), int(r["label"]), "comp_helpful_top200", rep_helpful)

    anti = pd.read_csv(anti_transfer_csv)
    for _, r in anti.iterrows():
        _upsert(str(r["path"]), int(r["label"]), "comp_anti_transfer", rep_anti)

    hard = _read_top_csv(hard_full_csv, n=hard_top_k, sort_col="hard_score")
    for _, r in hard.iterrows():
        _upsert(str(r["path"]), int(r["label"]), "comp_hard_top", rep_hard)

    rows: List[Dict[str, object]] = []
    for rec in path2rec.values():
        tags = sorted(list(rec["tags"]))
        rows.append(
            {
                "path": rec["path"],
                "label": int(rec["label"]),
                "repeat": int(rec["repeat"]),
                "tags": "|".join(tags),
                "source": "comp_anchor",
            }
        )
    return pd.DataFrame(rows).sort_values(["label", "repeat"], ascending=[True, False]).reset_index(drop=True)


def expand_with_repeat(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        rep = max(1, int(r.get("repeat", 1)))
        for _ in range(rep):
            rows.append({"path": str(r["path"]), "label": int(r["label"]), "source": str(r.get("source", "unk"))})
    return pd.DataFrame(rows)


def summarize_mix(mix_df: pd.DataFrame, df40_video_df: pd.DataFrame, comp_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    by_source = mix_df.groupby(["source", "label"]).size().reset_index(name="count")
    by_source["label_name"] = by_source["label"].map({0: "real", 1: "fake"})

    method_dist = (
        df40_video_df[df40_video_df["label"] == 1]["method"]
        .value_counts()
        .rename_axis("method")
        .reset_index(name="n_fake_videos")
    )
    method_dist["ratio"] = method_dist["n_fake_videos"] / max(1, method_dist["n_fake_videos"].sum())

    comp_tag = (
        comp_df.assign(tag_first=comp_df["tags"].astype(str).str.split("|").str[0])
        .groupby(["tag_first", "label"])
        .size()
        .reset_index(name="n_unique")
    )

    md_lines = []
    md_lines.append("# Robust Sampling Manifest Summary")
    md_lines.append("")
    md_lines.append("## Mixed manifest counts (after repeat expansion)")
    md_lines.append(by_source.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## DF40 fake method distribution (selected videos)")
    md_lines.append(method_dist.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## Competition anchor unique samples")
    md_lines.append(comp_tag.to_markdown(index=False))
    md = "\n".join(md_lines) + "\n"

    summary_df = by_source.copy()
    return summary_df, md


def build_outputs(args: argparse.Namespace) -> BuildOutputs:
    ts = args.tag
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return BuildOutputs(
        df40_video_csv=out_dir / f"robust_df40_video_selection_{ts}.csv",
        df40_path_csv=out_dir / f"robust_df40eqdiv_frames{args.frames_per_video}_{ts}.csv",
        comp_anchor_csv=out_dir / f"robust_comp_anchor_{ts}.csv",
        mix_csv=out_dir / f"robust_mix_manifest_{ts}.csv",
        summary_csv=out_dir / f"robust_mix_summary_{ts}.csv",
        summary_md=out_dir / f"robust_mix_summary_{ts}.md",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build robust sampling manifests from historical records.")
    p.add_argument(
        "--df40_scores_csv",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/df40_lb_proxy_video_scores_20260305.csv"),
    )
    p.add_argument(
        "--helpful_top_csv",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/train_manifest_lb_aligned_top200_20260305.csv"),
    )
    p.add_argument(
        "--anti_transfer_csv",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/train_ft276_anti_transfer_20260303.csv"),
    )
    p.add_argument(
        "--hard_full_csv",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/hardsample_val200_full_20260301.csv"),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score"),
    )
    p.add_argument("--tag", type=str, default="20260306")
    p.add_argument("--target_per_class", type=int, default=566)
    p.add_argument("--frames_per_video", type=int, default=8)
    p.add_argument("--fake_cap_ratio", type=float, default=0.22)
    p.add_argument("--quality_penalty", type=float, default=0.25)
    p.add_argument("--method_bonus_weight", type=float, default=0.40)
    p.add_argument("--hard_top_k", type=int, default=120)
    p.add_argument("--rep_helpful", type=int, default=2)
    p.add_argument("--rep_anti", type=int, default=2)
    p.add_argument("--rep_hard", type=int, default=3)
    p.add_argument("--seed", type=int, default=2048)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    out = build_outputs(args)

    df40_video_df = select_df40_videos(
        df40_scores_csv=args.df40_scores_csv,
        target_per_class=int(args.target_per_class),
        fake_cap_ratio=float(args.fake_cap_ratio),
        quality_penalty=float(args.quality_penalty),
        method_bonus_weight=float(args.method_bonus_weight),
    )
    _ensure_parent(out.df40_video_csv)
    df40_video_df.to_csv(out.df40_video_csv, index=False)

    df40_path_df = expand_df40_to_paths(df40_video_df, frames_per_video=int(args.frames_per_video))
    _ensure_parent(out.df40_path_csv)
    df40_path_df.to_csv(out.df40_path_csv, index=False)

    comp_df = build_comp_anchor(
        helpful_top_csv=args.helpful_top_csv,
        anti_transfer_csv=args.anti_transfer_csv,
        hard_full_csv=args.hard_full_csv,
        hard_top_k=int(args.hard_top_k),
        rep_helpful=int(args.rep_helpful),
        rep_anti=int(args.rep_anti),
        rep_hard=int(args.rep_hard),
    )
    _ensure_parent(out.comp_anchor_csv)
    comp_df.to_csv(out.comp_anchor_csv, index=False)

    comp_expanded = expand_with_repeat(comp_df)
    mix_df = pd.concat(
        [
            df40_path_df[["path", "label", "source"]].copy(),
            comp_expanded[["path", "label", "source"]].copy(),
        ],
        axis=0,
        ignore_index=True,
    )
    mix_df = mix_df.sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)
    _ensure_parent(out.mix_csv)
    mix_df.to_csv(out.mix_csv, index=False)

    summary_df, summary_md = summarize_mix(mix_df, df40_video_df, comp_df)
    summary_df.to_csv(out.summary_csv, index=False)
    out.summary_md.write_text(summary_md, encoding="utf-8")

    print(f"[ok] df40 videos: {out.df40_video_csv}")
    print(f"[ok] df40 paths:  {out.df40_path_csv}")
    print(f"[ok] comp anchor: {out.comp_anchor_csv}")
    print(f"[ok] mix manifest: {out.mix_csv}")
    print(f"[ok] summary csv: {out.summary_csv}")
    print(f"[ok] summary md:  {out.summary_md}")
    print(
        "[hint] train with: "
        f"python train.py --train_csv {out.mix_csv} --val_csv <your_val_csv> ..."
    )


if __name__ == "__main__":
    main()
