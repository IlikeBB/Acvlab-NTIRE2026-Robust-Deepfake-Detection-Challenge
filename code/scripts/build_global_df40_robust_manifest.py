#!/usr/bin/env python3
"""
Build DF40 robust sampling manifests using:
1) Cross-repo run inventory (app/app3/app4/app-isolated)
2) Multi-model val1000 inferbank (per-image predictions)
3) Existing DF40 LB-proxy video scores

Outputs are written under app/score and are ready for --train_csv usage.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg"}


def _z(x: pd.Series) -> pd.Series:
    s = float(x.std(ddof=0))
    if s < 1e-12:
        return pd.Series(np.zeros(len(x), dtype=np.float32), index=x.index)
    return (x - float(x.mean())) / s


def _sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def _laplacian_var_log(img: Image.Image, max_side: int = 256, eps: float = 1e-6) -> float:
    if max_side > 0:
        w, h = img.size
        m = max(w, h)
        if m > max_side:
            sc = float(max_side) / float(m)
            nw = max(1, int(round(w * sc)))
            nh = max(1, int(round(h * sc)))
            img = img.resize((nw, nh), resample=Image.BILINEAR)
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (
        -4.0 * g
        + np.roll(g, 1, axis=0)
        + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1)
        + np.roll(g, -1, axis=1)
    )
    v = float(np.var(lap))
    return float(np.log(max(v, eps)))


def _parse_df40_method_and_video(path_str: str) -> Tuple[str, str, int]:
    """
    Returns method, video_dir, label for DF40 path.
    Expects path containing /DF40_frames/train/DF40/(real|fake)/<method>/frames/...
    """
    p = Path(path_str)
    s = str(p)
    key = "/DF40_frames/train/DF40/"
    i = s.find(key)
    if i < 0:
        return "", "", -1
    tail = s[i + len(key) :].strip("/")
    parts = tail.split("/")
    if len(parts) < 3:
        return "", "", -1
    label_name = parts[0]
    method = parts[1]
    label = 1 if label_name == "fake" else 0
    if p.suffix.lower() in IMG_EXTS:
        video_dir = str(p.parent)
    else:
        video_dir = str(p)
    return method, video_dir, label


def read_manifest_method_share(train_csv: Path) -> Dict[str, float]:
    """Return fake-method shares among DF40 rows for one train_csv."""
    if not train_csv.exists():
        return {}
    try:
        df = pd.read_csv(train_csv)
    except Exception:
        return {}
    if "frames_dir" in df.columns:
        paths = df["frames_dir"].astype(str).tolist()
    elif "path" in df.columns:
        paths = df["path"].astype(str).tolist()
    else:
        return {}
    c = Counter()
    n = 0
    for s in paths:
        m, _, lab = _parse_df40_method_and_video(s)
        if lab != 1:
            continue
        c[m] += 1
        n += 1
    if n == 0:
        return {}
    return {k: float(v) / float(n) for k, v in c.items()}


def build_method_lift(
    infer_summary: pd.DataFrame,
    model_table: pd.DataFrame,
    train_inventory: pd.DataFrame,
    df40_video_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build method lift from model performance + per-run manifest composition.
    """
    ok = infer_summary[infer_summary["status"] == "ok"].copy()
    inv = train_inventory.copy()
    if "exp_name" not in inv.columns and "exp" in inv.columns:
        inv = inv.rename(columns={"exp": "exp_name"})
    ok = ok.merge(model_table[["model_id", "exp_name"]], on="model_id", how="left")
    ok = ok.merge(inv[["exp_name", "train_csv"]], on="exp_name", how="left")
    ok["train_csv"] = ok["train_csv"].astype(str)

    auc_med = float(ok["auc"].median())
    ok["w_pos"] = (ok["auc"] - auc_med).clip(lower=0.0)
    ok["w_neg"] = (auc_med - ok["auc"]).clip(lower=0.0)

    method_pos = defaultdict(float)
    method_neg = defaultdict(float)
    pos_sum = 0.0
    neg_sum = 0.0
    seen_cache: Dict[str, Dict[str, float]] = {}

    for _, r in ok.iterrows():
        tcsv = str(r["train_csv"])
        if tcsv not in seen_cache:
            seen_cache[tcsv] = read_manifest_method_share(Path(tcsv))
        share = seen_cache[tcsv]
        if not share:
            continue
        wp = float(r["w_pos"])
        wn = float(r["w_neg"])
        if wp > 0:
            pos_sum += wp
            for m, v in share.items():
                method_pos[m] += wp * float(v)
        if wn > 0:
            neg_sum += wn
            for m, v in share.items():
                method_neg[m] += wn * float(v)

    baseline = (
        df40_video_scores[df40_video_scores["label"] == 1]["method"].value_counts(normalize=True).to_dict()
    )
    methods = sorted(set(list(baseline.keys()) + list(method_pos.keys()) + list(method_neg.keys())))
    rows = []
    for m in methods:
        p = method_pos.get(m, 0.0) / max(pos_sum, 1e-8)
        n = method_neg.get(m, 0.0) / max(neg_sum, 1e-8)
        b = float(baseline.get(m, 1e-6))
        lift = math.log((p + 1e-6) / (b + 1e-6))
        contra = math.log((p + 1e-6) / (n + 1e-6))
        rows.append(
            {
                "method": m,
                "share_pos": p,
                "share_neg": n,
                "share_baseline": b,
                "lift_vs_baseline_log": lift,
                "lift_pos_vs_neg_log": contra,
            }
        )
    out = pd.DataFrame(rows).sort_values("lift_vs_baseline_log", ascending=False).reset_index(drop=True)
    return out


def build_val_difficulty(
    inferbank_dir: Path,
    q_cache_path: Path,
    hard_top_ratio: float = 0.30,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Return:
    - val_df with per-sample difficulty stats
    - hard_df top hard samples
    - label->bin_edges
    - label->hard_lift_by_bin
    """
    score_csvs = sorted([p for p in inferbank_dir.glob("*.csv") if p.name != "inferbank_summary.csv"])
    if not score_csvs:
        raise RuntimeError(f"No model score csv found in {inferbank_dir}")

    base = pd.read_csv(score_csvs[0])[["path", "label"]].copy()
    for p in score_csvs:
        m = p.stem
        d = pd.read_csv(p)[["path", "score"]].rename(columns={"score": f"s_{m}"})
        base = base.merge(d, on="path", how="inner")

    score_cols = [c for c in base.columns if c.startswith("s_")]
    S = base[score_cols].to_numpy(dtype=float)
    y = base["label"].astype(int).to_numpy()
    eps = 1e-6

    # per-sample difficulty from ensemble behavior
    mean_score = S.mean(axis=1)
    std_score = S.std(axis=1)
    pred_bin = (S >= 0.5).astype(int)
    err_rate = (pred_bin != y[:, None]).mean(axis=1)
    bce = -(y[:, None] * np.log(np.clip(S, eps, 1 - eps)) + (1 - y[:, None]) * np.log(np.clip(1 - S, eps, 1 - eps)))
    bce_mean = bce.mean(axis=1)
    bce_std = bce.std(axis=1)

    base["score_mean"] = mean_score
    base["score_std"] = std_score
    base["err_rate"] = err_rate
    base["bce_mean"] = bce_mean
    base["bce_std"] = bce_std

    # q_log cache
    if q_cache_path.exists():
        qdf = pd.read_csv(q_cache_path)
        qmap = dict(zip(qdf["path"].astype(str), qdf["q_log"].astype(float)))
        base["q_log"] = base["path"].astype(str).map(qmap)
    else:
        qvals = []
        for p in base["path"].astype(str):
            qvals.append(_laplacian_var_log(Image.open(p).convert("RGB"), max_side=256))
        base["q_log"] = qvals
        pd.DataFrame({"path": base["path"], "q_log": base["q_log"]}).to_csv(q_cache_path, index=False)

    # normalized difficulty score
    d = (
        0.50 * _z(base["bce_mean"])
        + 0.25 * _z(base["score_std"])
        + 0.25 * _z(base["err_rate"])
    )
    base["difficulty"] = d
    base = base.sort_values("difficulty", ascending=False).reset_index(drop=True)
    k = max(1, int(round(len(base) * hard_top_ratio)))
    hard_df = base.head(k).copy()

    # q-bin lift by label: hard_freq / all_freq
    bin_edges: Dict[int, np.ndarray] = {}
    lift_map: Dict[int, np.ndarray] = {}
    for lab in [0, 1]:
        all_lab = base[base["label"] == lab]["q_log"].astype(float).to_numpy()
        hard_lab = hard_df[hard_df["label"] == lab]["q_log"].astype(float).to_numpy()
        if len(all_lab) < 10 or len(hard_lab) < 5:
            edges = np.array([-np.inf, np.inf], dtype=float)
            lift = np.array([1.0], dtype=float)
            bin_edges[lab] = edges
            lift_map[lab] = lift
            continue
        qs = np.quantile(all_lab, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        edges = np.unique(qs)
        if len(edges) <= 2:
            edges = np.array([all_lab.min() - 1e-6, all_lab.max() + 1e-6], dtype=float)
        all_hist, _ = np.histogram(all_lab, bins=edges)
        hard_hist, _ = np.histogram(hard_lab, bins=edges)
        all_p = all_hist / max(all_hist.sum(), 1)
        hard_p = hard_hist / max(hard_hist.sum(), 1)
        lift = (hard_p + 1e-6) / (all_p + 1e-6)
        lift = lift / np.mean(lift)
        bin_edges[lab] = edges
        lift_map[lab] = lift

    return base, hard_df, bin_edges, lift_map


def score_df40_videos(
    df40_video_scores: pd.DataFrame,
    method_lift_df: pd.DataFrame,
    q_bin_edges: Dict[int, np.ndarray],
    q_lift_map: Dict[int, np.ndarray],
) -> pd.DataFrame:
    df = df40_video_scores.copy()
    mlift = dict(zip(method_lift_df["method"], method_lift_df["lift_vs_baseline_log"]))
    df["method_lift_raw"] = df["method"].map(mlift).fillna(0.0)

    q_lift_vals = []
    for _, r in df.iterrows():
        lab = int(r["label"])
        q = float(r["q_log_mean"])
        edges = q_bin_edges.get(lab, np.array([-np.inf, np.inf], dtype=float))
        lift = q_lift_map.get(lab, np.array([1.0], dtype=float))
        if len(edges) < 2:
            q_lift_vals.append(1.0)
            continue
        b = int(np.clip(np.digitize([q], edges[1:-1], right=False)[0], 0, len(lift) - 1))
        q_lift_vals.append(float(lift[b]))
    df["q_hard_lift"] = q_lift_vals

    # composite robust score
    df["z_proxy"] = _z(df["proxy_top20_mean"])
    df["z_mlift"] = _z(df["method_lift_raw"])
    df["z_qlift"] = _z(df["q_hard_lift"])
    # strong prior on LB-proxy, but include hard-profile and method lift
    df["robust_global_score"] = 0.58 * df["z_proxy"] + 0.24 * df["z_qlift"] + 0.18 * df["z_mlift"]
    return df


def select_equalclass_with_fake_caps(
    scored_df: pd.DataFrame,
    n_per_class: int,
    fake_max_share: float,
) -> pd.DataFrame:
    real = scored_df[scored_df["label"] == 0].copy().sort_values("robust_global_score", ascending=False)
    fake = scored_df[scored_df["label"] == 1].copy().sort_values("robust_global_score", ascending=False)

    n_real = min(n_per_class, len(real))
    n_fake = min(n_per_class, len(fake))
    real_sel = real.head(n_real).copy()

    cap = max(10, int(round(n_fake * fake_max_share)))
    used = Counter()
    fake_rows = []
    for _, r in fake.iterrows():
        m = str(r["method"])
        if used[m] >= cap:
            continue
        fake_rows.append(r)
        used[m] += 1
        if len(fake_rows) >= n_fake:
            break
    if len(fake_rows) < n_fake:
        picked = {str(r["video_abs_path"]) for r in fake_rows}
        for _, r in fake.iterrows():
            p = str(r["video_abs_path"])
            if p in picked:
                continue
            fake_rows.append(r)
            picked.add(p)
            if len(fake_rows) >= n_fake:
                break
    fake_sel = pd.DataFrame(fake_rows)
    out = pd.concat([real_sel, fake_sel], axis=0, ignore_index=True)
    out = out.sort_values(["label", "robust_global_score"], ascending=[True, False]).reset_index(drop=True)
    return out


def expand_video_to_paths(vdf: pd.DataFrame, frames_per_video: int) -> pd.DataFrame:
    rows = []
    for _, r in vdf.iterrows():
        v = Path(str(r["video_abs_path"]))
        if not v.is_dir():
            continue
        fs = sorted([p for p in v.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if not fs:
            continue
        if len(fs) > frames_per_video:
            idx = np.linspace(0, len(fs) - 1, num=frames_per_video, dtype=int)
            fs = [fs[int(i)] for i in idx]
        for p in fs:
            rows.append(
                {
                    "path": str(p),
                    "label": int(r["label"]),
                    "source": "df40_global_robust",
                    "video_abs_path": str(r["video_abs_path"]),
                    "method": str(r["method"]),
                    "robust_global_score": float(r["robust_global_score"]),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Global DF40 robust manifest builder.")
    p.add_argument(
        "--inferbank_dir",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/val1000_inferbank_20260306"),
    )
    p.add_argument(
        "--model_table_csv",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/global_val1000_infer_models_20260306.csv"),
    )
    p.add_argument(
        "--train_inventory_csv",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/global_traincsv_inventory_20260306.csv"),
    )
    p.add_argument(
        "--df40_video_scores_csv",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/df40_lb_proxy_video_scores_20260305.csv"),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score"),
    )
    p.add_argument("--tag", type=str, default="20260306_global")
    p.add_argument("--n_per_class", type=int, default=566)
    p.add_argument("--frames_per_video", type=int, default=8)
    p.add_argument("--fake_max_share", type=float, default=0.18)
    p.add_argument("--hard_top_ratio", type=float, default=0.30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    infer_summary = pd.read_csv(args.inferbank_dir / "inferbank_summary.csv")
    model_table = pd.read_csv(args.model_table_csv)
    train_inventory = pd.read_csv(args.train_inventory_csv)
    df40 = pd.read_csv(args.df40_video_scores_csv)

    # 1) method lift from global runs + inferbank performance
    method_lift = build_method_lift(
        infer_summary=infer_summary,
        model_table=model_table,
        train_inventory=train_inventory,
        df40_video_scores=df40,
    )
    method_lift_path = args.out_dir / f"global_method_lift_{args.tag}.csv"
    method_lift.to_csv(method_lift_path, index=False)

    # 2) val1000 difficulty profile from inferbank
    q_cache = args.out_dir / f"val1000_qlog_cache_{args.tag}.csv"
    val_df, hard_df, q_edges, q_lift = build_val_difficulty(
        inferbank_dir=args.inferbank_dir,
        q_cache_path=q_cache,
        hard_top_ratio=float(args.hard_top_ratio),
    )
    val_out = args.out_dir / f"val1000_difficulty_{args.tag}.csv"
    hard_out = args.out_dir / f"val1000_hard_top_{args.tag}.csv"
    val_df.to_csv(val_out, index=False)
    hard_df.to_csv(hard_out, index=False)

    # 3) score df40 videos and select robust subset
    scored = score_df40_videos(
        df40_video_scores=df40,
        method_lift_df=method_lift,
        q_bin_edges=q_edges,
        q_lift_map=q_lift,
    )
    scored_out = args.out_dir / f"df40_global_scored_{args.tag}.csv"
    scored.to_csv(scored_out, index=False)

    sel = select_equalclass_with_fake_caps(
        scored_df=scored,
        n_per_class=int(args.n_per_class),
        fake_max_share=float(args.fake_max_share),
    )
    sel_out = args.out_dir / f"df40_global_robust_videos_{args.tag}.csv"
    sel.to_csv(sel_out, index=False)

    # 4) expand to frame path manifest
    paths = expand_video_to_paths(sel, frames_per_video=int(args.frames_per_video))
    path_out = args.out_dir / f"df40_global_robust_frames{args.frames_per_video}_{args.tag}.csv"
    paths.to_csv(path_out, index=False)
    path_pl_out = args.out_dir / f"df40_global_robust_frames{args.frames_per_video}_pathlabel_{args.tag}.csv"
    paths[["path", "label"]].to_csv(path_pl_out, index=False)

    # 5) short report
    report_path = args.out_dir / f"df40_global_robust_report_{args.tag}.md"
    fake_methods = sel[sel["label"] == 1]["method"].value_counts().reset_index()
    fake_methods.columns = ["method", "n_fake_videos"]
    fake_methods["ratio"] = fake_methods["n_fake_videos"] / max(1, fake_methods["n_fake_videos"].sum())

    lines = []
    lines.append("# Global DF40 Robust Manifest Report")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- inferbank: `{args.inferbank_dir}`")
    lines.append(f"- model table: `{args.model_table_csv}`")
    lines.append(f"- train inventory: `{args.train_inventory_csv}`")
    lines.append(f"- df40 video scores: `{args.df40_video_scores_csv}`")
    lines.append("")
    lines.append("## Output summary")
    lines.append(f"- selected videos: `{len(sel)}` (real={(sel['label']==0).sum()}, fake={(sel['label']==1).sum()})")
    lines.append(f"- frame manifest rows: `{len(paths)}`")
    lines.append("")
    lines.append("## Fake method distribution (selected)")
    lines.append(fake_methods.to_markdown(index=False))
    lines.append("")
    lines.append("## Top method lift")
    lines.append(method_lift.head(20).to_markdown(index=False))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] method lift -> {method_lift_path}")
    print(f"[ok] val difficulty -> {val_out}")
    print(f"[ok] hard top -> {hard_out}")
    print(f"[ok] scored df40 -> {scored_out}")
    print(f"[ok] selected videos -> {sel_out}")
    print(f"[ok] frames manifest -> {path_out}")
    print(f"[ok] path+label manifest -> {path_pl_out}")
    print(f"[ok] report -> {report_path}")


if __name__ == "__main__":
    main()
