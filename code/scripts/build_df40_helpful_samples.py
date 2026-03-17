#!/usr/bin/env python3
"""
Build DF40 helpful-sample rankings with two routes:
1) Statistical: infer per-video usefulness from run-level LB residuals.
2) Non-statistical: temporal diversity + duplicate penalty + rarity priors.

Outputs are written to app/score and app/local_checkpoints/manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


IMG_EXTS = {".png", ".jpg", ".jpeg"}
DF40_MARKER = "/DF40_frames/train/DF40/"


def _z(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    std = float(x.std(ddof=0))
    if std < 1e-12:
        return pd.Series(np.zeros(len(x), dtype=np.float32), index=x.index)
    return (x - float(x.mean())) / std


def _sort_key(path: Path) -> Tuple[int, int | str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def normalize_df40_video_key(path_like: str) -> Optional[str]:
    s = str(path_like or "").replace("\\", "/").strip()
    if not s:
        return None
    i = s.find(DF40_MARKER)
    if i < 0:
        return None
    tail = s[i + len(DF40_MARKER) :].strip("/")
    if not tail:
        return None
    p = Path(tail)
    if p.suffix.lower() in IMG_EXTS:
        p = p.parent
    return str(p).replace("\\", "/").strip("/")


def parse_best_val_from_log(log_path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not log_path.exists():
        return out
    pat = re.compile(
        r"epoch\s+(\d+)\s+val acc=([0-9.]+).*?auc=([0-9.]+).*?ap=([0-9.]+)",
        re.IGNORECASE,
    )
    fpr_pat = re.compile(
        r"epoch\s+(\d+)\s+val fpr@tpr95=([0-9.]+).*?worst_quality_bin_auc=([0-9.]+).*?group_gap_auc=([0-9.]+)",
        re.IGNORECASE,
    )
    best_auc = -1.0
    best_ep = -1
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            ep = int(m.group(1))
            acc = float(m.group(2))
            auc = float(m.group(3))
            ap = float(m.group(4))
            if auc > best_auc:
                best_auc = auc
                best_ep = ep
                out["best_val_acc"] = acc
                out["best_val_auc"] = auc
                out["best_val_ap"] = ap
        m2 = fpr_pat.search(line)
        if m2 and int(m2.group(1)) == best_ep:
            out["best_fpr95"] = float(m2.group(2))
            out["best_worst_qbin_auc"] = float(m2.group(3))
            out["best_group_gap_auc"] = float(m2.group(4))
    if best_ep >= 0:
        out["best_epoch"] = float(best_ep)
    return out


def parse_extra_lb(items: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        s = str(item or "").strip()
        if not s:
            continue
        if ":" not in s:
            continue
        exp, val = s.rsplit(":", 1)
        exp = exp.strip()
        try:
            out[exp] = float(val.strip())
        except Exception:
            continue
    return out


def build_df40_universe(df40_video_scores: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], Set[str]]:
    uni = df40_video_scores.copy()
    uni = uni.drop_duplicates(subset=["video_abs_path"]).reset_index(drop=True)
    key_to_video: Dict[str, str] = {}
    for _, r in uni.iterrows():
        v = str(r["video_abs_path"])
        k1 = normalize_df40_video_key(v)
        k2 = normalize_df40_video_key(str(r.get("frame_dir", "")))
        if k1:
            key_to_video[k1] = v
        if k2:
            key_to_video[k2] = v
    return uni, key_to_video, set(uni["video_abs_path"].astype(str).tolist())


def read_train_csv_video_set(
    train_csv: str,
    key_to_video: Dict[str, str],
    full_video_set: Set[str],
    full_df40_csv: Path,
) -> Set[str]:
    p = Path(str(train_csv or ""))
    if not p.exists():
        return set()
    try:
        if p.resolve() == full_df40_csv.resolve():
            return set(full_video_set)
    except Exception:
        pass

    out: Set[str] = set()
    try:
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            if "video_abs_path" in fields:
                for row in reader:
                    k = normalize_df40_video_key(row.get("video_abs_path", ""))
                    if k and k in key_to_video:
                        out.add(key_to_video[k])
                return out
            if "frames_dir" in fields:
                for row in reader:
                    k = normalize_df40_video_key(row.get("frames_dir", ""))
                    if k and k in key_to_video:
                        out.add(key_to_video[k])
                return out
            if "path" in fields:
                for row in reader:
                    path_str = str(row.get("path", "")).strip()
                    if not path_str:
                        continue
                    k = normalize_df40_video_key(path_str)
                    if k is None:
                        k = normalize_df40_video_key(str(Path(path_str).parent))
                    if k and k in key_to_video:
                        out.add(key_to_video[k])
                return out
    except Exception:
        return set()
    return out


def fit_lb_from_val_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float]]:
    numeric = [
        "best_val_auc",
        "best_fpr95",
        "best_worst_qbin_auc",
        "best_group_gap_auc",
        "best_val_ap",
        "best_val_acc",
    ]
    categorical = [
        "route",
        "multi_expert_enable",
        "use_groupdro",
        "groupdro_use_cvar",
        "val_micro_tta",
        "data_aug",
        "frame_mode",
    ]

    X = df[numeric + categorical].copy()
    y = df["lb_obs"].astype(float).values

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    model = RidgeCV(alphas=np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0], dtype=float))
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X, y)

    pred = pipe.predict(X)
    stats = {
        "train_r2": float(r2_score(y, pred)),
        "train_mae": float(mean_absolute_error(y, pred)),
        "n_labeled": float(len(df)),
        "alpha": float(model.alpha_),
    }
    return pipe, stats


def _dhash64(gray_img: Image.Image) -> int:
    img = gray_img.resize((9, 8), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    bits = arr[:, 1:] > arr[:, :-1]
    out = 0
    for b in bits.flatten().tolist():
        out = (out << 1) | int(bool(b))
    return int(out)


def compute_temporal_stats(video_dir: Path, frames_per_video: int = 8, side: int = 48) -> Tuple[float, int]:
    if not video_dir.is_dir():
        return (0.0, 0)
    fs = sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS], key=_sort_key)
    if not fs:
        return (0.0, 0)
    if len(fs) > frames_per_video:
        idx = np.linspace(0, len(fs) - 1, num=frames_per_video, dtype=int)
        fs = [fs[int(i)] for i in idx]

    vals: List[np.ndarray] = []
    for p in fs:
        try:
            g = Image.open(p).convert("L").resize((side, side), resample=Image.BILINEAR)
            vals.append(np.asarray(g, dtype=np.float32) / 255.0)
        except Exception:
            continue
    if len(vals) <= 1:
        try:
            h = _dhash64(Image.open(fs[0]).convert("L"))
        except Exception:
            h = 0
        return (0.0, h)
    diffs = []
    for i in range(1, len(vals)):
        diffs.append(float(np.mean(np.abs(vals[i] - vals[i - 1]))))
    try:
        h = _dhash64(Image.open(fs[0]).convert("L"))
    except Exception:
        h = 0
    return (float(np.mean(diffs)), h)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DF40 helpful sample ranking (statistical + non-statistical).")
    p.add_argument(
        "--run_composition_csv",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/global_df40_run_composition_20260306.csv"),
    )
    p.add_argument(
        "--lb_runs_csv",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/lb_val_runs_with_metrics_20260304_refresh.csv"),
    )
    p.add_argument(
        "--df40_video_scores_csv",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score/df40_lb_proxy_video_scores_20260305.csv"),
    )
    p.add_argument(
        "--df40_master_csv",
        type=Path,
        default=Path("/home/u1657859/work/ntire_RDFC_2026_cvpr/Dataset/DF40_frames/image_frame_list_face.csv"),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/work/u1657859/Jess/app/score"),
    )
    p.add_argument(
        "--manifest_out_dir",
        type=Path,
        default=Path("/work/u1657859/Jess/app/local_checkpoints/manifests"),
    )
    p.add_argument("--tag", type=str, default="20260307_v1")
    p.add_argument(
        "--extra_exp_lb",
        action="append",
        default=[],
        help="Manual exp lb mapping in format ExpName:0.74. Can be repeated.",
    )
    p.add_argument(
        "--extra_exp_dir",
        action="append",
        default=[],
        help="Extra local checkpoint run dir to append (reads config.json + train_*.log).",
    )
    p.add_argument("--pseudo_weight", type=float, default=0.35)
    p.add_argument("--min_identify_weight", type=float, default=3.0)
    p.add_argument("--frames_per_video", type=int, default=8)
    p.add_argument("--temporal_side", type=int, default=48)
    p.add_argument("--temporal_cache", type=Path, default=None)
    p.add_argument("--no_temporal_scan", action="store_true")
    p.add_argument("--top_pcts", type=str, default="0.30,0.40,0.50")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_out_dir.mkdir(parents=True, exist_ok=True)

    comp = pd.read_csv(args.run_composition_csv).copy()
    lb_runs = pd.read_csv(args.lb_runs_csv).copy()
    df40 = pd.read_csv(args.df40_video_scores_csv).copy()
    full_df40 = pd.read_csv(args.df40_master_csv).copy()

    # Append any ad-hoc runs not present in composition table yet.
    extra_rows: List[dict] = []
    for run_dir_s in args.extra_exp_dir:
        run_dir = Path(str(run_dir_s))
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        logs = sorted(run_dir.glob("train_*.log"))
        best_metrics = parse_best_val_from_log(logs[-1]) if logs else {}
        extra_rows.append(
            {
                "exp_name": str(cfg.get("name", run_dir.name)),
                "repo": "manual",
                "exp_dir": str(run_dir),
                "config_path": str(cfg_path),
                "log_path": str(logs[-1]) if logs else "",
                "route": str(cfg.get("multi_expert_route", "")),
                "multi_expert_enable": bool(cfg.get("multi_expert_enable", False)),
                "multi_expert_k": int(cfg.get("multi_expert_k", 1)),
                "use_groupdro": bool(cfg.get("use_groupdro", False)),
                "groupdro_use_cvar": bool(cfg.get("groupdro_use_cvar", False)),
                "groupdro_group_key": str(cfg.get("groupdro_group_key", "")),
                "quality_metric": str(cfg.get("quality_metric", "")),
                "quality_bins": int(cfg.get("quality_bins", 3)),
                "degrade_bins": int(cfg.get("degrade_bins", 3)),
                "frame_mode": str(cfg.get("train_frame_sample_mode", "uniform")),
                "frame_min_gap_ratio": float(cfg.get("train_frame_min_gap_ratio", 0.0)),
                "frame_start_offset": int(cfg.get("train_frame_start_offset", 0)),
                "train_frame_num": int(cfg.get("train_frame_num", 8)),
                "hflip_prob": float(cfg.get("hflip_prob", 0.0)),
                "val_micro_tta": bool(cfg.get("val_micro_tta", False)),
                "data_aug": bool(cfg.get("data_aug", False)),
                "blur_prob": float(cfg.get("blur_prob", 0.0)),
                "jpg_prob": float(cfg.get("jpg_prob", 0.0)),
                "batch_size": int(cfg.get("batch_size", 24)),
                "seed": int(cfg.get("seed", 42)),
                "train_csv": str(cfg.get("train_csv", "")),
                "train_csv_exists": Path(str(cfg.get("train_csv", ""))).exists(),
                "df40_rows": np.nan,
                "linked_lb": np.nan,
                "best_epoch": best_metrics.get("best_epoch", np.nan),
                "best_val_auc": best_metrics.get("best_val_auc", np.nan),
                "best_val_acc": best_metrics.get("best_val_acc", np.nan),
                "best_val_ap": best_metrics.get("best_val_ap", np.nan),
                "best_fpr95": best_metrics.get("best_fpr95", np.nan),
                "best_worst_qbin_auc": best_metrics.get("best_worst_qbin_auc", np.nan),
                "best_group_gap_auc": best_metrics.get("best_group_gap_auc", np.nan),
            }
        )
    if extra_rows:
        comp = pd.concat([comp, pd.DataFrame(extra_rows)], ignore_index=True, sort=False)

    # Reliable LB mapping at experiment level.
    lb_exp = (
        lb_runs[lb_runs["mapped_exp"].notna()]
        .groupby("mapped_exp", as_index=False)
        .agg(lb_obs=("public_lb", "median"), lb_n=("public_lb", "count"), lb_std=("public_lb", "std"))
    )
    runs = comp.merge(lb_exp, left_on="exp_name", right_on="mapped_exp", how="left")
    runs.drop(columns=["mapped_exp"], inplace=True, errors="ignore")

    manual_lb = parse_extra_lb(args.extra_exp_lb)
    if manual_lb:
        for exp, val in manual_lb.items():
            m = runs["exp_name"].astype(str) == str(exp)
            runs.loc[m, "lb_obs"] = float(val)
            runs.loc[m, "lb_n"] = 1.0
            runs.loc[m, "lb_std"] = 0.0

    # Build LB surrogate from val-side diagnostics (not training-set composition).
    labeled = runs[runs["lb_obs"].notna()].copy()
    if len(labeled) < 6:
        raise RuntimeError(f"Not enough labeled runs for LB surrogate: {len(labeled)}")
    lb_model, lb_model_stats = fit_lb_from_val_model(labeled)

    feat_numeric = [
        "best_val_auc",
        "best_fpr95",
        "best_worst_qbin_auc",
        "best_group_gap_auc",
        "best_val_ap",
        "best_val_acc",
    ]
    feat_cat = [
        "route",
        "multi_expert_enable",
        "use_groupdro",
        "groupdro_use_cvar",
        "val_micro_tta",
        "data_aug",
        "frame_mode",
    ]
    runs["lb_pseudo"] = lb_model.predict(runs[feat_numeric + feat_cat].copy())
    runs["lb_target"] = runs["lb_obs"].where(runs["lb_obs"].notna(), runs["lb_pseudo"])
    runs["lb_source"] = np.where(runs["lb_obs"].notna(), "observed", "pseudo")

    pseudo_w = float(args.pseudo_weight)
    has_val = runs["best_val_auc"].notna().astype(float)
    runs["run_weight"] = np.where(
        runs["lb_obs"].notna(),
        1.0,
        pseudo_w * (0.5 + 0.5 * has_val),
    )
    runs["lb_pred_from_val"] = runs["lb_pseudo"]
    runs["lb_residual"] = runs["lb_target"] - runs["lb_pred_from_val"]

    # DF40 universe and manifest parsing.
    uni, key_to_video, full_video_set = build_df40_universe(df40)
    full_videos = sorted(list(full_video_set))
    vid_to_idx = {v: i for i, v in enumerate(full_videos)}

    parse_cache: Dict[str, Set[str]] = {}

    def _video_set_for_csv(csv_path: str, df40_rows_hint: float) -> Set[str]:
        key = str(csv_path or "")
        if key in parse_cache:
            return parse_cache[key]
        vids = read_train_csv_video_set(key, key_to_video, full_video_set, args.df40_master_csv)
        if (not vids) and (pd.notna(df40_rows_hint)) and float(df40_rows_hint) >= len(full_videos) * 0.95:
            vids = set(full_video_set)
        parse_cache[key] = vids
        return vids

    runs["parsed_df40_videos"] = 0
    run_video_sets: List[Set[str]] = []
    for _, r in runs.iterrows():
        vids = _video_set_for_csv(str(r.get("train_csv", "")), float(r.get("df40_rows", np.nan)))
        run_video_sets.append(vids)
    runs["parsed_df40_videos"] = [len(s) for s in run_video_sets]

    use_mask = (runs["lb_target"].notna()) & (runs["parsed_df40_videos"] > 0)
    use_runs = runs[use_mask].reset_index(drop=True)
    use_sets = [run_video_sets[i] for i in runs.index[use_mask].tolist()]
    if len(use_runs) < 8:
        raise RuntimeError(f"Too few usable runs for attribution: {len(use_runs)}")

    # Inclusion matrix: run x video.
    R = len(use_runs)
    V = len(full_videos)
    X = np.zeros((R, V), dtype=np.uint8)
    for i, vids in enumerate(use_sets):
        idxs = [vid_to_idx[v] for v in vids if v in vid_to_idx]
        if idxs:
            X[i, idxs] = 1

    w = use_runs["run_weight"].astype(float).to_numpy()
    y = use_runs["lb_residual"].astype(float).to_numpy()
    wy = w * y
    sum_w = float(w.sum())
    sum_wy = float(wy.sum())

    sum_w_in = X.T.dot(w)
    sum_wy_in = X.T.dot(wy)
    sum_w_out = sum_w - sum_w_in
    sum_wy_out = sum_wy - sum_wy_in

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_in = np.divide(sum_wy_in, sum_w_in, out=np.zeros_like(sum_w_in), where=sum_w_in > 1e-12)
        mean_out = np.divide(sum_wy_out, sum_w_out, out=np.zeros_like(sum_w_out), where=sum_w_out > 1e-12)
    delta = mean_in - mean_out
    identify = np.minimum(sum_w_in, sum_w_out)
    shrink = (sum_w_in / (sum_w_in + 3.0)) * (sum_w_out / (sum_w_out + 3.0))
    stat_score = delta * shrink
    stat_valid = (sum_w_in >= float(args.min_identify_weight)) & (sum_w_out >= float(args.min_identify_weight))

    rank = uni.copy()
    rank = rank.set_index("video_abs_path").loc[full_videos].reset_index()
    rank["stat_delta"] = delta
    rank["stat_identify_weight"] = identify
    rank["stat_score"] = stat_score
    rank["stat_valid"] = stat_valid.astype(int)

    # Non-statistical route.
    fake_freq = rank[rank["label"] == 1]["method"].value_counts(normalize=True).to_dict()
    rank["method_rarity"] = rank.apply(
        lambda r: -math.log(max(1e-8, float(fake_freq.get(str(r["method"]), 1e-8)))) if int(r["label"]) == 1 else 0.0,
        axis=1,
    )

    if args.temporal_cache is None:
        args.temporal_cache = args.out_dir / f"df40_temporal_cache_{args.tag}.csv"
    if args.temporal_cache.exists():
        tdf = pd.read_csv(args.temporal_cache)
        tmap_div = dict(zip(tdf["video_abs_path"].astype(str), tdf["temporal_div"].astype(float)))
        tmap_hash = dict(zip(tdf["video_abs_path"].astype(str), tdf["dhash64"].astype("Int64").fillna(0).astype(int)))
    else:
        tmap_div, tmap_hash = {}, {}

    if not args.no_temporal_scan:
        miss = [v for v in rank["video_abs_path"].astype(str).tolist() if v not in tmap_div]
        recs = []
        for i, v in enumerate(miss, start=1):
            td, dh = compute_temporal_stats(
                Path(v),
                frames_per_video=int(args.frames_per_video),
                side=int(args.temporal_side),
            )
            tmap_div[v] = float(td)
            tmap_hash[v] = int(dh)
            if i % 1000 == 0:
                pass
        all_rows = [{"video_abs_path": v, "temporal_div": float(tmap_div.get(v, 0.0)), "dhash64": int(tmap_hash.get(v, 0))} for v in full_videos]
        pd.DataFrame(all_rows).to_csv(args.temporal_cache, index=False)

    rank["temporal_div"] = rank["video_abs_path"].astype(str).map(lambda v: float(tmap_div.get(v, 0.0)))
    rank["dhash64"] = rank["video_abs_path"].astype(str).map(lambda v: int(tmap_hash.get(v, 0)))
    hfreq = rank["dhash64"].value_counts().to_dict()
    rank["hash_freq"] = rank["dhash64"].map(lambda h: int(hfreq.get(int(h), 1)))
    rank["dup_penalty"] = np.log1p(rank["hash_freq"].astype(float))

    # Keep non-stat route numerically stable even if any source column has missing values.
    qstd = rank["q_log_std"].astype(float).fillna(float(rank["q_log_std"].astype(float).median()))
    proxy = rank["proxy_top20_mean"].astype(float).fillna(float(rank["proxy_top20_mean"].astype(float).median()))
    rarity = rank["method_rarity"].astype(float).fillna(0.0)
    tdiv = rank["temporal_div"].astype(float).fillna(float(rank["temporal_div"].astype(float).median()))
    dpen = rank["dup_penalty"].astype(float).fillna(float(rank["dup_penalty"].astype(float).median()))

    rank["nonstat_score"] = (
        0.55 * _z(tdiv)
        + 0.20 * _z(qstd)
        + 0.15 * _z(rarity)
        + 0.10 * _z(proxy)
        - 0.20 * _z(dpen)
    )

    stat_fill = rank["stat_score"].copy()
    stat_fill[rank["stat_valid"] == 0] = float(np.nanmedian(stat_fill.to_numpy()))
    rank["stat_z"] = _z(stat_fill.fillna(float(np.nanmedian(stat_fill.to_numpy()))))
    rank["nonstat_z"] = _z(rank["nonstat_score"])

    p80 = float(np.nanquantile(rank["stat_identify_weight"].to_numpy(), 0.80))
    denom = max(1e-6, p80 - float(args.min_identify_weight))
    id_w = (rank["stat_identify_weight"] - float(args.min_identify_weight)) / denom
    id_w = np.clip(id_w, 0.0, 1.0)
    rank["identify_w"] = id_w
    rank["final_help_score"] = (
        rank["identify_w"] * (0.72 * rank["stat_z"] + 0.28 * rank["nonstat_z"])
        + (1.0 - rank["identify_w"]) * (0.15 * rank["stat_z"] + 0.85 * rank["nonstat_z"])
    )
    rank["final_help_score"] = rank["final_help_score"].astype(float).fillna(float(rank["final_help_score"].median()))

    rank = rank.sort_values("final_help_score", ascending=False).reset_index(drop=True)

    rank_out = args.out_dir / f"df40_video_helpfulness_{args.tag}.csv"
    rank.to_csv(rank_out, index=False)

    run_out = args.out_dir / f"df40_run_residuals_{args.tag}.csv"
    use_runs_out = use_runs.copy()
    use_runs_out["parsed_df40_ratio"] = use_runs_out["parsed_df40_videos"] / max(1, len(full_videos))
    use_runs_out.to_csv(run_out, index=False)

    # Build top-pct manifests, preserving original DF40 class ratio.
    base_cols = ["split", "label", "label_name", "dataset", "subject_id", "video_abs_path", "frames_dir"]
    base_map = full_df40[base_cols].drop_duplicates(subset=["video_abs_path"]).copy()
    base_map["video_abs_path"] = base_map["video_abs_path"].astype(str)

    total_real = int((base_map["label"] == 0).sum())
    total_fake = int((base_map["label"] == 1).sum())
    pct_vals = []
    for x in str(args.top_pcts).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            pct_vals.append(float(x))
        except Exception:
            continue
    pct_vals = [p for p in pct_vals if p > 0]
    pct_vals = sorted(set(pct_vals))

    manifest_rows = []
    for pct in pct_vals:
        n_real = max(1, int(round(total_real * pct)))
        n_fake = max(1, int(round(total_fake * pct)))
        rsel = rank[rank["label"] == 0].head(n_real)
        fsel = rank[rank["label"] == 1].head(n_fake)
        sel = pd.concat([rsel, fsel], axis=0, ignore_index=True)
        out = base_map.merge(sel[["video_abs_path", "final_help_score", "stat_score", "nonstat_score"]], on="video_abs_path", how="inner")
        out = out.sort_values(["label", "final_help_score"], ascending=[True, False]).reset_index(drop=True)
        tag_pct = f"p{int(round(pct * 100)):02d}"
        out_csv = args.out_dir / f"df40_helpful_{tag_pct}_videos_{args.tag}.csv"
        out.to_csv(out_csv, index=False)
        out_csv2 = args.manifest_out_dir / f"df40_helpful_{tag_pct}_videos_{args.tag}.csv"
        out.to_csv(out_csv2, index=False)
        manifest_rows.append(
            {
                "pct": pct,
                "n_videos": len(out),
                "n_real": int((out["label"] == 0).sum()),
                "n_fake": int((out["label"] == 1).sum()),
                "out_csv": str(out_csv),
                "manifest_csv": str(out_csv2),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_summary = args.out_dir / f"df40_helpful_manifest_summary_{args.tag}.csv"
    manifest_df.to_csv(manifest_summary, index=False)

    # Short report.
    report = args.out_dir / f"df40_helpful_report_{args.tag}.md"
    lines: List[str] = []
    lines.append("# DF40 Helpful Sample Mining Report")
    lines.append("")
    lines.append("## Run-level model")
    lines.append(f"- labeled runs: `{int(lb_model_stats['n_labeled'])}`")
    lines.append(f"- lb surrogate alpha: `{lb_model_stats['alpha']:.4f}`")
    lines.append(f"- lb surrogate train_r2: `{lb_model_stats['train_r2']:.4f}`")
    lines.append(f"- lb surrogate train_mae: `{lb_model_stats['train_mae']:.4f}`")
    lines.append(f"- usable runs for attribution: `{len(use_runs)}`")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- ranking csv: `{rank_out}`")
    lines.append(f"- run residual csv: `{run_out}`")
    lines.append(f"- manifest summary: `{manifest_summary}`")
    lines.append("")
    lines.append("## Top manifests")
    if not manifest_df.empty:
        lines.append(manifest_df.to_markdown(index=False))
    else:
        lines.append("- none")
    lines.append("")
    top_fake = (
        rank[rank["label"] == 1]
        .head(500)["method"]
        .value_counts(normalize=True)
        .reset_index()
    )
    top_fake.columns = ["method", "ratio_top500_fake"]
    lines.append("## Top500 fake method mix")
    lines.append(top_fake.head(20).to_markdown(index=False))
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] ranking -> {rank_out}")
    print(f"[ok] runs -> {run_out}")
    print(f"[ok] manifests -> {manifest_summary}")
    print(f"[ok] report -> {report}")


if __name__ == "__main__":
    main()
