#!/usr/bin/env python3
"""
LB proxy recalibration from historical submissions.

Design goals:
- Use only local historical files (no hidden labels).
- Be robust to missing val_auc by using score-distribution features.
- Export a lightweight linear model JSON for shell scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_REGISTRY = Path("/work/u1657859/Jess/app/score/submission_lb_registry.csv")
DEFAULT_VAL_MAP = Path("/work/u1657859/Jess/app/score/merged_lb_table_app_app4.csv")
DEFAULT_MODEL_OUT = Path("/work/u1657859/Jess/app/score/lb_proxy_linear_v2.json")

FEATURES = [
    "val_auc",
    "min",
    "p01",
    "p05",
    "p10",
    "p50",
    "p90",
    "p95",
    "p99",
    "max",
    "spread_90_10",
    "spread_99_01",
    "center_abs",
    "frac_le_1e3",
    "frac_ge_0999",
]


def _safe_float(v: object) -> float:
    try:
        if v is None:
            return float("nan")
        s = str(v).strip()
        if s == "":
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if (not path.exists()) or (not path.is_file()):
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_scores(path: Path) -> np.ndarray:
    p = path.resolve()
    lines: List[str]
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            if "submission.txt" not in zf.namelist():
                raise ValueError(f"{p} missing submission.txt")
            lines = zf.read("submission.txt").decode("utf-8").splitlines()
    else:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    vals: List[float] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        vals.append(float(s.split(",")[0].strip()))
    if len(vals) == 0:
        raise ValueError(f"empty submission: {p}")
    arr = np.asarray(vals, dtype=np.float64)
    return np.clip(arr, 0.0, 1.0)


def _dist_features_from_scores(scores: np.ndarray) -> Dict[str, float]:
    q = np.quantile(scores, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    n = float(max(1, int(scores.size)))
    feat = {
        "min": float(scores.min()),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p10": float(q[2]),
        "p50": float(q[3]),
        "p90": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": float(scores.max()),
        "spread_90_10": float(q[4] - q[2]),
        "spread_99_01": float(q[6] - q[0]),
        "center_abs": float(abs(q[3] - 0.5)),
        "frac_le_1e3": float((scores <= 1e-3).sum() / n),
        "frac_ge_0999": float((scores >= 0.999).sum() / n),
    }
    return feat


def _kfold_indices(n: int, k: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i], axis=0)
        out.append((tr, va))
    return out


def _knn_predict_one(
    z_query: np.ndarray,
    z_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> float:
    k = int(max(1, min(int(k), int(z_train.shape[0]))))
    d = np.linalg.norm(z_train - z_query[None, :], axis=1)
    idx = np.argsort(d)[:k]
    d_k = d[idx]
    y_k = y_train[idx]
    w = 1.0 / np.clip(d_k, 1e-6, None)
    w = w / np.sum(w)
    return float(np.sum(w * y_k))


def _knn_predict_batch(
    z_query: np.ndarray,
    z_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> np.ndarray:
    out = []
    for i in range(int(z_query.shape[0])):
        out.append(_knn_predict_one(z_query[i], z_train, y_train, k=k))
    return np.asarray(out, dtype=np.float64)


def _prepare_xy(
    registry_rows: List[Dict[str, str]],
    val_map: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats: List[List[float]] = []
    ys: List[float] = []
    run_ids: List[str] = []
    for r in registry_rows:
        lb = _safe_float(r.get("public_lb"))
        if not np.isfinite(lb):
            continue
        n_values = _safe_float(r.get("n_values"))
        if not np.isfinite(n_values) or n_values <= 0:
            n_values = 100.0

        row_feat: Dict[str, float] = {
            "val_auc": val_map.get(str(r.get("run_id", "")).strip(), float("nan")),
            "min": _safe_float(r.get("min")),
            "p01": _safe_float(r.get("p01")),
            "p05": _safe_float(r.get("p05")),
            "p10": _safe_float(r.get("p10")),
            "p50": _safe_float(r.get("p50")),
            "p90": _safe_float(r.get("p90")),
            "p95": _safe_float(r.get("p95")),
            "p99": _safe_float(r.get("p99")),
            "max": _safe_float(r.get("max")),
            "frac_le_1e3": _safe_float(r.get("cnt_le_1e3")) / float(n_values),
            "frac_ge_0999": _safe_float(r.get("cnt_ge_0999")) / float(n_values),
        }
        row_feat["spread_90_10"] = row_feat["p90"] - row_feat["p10"] if np.isfinite(row_feat["p90"]) and np.isfinite(row_feat["p10"]) else float("nan")
        row_feat["spread_99_01"] = row_feat["p99"] - row_feat["p01"] if np.isfinite(row_feat["p99"]) and np.isfinite(row_feat["p01"]) else float("nan")
        row_feat["center_abs"] = abs(row_feat["p50"] - 0.5) if np.isfinite(row_feat["p50"]) else float("nan")

        vec = [float(row_feat.get(k, float("nan"))) for k in FEATURES]
        # Keep rows with at least distribution core features.
        core_ok = np.isfinite(vec[1:10]).sum() >= 7
        if not core_ok:
            continue

        feats.append(vec)
        ys.append(float(lb))
        run_ids.append(str(r.get("run_id", "")).strip())

    if len(ys) < 12:
        raise RuntimeError(f"Not enough labeled LB rows for calibration: {len(ys)}")

    x = np.asarray(feats, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    ids = np.asarray(run_ids, dtype=object)
    return x, y, ids


def _impute_and_scale_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x2 = x.copy()
    med = np.nanmedian(x2, axis=0)
    for j in range(x2.shape[1]):
        miss = ~np.isfinite(x2[:, j])
        if miss.any():
            x2[miss, j] = med[j]
    mu = x2.mean(axis=0)
    sd = x2.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (x2 - mu) / sd
    return z, med, mu, sd


def _transform_with_stats(x: np.ndarray, med: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x2 = x.copy()
    for j in range(x2.shape[1]):
        miss = ~np.isfinite(x2[:, j])
        if miss.any():
            x2[miss, j] = med[j]
    return (x2 - mu) / sd


def cmd_fit(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry).resolve()
    val_map_path = Path(args.val_map).resolve()
    out_json = Path(args.out_json).resolve()

    registry = _read_csv(registry_path)
    if not registry:
        raise RuntimeError(f"registry not found or empty: {registry_path}")

    val_rows = _read_csv(val_map_path)
    val_map: Dict[str, float] = {}
    for r in val_rows:
        rid = str(r.get("run_id", "")).strip()
        if not rid:
            continue
        v = _safe_float(r.get("val_auc"))
        if np.isfinite(v):
            val_map[rid] = float(v)

    x_raw, y, ids = _prepare_xy(registry, val_map)
    z, med, mu, sd = _impute_and_scale_fit(x_raw)

    k_list = [5, 7, 9, 11, 15, 21]
    folds = _kfold_indices(len(y), k=min(5, max(3, len(y) // 10)), seed=int(args.seed))

    best = None
    for k in k_list:
        mae_list: List[float] = []
        for tr, va in folds:
            pred = _knn_predict_batch(z[va], z[tr], y[tr], k=int(k))
            mae_list.append(float(np.mean(np.abs(pred - y[va]))))
        mae = float(np.mean(mae_list))
        if best is None or mae < best[0]:
            best = (mae, k)

    assert best is not None
    cv_mae, best_k = best

    # Leave-one-out residual as uncertainty proxy.
    loo_pred = np.zeros_like(y)
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        loo_pred[i] = _knn_predict_one(z[i], z[mask], y[mask], k=int(best_k))
    resid = y - loo_pred
    resid_std = float(np.std(resid))
    train_mae = float(np.mean(np.abs(resid)))
    val_col = x_raw[:, 0]
    alpha_val_auc = 0.0
    finite_val = np.isfinite(val_col)
    if int(np.sum(finite_val)) >= 8:
        alpha_grid = [0.0, 0.15, 0.30, 0.45, 0.60]
        best_a = None
        for a in alpha_grid:
            p = (1.0 - a) * loo_pred[finite_val] + a * val_col[finite_val]
            mae = float(np.mean(np.abs(p - y[finite_val])))
            if best_a is None or mae < best_a[0]:
                best_a = (mae, a)
        if best_a is not None:
            alpha_val_auc = float(best_a[1])

    pred_tr = loo_pred.copy()
    if alpha_val_auc > 0 and np.any(finite_val):
        pred_tr[finite_val] = (1.0 - alpha_val_auc) * pred_tr[finite_val] + alpha_val_auc * val_col[finite_val]
    train_r = float(np.corrcoef(y, pred_tr)[0, 1]) if len(y) >= 3 else float("nan")

    payload = {
        "version": 3,
        "model_type": "knn",
        "feature_names": FEATURES,
        "medians": med.tolist(),
        "means": mu.tolist(),
        "stds": sd.tolist(),
        "z_train": z.tolist(),
        "y_train": y.tolist(),
        "k": int(best_k),
        "alpha_val_auc": float(alpha_val_auc),
        "cv_mae": float(cv_mae),
        "train_mae": float(train_mae),
        "train_r": float(train_r),
        "resid_std": float(resid_std),
        "n_train": int(len(y)),
        "seed": int(args.seed),
        "registry_path": str(registry_path),
        "val_map_path": str(val_map_path),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[ok] wrote model -> {out_json}")
    print(
        f"[fit] n={len(y)} cv_mae={cv_mae:.4f} train_mae={train_mae:.4f} "
        f"train_r={train_r:.3f} k={best_k} alpha_val={alpha_val_auc:.2f}"
    )


def _build_feature_vector_for_submission(submission: Path, val_auc: Optional[float]) -> np.ndarray:
    scores = _read_scores(submission)
    feat = _dist_features_from_scores(scores)
    feat["val_auc"] = float(val_auc) if (val_auc is not None and np.isfinite(val_auc)) else float("nan")
    vec = np.asarray([float(feat.get(k, float("nan"))) for k in FEATURES], dtype=np.float64).reshape(1, -1)
    return vec


def cmd_predict(args: argparse.Namespace) -> None:
    model_path = Path(args.model_json).resolve()
    sub_path = Path(args.submission).resolve()
    payload = json.loads(model_path.read_text(encoding="utf-8"))

    med = np.asarray(payload["medians"], dtype=np.float64)
    mu = np.asarray(payload["means"], dtype=np.float64)
    sd = np.asarray(payload["stds"], dtype=np.float64)
    model_type = str(payload.get("model_type", "linear")).lower()
    resid_std = float(payload.get("resid_std", 0.01))

    val_auc = _safe_float(args.val_auc)
    if not np.isfinite(val_auc):
        val_auc = float("nan")
    x = _build_feature_vector_for_submission(sub_path, val_auc=val_auc)
    z = _transform_with_stats(x, med=med, mu=mu, sd=sd)
    if model_type == "knn":
        z_train = np.asarray(payload["z_train"], dtype=np.float64)
        y_train = np.asarray(payload["y_train"], dtype=np.float64)
        k = int(payload.get("k", 11))
        pred = float(_knn_predict_one(z[0], z_train, y_train, k=k))
        a = float(payload.get("alpha_val_auc", 0.0))
        if np.isfinite(val_auc):
            a = max(a, 0.25)
        if np.isfinite(val_auc) and a > 0:
            pred = (1.0 - a) * pred + a * float(val_auc)
    else:
        # Backward compatibility for old linear proxy json.
        w = np.asarray(payload["weights"], dtype=np.float64)
        b = float(payload["bias"])
        pred = float((z @ w + b)[0])
    pred = float(np.clip(pred, 0.0, 1.0))
    lo = pred - 1.28 * resid_std
    hi = pred + 1.28 * resid_std

    print(
        json.dumps(
            {
                "submission": str(sub_path),
                "pred_lb": pred,
                "pred_lb_lo80": lo,
                "pred_lb_hi80": hi,
                "val_auc": None if not np.isfinite(val_auc) else float(val_auc),
                "model_json": str(model_path),
            },
            ensure_ascii=True,
        )
    )

    if args.out_csv:
        out_csv = Path(args.out_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_csv.exists()
        with out_csv.open("a", encoding="utf-8", newline="") as f:
            wtr = csv.writer(f)
            if write_header:
                wtr.writerow(["submission", "val_auc", "pred_lb", "pred_lb_lo80", "pred_lb_hi80", "model_json"])
            wtr.writerow(
                [
                    str(sub_path),
                    "" if not np.isfinite(val_auc) else f"{float(val_auc):.6f}",
                    f"{pred:.6f}",
                    f"{lo:.6f}",
                    f"{hi:.6f}",
                    str(model_path),
                ]
            )
        print(f"[ok] appended prediction -> {out_csv}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Recalibrate and apply LB proxy model.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_fit = sub.add_parser("fit", help="Fit LB proxy model from historical records.")
    ap_fit.add_argument("--registry", type=str, default=str(DEFAULT_REGISTRY))
    ap_fit.add_argument("--val_map", type=str, default=str(DEFAULT_VAL_MAP))
    ap_fit.add_argument("--out_json", type=str, default=str(DEFAULT_MODEL_OUT))
    ap_fit.add_argument("--seed", type=int, default=42)
    ap_fit.set_defaults(func=cmd_fit)

    ap_pred = sub.add_parser("predict", help="Predict LB for one submission.")
    ap_pred.add_argument("--model_json", type=str, default=str(DEFAULT_MODEL_OUT))
    ap_pred.add_argument("--submission", type=str, required=True)
    ap_pred.add_argument("--val_auc", type=str, default="")
    ap_pred.add_argument("--out_csv", type=str, default="")
    ap_pred.set_defaults(func=cmd_predict)

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
