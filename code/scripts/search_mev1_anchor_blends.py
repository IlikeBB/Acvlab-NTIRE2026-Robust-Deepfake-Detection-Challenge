#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np


def _read_submission(path: Path) -> np.ndarray:
    path = path.resolve()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            raw = zf.read("submission.txt").decode("utf-8").splitlines()
    else:
        raw = path.read_text(encoding="utf-8").splitlines()
    vals = []
    for i, line in enumerate(raw, start=1):
        s = line.strip()
        if not s:
            raise ValueError(f"empty line at {i} in {path}")
        v = float(s.split(",")[0].strip())
        vals.append(min(1.0 - 1e-6, max(1e-6, v)))
    if not vals:
        raise ValueError(f"no values in {path}")
    return np.asarray(vals, dtype=np.float64)


def _write_submission_txt(path: Path, probs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in probs:
            vv = min(1.0, max(0.0, float(v)))
            f.write(f"{vv:.6f}\n")


def _write_submission_zip(txt_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname="submission.txt")


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _average_tie_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _average_tie_ranks(np.asarray(a, dtype=np.float64))
    rb = _average_tie_ranks(np.asarray(b, dtype=np.float64))
    sa = ra - ra.mean()
    sb = rb - rb.mean()
    den = float(np.sqrt(np.dot(sa, sa) * np.dot(sb, sb)))
    if den <= 0.0:
        return 1.0
    return float(np.dot(sa, sb) / den)


class ProxyModel:
    def __init__(self, obj: dict):
        self.feature_names = list(obj["feature_names"])
        self.medians = np.asarray(obj["medians"], dtype=np.float64)
        self.means = np.asarray(obj["means"], dtype=np.float64)
        self.stds = np.asarray(obj["stds"], dtype=np.float64)
        self.z_train = np.asarray(obj["z_train"], dtype=np.float64)
        self.y_train = np.asarray(obj["y_train"], dtype=np.float64)
        self.k = int(obj["k"])

    @classmethod
    def load(cls, path: Path) -> "ProxyModel":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def predict(self, probs: np.ndarray) -> tuple[float, dict]:
        a = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
        q = np.quantile(a, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
        feat = {
            "val_auc": np.nan,
            "min": float(a.min()),
            "p01": float(q[0]),
            "p05": float(q[1]),
            "p10": float(q[2]),
            "p50": float(q[3]),
            "p90": float(q[4]),
            "p95": float(q[5]),
            "p99": float(q[6]),
            "max": float(a.max()),
        }
        feat["spread_90_10"] = feat["p90"] - feat["p10"]
        feat["spread_99_01"] = feat["p99"] - feat["p01"]
        feat["center_abs"] = abs(feat["p50"] - 0.5)
        feat["frac_le_1e3"] = float((a <= 1e-3).mean())
        feat["frac_ge_0999"] = float((a >= 0.999).mean())

        x = np.asarray([feat.get(name, np.nan) for name in self.feature_names], dtype=np.float64)
        miss = ~np.isfinite(x)
        x[miss] = self.medians[miss]
        zq = (x - self.means) / self.stds
        kk = max(1, min(self.k, int(self.y_train.shape[0])))
        dist = np.linalg.norm(self.z_train - zq[None, :], axis=1)
        idx = np.argsort(dist)[:kk]
        w = 1.0 / np.clip(dist[idx], 1e-6, None)
        w = w / w.sum()
        pred = float((w * self.y_train[idx]).sum())
        return pred, feat


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _prob_mix(anchor: np.ndarray, alt: np.ndarray, w: float) -> np.ndarray:
    return np.clip((1.0 - float(w)) * anchor + float(w) * alt, 1e-6, 1.0 - 1e-6)


def _logit_mix(anchor: np.ndarray, alt: np.ndarray, w: float) -> np.ndarray:
    return _sigmoid((1.0 - float(w)) * _logit(anchor) + float(w) * _logit(alt))


def _prob_mix2(anchor: np.ndarray, a: np.ndarray, b: np.ndarray, wa: float, wb: float) -> np.ndarray:
    base = 1.0 - float(wa) - float(wb)
    return np.clip(base * anchor + float(wa) * a + float(wb) * b, 1e-6, 1.0 - 1e-6)


def _logit_mix2(anchor: np.ndarray, a: np.ndarray, b: np.ndarray, wa: float, wb: float) -> np.ndarray:
    base = 1.0 - float(wa) - float(wb)
    return _sigmoid(base * _logit(anchor) + float(wa) * _logit(a) + float(wb) * _logit(b))


def _gate_indices_lowconf(anchor: np.ndarray, frac: float) -> np.ndarray:
    n = max(1, int(round(float(frac) * anchor.size)))
    return np.argsort(np.abs(anchor - 0.5))[:n]


def _gate_indices_delta(anchor: np.ndarray, alt: np.ndarray, frac: float) -> np.ndarray:
    n = max(1, int(round(float(frac) * anchor.size)))
    return np.argsort(-np.abs(anchor - alt))[:n]


def _gated(anchor: np.ndarray, alt: np.ndarray, idx: np.ndarray, g: float) -> np.ndarray:
    out = anchor.copy()
    gg = float(g)
    out[idx] = np.clip((1.0 - gg) * anchor[idx] + gg * alt[idx], 1e-6, 1.0 - 1e-6)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", type=str, required=True)
    p.add_argument("--alts", type=str, nargs="+", required=True)
    p.add_argument(
        "--model_json",
        type=str,
        default="/work/u1657859/Jess/app/score/lb_proxy_linear_v2.json",
    )
    p.add_argument("--tag", type=str, default="codex_anchor_search")
    p.add_argument(
        "--out_dir",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/submissions",
    )
    p.add_argument("--weights", type=str, default="0.03,0.05,0.07,0.10,0.12,0.15,0.18,0.20")
    p.add_argument("--pair_weights", type=str, default="0.03,0.05,0.07,0.10,0.12")
    p.add_argument("--gate_fracs", type=str, default="0.05,0.10,0.15,0.20")
    p.add_argument("--gate_strengths", type=str, default="0.5,0.7,1.0")
    p.add_argument("--max_pair_total", type=float, default=0.20)
    p.add_argument("--top_k", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    proxy = ProxyModel.load(Path(args.model_json).resolve())
    anchor_path = Path(args.anchor).resolve()
    anchor = _read_submission(anchor_path)

    alt_items = []
    for raw in args.alts:
        p = Path(raw).resolve()
        a = _read_submission(p)
        if a.shape != anchor.shape:
            raise ValueError(f"length mismatch: {p}")
        alt_items.append((p, a))

    weights = _parse_float_list(args.weights)
    pair_weights = _parse_float_list(args.pair_weights)
    gate_fracs = _parse_float_list(args.gate_fracs)
    gate_strengths = _parse_float_list(args.gate_strengths)

    seen = set()
    rows = []

    def emit(recipe: str, probs: np.ndarray) -> None:
        rounded = np.round(np.asarray(probs, dtype=np.float64), 6)
        digest = hashlib.sha1(rounded.tobytes()).hexdigest()
        if digest in seen:
            return
        seen.add(digest)
        pred, feat = proxy.predict(rounded)
        sp = _spearman(anchor, rounded)
        rows.append(
            {
                "pred_lb": pred,
                "spearman_to_anchor": sp,
                "min": feat["min"],
                "p50": feat["p50"],
                "p90": feat["p90"],
                "p95": feat["p95"],
                "p99": feat["p99"],
                "max": feat["max"],
                "mean_abs_delta": float(np.mean(np.abs(rounded - anchor))),
                "recipe": recipe,
                "probs": rounded,
            }
        )

    emit(f"anchor:{anchor_path.name}", anchor)

    for alt_path, alt in alt_items:
        emit(f"alt:{alt_path.name}", alt)
        for w in weights:
            emit(f"lin_prob:{1.0-w:.2f}*{anchor_path.name}+{w:.2f}*{alt_path.name}", _prob_mix(anchor, alt, w))
            emit(f"lin_logit:{1.0-w:.2f}*{anchor_path.name}+{w:.2f}*{alt_path.name}", _logit_mix(anchor, alt, w))
        for frac in gate_fracs:
            idx_lowconf = _gate_indices_lowconf(anchor, frac)
            idx_delta = _gate_indices_delta(anchor, alt, frac)
            for g in gate_strengths:
                emit(
                    f"gate_lowconf:q{frac:.2f}:g{g:.2f}:{alt_path.name}",
                    _gated(anchor, alt, idx_lowconf, g),
                )
                emit(
                    f"gate_delta:q{frac:.2f}:g{g:.2f}:{alt_path.name}",
                    _gated(anchor, alt, idx_delta, g),
                )

    for (p1, a1), (p2, a2) in combinations(alt_items, 2):
        for w1 in pair_weights:
            for w2 in pair_weights:
                if (w1 + w2) > float(args.max_pair_total):
                    continue
                recipe_prob = (
                    f"pair_prob:{1.0-w1-w2:.2f}*{anchor_path.name}"
                    f"+{w1:.2f}*{p1.name}+{w2:.2f}*{p2.name}"
                )
                recipe_logit = (
                    f"pair_logit:{1.0-w1-w2:.2f}*{anchor_path.name}"
                    f"+{w1:.2f}*{p1.name}+{w2:.2f}*{p2.name}"
                )
                emit(recipe_prob, _prob_mix2(anchor, a1, a2, w1, w2))
                emit(recipe_logit, _logit_mix2(anchor, a1, a2, w1, w2))

    rows.sort(key=lambda r: (r["pred_lb"], r["spearman_to_anchor"]), reverse=True)
    keep = rows[: max(1, int(args.top_k))]

    summary_path = out_dir / f"{args.tag}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "pred_lb",
                "spearman_to_anchor",
                "mean_abs_delta",
                "min",
                "p50",
                "p90",
                "p95",
                "p99",
                "max",
                "recipe",
                "txt_path",
                "zip_path",
            ]
        )
        for i, row in enumerate(keep, start=1):
            stem = f"{args.tag}_r{i:02d}"
            txt_path = out_dir / f"{stem}.txt"
            zip_path = out_dir / f"{stem}.zip"
            _write_submission_txt(txt_path, row["probs"])
            _write_submission_zip(txt_path, zip_path)
            w.writerow(
                [
                    i,
                    f"{row['pred_lb']:.6f}",
                    f"{row['spearman_to_anchor']:.6f}",
                    f"{row['mean_abs_delta']:.6f}",
                    f"{row['min']:.6f}",
                    f"{row['p50']:.6f}",
                    f"{row['p90']:.6f}",
                    f"{row['p95']:.6f}",
                    f"{row['p99']:.6f}",
                    f"{row['max']:.6f}",
                    row["recipe"],
                    str(txt_path),
                    str(zip_path),
                ]
            )

    print(f"[done] wrote {len(keep)} candidates -> {summary_path}")
    for i, row in enumerate(keep, start=1):
        print(
            f"{i:02d}\tpred={row['pred_lb']:.6f}\tsp={row['spearman_to_anchor']:.6f}"
            f"\tdelta={row['mean_abs_delta']:.6f}\t{row['recipe']}"
        )


if __name__ == "__main__":
    main()
