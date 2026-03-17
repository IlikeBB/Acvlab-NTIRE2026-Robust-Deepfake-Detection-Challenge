#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile

import numpy as np


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_submission_values(path: Path) -> np.ndarray:
    path = path.resolve()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if "submission.txt" not in names:
                raise ValueError(f"{path} does not contain submission.txt")
            raw = zf.read("submission.txt").decode("utf-8").splitlines()
    else:
        raw = path.read_text(encoding="utf-8").splitlines()

    vals: List[float] = []
    for i, line in enumerate(raw, start=1):
        s = line.strip()
        if not s:
            raise ValueError(f"Empty line at {i} in {path}")
        try:
            v = float(s.split(",")[0].strip())
        except Exception as e:
            raise ValueError(f"Invalid float at line {i} in {path}: {s!r}") from e
        if (v < 0.0) or (v > 1.0):
            raise ValueError(f"Out-of-range value at line {i} in {path}: {v}")
        vals.append(v)
    if not vals:
        raise ValueError(f"No values in {path}")
    return np.asarray(vals, dtype=np.float64)


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ensure_dirs(root: Path) -> Dict[str, Path]:
    db = root.resolve()
    vec_dir = db / "vectors"
    sub_dir = db / "submissions"
    db.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)
    return {"db": db, "vec": vec_dir, "sub": sub_dir}


def _meta_path(db: Path) -> Path:
    return db / "runs.jsonl"


def _load_meta(db: Path) -> List[Dict]:
    p = _meta_path(db)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def _save_meta(db: Path, rows: List[Dict]) -> None:
    p = _meta_path(db)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def _find_row(rows: List[Dict], run_id: str) -> Optional[Dict]:
    for r in rows:
        if str(r.get("run_id")) == str(run_id):
            return r
    return None


def _next_run_id(rows: List[Dict], stem: str) -> str:
    base = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if _find_row(rows, base) is None:
        return base
    i = 1
    while True:
        cand = f"{base}_{i}"
        if _find_row(rows, cand) is None:
            return cand
        i += 1


def cmd_add(args):
    paths = _ensure_dirs(Path(args.db_dir))
    db = paths["db"]
    vec_dir = paths["vec"]
    sub_dir = paths["sub"]
    rows = _load_meta(db)

    sub_path = Path(args.submission).resolve()
    vals = _read_submission_values(sub_path)

    if rows:
        n0 = int(rows[0]["n_values"])
        if int(vals.size) != n0:
            raise ValueError(f"Submission length mismatch: got {vals.size}, expected {n0}")

    stem = sub_path.stem
    run_id = args.run_id or _next_run_id(rows, stem)
    if _find_row(rows, run_id) is not None:
        raise ValueError(f"run_id exists: {run_id}")

    vec_path = vec_dir / f"{run_id}.npy"
    np.save(vec_path, vals)

    archived_name = f"{run_id}{sub_path.suffix.lower()}"
    archived_path = sub_dir / archived_name
    shutil.copy2(sub_path, archived_path)

    row = {
        "run_id": run_id,
        "created_at": _now_iso(),
        "submission_path": str(archived_path),
        "source_path": str(sub_path),
        "sha1": _sha1_file(sub_path),
        "n_values": int(vals.size),
        "public_score": float(args.public_score) if args.public_score is not None else None,
        "private_score": float(args.private_score) if args.private_score is not None else None,
        "note": str(args.note or ""),
    }
    rows.append(row)
    _save_meta(db, rows)
    print(f"[ok] added run_id={run_id} n={vals.size} db={db}")


def cmd_set_score(args):
    db = Path(args.db_dir).resolve()
    rows = _load_meta(db)
    row = _find_row(rows, args.run_id)
    if row is None:
        raise ValueError(f"run_id not found: {args.run_id}")
    if args.public_score is not None:
        row["public_score"] = float(args.public_score)
    if args.private_score is not None:
        row["private_score"] = float(args.private_score)
    if args.note is not None:
        old = str(row.get("note") or "")
        add = str(args.note)
        row["note"] = (old + " | " + add).strip(" |") if old else add
    row["updated_at"] = _now_iso()
    _save_meta(db, rows)
    print(f"[ok] updated run_id={args.run_id}")


def _metric_value(row: Dict, metric: str) -> Optional[float]:
    pub = row.get("public_score")
    prv = row.get("private_score")
    if metric == "public":
        return None if pub is None else float(pub)
    if metric == "private":
        return None if prv is None else float(prv)
    if prv is not None:
        return float(prv)
    if pub is not None:
        return float(pub)
    return None


def _id_list_from_source(id_source: Optional[str], n: int) -> List[str]:
    if not id_source:
        return [f"{i:04d}" for i in range(n)]
    p = Path(id_source)
    if p.is_dir():
        names = sorted([x.name for x in p.glob("*") if x.is_file()])
        if len(names) == n:
            return names
    if p.is_file():
        raw = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(raw) == n:
            return raw
    return [f"{i:04d}" for i in range(n)]


def _weighted_stats(arr: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w = w / w.sum()
    mu = (w[:, None] * arr).sum(axis=0)
    var = (w[:, None] * (arr - mu[None, :]) ** 2).sum(axis=0)
    return mu, np.sqrt(np.maximum(0.0, var))


def cmd_report(args):
    db = Path(args.db_dir).resolve()
    rows = _load_meta(db)
    if not rows:
        raise ValueError(f"No runs in {db}")

    items = []
    for r in rows:
        score = _metric_value(r, args.metric)
        vec_path = db / "vectors" / f"{r['run_id']}.npy"
        if not vec_path.exists():
            continue
        vec = np.load(vec_path).astype(np.float64)
        items.append((r, score, vec))
    if not items:
        raise ValueError("No vectors found")

    # Use scored runs first; fallback to all runs with equal weights.
    scored = [(r, s, v) for (r, s, v) in items if s is not None and np.isfinite(s)]
    if scored:
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        use = scored[: int(args.topk)] if int(args.topk) > 0 else scored
        scores = np.asarray([float(x[1]) for x in use], dtype=np.float64)
        s0 = float(scores.min())
        weights = scores - s0 + 1e-6
    else:
        use = items[: int(args.topk)] if int(args.topk) > 0 else items
        weights = np.ones(len(use), dtype=np.float64)

    if not use:
        raise ValueError("No runs selected")

    mat = np.stack([x[2] for x in use], axis=0)
    n_runs, n_files = mat.shape
    mu, sd = _weighted_stats(mat, weights)
    ids = _id_list_from_source(args.id_source, n_files)
    run_scale = float(min(1.0, float(n_runs) / 6.0))

    out_rows = []
    for i in range(n_files):
        tendency = "fake" if mu[i] >= 0.5 else "real"
        margin = abs(float(mu[i]) - 0.5) * 2.0
        stability = float(max(0.0, 1.0 - (float(sd[i]) / 0.5)))
        confidence = float(max(0.0, min(1.0, margin * stability * run_scale)))
        out_rows.append(
            {
                "index": i,
                "id": ids[i] if i < len(ids) else f"{i:04d}",
                "mean_prob_fake": float(mu[i]),
                "std_prob": float(sd[i]),
                "tendency": tendency,
                "confidence": confidence,
            }
        )

    out_rows.sort(key=lambda r: r["confidence"], reverse=True)
    out_csv = Path(args.out_csv) if args.out_csv else (db / "file_tendency.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["index", "id", "mean_prob_fake", "std_prob", "tendency", "confidence"],
        )
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"[ok] report saved: {out_csv} (runs={n_runs}, files={n_files})")
    if n_runs < 4:
        print(f"[warn] only {n_runs} runs; confidence is down-scaled and should be treated as preliminary")
    print("[top confidence]")
    for r in out_rows[: min(10, len(out_rows))]:
        print(
            f"  {r['id']}: tendency={r['tendency']} "
            f"mean={r['mean_prob_fake']:.4f} std={r['std_prob']:.4f} conf={r['confidence']:.3f}"
        )


def cmd_list(args):
    db = Path(args.db_dir).resolve()
    rows = _load_meta(db)
    if not rows:
        print("[info] empty")
        return
    rows.sort(key=lambda r: str(r.get("created_at", "")))
    print("run_id\tpublic\tprivate\tn\tnote")
    for r in rows:
        print(
            f"{r.get('run_id','')}\t{r.get('public_score')}\t{r.get('private_score')}\t"
            f"{r.get('n_values')}\t{r.get('note','')}"
        )


def cmd_diff(args):
    db = Path(args.db_dir).resolve()
    rows = _load_meta(db)
    row_a = _find_row(rows, args.run_a)
    row_b = _find_row(rows, args.run_b)
    if row_a is None or row_b is None:
        raise ValueError("run_a or run_b not found")

    va = np.load(db / "vectors" / f"{args.run_a}.npy").astype(np.float64)
    vb = np.load(db / "vectors" / f"{args.run_b}.npy").astype(np.float64)
    if va.shape != vb.shape:
        raise ValueError("vector shape mismatch")

    ids = _id_list_from_source(args.id_source, va.size)
    delta = vb - va
    rows_out = []
    for i in range(va.size):
        rows_out.append(
            {
                "index": i,
                "id": ids[i] if i < len(ids) else f"{i:04d}",
                "score_a": float(va[i]),
                "score_b": float(vb[i]),
                "delta_b_minus_a": float(delta[i]),
                "abs_delta": float(abs(delta[i])),
            }
        )
    rows_out.sort(key=lambda r: r["abs_delta"], reverse=True)
    topk = int(args.topk or 20)

    out_csv = Path(args.out_csv) if args.out_csv else (db / f"diff_{args.run_a}_vs_{args.run_b}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["index", "id", "score_a", "score_b", "delta_b_minus_a", "abs_delta"],
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"[ok] diff saved: {out_csv}")
    print(f"[top {topk}]")
    for r in rows_out[:topk]:
        trend = "more_fake_in_b" if r["delta_b_minus_a"] > 0 else "more_real_in_b"
        print(
            f"  {r['id']}: a={r['score_a']:.4f} b={r['score_b']:.4f} "
            f"delta={r['delta_b_minus_a']:+.4f} ({trend})"
        )


def cmd_impact(args):
    db = Path(args.db_dir).resolve()
    rows = _load_meta(db)
    if not rows:
        raise ValueError(f"No runs in {db}")

    scored_items = []
    for r in rows:
        score = _metric_value(r, args.metric)
        if score is None or (not np.isfinite(float(score))):
            continue
        vec_path = db / "vectors" / f"{r['run_id']}.npy"
        if not vec_path.exists():
            continue
        vec = np.load(vec_path).astype(np.float64)
        scored_items.append((r, float(score), vec))

    if len(scored_items) < 3:
        raise ValueError("Need at least 3 scored runs for impact analysis")

    scored_items.sort(key=lambda x: x[1], reverse=True)
    use = scored_items[: int(args.topk)] if int(args.topk) > 0 else scored_items
    if len(use) < 3:
        raise ValueError("Selected runs < 3; increase --topk or add more scored runs")

    mat = np.stack([x[2] for x in use], axis=0)
    y = np.asarray([x[1] for x in use], dtype=np.float64)
    n_runs, n_files = mat.shape
    ids = _id_list_from_source(args.id_source, n_files)

    yc = y - y.mean()
    vary = float(np.sum(yc * yc))
    if vary <= 1e-12:
        raise ValueError("Score variance is ~0, cannot estimate impact")

    rows_out = []
    scale_n = float(np.sqrt(float(n_runs) / float(n_runs + 3)))
    for i in range(n_files):
        x = mat[:, i]
        xc = x - x.mean()
        varx = float(np.sum(xc * xc))
        if varx <= 1e-12:
            corr = 0.0
            slope = 0.0
        else:
            cov = float(np.sum(xc * yc))
            corr = float(cov / np.sqrt(max(1e-12, varx * vary)))
            slope = float(cov / varx)

        tendency = "fake" if corr > 0.0 else "real"
        confidence = float(max(0.0, min(1.0, abs(corr) * scale_n)))
        rows_out.append(
            {
                "index": i,
                "id": ids[i] if i < len(ids) else f"{i:04d}",
                "corr_score_vs_prob": corr,
                "slope_score_per_prob": slope,
                "mean_prob_fake": float(x.mean()),
                "std_prob": float(x.std()),
                "tendency": tendency,
                "confidence": confidence,
                "n_runs": int(n_runs),
            }
        )

    rows_out.sort(key=lambda r: abs(float(r["corr_score_vs_prob"])), reverse=True)
    out_csv = Path(args.out_csv) if args.out_csv else (db / "file_impact.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "id",
                "corr_score_vs_prob",
                "slope_score_per_prob",
                "mean_prob_fake",
                "std_prob",
                "tendency",
                "confidence",
                "n_runs",
            ],
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    topn = min(10, len(rows_out))
    print(f"[ok] impact saved: {out_csv} (runs={n_runs}, files={n_files})")
    print(f"[top {topn}]")
    for r in rows_out[:topn]:
        print(
            f"  {r['id']}: tendency={r['tendency']} corr={r['corr_score_vs_prob']:+.4f} "
            f"slope={r['slope_score_per_prob']:+.4f} conf={r['confidence']:.3f}"
        )


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--db_dir",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/submissions/history",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add")
    p_add.add_argument("--submission", type=str, required=True)
    p_add.add_argument("--run_id", type=str, default=None)
    p_add.add_argument("--public_score", type=float, default=None)
    p_add.add_argument("--private_score", type=float, default=None)
    p_add.add_argument("--note", type=str, default="")
    p_add.set_defaults(func=cmd_add)

    p_set = sub.add_parser("set-score")
    p_set.add_argument("--run_id", type=str, required=True)
    p_set.add_argument("--public_score", type=float, default=None)
    p_set.add_argument("--private_score", type=float, default=None)
    p_set.add_argument("--note", type=str, default=None)
    p_set.set_defaults(func=cmd_set_score)

    p_rep = sub.add_parser("report")
    p_rep.add_argument("--metric", type=str, default="best", choices=["best", "public", "private"])
    p_rep.add_argument("--topk", type=int, default=0, help="0 means use all selected runs")
    p_rep.add_argument("--id_source", type=str, default=None, help="Optional directory or file-name list")
    p_rep.add_argument("--out_csv", type=str, default=None)
    p_rep.set_defaults(func=cmd_report)

    p_list = sub.add_parser("list")
    p_list.set_defaults(func=cmd_list)

    p_diff = sub.add_parser("diff")
    p_diff.add_argument("--run_a", type=str, required=True)
    p_diff.add_argument("--run_b", type=str, required=True)
    p_diff.add_argument("--id_source", type=str, default=None)
    p_diff.add_argument("--topk", type=int, default=20)
    p_diff.add_argument("--out_csv", type=str, default=None)
    p_diff.set_defaults(func=cmd_diff)

    p_imp = sub.add_parser("impact")
    p_imp.add_argument("--metric", type=str, default="best", choices=["best", "public", "private"])
    p_imp.add_argument("--topk", type=int, default=0, help="0 means use all scored runs")
    p_imp.add_argument("--id_source", type=str, default=None)
    p_imp.add_argument("--out_csv", type=str, default=None)
    p_imp.set_defaults(func=cmd_impact)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
