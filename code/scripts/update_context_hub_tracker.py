#!/usr/bin/env python3
"""
Generate a local Context Hub-compatible tracker from experiment outputs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_SCORE_DIR = "/work/u1657859/Jess/app/score"
DEFAULT_CONTENT_ROOT = "/work/u1657859/Jess/app/context-hub-content"
DOC_REL = "jesslab/docs/ntire-mev1-tracker"
DOC_NAME = "ntire-mev1-tracker"


def _safe_float(v: str | None, fallback: float = -1.0) -> float:
    try:
        if v is None:
            return fallback
        s = str(v).strip()
        if s == "":
            return fallback
        return float(s)
    except Exception:
        return fallback


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if (not path.exists()) or (not path.is_file()):
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _latest_file(score_dir: Path, pattern: str) -> Path | None:
    files = [Path(p) for p in glob.glob(str(score_dir / pattern))]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _top_lb_rows(merged_lb_csv: Path, limit: int = 30) -> List[Dict[str, str]]:
    rows = _read_csv(merged_lb_csv)
    if not rows:
        return []
    rows.sort(key=lambda r: _safe_float(r.get("lb")), reverse=True)
    return rows[:limit]


def _parse_revision(doc_path: Path) -> int:
    if not doc_path.exists():
        return 1
    text = doc_path.read_text(encoding="utf-8", errors="ignore")
    marker = "revision:"
    idx = text.find(marker)
    if idx < 0:
        return 1
    tail = text[idx + len(marker) : idx + len(marker) + 20]
    digits = "".join(ch for ch in tail if ch.isdigit())
    if digits == "":
        return 1
    try:
        return int(digits) + 1
    except Exception:
        return 1


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_No data._"
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def _collect_process_snapshot() -> str:
    cmd = [
        "bash",
        "-lc",
        "pgrep -af 'run_mev1_diverse_pack.sh|run_mev1_softsvd_full_all.sh|Exp_M1div_|Exp_M1softFULL_|Exp_M1test_debug' || true",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        return out if out else "(no matched processes)"
    except Exception as e:
        return f"(process snapshot failed: {e})"


def _write_csv_rows(path: Path, headers: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def _copy_if_exists(src: Path | None, dst: Path) -> None:
    if src is None or (not src.exists()):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def render_doc(
    doc_path: Path,
    score_dir: Path,
    merged_lb_csv: Path,
    top_lb: List[Dict[str, str]],
    diverse_csv: Path | None,
    diverse_rows: List[Dict[str, str]],
    softsvd_csv: Path | None,
    softsvd_rows: List[Dict[str, str]],
    proc_snapshot: str,
) -> None:
    now = dt.datetime.now()
    today = now.date().isoformat()
    rev = _parse_revision(doc_path)

    top_lb_table_rows: List[List[str]] = []
    for i, r in enumerate(top_lb, start=1):
        top_lb_table_rows.append(
            [
                str(i),
                f"{_safe_float(r.get('lb'), 0.0):.4f}",
                (r.get("run_id") or "").strip(),
                (r.get("val_auc") or "").strip(),
                (r.get("source_path") or "").strip(),
            ]
        )

    diverse_table_rows: List[List[str]] = []
    for r in diverse_rows[:20]:
        diverse_table_rows.append(
            [
                (r.get("case_id") or "").strip(),
                (r.get("run_name") or "").strip(),
                (r.get("val_auc_epoch0") or "").strip(),
                (r.get("submission_status") or "").strip(),
            ]
        )

    softsvd_table_rows: List[List[str]] = []
    for r in softsvd_rows[:20]:
        softsvd_table_rows.append(
            [
                (r.get("run_name") or "").strip(),
                (r.get("adapter_last_n_layers") or "").strip(),
                (r.get("svd_residual_mode") or "").strip(),
                (r.get("val_auc_epoch0") or "").strip(),
                (r.get("submission_status") or "").strip(),
            ]
        )

    text = f"""---
name: {DOC_NAME}
description: "NTIRE RDFC MEv1 experiment tracker and leaderboard records."
metadata:
  languages: "markdown"
  versions: "2026.03"
  revision: {rev}
  updated-on: "{today}"
  source: maintainer
  tags: "ntire,rdfc,mev1,leaderboard,experiments"
---

# NTIRE MEv1 Tracker

Generated at: `{now.isoformat(timespec='seconds')}`

Score directory: `{score_dir}`

## Top LB Submissions

Source: `{merged_lb_csv}`

{_markdown_table(['rank', 'lb', 'run_id', 'val_auc', 'source_path'], top_lb_table_rows)}

## Latest Diverse Pack Sweep

Latest file: `{diverse_csv if diverse_csv else '(none)'}`

{_markdown_table(['case_id', 'run_name', 'val_auc_epoch0', 'status'], diverse_table_rows)}

## Latest SoftSVD Sweep

Latest file: `{softsvd_csv if softsvd_csv else '(none)'}`

{_markdown_table(['run_name', 'adapter_n', 'svd_mode', 'val_auc_epoch0', 'status'], softsvd_table_rows)}

## Active Training Processes

```
{proc_snapshot}
```

## Notes

- This document is auto-generated from local experiment outputs.
- Keep this entry as the stable memory anchor for future runs.
"""
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(text, encoding="utf-8")


def run_build(content_root: Path, output_dir: Path, validate_only: bool) -> Tuple[int, str]:
    cmd = [
        "npx",
        "-y",
        "@aisuite/chub",
        "build",
        str(content_root),
        "-o",
        str(output_dir),
    ]
    if validate_only:
        cmd.append("--validate-only")
    env = os.environ.copy()
    npm_cache = content_root / ".npm-cache"
    npm_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("npm_config_cache", str(npm_cache))
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, env=env)
        return 0, out.strip()
    except subprocess.CalledProcessError as e:
        return int(e.returncode), (e.output or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync experiment outputs into Context Hub-compatible tracker.")
    ap.add_argument("--score-dir", default=DEFAULT_SCORE_DIR)
    ap.add_argument("--content-root", default=DEFAULT_CONTENT_ROOT)
    ap.add_argument("--build", action="store_true", help="Run `chub build` after generating content.")
    ap.add_argument("--validate-only", action="store_true", help="Use `chub build --validate-only`.")
    args = ap.parse_args()

    score_dir = Path(args.score_dir).resolve()
    content_root = Path(args.content_root).resolve()
    doc_dir = content_root / DOC_REL
    doc_path = doc_dir / "DOC.md"
    ref_dir = doc_dir / "references"
    dist_dir = content_root / "dist"

    merged_lb_csv = score_dir / "merged_lb_table_app_app4.csv"
    top_lb = _top_lb_rows(merged_lb_csv, limit=30)

    diverse_csv = _latest_file(score_dir, "mev1_diverse_pack_*.csv")
    diverse_rows = _read_csv(diverse_csv) if diverse_csv else []

    softsvd_csv = _latest_file(score_dir, "mev1_softsvd_full_all_*.csv")
    softsvd_rows = _read_csv(softsvd_csv) if softsvd_csv else []

    proc_snapshot = _collect_process_snapshot()

    # Reference artifacts for quick diff/audit.
    ref_dir.mkdir(parents=True, exist_ok=True)
    lb_rows_for_csv: List[List[str]] = []
    for r in top_lb:
        lb_rows_for_csv.append(
            [
                r.get("run_id", ""),
                r.get("lb", ""),
                r.get("val_auc", ""),
                r.get("source_path", ""),
            ]
        )
    _write_csv_rows(ref_dir / "top_lb_submissions.csv", ["run_id", "lb", "val_auc", "source_path"], lb_rows_for_csv)
    _copy_if_exists(diverse_csv, ref_dir / "latest_diverse_pack.csv")
    _copy_if_exists(softsvd_csv, ref_dir / "latest_softsvd.csv")
    (ref_dir / "active_processes.txt").write_text(proc_snapshot + "\n", encoding="utf-8")

    render_doc(
        doc_path=doc_path,
        score_dir=score_dir,
        merged_lb_csv=merged_lb_csv,
        top_lb=top_lb,
        diverse_csv=diverse_csv,
        diverse_rows=diverse_rows,
        softsvd_csv=softsvd_csv,
        softsvd_rows=softsvd_rows,
        proc_snapshot=proc_snapshot,
    )

    print(f"[ok] wrote doc: {doc_path}")
    print(f"[ok] wrote refs: {ref_dir}")

    if args.build:
        code, out = run_build(content_root=content_root, output_dir=dist_dir, validate_only=bool(args.validate_only))
        if out:
            print(out)
        if code != 0:
            raise SystemExit(code)
        print(f"[ok] build output: {dist_dir}")


if __name__ == "__main__":
    main()
