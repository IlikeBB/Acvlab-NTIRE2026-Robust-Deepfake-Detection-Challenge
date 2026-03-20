#!/usr/bin/env python3

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np


DEFAULT_INPUTS = [
    "submissions/pred_model1_lb0.8568.csv",
    "submissions/pred_model2_lb0.8572.csv",
    "submissions/pred_model3_lb0.8568strict.csv",
]


def load_predictions(path: Path) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))
    if not values:
        raise ValueError(f"prediction file is empty: {path}")
    return np.asarray(values, dtype=np.float64)


def write_zip(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submission.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Average 3 prediction CSV files and create Codabench zip.")
    parser.add_argument(
        "--inputs",
        nargs=3,
        default=DEFAULT_INPUTS,
        help="Exactly 3 prediction CSV files, averaged as (pred1 + pred2 + pred3) / 3.",
    )
    parser.add_argument(
        "--output_csv",
        default="submissions/submission.txt",
        help="Output averaged prediction text file.",
    )
    parser.add_argument(
        "--output_zip",
        default="submissions/final_submission_equal_average.zip",
        help="Output zip file containing submission.txt.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"missing prediction file: {path}")

    preds = [load_predictions(path) for path in input_paths]
    lengths = {arr.shape[0] for arr in preds}
    if len(lengths) != 1:
        raise ValueError(f"prediction lengths do not match: {[arr.shape[0] for arr in preds]}")

    averaged = (preds[0] + preds[1] + preds[2]) / 3.0

    output_csv = Path(args.output_csv)
    output_zip = Path(args.output_zip)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8") as f:
        for value in averaged:
            f.write(f"{value:.6f}\n")

    write_zip(output_csv, output_zip)

    print("[done] simple average ensemble completed")
    print(f"[done] inputs={', '.join(str(p) for p in input_paths)}")
    print(f"[done] output_csv={output_csv}")
    print(f"[done] output_zip={output_zip}")
    print(f"[done] count={averaged.shape[0]}")
    print(f"[done] min={averaged.min():.6f} max={averaged.max():.6f} mean={averaged.mean():.6f}")


if __name__ == "__main__":
    main()
