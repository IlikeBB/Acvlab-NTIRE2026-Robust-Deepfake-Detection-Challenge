#!/usr/bin/env python3
"""Ensemble with best 0.8688 model included"""
import numpy as np
import zipfile
from pathlib import Path

def load_predictions(csv_path):
    """Load predictions from CSV"""
    preds = []
    with open(csv_path, 'r') as f:
        for line in f:
            preds.append(float(line.strip()))
    return np.array(preds)

def ensemble_predictions(pred_files, weights=None):
    """Ensemble predictions with optional weights"""
    all_preds = []
    for f in pred_files:
        if Path(f).exists():
            all_preds.append(load_predictions(f))
            print(f"Loaded {f}: {len(all_preds[-1])} predictions")
        else:
            print(f"Warning: {f} not found, skipping")

    if not all_preds:
        raise ValueError("No predictions found!")

    # Stack predictions
    preds = np.stack(all_preds, axis=0)  # (n_models, n_samples)

    if weights is None:
        weights = np.ones(len(all_preds)) / len(all_preds)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()

    print(f"\nUsing weights: {weights}")

    # Weighted average
    ensemble = np.average(preds, axis=0, weights=weights)
    return ensemble

def create_submission_zip(predictions, output_zip):
    """Create submission.txt and zip it"""
    # Write submission.txt
    submission_txt = "submission.txt"
    with open(submission_txt, 'w') as f:
        for pred in predictions:
            f.write(f"{pred:.6f}\n")

    # Create zip
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_txt, arcname="submission.txt")

    print(f"\nCreated {output_zip} with {len(predictions)} predictions")
    print(f"Stats: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")

    # Clean up temp file
    Path(submission_txt).unlink()

if __name__ == "__main__":
    # Best 3 confirmed pure models with LB >= 0.85
    pred_files = [
        "../submissions/pred_model1_lb0.8568.csv",  # LB 0.8568
        "../submissions/pred_model2_lb0.8572.csv",  # LB 0.8572 - highest!
        "../submissions/pred_model3_lb0.8568strict.csv",  # LB 0.8568 strict
    ]

    existing = [f for f in pred_files if Path(f).exists()]
    print(f"Found {len(existing)}/{len(pred_files)} prediction files")

    if len(existing) == 0:
        print("No predictions ready yet!")
        exit(1)

    # Equal weight ensemble
    ensemble_equal = ensemble_predictions(existing, weights=None)
    create_submission_zip(ensemble_equal, "final_submission_best3_equal.zip")

    # Weighted ensemble (more weight on highest LB model)
    if len(existing) == 3:
        # Give more weight to LB 0.8572 model
        ensemble_weighted = ensemble_predictions(existing, weights=[0.3, 0.4, 0.3])
        create_submission_zip(ensemble_weighted, "final_submission_best3_weighted.zip")
        print("\nCreated 2 versions:")
        print("  - final_submission_best3_equal.zip (equal weights 1/3, 1/3, 1/3)")
        print("  - final_submission_best3_weighted.zip (0.3, 0.4, 0.3 - more weight on LB 0.8572)")
    else:
        print(f"\nCreated ensemble with {len(existing)} models only")
