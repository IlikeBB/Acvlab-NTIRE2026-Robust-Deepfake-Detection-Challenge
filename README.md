# Select and Detect: Multi-Level Quality Routing for Robust Deepfake Detection
<hr>
## Final Submission Easy Run

This package is a simplified copy of the original submission package.

- The original inference code stays in `code/` and is kept as-is.
- The easy-run layer only adds wrapper scripts and clearer docs.
- The official ensemble rule in this package is:

```text
final = (pred1 + pred2 + pred3) / 3
```

## Folder Layout

```text
final_submission_easyrun/
├── checkpoints/            # 3 fixed checkpoints
├── code/                   # original inference/training source code
├── docs/QUICKSTART.md      # short usage guide
├── submissions/            # prediction csv and final zip outputs
├── run_model.sh            # run one checkpoint
├── run_all_models.sh       # run all 3 checkpoints
├── run_ensemble.py         # average the 3 csv files
├── run_submission.sh       # full end-to-end pipeline
└── SETUP_AND_VERIFY.sh     # dependency / package / optional smoke test
```

## Environment

```bash
conda create -n deepfake python=3.10 -y
conda activate deepfake
pip install -r requirements.txt
```

Required local resources:

- `checkpoints/*.pt`
- a local CLIP model directory
- an input image directory for inference

Current machine usable CLIP path:

```bash
/ssd2/ntire_RDFC_2026_cvpr/clip-vit-large-patch14
```

If that path exists, the wrapper scripts use it automatically. Otherwise pass `--clip_model /your/path`.

## Quick Commands

Single model, best checkpoint:

```bash
cd /ssd3/ntire_RDFC_2026_cvpr_fan/final_submission_easyrun

./run_model.sh \
  --model 2 \
  --input_dir /path/to/test_images \
  --batch_size 32 \
  --num_workers 4
```

This produces:

```text
submissions/pred_model2_lb0.8572.csv
```

Run all 3 models:

```bash
./run_all_models.sh \
  --input_dir /path/to/test_images \
  --batch_size 32 \
  --num_workers 4
```

This produces:

```text
submissions/pred_model1_lb0.8568.csv
submissions/pred_model2_lb0.8572.csv
submissions/pred_model3_lb0.8568strict.csv
```

Average ensemble only:

```bash
python run_ensemble.py
```

This produces:

```text
submissions/submission.txt
submissions/final_submission_equal_average.zip
```

Full pipeline in one command:

```bash
./run_submission.sh \
  --input_dir /path/to/test_images \
  --batch_size 32 \
  --num_workers 4
```

## Full Inference Parameters

These wrapper parameters are the ones most users need:

- `--input_dir`: image folder for inference
- `--model`: choose `1`, `2`, or `3` in `run_model.sh`
- `--batch_size`: default `32`
- `--num_workers`: default `4`
- `--clip_model`: CLIP model directory
- `--python`: select a specific Python executable

If you need raw inference options from `code/full_submission.py`, pass them after `--`.

Example:

```bash
./run_model.sh \
  --model 2 \
  --input_dir /path/to/test_images \
  --batch_size 16 \
  --num_workers 2 \
  -- \
  --tri_view \
  --tri_blur_sig 0.2 1.2 \
  --tri_jpeg_qual 70 95 \
  --tri_unsharp_amount 0.35
```

Common original options from `full_submission.py`:

- `--batch_size 32`
- `--num_workers 4`
- `--multi_expert_enable`
- `--multi_expert_k 1`
- `--multi_expert_route quality_bin`
- `--infer_quality_bin`
- `--quality_metric laplacian`
- `--tta_mode hflip_resize`
- `--tta_scales 0.94 1.06`
- `--tta_logit_agg mean`
- `--tri_view`
- `--tri_blur_sig 0.2 1.2`
- `--tri_jpeg_qual 70 95`
- `--tri_unsharp_amount 0.35`

## Checkpoint Mapping

- Model 1: `checkpoints/model1_lb0.8568_best_auc.pt`
- Model 2: `checkpoints/model2_lb0.8572_best_auc.pt`
- Model 3: `checkpoints/model3_lb0.8568strict_best_auc.pt`

## Verification

Basic verification:

```bash
./SETUP_AND_VERIFY.sh
```

Smoke inference on a small image folder:

```bash
./SETUP_AND_VERIFY.sh \
  --run_smoke_test \
  --test_input_dir /path/to/small_test_images
```

## Output Rule

This package treats the following as the final submission workflow:

1. Run 3 checkpoints independently.
2. Save 3 prediction csv files.
3. Average them equally.
4. Pack `submission.txt` into `submissions/final_submission_equal_average.zip`.

Weighted ensemble is not used in this simplified package.


- Verified:
  - `checkpoint -> submission.txt` reproduction
- Not verified here:
  - full training rerun from scratch
