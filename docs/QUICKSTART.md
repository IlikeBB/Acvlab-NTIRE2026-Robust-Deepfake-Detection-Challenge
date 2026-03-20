# Quick Start

## 1. Install

```bash
cd /ssd3/ntire_RDFC_2026_cvpr_fan/final_submission_easyrun
conda activate deepfake
pip install -r requirements.txt
```

## 2. Verify Package

```bash
./SETUP_AND_VERIFY.sh
```

## 3. Run Best Single Model

```bash
./run_model.sh \
  --model 2 \
  --input_dir /path/to/test_images \
  --batch_size 32 \
  --num_workers 4
```

Output:

```text
submissions/pred_model2_lb0.8572.csv
```

## 4. Run Final Submission

```bash
./run_submission.sh \
  --input_dir /path/to/test_images \
  --batch_size 32 \
  --num_workers 4
```

Outputs:

```text
submissions/pred_model1_lb0.8568.csv
submissions/pred_model2_lb0.8572.csv
submissions/pred_model3_lb0.8568strict.csv
submissions/submission.txt
submissions/final_submission_equal_average.zip
```

## 5. Ensemble Rule

```text
final = (pred1 + pred2 + pred3) / 3
```

## Optional: pass original inference flags

```bash
./run_model.sh \
  --model 2 \
  --input_dir /path/to/test_images \
  -- \
  --tri_view \
  --tri_blur_sig 0.2 1.2 \
  --tri_jpeg_qual 70 95 \
  --tri_unsharp_amount 0.35
```
