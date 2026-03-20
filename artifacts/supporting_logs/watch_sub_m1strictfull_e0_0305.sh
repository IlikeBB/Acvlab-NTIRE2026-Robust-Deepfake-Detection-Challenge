#!/usr/bin/env bash
set -euo pipefail
REPO=/work/u1657859/Jess/app-isolated-20260227-191252
RUN_DIR=/work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_strict_full_recheck_s2048_20260305
CKPT="$RUN_DIR/epoch_0.pt"
VAL_DIR=/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final
OUT_TXT=/work/u1657859/Jess/app/local_checkpoints/submissions/sub_m1strictfull_e0_0305.txt
OUT_ZIP=/work/u1657859/Jess/app/local_checkpoints/submissions/sub_m1strictfull_e0_0305.zip
LOG=/work/u1657859/Jess/app/local_checkpoints/logs/watch_sub_m1strictfull_e0_0305.log

echo "[$(date '+%F %T')] waiting ckpt: $CKPT" | tee -a "$LOG"
while [[ ! -f "$CKPT" ]]; do sleep 30; done

echo "[$(date '+%F %T')] ckpt ready; building submission" | tee -a "$LOG"
/home/u1657859/miniconda3/envs/eamamba/bin/python "$REPO/full_submission.py" \
  --ckpt "$CKPT" \
  --txt_path "$VAL_DIR" \
  --root_dir "$VAL_DIR" \
  --out_csv "$OUT_TXT" \
  --zip_submission \
  --zip_path "$OUT_ZIP" \
  --batch_size 32 \
  --num_workers 4 \
  --multi_expert_enable \
  --multi_expert_k 3 \
  --multi_expert_route quality_bin \
  --infer_quality_bin \
  --tta_mode none \
  --tta_logit_agg mean \
  --on_error skip >> "$LOG" 2>&1

echo "[$(date '+%F %T')] done: $OUT_ZIP" | tee -a "$LOG"
