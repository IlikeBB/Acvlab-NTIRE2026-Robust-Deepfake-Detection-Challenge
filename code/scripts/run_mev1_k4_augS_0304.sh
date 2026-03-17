#!/usr/bin/env bash
set -euo pipefail

REPO="/work/u1657859/Jess/app-isolated-20260227-191252"
BASE_CFG="/work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_full6_8f_s2048_local/config.json"
CKPT_DIR="/work/u1657859/Jess/app/local_checkpoints"
NAME="Exp_MEv1_k4_augS_s2048_0304"

export TRANSFORMERS_NO_FLASH_ATTN=1
export PYTHONPATH="${REPO}/.deps:${PYTHONPATH:-}"

cd "${REPO}"

python scripts/run_multiexpert_local.py \
  --repo "${REPO}" \
  --base_config "${BASE_CFG}" \
  --name "${NAME}" \
  --checkpoints_dir "${CKPT_DIR}" \
  --seed 2048 \
  --niter 6 \
  --earlystop_epoch 0 \
  --batch_size 24 \
  --num_threads 4 \
  --train_frame_num 8 \
  --val_frame_num 1 \
  --multi_expert_k 4 \
  --multi_expert_route quality_bin \
  --multi_expert_div_lambda 0.002 \
  --set rank_loss_enable=false \
  --set max_aug_per_image=2 \
  --set lowres_prob=0.18 \
  --set motion_blur_prob=0.18 \
  --set blur_prob=0.25 \
  --set blur_sig_mild='[0.2,1.8]' \
  --set blur_sig_hard='[1.8,3.4]' \
  --set blur_hard_prob=0.30 \
  --set jpg_prob=0.30 \
  --set jpg_qual_mild='[35,85]' \
  --set jpg_qual_hard='[8,30]' \
  --set jpg_hard_prob=0.35 \
  --set lowres_scale_mild='[0.45,0.80]' \
  --set lowres_scale_hard='[0.22,0.45]' \
  --set lowres_hard_prob=0.35 \
  --set noise_prob=0.20 \
  --set iso_noise_prob=0.30 \
  --set grainy_group_prob=0.18 \
  --set mono_group_prob=0.12 \
  --set motion_grainy_group_prob=0.12 \
  --set lowlight_group_prob=0.10
