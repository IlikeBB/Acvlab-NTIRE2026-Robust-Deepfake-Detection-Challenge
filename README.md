# Acvlab-NTIRE2026-Robust-Deepfake-Detection-Challenge
Historical public LB: `0.8568`

This bundle is for reproducing the historical submission content of the single-model run recorded as `m1strictfull_e0_0305_userlb_08568`.

## Included

- `code/`
  - Minimal code snapshot used for the verified reproduction run.
- `artifacts/checkpoint/best_auc.pt`
- `artifacts/run_dir/config.json`
- `artifacts/run_dir/train_20260305-125022.log`
- `artifacts/original_submission/sub_m1strictfull_e0_0305.txt`
- `artifacts/original_submission/sub_m1strictfull_e0_0305.zip`
- `artifacts/verified_repro/`
  - Reproduction outputs generated and verified on 2026-03-14.
- `artifacts/supporting_logs/`
  - Includes the historical watch script that links this run to the submission path.
- `run_repro_submission.sh`
- `verify_repro.sh`

## External Requirements

This bundle is not fully self-contained. The receiving side still needs:

- competition validation images, for example:
  - `/path/to/validation_data_final`
- CLIP backbone weights:
  - `/path/to/clip-vit-large-patch14`
- a Python environment with compatible packages

## Tested Environment

- Python `3.12.2`
- `torch==2.2.0+cu121`
- `torchvision==0.17.0+cu121`
- `transformers==4.44.2`
- `numpy==1.26.4`
- `Pillow==10.2.0`
- `tqdm==4.66.5`
- GPU used for verification: `Tesla V100-SXM2-32GB`

## Verified Result

On 2026-03-14, this bundle reproduced the historical submission text exactly.

- original txt SHA256:
  - `6bfca0e13c0a0fdb1db2a660e5468ed1250dfc51a66dae4b4592cef308642adf`
- reproduced txt SHA256:
  - `6bfca0e13c0a0fdb1db2a660e5468ed1250dfc51a66dae4b4592cef308642adf`

The outer zip bytes are not guaranteed to match because zip timestamps differ, but the inner `submission.txt` content matches exactly.

## Quick Start

```bash
export DATA_ROOT=/path/to/validation_data_final
export CLIP_ROOT=/path/to/clip-vit-large-patch14
export PYTHON_BIN=python

bash run_repro_submission.sh
bash verify_repro.sh
```

## Scope

- Verified:
  - `checkpoint -> submission.txt` reproduction
- Not verified here:
  - full training rerun from scratch
