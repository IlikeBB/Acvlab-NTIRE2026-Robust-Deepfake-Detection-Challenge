# ---------------- Environment ----------------
CUDA_DEVICE_ORDER = "PCI_BUS_ID"
CUDA_VISIBLE_DEVICES = "0"

# ---------------- Reproducibility ----------------
SEED = 418
DETERMINISTIC = True

# ---------------- Experiment ----------------
EXP_NAME = "Exp_app4_datafirst_phaseA_e6_s418_epzip"
CHECKPOINTS_DIR = "./checkpoints"

# ---------------- Training ----------------
TRAIN_BATCH_SIZE = 32
NUM_WORKERS = 4
NITER = 6
LR = 2e-4
RESIDUAL_LR = None
WEIGHT_DECAY = 0.0
USE_AMP = True
GRAD_ACCUM_STEPS = 1
GRAD_CLIP = 1.0
CLEAR_CACHE_EACH_EPOCH = True
LR_SCHEDULER = "cosine"  # none | cosine
COSINE_T_MAX = None
COSINE_MIN_LR = 1e-6
LR_WARMUP_STEPS = 0.0
EARLYSTOP_EPOCH = 5
LOSS_FREQ = 200
SAVE_EPOCH_FREQ = 1

# ---------------- Epoch Submission Packaging ----------------
SUBMIT_EACH_EPOCH = True
SUBMIT_BUILDER_SCRIPT = "/work/u1657859/Jess/app4/scripts/build_submission_mev1_strict.sh"
SUBMIT_OUT_PREFIX = "app_Exp_app4_datafirst_phaseA_e6_s418_epzip"
SUBMIT_OUT_DIR = "/work/u1657859/Jess/app/local_checkpoints/submissions"
SUBMIT_APP = "/work/u1657859/Jess/app4"
SUBMIT_PYTHON = "/home/u1657859/miniconda3/envs/eamamba/bin/python"
SUBMIT_VAL_DIR = "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final"
SUBMIT_BATCH_SIZE = 64
SUBMIT_NUM_WORKERS = 4
SUBMIT_TTA_MODE = "none"
SUBMIT_TTA_LOGIT_AGG = "mean"

# ---------------- Data Paths ----------------
TRAIN_CSV = "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/DF40_frames/image_frame_list_face.csv"
VAL_DATA_ROOT = "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/training_data_final"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TRAIN_FRAME_NUM = 8
VAL_FRAME_NUM = 1
TRAIN_FRAME_SAMPLE_MODE = "uniform"
VAL_FRAME_SAMPLE_MODE = "uniform"
FRAME_SAMPLE_SEED = 42
IMAGE_SIZE = 224
ON_ERROR = "skip"

# ---------------- Augmentation ----------------
HFLIP_PROB = 0.2
BLUR_PROB = 0.2
BLUR_SIG = [0.0, 2.0]
JPG_PROB = 0.2
JPG_QUAL = [30, 100]
AUGMENT_PROB = None
# PMM exclusive augmentation group: when hit, return immediately without stacking classic blur/jpeg/noise.
USE_PMM_AUG = False
PMM_GROUP_PROB = 0.0
PMM_PDM_TYPE = "ours"  # ours | bsrgan | ddrc | noise | blur | jpeg | resize | motion | brightness | speckle
PMM_STRENGTH = 0.5
PMM_USE_BETA = False
PMM_BETA_A = 0.5
PMM_BETA_B = 0.5
PMM_DISTRACTOR_P = 0.0

# ---------------- GroupDRO / Quality Groups ----------------
QUALITY_BALANCE = False
USE_STRATIFIED_SAMPLER = False
STRATIFIED_SAMPLER_KEY = "stratum_id"  # stratum_id | quality_bin | degrade_bin (label-free)
USE_REWEIGHTED_ERM = False
REWEIGHT_USE_SQRT = False
REWEIGHT_CLIP_MIN = 0.2
REWEIGHT_CLIP_MAX = 5.0
VAL_QUALITY_DIAG = True
QUALITY_BINS = 3
QUALITY_MAX_SIDE = 256
QUALITY_METRIC = "laplacian"  # laplacian | clipiqa
QUALITY_CLIPIQA_BATCH_SIZE = 64
QUALITY_CLIPIQA_DEVICE = "cuda"
QUALITY_CLIPIQA_WEIGHTS = [0.5, 0.3, 0.2]  # quality, sharpness, (1-noisiness)
QUALITY_CACHE_PATH = None
QUALITY_CACHE_TRAIN_PATH = "cache/q_train.csv"
QUALITY_CACHE_VAL_PATH = "cache/q_val.csv"
QUALITY_CUTS = None  # fixed cutpoints from train; populated at runtime and saved to config.json
DEGRADE_BINS = 3
DEGRADE_CACHE_PATH = None
DEGRADE_CACHE_TRAIN_PATH = "cache/degrade_train.csv"
DEGRADE_CACHE_VAL_PATH = "cache/degrade_val.csv"

# ---------------- GroupDRO (robust worst-group) ----------------
# Group key should be label-free if your goal is shortcut reduction (e.g. "quality_bin").
USE_GROUPDRO = False
GROUPDRO_GROUP_KEY = "quality_bin"   # quality_bin | stratum_id | degrade_bin
GROUPDRO_CLASS_BALANCE = False       # class-balanced aggregation within each group
GROUPDRO_NUM_GROUPS = None           # optional override; None -> inferred
GROUPDRO_ETA = 0.10                  # step size for exponentiated-gradient
GROUPDRO_EMA_BETA = 0.90             # EMA beta for per-group loss
GROUPDRO_Q_FLOOR = 0.00              # optional floor added before normalize
GROUPDRO_LOGQ_CLIP = None            # optional float: clamp log_q to [-c, c] before softmax
GROUPDRO_MIN_COUNT_GROUP = 1         # minimum samples in a group to update EMA/q
GROUPDRO_MIN_COUNT_CLASS = 1         # (class-balanced mode) minimum samples per class within group
GROUPDRO_WARMUP_STEPS = 0            # keep ERM for first N steps
GROUPDRO_UPDATE_EVERY = 1            # update q every N steps

# ---------------- Validation Micro-TTA (tensor-space) ----------------
VAL_MICRO_TTA = False
VAL_MICRO_TTA_K = 9
VAL_MICRO_TTA_SEED = 0
VAL_MICRO_TTA_BLUR_MAX_SIG = 1.5
VAL_MICRO_TTA_JPEG_QUAL = [60, 95]
VAL_MICRO_TTA_CONTRAST = [0.9, 1.1]
VAL_MICRO_TTA_GAMMA = [0.9, 1.1]
VAL_MICRO_TTA_RESIZE_JITTER = 8

# ---------------- Tri-view Consistency (train-time invariance) ----------------
TRI_CONSISTENCY_ENABLE = False
TRI_CONSISTENCY_APPLY_PROB = 0.5
TRI_CONSISTENCY_CE_WEIGHT = 0.25
TRI_CONSISTENCY_ALIGN_WEIGHT = 0.15
TRI_CONSISTENCY_BLUR_SIG = [0.2, 1.2]
TRI_CONSISTENCY_JPEG_QUAL = [70, 95]
TRI_CONSISTENCY_RESIZE_JITTER = 4
TRI_CONSISTENCY_NOISE_STD = 0.005
TRI_CONSISTENCY_UNSHARP_AMOUNT = 0.35
TRI_CONSISTENCY_RESTORE_MIX = 0.25

# ---------------- Noise-Robust Fine-Tune ----------------
NOISE_ROBUST_ENABLE = False
NOISE_ROBUST_WARMUP_STEPS = 0
NOISE_ROBUST_CONF_THRESH = 0.65
NOISE_ROBUST_MIN_WEIGHT = 0.20
NOISE_ROBUST_SOFT_WEIGHT = 0.50
NOISE_ROBUST_LABEL_SMOOTH = 0.02
NOISE_ROBUST_GROUP_KEY = "quality_bin"  # none|quality_bin|stratum_id|degrade_bin
NOISE_ROBUST_GROUP_MOMENTUM = 0.90

# E3/E4 switch: False -> resize then augment, True -> augment then resize
AUG_BEFORE_RESIZE = False

# ---------------- Model ----------------
MODEL_FAMILY = "effort"  # effort | dinov3
CLIP_MODEL = "/work/u1657859/ntire_RDFC_2026_cvpr/clip-vit-large-patch14/"
FIX_BACKBONE = True
SVD_CACHE_ENABLE = True
SVD_CACHE_DIR = "cache"
SVD_INIT_DEVICE = "auto"   # auto|cpu|cuda
SVD_SHOW_PROGRESS = True
MULTI_EXPERT_ENABLE = False
MULTI_EXPERT_K = 1
MULTI_EXPERT_ROUTE = "quality_bin"  # quality_bin|quality_soft|degrade_bin|uniform|router|hybrid|degrade_router|degrade_hybrid
MULTI_EXPERT_ROUTE_TEMP = 1.0
MULTI_EXPERT_ROUTE_DETACH = True
MULTI_EXPERT_HYBRID_MIX = 0.0
MULTI_EXPERT_ROUTE_SUP_LAMBDA = 0.0
MULTI_EXPERT_BALANCE_LAMBDA = 0.0
MULTI_EXPERT_DIV_LAMBDA = 0.0
# Inverse-MEv1 (optional): MEv1 teacher guidance + adapter divergence from anchor ckpt.
INV_MEV1_ENABLE = False
INV_MEV1_ANCHOR_CKPT = None
INV_MEV1_TEACHER_CKPT = None
INV_MEV1_DIV_LAMBDA = 0.0
INV_MEV1_DIV_TARGET_COS = 0.995
INV_MEV1_KD_LAMBDA = 0.0
INV_MEV1_KD_TEMP = 2.0
INV_MEV1_TEACHER_EVERY = 1
INV_MEV1_WARMUP_STEPS = 0.0
INV_MEV1_RAMP_STEPS = 0.0
DINO_SOURCE = "timm"  # timm | torchhub
DINO_REPO = "facebookresearch/dinov3"
# For this env, the convnext DINOv3 checkpoint is the most reproducible ungated option.
DINO_MODEL_NAME = "hf-hub:timm/convnext_base.dinov3_lvd1689m"
DINO_PRETRAINED = True
DINO_LOCAL_WEIGHTS = None
DINO_FORCE_IMAGENET_NORM = True

# ---------------- Patch Pooling ----------------
PATCH_POOL_TAU = 1.8
PATCH_POOL_MODE = "lse"
PATCH_TRIM_P = 0.2
PATCH_QUALITY = "cos_norm"
PATCH_POOL_GAMMA = 0.08
PATCH_WARMUP_STEPS = 0.10
PATCH_RAMP_STEPS = 0.30
PATCH_AUX_LAMBDA = 0.0 #0.5

# ---------------- OSD Regularizers ----------------
OSD_REGS = True
LAMBDA_ORTH = 5e-2
LAMBDA_KSV = 1e-2
REG_WARMUP_STEPS = 0.10
REG_RAMP_STEPS = 0.20

# ---------------- Parser Defaults ----------------
# Defaults from options/base_options.py + options/train_options.py
PARSER_DEFAULTS = {
    # base options
    "head_type": "attention",
    "no_augment": False,
    "patch_base": False,
    "shuffle": False,
    "shuffle_times": 1,
    "original_times": 1,
    "patch_size": [14],
    "resume_path": None,
    "load_whole_model": False,
    "scale": 1.0,
    "mode": "binary",
    "arch": "CLIP:ViT-L/14",
    "fix_backbone": False,
    "model_family": "effort",
    "clip_model": None,
    "svd_cache_enable": True,
    "svd_cache_dir": "cache",
    "svd_init_device": "auto",
    "svd_show_progress": True,
    "multi_expert_enable": False,
    "multi_expert_k": 1,
    "multi_expert_route": "quality_bin",
    "multi_expert_route_temp": 1.0,
    "multi_expert_route_detach": True,
    "multi_expert_hybrid_mix": 0.0,
    "multi_expert_route_sup_lambda": 0.0,
    "multi_expert_balance_lambda": 0.0,
    "multi_expert_div_lambda": 0.0,
    "inv_mev1_enable": False,
    "inv_mev1_anchor_ckpt": None,
    "inv_mev1_teacher_ckpt": None,
    "inv_mev1_div_lambda": 0.0,
    "inv_mev1_div_target_cos": 0.995,
    "inv_mev1_kd_lambda": 0.0,
    "inv_mev1_kd_temp": 2.0,
    "inv_mev1_teacher_every": 1,
    "inv_mev1_warmup_steps": 0.0,
    "inv_mev1_ramp_steps": 0.0,
    "dino_source": "timm",
    "dino_repo": "facebookresearch/dinov3",
    "dino_model_name": "hf-hub:timm/convnext_base.dinov3_lvd1689m",
    "dino_pretrained": True,
    "dino_local_weights": None,
    "dino_force_imagenet_norm": True,
    # patch pooling
    "patch_pool_tau": 1.5,
    "patch_pool_mode": "lse",
    "patch_trim_p": 0.2,
    "patch_quality": "none",
    "patch_pool_gamma": 0.0,
    "patch_warmup_steps": 0.0,
    "patch_ramp_steps": 0.0,
    "patch_aux_lambda": 0.0,
    # OSD regularizers
    "osd_regs": True,
    "no_osd_regs": False,
    "lambda_orth": 5e-2,
    "lambda_ksv": 1e-2,
    "reg_warmup_steps": 0.0,
    "reg_ramp_steps": 0.0,
    # augmentation
    "rz_interp": "bilinear",
    "blur_prob": 0.2,
    "blur_sig": [0.0, 2.0],
    "jpg_prob": 0.2,
    "jpg_method": "cv2,pil",
    "jpg_qual": [30, 100],
    "hflip_prob": 0.2,
    "augment_prob": None,
    "use_pmm_aug": False,
    "pmm_group_prob": 0.0,
    "pmm_pdm_type": "ours",
    "pmm_strength": 0.5,
    "pmm_use_beta": False,
    "pmm_beta_a": 0.5,
    "pmm_beta_b": 0.5,
    "pmm_distractor_p": 0.0,
    # data inputs
    "csv": None,
    "train_csv": None,
    "val_csv": None,
    "folder_root": None,
    "val_data_root": None,
    "train_frame_num": 8,
    "val_frame_num": 1,
    "train_frame_sample_mode": "uniform",
    "val_frame_sample_mode": "uniform",
    "frame_sample_seed": 42,
    "train_frame_sample_seed": None,
    "val_frame_sample_seed": None,
    "image_size": 224,
    "on_error": "skip",
    "quality_balance": False,
    "use_stratified_sampler": False,
    "stratified_sampler_key": "stratum_id",
    "use_reweighted_erm": False,
    "reweight_use_sqrt": False,
    "reweight_clip_min": 0.2,
    "reweight_clip_max": 5.0,
    "val_quality_diag": True,
    "quality_bins": 3,
    "quality_max_side": 256,
    "quality_metric": "laplacian",
    "quality_clipiqa_batch_size": 64,
    "quality_clipiqa_device": "cuda",
    "quality_clipiqa_weights": [0.5, 0.3, 0.2],
    "quality_cache_path": None,
    "quality_cache_train_path": None,
    "quality_cache_val_path": None,
    "quality_cuts": None,
    "degrade_bins": 3,
    "degrade_cache_path": None,
    "degrade_cache_train_path": None,
    "degrade_cache_val_path": None,
    "aug_before_resize": False,
    # val micro-tta
    "val_micro_tta": False,
    "val_micro_tta_k": 9,
    "val_micro_tta_seed": 0,
    "val_micro_tta_blur_max_sig": 1.5,
    "val_micro_tta_jpeg_qual": [60, 95],
    "val_micro_tta_contrast": [0.9, 1.1],
    "val_micro_tta_gamma": [0.9, 1.1],
    "val_micro_tta_resize_jitter": 8,
    # tri-view consistency
    "tri_consistency_enable": False,
    "tri_consistency_apply_prob": 0.5,
    "tri_consistency_ce_weight": 0.25,
    "tri_consistency_align_weight": 0.15,
    "tri_consistency_blur_sig": [0.2, 1.2],
    "tri_consistency_jpeg_qual": [70, 95],
    "tri_consistency_resize_jitter": 4,
    "tri_consistency_noise_std": 0.005,
    "tri_consistency_unsharp_amount": 0.35,
    "tri_consistency_restore_mix": 0.25,
    # noise-robust fine-tune
    "noise_robust_enable": False,
    "noise_robust_warmup_steps": 0,
    "noise_robust_conf_thresh": 0.65,
    "noise_robust_min_weight": 0.20,
    "noise_robust_soft_weight": 0.50,
    "noise_robust_label_smooth": 0.02,
    "noise_robust_group_key": "quality_bin",
    "noise_robust_group_momentum": 0.90,
    # groupdro
    "use_groupdro": False,
    "groupdro_group_key": "quality_bin",
    "groupdro_class_balance": False,
    "groupdro_num_groups": None,
    "groupdro_eta": 0.10,
    "groupdro_ema_beta": 0.90,
    "groupdro_q_floor": 0.00,
    "groupdro_logq_clip": None,
    "groupdro_min_count_group": 1,
    "groupdro_min_count_class": 1,
    "groupdro_warmup_steps": 0,
    "groupdro_update_every": 1,
    # legacy data options
    "real_list_path": None,
    "fake_list_path": None,
    "wang2020_data_path": None,
    "data_mode": "ours",
    "data_label": "train",
    "weight_decay": 0.0,
    "class_bal": False,
    # misc
    "batch_size": 32,
    "grad_accum_steps": 1,
    "loadSize": 256,
    "cropSize": 224,
    "gpu_ids": "0",
    "name": "experiment_name",
    "num_threads": 4,
    "checkpoints_dir": "./checkpoints",
    "serial_batches": False,
    "resize_or_crop": "scale_and_crop",
    "no_flip": False,
    "init_type": "normal",
    "init_gain": 0.02,
    "suffix": "",
    # train options
    "earlystop_epoch": 5,
    "data_aug": False,
    "optim": "adam",
    "new_optim": False,
    "loss_freq": 400,
    "save_epoch_freq": 1,
    "epoch_count": 1,
    "last_epoch": -1,
    "train_split": "train",
    "val_split": "val",
    "niter": 50,
    "beta1": 0.9,
    "lr": 0.0002,
    "residual_lr": None,
    "grad_clip": 1.0,
    "use_amp": False,
    "clear_cache_each_epoch": True,
    "no_clear_cache_each_epoch": False,
    "seed": 42,
    "deterministic": True,
    "no_deterministic": False,
    "lr_scheduler": "none",
    "cosine_t_max": None,
    "cosine_min_lr": 1e-6,
    "lr_warmup_steps": 0.0,
    "submit_each_epoch": False,
    "submit_builder_script": "/work/u1657859/Jess/app4/scripts/build_submission_mev1_strict.sh",
    "submit_out_prefix": None,
    "submit_out_dir": "/work/u1657859/Jess/app/local_checkpoints/submissions",
    "submit_app": None,
    "submit_python": None,
    "submit_val_dir": "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final",
    "submit_batch_size": 64,
    "submit_num_workers": 4,
    "submit_tta_mode": "none",
    "submit_tta_logit_agg": "mean",
}

# ---------------- TrainOptions Overrides ----------------
OPT_OVERRIDES = {
    # training/core
    "seed": SEED,
    "deterministic": DETERMINISTIC,
    "use_amp": USE_AMP,
    "clear_cache_each_epoch": CLEAR_CACHE_EACH_EPOCH,
    "lr_scheduler": LR_SCHEDULER,
    "cosine_t_max": COSINE_T_MAX,
    "cosine_min_lr": COSINE_MIN_LR,
    "lr_warmup_steps": LR_WARMUP_STEPS,
    "submit_each_epoch": SUBMIT_EACH_EPOCH,
    "submit_builder_script": SUBMIT_BUILDER_SCRIPT,
    "submit_out_prefix": SUBMIT_OUT_PREFIX,
    "submit_out_dir": SUBMIT_OUT_DIR,
    "submit_app": SUBMIT_APP,
    "submit_python": SUBMIT_PYTHON,
    "submit_val_dir": SUBMIT_VAL_DIR,
    "submit_batch_size": SUBMIT_BATCH_SIZE,
    "submit_num_workers": SUBMIT_NUM_WORKERS,
    "submit_tta_mode": SUBMIT_TTA_MODE,
    "submit_tta_logit_agg": SUBMIT_TTA_LOGIT_AGG,
    "earlystop_epoch": EARLYSTOP_EPOCH,
    "loss_freq": LOSS_FREQ,
    "save_epoch_freq": SAVE_EPOCH_FREQ,
    "batch_size": TRAIN_BATCH_SIZE,
    "grad_accum_steps": GRAD_ACCUM_STEPS,
    "num_threads": NUM_WORKERS,
    "niter": NITER,
    "lr": LR,
    "residual_lr": RESIDUAL_LR,
    "weight_decay": WEIGHT_DECAY,
    "grad_clip": GRAD_CLIP,
    "checkpoints_dir": CHECKPOINTS_DIR,
    "name": EXP_NAME,
    "gpu_ids": "0",
    # data
    "train_csv": TRAIN_CSV,
    "val_data_root": VAL_DATA_ROOT,
    "train_split": TRAIN_SPLIT,
    "val_split": VAL_SPLIT,
    "train_frame_num": TRAIN_FRAME_NUM,
    "val_frame_num": VAL_FRAME_NUM,
    "train_frame_sample_mode": TRAIN_FRAME_SAMPLE_MODE,
    "val_frame_sample_mode": VAL_FRAME_SAMPLE_MODE,
    "frame_sample_seed": FRAME_SAMPLE_SEED,
    "image_size": IMAGE_SIZE,
    "on_error": ON_ERROR,
    "hflip_prob": HFLIP_PROB,
    "blur_prob": BLUR_PROB,
    "blur_sig": BLUR_SIG,
    "jpg_prob": JPG_PROB,
    "jpg_qual": JPG_QUAL,
    "augment_prob": AUGMENT_PROB,
    "use_pmm_aug": USE_PMM_AUG,
    "pmm_group_prob": PMM_GROUP_PROB,
    "pmm_pdm_type": PMM_PDM_TYPE,
    "pmm_strength": PMM_STRENGTH,
    "pmm_use_beta": PMM_USE_BETA,
    "pmm_beta_a": PMM_BETA_A,
    "pmm_beta_b": PMM_BETA_B,
    "pmm_distractor_p": PMM_DISTRACTOR_P,
    "quality_balance": QUALITY_BALANCE,
    "use_stratified_sampler": USE_STRATIFIED_SAMPLER,
    "stratified_sampler_key": STRATIFIED_SAMPLER_KEY,
    "use_reweighted_erm": USE_REWEIGHTED_ERM,
    "reweight_use_sqrt": REWEIGHT_USE_SQRT,
    "reweight_clip_min": REWEIGHT_CLIP_MIN,
    "reweight_clip_max": REWEIGHT_CLIP_MAX,
    "val_quality_diag": VAL_QUALITY_DIAG,
    "quality_bins": QUALITY_BINS,
    "quality_max_side": QUALITY_MAX_SIDE,
    "quality_metric": QUALITY_METRIC,
    "quality_clipiqa_batch_size": QUALITY_CLIPIQA_BATCH_SIZE,
    "quality_clipiqa_device": QUALITY_CLIPIQA_DEVICE,
    "quality_clipiqa_weights": QUALITY_CLIPIQA_WEIGHTS,
    "quality_cache_path": QUALITY_CACHE_PATH,
    "quality_cache_train_path": QUALITY_CACHE_TRAIN_PATH,
    "quality_cache_val_path": QUALITY_CACHE_VAL_PATH,
    "quality_cuts": QUALITY_CUTS,
    "degrade_bins": DEGRADE_BINS,
    "degrade_cache_path": DEGRADE_CACHE_PATH,
    "degrade_cache_train_path": DEGRADE_CACHE_TRAIN_PATH,
    "degrade_cache_val_path": DEGRADE_CACHE_VAL_PATH,
    "aug_before_resize": AUG_BEFORE_RESIZE,
    # val micro-tta
    "val_micro_tta": VAL_MICRO_TTA,
    "val_micro_tta_k": VAL_MICRO_TTA_K,
    "val_micro_tta_seed": VAL_MICRO_TTA_SEED,
    "val_micro_tta_blur_max_sig": VAL_MICRO_TTA_BLUR_MAX_SIG,
    "val_micro_tta_jpeg_qual": VAL_MICRO_TTA_JPEG_QUAL,
    "val_micro_tta_contrast": VAL_MICRO_TTA_CONTRAST,
    "val_micro_tta_gamma": VAL_MICRO_TTA_GAMMA,
    "val_micro_tta_resize_jitter": VAL_MICRO_TTA_RESIZE_JITTER,
    # tri-view consistency
    "tri_consistency_enable": TRI_CONSISTENCY_ENABLE,
    "tri_consistency_apply_prob": TRI_CONSISTENCY_APPLY_PROB,
    "tri_consistency_ce_weight": TRI_CONSISTENCY_CE_WEIGHT,
    "tri_consistency_align_weight": TRI_CONSISTENCY_ALIGN_WEIGHT,
    "tri_consistency_blur_sig": TRI_CONSISTENCY_BLUR_SIG,
    "tri_consistency_jpeg_qual": TRI_CONSISTENCY_JPEG_QUAL,
    "tri_consistency_resize_jitter": TRI_CONSISTENCY_RESIZE_JITTER,
    "tri_consistency_noise_std": TRI_CONSISTENCY_NOISE_STD,
    "tri_consistency_unsharp_amount": TRI_CONSISTENCY_UNSHARP_AMOUNT,
    "tri_consistency_restore_mix": TRI_CONSISTENCY_RESTORE_MIX,
    # noise-robust fine-tune
    "noise_robust_enable": NOISE_ROBUST_ENABLE,
    "noise_robust_warmup_steps": NOISE_ROBUST_WARMUP_STEPS,
    "noise_robust_conf_thresh": NOISE_ROBUST_CONF_THRESH,
    "noise_robust_min_weight": NOISE_ROBUST_MIN_WEIGHT,
    "noise_robust_soft_weight": NOISE_ROBUST_SOFT_WEIGHT,
    "noise_robust_label_smooth": NOISE_ROBUST_LABEL_SMOOTH,
    "noise_robust_group_key": NOISE_ROBUST_GROUP_KEY,
    "noise_robust_group_momentum": NOISE_ROBUST_GROUP_MOMENTUM,
    # groupdro
    "use_groupdro": USE_GROUPDRO,
    "groupdro_group_key": GROUPDRO_GROUP_KEY,
    "groupdro_class_balance": GROUPDRO_CLASS_BALANCE,
    "groupdro_num_groups": GROUPDRO_NUM_GROUPS,
    "groupdro_eta": GROUPDRO_ETA,
    "groupdro_ema_beta": GROUPDRO_EMA_BETA,
    "groupdro_q_floor": GROUPDRO_Q_FLOOR,
    "groupdro_logq_clip": GROUPDRO_LOGQ_CLIP,
    "groupdro_min_count_group": GROUPDRO_MIN_COUNT_GROUP,
    "groupdro_min_count_class": GROUPDRO_MIN_COUNT_CLASS,
    "groupdro_warmup_steps": GROUPDRO_WARMUP_STEPS,
    "groupdro_update_every": GROUPDRO_UPDATE_EVERY,
    # model
    "model_family": MODEL_FAMILY,
    "clip_model": CLIP_MODEL,
    "fix_backbone": FIX_BACKBONE,
    "svd_cache_enable": SVD_CACHE_ENABLE,
    "svd_cache_dir": SVD_CACHE_DIR,
    "svd_init_device": SVD_INIT_DEVICE,
    "svd_show_progress": SVD_SHOW_PROGRESS,
    "multi_expert_enable": MULTI_EXPERT_ENABLE,
    "multi_expert_k": MULTI_EXPERT_K,
    "multi_expert_route": MULTI_EXPERT_ROUTE,
    "multi_expert_route_temp": MULTI_EXPERT_ROUTE_TEMP,
    "multi_expert_route_detach": MULTI_EXPERT_ROUTE_DETACH,
    "multi_expert_hybrid_mix": MULTI_EXPERT_HYBRID_MIX,
    "multi_expert_route_sup_lambda": MULTI_EXPERT_ROUTE_SUP_LAMBDA,
    "multi_expert_balance_lambda": MULTI_EXPERT_BALANCE_LAMBDA,
    "multi_expert_div_lambda": MULTI_EXPERT_DIV_LAMBDA,
    "inv_mev1_enable": INV_MEV1_ENABLE,
    "inv_mev1_anchor_ckpt": INV_MEV1_ANCHOR_CKPT,
    "inv_mev1_teacher_ckpt": INV_MEV1_TEACHER_CKPT,
    "inv_mev1_div_lambda": INV_MEV1_DIV_LAMBDA,
    "inv_mev1_div_target_cos": INV_MEV1_DIV_TARGET_COS,
    "inv_mev1_kd_lambda": INV_MEV1_KD_LAMBDA,
    "inv_mev1_kd_temp": INV_MEV1_KD_TEMP,
    "inv_mev1_teacher_every": INV_MEV1_TEACHER_EVERY,
    "inv_mev1_warmup_steps": INV_MEV1_WARMUP_STEPS,
    "inv_mev1_ramp_steps": INV_MEV1_RAMP_STEPS,
    "dino_source": DINO_SOURCE,
    "dino_repo": DINO_REPO,
    "dino_model_name": DINO_MODEL_NAME,
    "dino_pretrained": DINO_PRETRAINED,
    "dino_local_weights": DINO_LOCAL_WEIGHTS,
    "dino_force_imagenet_norm": DINO_FORCE_IMAGENET_NORM,
    # patch pooling
    "patch_aux_lambda": PATCH_AUX_LAMBDA,
    "patch_pool_tau": PATCH_POOL_TAU,
    "patch_pool_mode": PATCH_POOL_MODE,
    "patch_trim_p": PATCH_TRIM_P,
    "patch_quality": PATCH_QUALITY,
    "patch_pool_gamma": PATCH_POOL_GAMMA,
    "patch_warmup_steps": PATCH_WARMUP_STEPS,
    "patch_ramp_steps": PATCH_RAMP_STEPS,
    # regularizers
    "osd_regs": OSD_REGS,
    "lambda_orth": LAMBDA_ORTH,
    "lambda_ksv": LAMBDA_KSV,
    "reg_warmup_steps": REG_WARMUP_STEPS,
    "reg_ramp_steps": REG_RAMP_STEPS,
}


def _set_gpu_ids(opt, gpu_ids_value):
    if gpu_ids_value is None:
        return
    import torch
    if isinstance(gpu_ids_value, (list, tuple)):
        opt.gpu_ids = list(gpu_ids_value)
    else:
        str_ids = str(gpu_ids_value).split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            if str_id.strip() == "":
                continue
            idx = int(str_id)
            if idx >= 0:
                opt.gpu_ids.append(idx)
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_ids[0])


def apply_overrides(opt, overrides, defaults=None):
    if defaults:
        for key, value in defaults.items():
            if not hasattr(opt, key):
                setattr(opt, key, value)
    for key, value in overrides.items():
        if value is not None:
            setattr(opt, key, value)
        elif not hasattr(opt, key):
            setattr(opt, key, None)
    _set_gpu_ids(opt, getattr(opt, "gpu_ids", None))
