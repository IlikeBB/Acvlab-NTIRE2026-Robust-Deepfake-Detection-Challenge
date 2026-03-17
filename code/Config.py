# ---------------- Environment ----------------
CUDA_DEVICE_ORDER = "PCI_BUS_ID"
CUDA_VISIBLE_DEVICES = "0"

# ---------------- Reproducibility ----------------
SEED = 418
DETERMINISTIC = True

# ---------------- Experiment ----------------
EXP_NAME = "Effort_aug_pooling_baseline_DF40_tuneoff-AUXLoss"
CHECKPOINTS_DIR = "./checkpoints"

# ---------------- Training ----------------
TRAIN_BATCH_SIZE = 32
NUM_WORKERS = 4
NITER = 6
MAX_TRAIN_STEPS_PER_EPOCH = 0
LR = 2e-4
RESIDUAL_LR = None
WEIGHT_DECAY = 0.0
USE_AMP = True
GRAD_CLIP = 1.0
CLEAR_CACHE_EACH_EPOCH = True
LR_SCHEDULER = "cosine"  # none | cosine
COSINE_T_MAX = None
COSINE_MIN_LR = 1e-6
EARLYSTOP_EPOCH = 5
LOSS_FREQ = 200
SAVE_EPOCH_FREQ = 1

# ---------------- Data Paths ----------------
TRAIN_CSV = "/ssd1/chihyi111/Dataset/DeepFake_Dataset/DF40_frames/image_frame_list_face.csv"
VAL_DATA_ROOT = "/ssd2/ntire_RDFC_2026_cvpr/challange_data/training_data_final"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TRAIN_FRAME_NUM = 8
VAL_FRAME_NUM = 1
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORMALIZE_STD = [0.26862954, 0.26130258, 0.27577711]
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
STRATIFIED_SAMPLER_KEY = "stratum_id"  # stratum_id | quality_bin (label-free)
USE_REWEIGHTED_ERM = False
REWEIGHT_USE_SQRT = False
REWEIGHT_CLIP_MIN = 0.2
REWEIGHT_CLIP_MAX = 5.0
VAL_QUALITY_DIAG = True
QUALITY_BINS = 3
QUALITY_MAX_SIDE = 256
QUALITY_CACHE_PATH = None
QUALITY_CACHE_TRAIN_PATH = "cache/q_train.csv"
QUALITY_CACHE_VAL_PATH = "cache/q_val.csv"
QUALITY_CUTS = None  # fixed cutpoints from train; populated at runtime and saved to config.json

# ---------------- GroupDRO (robust worst-group) ----------------
# Group key should be label-free if your goal is shortcut reduction (e.g. "quality_bin").
USE_GROUPDRO = False
GROUPDRO_GROUP_KEY = "quality_bin"   # quality_bin | stratum_id
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
GROUPDRO_Q_TEMP = 1.0                # temperature for q update; >=1 flattens q
GROUPDRO_Q_MIX = 0.0                 # q <- (1-mix) q_old + mix q_new, to reduce variance
GROUPDRO_USE_CVAR = False            # enable within-group CVaR tail focus
GROUPDRO_CVAR_ALPHA = 0.30           # tail fraction in (0,1]
GROUPDRO_CVAR_LAMBDA = 1.0           # max mixing weight for tail component (0..1)
GROUPDRO_CVAR_WARMUP_STEPS = 0       # warmup before enabling CVaR
GROUPDRO_CVAR_RAMP_STEPS = 0         # ramp length to max lambda
GROUPDRO_CVAR_CAP_P = 0.99           # winsorize/trim threshold quantile
GROUPDRO_CVAR_TRIM = False           # True: trim > cap_p; False: winsorize cap
GROUPDRO_CVAR_MIN_K = 1              # minimum tail sample count per group/class

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

# E3/E4 switch: False -> resize then augment, True -> augment then resize
AUG_BEFORE_RESIZE = False

# ---------------- Model ----------------
CLIP_MODEL = "/work/u1657859/ntire_RDFC_2026_cvpr/clip-vit-large-patch14/"
FIX_BACKBONE = True
SVD_CACHE_ENABLE = True
SVD_CACHE_DIR = "cache"
SVD_INIT_DEVICE = "auto"   # auto|cpu|cuda
SVD_SHOW_PROGRESS = True
SVD_RANK_CAP = None
ADAPTER_LAST_N_LAYERS = 0
SVD_RESIDUAL_MODE = "add"  # add | gated | softplus
SVD_RESIDUAL_GATE_INIT = 0.0
SVD_RESIDUAL_TRAIN_MODE = "full"  # full | sigma_only | uv_only
PEFT_UNFREEZE_LAST_N_BLOCKS = 0
PEFT_UNFREEZE_LAYERNORM = True
PEFT_UNFREEZE_BIAS = False
MULTI_EXPERT_ENABLE = False
MULTI_EXPERT_K = 1
MULTI_EXPERT_ROUTE = "quality_bin"  # quality_bin|quality_soft|uniform|router|hybrid|degrade_router|degrade_hybrid
MULTI_EXPERT_ROUTE_TEMP = 1.0
MULTI_EXPERT_ROUTE_DETACH = True
MULTI_EXPERT_HYBRID_MIX = 0.0
MULTI_EXPERT_ROUTE_SUP_LAMBDA = 0.0
MULTI_EXPERT_BALANCE_LAMBDA = 0.0
MULTI_EXPERT_DIV_LAMBDA = 0.0
RANK_LOSS_ENABLE = False
RANK_LOSS_LAMBDA = 0.0
RANK_LOSS_MARGIN = 0.2
RANK_LOSS_MODE = "logistic"  # logistic|hinge
RANK_LOSS_MAX_PAIRS = 1024
RANK_LOSS_WARMUP_STEPS = 0.0
RANK_LOSS_RAMP_STEPS = 0.0

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
    "clip_model": None,
    "svd_cache_enable": True,
    "svd_cache_dir": "cache",
    "svd_init_device": "auto",
    "svd_show_progress": True,
    "svd_rank_cap": None,
    "adapter_last_n_layers": 0,
    "svd_residual_mode": "add",
    "svd_residual_gate_init": 0.0,
    "svd_residual_train_mode": "full",
    "peft_unfreeze_last_n_blocks": 0,
    "peft_unfreeze_layernorm": True,
    "peft_unfreeze_bias": False,
    "multi_expert_enable": False,
    "multi_expert_k": 1,
    "multi_expert_route": "quality_bin",
    "multi_expert_route_temp": 1.0,
    "multi_expert_route_detach": True,
    "multi_expert_hybrid_mix": 0.0,
    "multi_expert_route_sup_lambda": 0.0,
    "multi_expert_balance_lambda": 0.0,
    "multi_expert_div_lambda": 0.0,
    "rank_loss_enable": False,
    "rank_loss_lambda": 0.0,
    "rank_loss_margin": 0.2,
    "rank_loss_mode": "logistic",
    "rank_loss_max_pairs": 1024,
    "rank_loss_warmup_steps": 0.0,
    "rank_loss_ramp_steps": 0.0,
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
    "image_size": 224,
    "normalize_mean": [0.48145466, 0.4578275, 0.40821073],
    "normalize_std": [0.26862954, 0.26130258, 0.27577711],
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
    "quality_cache_path": None,
    "quality_cache_train_path": None,
    "quality_cache_val_path": None,
    "quality_cuts": None,
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
    "groupdro_q_temp": 1.0,
    "groupdro_q_mix": 0.0,
    "groupdro_use_cvar": False,
    "groupdro_cvar_alpha": 0.30,
    "groupdro_cvar_lambda": 1.0,
    "groupdro_cvar_warmup_steps": 0,
    "groupdro_cvar_ramp_steps": 0,
    "groupdro_cvar_cap_p": 0.99,
    "groupdro_cvar_trim": False,
    "groupdro_cvar_min_k": 1,
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
    "max_train_steps_per_epoch": 0,
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
    "earlystop_epoch": EARLYSTOP_EPOCH,
    "loss_freq": LOSS_FREQ,
    "save_epoch_freq": SAVE_EPOCH_FREQ,
    "batch_size": TRAIN_BATCH_SIZE,
    "num_threads": NUM_WORKERS,
    "niter": NITER,
    "max_train_steps_per_epoch": MAX_TRAIN_STEPS_PER_EPOCH,
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
    "image_size": IMAGE_SIZE,
    "normalize_mean": NORMALIZE_MEAN,
    "normalize_std": NORMALIZE_STD,
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
    "quality_cache_path": QUALITY_CACHE_PATH,
    "quality_cache_train_path": QUALITY_CACHE_TRAIN_PATH,
    "quality_cache_val_path": QUALITY_CACHE_VAL_PATH,
    "quality_cuts": QUALITY_CUTS,
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
    "groupdro_q_temp": GROUPDRO_Q_TEMP,
    "groupdro_q_mix": GROUPDRO_Q_MIX,
    "groupdro_use_cvar": GROUPDRO_USE_CVAR,
    "groupdro_cvar_alpha": GROUPDRO_CVAR_ALPHA,
    "groupdro_cvar_lambda": GROUPDRO_CVAR_LAMBDA,
    "groupdro_cvar_warmup_steps": GROUPDRO_CVAR_WARMUP_STEPS,
    "groupdro_cvar_ramp_steps": GROUPDRO_CVAR_RAMP_STEPS,
    "groupdro_cvar_cap_p": GROUPDRO_CVAR_CAP_P,
    "groupdro_cvar_trim": GROUPDRO_CVAR_TRIM,
    "groupdro_cvar_min_k": GROUPDRO_CVAR_MIN_K,
    # model
    "clip_model": CLIP_MODEL,
    "fix_backbone": FIX_BACKBONE,
    "svd_cache_enable": SVD_CACHE_ENABLE,
    "svd_cache_dir": SVD_CACHE_DIR,
    "svd_init_device": SVD_INIT_DEVICE,
    "svd_show_progress": SVD_SHOW_PROGRESS,
    "svd_rank_cap": SVD_RANK_CAP,
    "adapter_last_n_layers": ADAPTER_LAST_N_LAYERS,
    "svd_residual_mode": SVD_RESIDUAL_MODE,
    "svd_residual_gate_init": SVD_RESIDUAL_GATE_INIT,
    "svd_residual_train_mode": SVD_RESIDUAL_TRAIN_MODE,
    "peft_unfreeze_last_n_blocks": PEFT_UNFREEZE_LAST_N_BLOCKS,
    "peft_unfreeze_layernorm": PEFT_UNFREEZE_LAYERNORM,
    "peft_unfreeze_bias": PEFT_UNFREEZE_BIAS,
    "multi_expert_enable": MULTI_EXPERT_ENABLE,
    "multi_expert_k": MULTI_EXPERT_K,
    "multi_expert_route": MULTI_EXPERT_ROUTE,
    "multi_expert_route_temp": MULTI_EXPERT_ROUTE_TEMP,
    "multi_expert_route_detach": MULTI_EXPERT_ROUTE_DETACH,
    "multi_expert_hybrid_mix": MULTI_EXPERT_HYBRID_MIX,
    "multi_expert_route_sup_lambda": MULTI_EXPERT_ROUTE_SUP_LAMBDA,
    "multi_expert_balance_lambda": MULTI_EXPERT_BALANCE_LAMBDA,
    "multi_expert_div_lambda": MULTI_EXPERT_DIV_LAMBDA,
    "rank_loss_enable": RANK_LOSS_ENABLE,
    "rank_loss_lambda": RANK_LOSS_LAMBDA,
    "rank_loss_margin": RANK_LOSS_MARGIN,
    "rank_loss_mode": RANK_LOSS_MODE,
    "rank_loss_max_pairs": RANK_LOSS_MAX_PAIRS,
    "rank_loss_warmup_steps": RANK_LOSS_WARMUP_STEPS,
    "rank_loss_ramp_steps": RANK_LOSS_RAMP_STEPS,
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
