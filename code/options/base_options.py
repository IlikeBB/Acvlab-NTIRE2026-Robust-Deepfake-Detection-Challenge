import argparse
import os
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--head_type', default='attention')
        parser.add_argument('--no_augment', action='store_true', default=False)
        parser.add_argument('--patch_base', action='store_true', default=False)
        parser.add_argument('--shuffle', action='store_true', default=False)
        parser.add_argument('--shuffle_times', type=int, default=1)
        parser.add_argument('--original_times', type=int, default=1)
        parser.add_argument('--patch_size', nargs='+', type=int, default=[14])
        parser.add_argument('--resume_path', type=str, default=None)
        parser.add_argument('--load_whole_model', action='store_true', default=False)
        parser.add_argument('--scale', type=float, default=1.0)

        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
        parser.add_argument('--fix_backbone', action='store_true')

        # Effort model options
        parser.add_argument('--clip_model', type=str, default=None)
        parser.add_argument('--svd_cache_enable', action='store_true', default=True)
        parser.add_argument('--svd_cache_dir', type=str, default='cache')
        parser.add_argument('--svd_init_device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
        parser.add_argument('--svd_show_progress', action='store_true', default=True)
        parser.add_argument('--svd_rank_cap', type=int, default=None)
        parser.add_argument('--adapter_last_n_layers', type=int, default=0, help='If >0, inject SVD adapters only in last N transformer blocks.')
        parser.add_argument('--svd_residual_mode', type=str, default='add', choices=['add', 'gated', 'softplus'])
        parser.add_argument('--svd_residual_gate_init', type=float, default=0.0)
        parser.add_argument(
            '--svd_residual_train_mode',
            type=str,
            default='full',
            choices=['full', 'sigma_only', 'uv_only'],
            help='Trainable SVD residual factors: full(U/S/V), sigma_only(S), uv_only(U/V).',
        )
        parser.add_argument('--peft_unfreeze_last_n_blocks', type=int, default=0, help='When fix_backbone=True, also unfreeze last N blocks.')
        parser.add_argument('--peft_unfreeze_layernorm', action='store_true', default=True)
        parser.add_argument('--no_peft_unfreeze_layernorm', dest='peft_unfreeze_layernorm', action='store_false')
        parser.add_argument('--peft_unfreeze_bias', action='store_true', default=False)
        parser.add_argument('--multi_expert_enable', action='store_true', default=False)
        parser.add_argument('--multi_expert_k', type=int, default=1)
        parser.add_argument(
            '--multi_expert_route',
            type=str,
            default='quality_bin',
            choices=['quality_bin', 'quality_soft', 'uniform', 'router', 'hybrid', 'degrade_router', 'degrade_hybrid'],
        )
        parser.add_argument('--multi_expert_route_temp', type=float, default=1.0)
        parser.add_argument('--multi_expert_route_detach', action='store_true', default=True)
        parser.add_argument('--no_multi_expert_route_detach', dest='multi_expert_route_detach', action='store_false')
        parser.add_argument('--multi_expert_hybrid_mix', type=float, default=0.0)
        parser.add_argument('--multi_expert_route_sup_lambda', type=float, default=0.0)
        parser.add_argument('--multi_expert_balance_lambda', type=float, default=0.0)
        parser.add_argument('--multi_expert_div_lambda', type=float, default=0.0)
        # Pairwise ranking loss (AUC surrogate)
        parser.add_argument('--rank_loss_enable', action='store_true', default=False)
        parser.add_argument('--rank_loss_lambda', type=float, default=0.0)
        parser.add_argument('--rank_loss_margin', type=float, default=0.2)
        parser.add_argument('--rank_loss_mode', type=str, default='logistic', choices=['logistic', 'hinge'])
        parser.add_argument('--rank_loss_max_pairs', type=int, default=1024)
        parser.add_argument('--rank_loss_warmup_steps', type=float, default=0.0)
        parser.add_argument('--rank_loss_ramp_steps', type=float, default=0.0)

        # Patch pooling
        parser.add_argument('--patch_pool_tau', type=float, default=1.5)
        parser.add_argument('--patch_pool_mode', type=str, default='lse', choices=['lse'])
        parser.add_argument('--patch_trim_p', type=float, default=0.2)
        parser.add_argument('--patch_quality', type=str, default='none', choices=['none', 'cos', 'cos_norm'])
        parser.add_argument('--patch_pool_gamma', type=float, default=0.0)
        parser.add_argument('--patch_warmup_steps', type=float, default=0.0)
        parser.add_argument('--patch_ramp_steps', type=float, default=0.0)

        # OSD regularizers
        parser.add_argument('--osd_regs', action='store_true', default=True)
        parser.add_argument('--no_osd_regs', action='store_true')
        parser.add_argument('--lambda_orth', type=float, default=5e-2)
        parser.add_argument('--lambda_ksv', type=float, default=1e-2)
        parser.add_argument('--reg_warmup_steps', type=float, default=0.0)
        parser.add_argument('--reg_ramp_steps', type=float, default=0.0)

        # Data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0.2)
        parser.add_argument('--blur_sig', type=float, nargs='+', default=[0.0, 2.0])
        parser.add_argument('--jpg_prob', type=float, default=0.2)
        parser.add_argument('--jpg_method', default='cv2,pil')
        parser.add_argument('--jpg_qual', type=int, nargs='+', default=[30, 100])
        parser.add_argument('--hflip_prob', type=float, default=0.2)
        parser.add_argument('--augment_prob', type=float, default=None)
        parser.add_argument('--use_pmm_aug', action='store_true', default=False)
        parser.add_argument('--pmm_group_prob', type=float, default=0.0)
        parser.add_argument('--pmm_pdm_type', type=str, default='ours')
        parser.add_argument('--pmm_strength', type=float, default=0.5)
        parser.add_argument('--pmm_use_beta', action='store_true', default=False)
        parser.add_argument('--pmm_beta_a', type=float, default=0.5)
        parser.add_argument('--pmm_beta_b', type=float, default=0.5)
        parser.add_argument('--pmm_distractor_p', type=float, default=0.0)

        # Data inputs
        parser.add_argument('--csv', type=str, default=None, help='CSV with frames_dir/split/label')
        parser.add_argument('--train_csv', type=str, default=None, help='CSV with path,label or frames_dir')
        parser.add_argument('--val_csv', type=str, default=None, help='CSV with path,label or frames_dir')
        parser.add_argument('--folder_root', type=str, default=None, help='Folder root with train/val/test')
        parser.add_argument('--val_data_root', type=str, default=None, help='Validation folder(s) with real/fake in filenames')
        parser.add_argument('--train_frame_num', type=int, default=8)
        parser.add_argument('--val_frame_num', type=int, default=1)
        parser.add_argument('--image_size', type=int, default=224)
        parser.add_argument('--normalize_mean', type=float, nargs='+', default=None, help='Input normalize mean (RGB).')
        parser.add_argument('--normalize_std', type=float, nargs='+', default=None, help='Input normalize std (RGB).')
        parser.add_argument('--on_error', type=str, default='skip', choices=['skip', 'raise'])
        parser.add_argument('--quality_balance', action='store_true', default=False)
        parser.add_argument('--use_stratified_sampler', action='store_true', default=False)
        parser.add_argument('--stratified_sampler_key', type=str, default='stratum_id', choices=['stratum_id', 'quality_bin'])
        parser.add_argument('--use_reweighted_erm', action='store_true', default=False)
        parser.add_argument('--reweight_use_sqrt', action='store_true', default=False)
        parser.add_argument('--reweight_clip_min', type=float, default=0.2)
        parser.add_argument('--reweight_clip_max', type=float, default=5.0)
        parser.add_argument('--val_quality_diag', action='store_true', default=True)
        parser.add_argument('--quality_bins', type=int, default=3)
        parser.add_argument('--quality_max_side', type=int, default=256)
        parser.add_argument('--quality_cache_path', type=str, default=None)
        parser.add_argument('--quality_cache_train_path', type=str, default=None)
        parser.add_argument('--quality_cache_val_path', type=str, default=None)
        parser.add_argument('--quality_cuts', type=float, nargs='+', default=None)
        parser.add_argument('--aug_before_resize', action='store_true', default=False)

        # Validation micro-TTA (tensor-space)
        parser.add_argument('--val_micro_tta', action='store_true', default=False)
        parser.add_argument('--val_micro_tta_k', type=int, default=9)
        parser.add_argument('--val_micro_tta_seed', type=int, default=0)
        parser.add_argument('--val_micro_tta_blur_max_sig', type=float, default=1.5)
        parser.add_argument('--val_micro_tta_jpeg_qual', type=int, nargs='+', default=[60, 95])
        parser.add_argument('--val_micro_tta_contrast', type=float, nargs='+', default=[0.9, 1.1])
        parser.add_argument('--val_micro_tta_gamma', type=float, nargs='+', default=[0.9, 1.1])
        parser.add_argument('--val_micro_tta_resize_jitter', type=int, default=8)

        # GroupDRO (robust worst-group optimization)
        parser.add_argument('--use_groupdro', action='store_true', default=False)
        parser.add_argument('--groupdro_group_key', type=str, default='quality_bin', choices=['quality_bin', 'stratum_id'])
        parser.add_argument('--groupdro_class_balance', action='store_true', default=False)
        parser.add_argument('--groupdro_num_groups', type=int, default=None)
        parser.add_argument('--groupdro_eta', type=float, default=0.10)
        parser.add_argument('--groupdro_ema_beta', type=float, default=0.90)
        parser.add_argument('--groupdro_q_floor', type=float, default=0.00)
        parser.add_argument('--groupdro_logq_clip', type=float, default=None)
        parser.add_argument('--groupdro_min_count_group', type=int, default=1)
        parser.add_argument('--groupdro_min_count_class', type=int, default=1)
        parser.add_argument('--groupdro_warmup_steps', type=int, default=0)
        parser.add_argument('--groupdro_update_every', type=int, default=1)
        parser.add_argument('--groupdro_q_temp', type=float, default=1.0)
        parser.add_argument('--groupdro_q_mix', type=float, default=0.0)

        # GroupDRO: within-group CVaR tail focus (optional)
        parser.add_argument('--groupdro_use_cvar', action='store_true', default=False)
        parser.add_argument('--groupdro_cvar_alpha', type=float, default=0.0)
        parser.add_argument('--groupdro_cvar_lambda', type=float, default=1.0)
        parser.add_argument('--groupdro_cvar_warmup_steps', type=float, default=0.0)
        parser.add_argument('--groupdro_cvar_ramp_steps', type=float, default=0.0)
        parser.add_argument('--groupdro_cvar_cap_p', type=float, default=0.99)
        parser.add_argument('--groupdro_cvar_trim', action='store_true', default=False)
        parser.add_argument('--groupdro_cvar_min_k', type=int, default=1)

        # Legacy data options (kept for compatibility)
        parser.add_argument('--real_list_path', default=None)
        parser.add_argument('--fake_list_path', default=None)
        parser.add_argument('--wang2020_data_path', default=None)
        parser.add_argument('--data_mode', default='ours')
        parser.add_argument('--data_label', default='train')
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--class_bal', action='store_true')

        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--loadSize', type=int, default=256)
        parser.add_argument('--cropSize', type=int, default=224)
        parser.add_argument('--gpu_ids', type=str, default='0')
        parser.add_argument('--name', type=str, default='experiment_name')
        parser.add_argument('--num_threads', type=int, default=4)
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        parser.add_argument('--serial_batches', action='store_true')
        parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop')
        parser.add_argument('--no_flip', action='store_true')
        parser.add_argument('--init_type', type=str, default='normal')
        parser.add_argument('--init_gain', type=float, default=0.02)
        parser.add_argument('--suffix', default='', type=str)
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\\n')

    def parse(self, print_options=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = str(opt.gpu_ids).split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            if str_id.strip() == '':
                continue
            idx = int(str_id)
            if idx >= 0:
                opt.gpu_ids.append(idx)
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu_ids[0])

        def _split_csv(value):
            if isinstance(value, str):
                return [v.strip() for v in value.split(',') if v.strip()]
            if isinstance(value, (list, tuple)):
                return list(value)
            return [value]

        def _as_float_list(value):
            if isinstance(value, (list, tuple)):
                return [float(v) for v in value]
            if isinstance(value, str):
                return [float(v) for v in value.split(',') if v.strip()]
            return [float(value)]

        def _as_int_list(value):
            if isinstance(value, (list, tuple)):
                return [int(v) for v in value]
            if isinstance(value, str):
                return [int(v) for v in value.split(',') if v.strip()]
            return [int(value)]

        opt.rz_interp = _split_csv(opt.rz_interp)
        opt.blur_sig = _as_float_list(opt.blur_sig)
        opt.jpg_method = _split_csv(opt.jpg_method)
        opt.jpg_qual = _as_int_list(opt.jpg_qual)
        if isinstance(opt.patch_size, int):
            opt.patch_size = [opt.patch_size]

        if opt.no_osd_regs:
            opt.osd_regs = False

        self.opt = opt
        return self.opt
