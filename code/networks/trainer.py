import logging
import math
import os
from contextlib import nullcontext
import torch
import torch.nn.functional as F

try:
    from torch.amp import autocast as _autocast_new
except Exception:
    _autocast_new = None

try:
    from torch.amp import GradScaler as _GradScalerNew
except Exception:
    _GradScalerNew = None

try:
    from torch.cuda.amp import autocast as _autocast_cuda
except Exception:
    _autocast_cuda = None

try:
    from torch.cuda.amp import GradScaler as _GradScalerCuda
except Exception:
    _GradScalerCuda = None

from effort.svd_residual import SVDResidualLinear
from effort.utils import load_ckpt, save_checkpoint
from models import get_model

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _autocast_ctx(*, device_type: str, enabled: bool):
    if _autocast_new is not None:
        return _autocast_new(device_type=device_type, enabled=enabled)
    if device_type == "cuda" and _autocast_cuda is not None:
        return _autocast_cuda(enabled=enabled)
    return nullcontext()


def _make_grad_scaler(*, enabled: bool):
    if _GradScalerNew is not None:
        try:
            return _GradScalerNew(device_type="cuda", enabled=enabled)
        except TypeError:
            return _GradScalerNew(enabled=enabled)
    if _GradScalerCuda is not None:
        return _GradScalerCuda(enabled=enabled)
    return None


def _ramp_scale(global_step, warmup_steps, ramp_steps):
    warmup_steps = int(warmup_steps or 0)
    ramp_steps = int(ramp_steps or 0)
    if warmup_steps <= 0 and ramp_steps <= 0:
        return 1.0
    if global_step < warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    t = (global_step - warmup_steps) / float(max(1, ramp_steps))
    return float(max(0.0, min(1.0, t)))


def _is_residual_param_name(name: str) -> bool:
    return ("S_residual" in name) or ("U_residual" in name) or ("V_residual" in name)


def _as_pair(x, default):
    if x is None:
        return default
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return float(x[0]), float(x[1])
    try:
        v = float(x)
        return v, v
    except Exception:
        return default


def _to_float_range(x, default):
    lo, hi = _as_pair(x, default)
    lo, hi = float(lo), float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _to_int_range(x, default):
    lo, hi = _as_pair(x, default)
    lo, hi = int(lo), int(hi)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _clip_stats(device, dtype):
    mean = torch.tensor(CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def _denorm_clip(images_norm: torch.Tensor) -> torch.Tensor:
    mean, std = _clip_stats(images_norm.device, images_norm.dtype)
    return (images_norm * std + mean).clamp(0.0, 1.0)


def _norm_clip(images: torch.Tensor) -> torch.Tensor:
    mean, std = _clip_stats(images.device, images.dtype)
    return (images - mean) / std


def _gaussian_blur_rgb(x: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 1e-6:
        return x
    k = int(2 * int(math.ceil(3.0 * sigma)) + 1)
    k = max(3, min(k, 31))
    t = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2.0
    kern1d = torch.exp(-(t ** 2) / (2.0 * (sigma ** 2)))
    kern1d = kern1d / kern1d.sum().clamp_min(1e-12)
    ky = kern1d.view(1, 1, k, 1).expand(3, 1, k, 1)
    kx = kern1d.view(1, 1, 1, k).expand(3, 1, 1, k)
    pad = k // 2
    x = F.conv2d(x, ky, padding=(pad, 0), groups=3)
    x = F.conv2d(x, kx, padding=(0, pad), groups=3)
    return x


def _jpegish(x: torch.Tensor, q_lo: int, q_hi: int) -> torch.Tensor:
    q_lo = int(max(1, min(100, q_lo)))
    q_hi = int(max(1, min(100, q_hi)))
    if q_hi < q_lo:
        q_lo, q_hi = q_hi, q_lo
    qv = int(torch.randint(low=q_lo, high=q_hi + 1, size=(), device=x.device).item())
    strength = float(100 - qv) / 100.0
    if strength <= 0.0:
        return x
    step = 1.0 + strength * 10.0
    x8 = torch.round(x * 255.0)
    xq = torch.round(x8 / step) * step
    x = (xq / 255.0).clamp(0.0, 1.0)
    blk = F.avg_pool2d(x, kernel_size=8, stride=8)
    blk = F.interpolate(blk, size=x.shape[-2:], mode="nearest")
    return ((1.0 - strength) * x + strength * blk).clamp(0.0, 1.0)


@torch.no_grad()
def _tri_view_degrade(
    images_norm: torch.Tensor,
    *,
    blur_sig=(0.2, 1.2),
    jpeg_qual=(70, 95),
    resize_jitter: int = 4,
    noise_std: float = 0.0,
) -> torch.Tensor:
    x = _denorm_clip(images_norm).float()

    b_lo, b_hi = _to_float_range(blur_sig, (0.2, 1.2))
    if b_hi > 0.0:
        sigma = float(torch.empty((), device=x.device).uniform_(b_lo, b_hi).item())
        x = _gaussian_blur_rgb(x, sigma)

    rj = int(resize_jitter or 0)
    if rj > 0:
        _, _, h, w = x.shape
        delta = int(torch.randint(low=-rj, high=rj + 1, size=(), device=x.device).item())
        nh = max(8, h + delta)
        nw = max(8, w + delta)
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    q_lo, q_hi = _to_int_range(jpeg_qual, (70, 95))
    x = _jpegish(x, q_lo, q_hi)

    ns = float(noise_std or 0.0)
    if ns > 0.0:
        x = (x + torch.randn_like(x) * ns).clamp(0.0, 1.0)

    return _norm_clip(x.to(dtype=images_norm.dtype))


@torch.no_grad()
def _tri_view_restore(
    images_norm_deg: torch.Tensor,
    *,
    original_norm: torch.Tensor | None = None,
    unsharp_amount: float = 0.35,
    restore_mix: float = 0.25,
) -> torch.Tensor:
    x = _denorm_clip(images_norm_deg).float()
    amt = float(max(0.0, unsharp_amount))
    if amt > 0:
        base = _gaussian_blur_rgb(x, sigma=1.0)
        x = (x + amt * (x - base)).clamp(0.0, 1.0)

    mix = float(max(0.0, min(1.0, restore_mix)))
    if (original_norm is not None) and (mix > 0.0):
        xo = _denorm_clip(original_norm).float()
        x = ((1.0 - mix) * x + mix * xo).clamp(0.0, 1.0)

    return _norm_clip(x.to(dtype=images_norm_deg.dtype))


def _quantile_threshold(x: torch.Tensor, p: float) -> torch.Tensor:
    """Approximate quantile threshold for 1D tensor via sorting.

    Note: uses index floor(p*(n-1)) so p close to 1.0 doesn't always pick max for small n.
    """
    x = x.flatten()
    n = int(x.numel())
    if n <= 1:
        return x.max() if n == 1 else torch.tensor(0.0, device=x.device, dtype=x.dtype)
    p = float(p)
    p = max(0.0, min(1.0, p))
    s, _ = torch.sort(x)
    j = int(math.floor(p * float(n - 1)))
    j = max(0, min(n - 1, j))
    return s[j]


def _apply_cap_or_trim(x: torch.Tensor, cap_p: float | None, trim: bool, required_k: int) -> torch.Tensor:
    """Winsorize (cap) or trim extreme tail values above cap_p quantile.

    - cap: x <- min(x, q_p)
    - trim: drop values > q_p, but only if remaining count >= required_k; otherwise fallback to cap.
    """
    if cap_p is None:
        return x
    try:
        cap_p_f = float(cap_p)
    except Exception:
        return x
    if not (0.0 < cap_p_f < 1.0) or int(x.numel()) <= 1:
        return x

    thr = _quantile_threshold(x, cap_p_f)
    if bool(trim):
        x2 = x[x <= thr]
        if int(x2.numel()) >= int(max(1, required_k)):
            return x2
    return torch.minimum(x, thr)


def _topk_mean(x: torch.Tensor, k: int) -> torch.Tensor:
    x = x.flatten()
    n = int(x.numel())
    if n <= 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    k = int(k)
    if k <= 1:
        return x.max()
    if k >= n:
        return x.mean()
    vals = torch.topk(x, k, largest=True).values
    return vals.mean()


def _cvar_tail_mean(
    x: torch.Tensor,
    tail_frac: float,
    cap_p: float | None = None,
    trim: bool = False,
    min_k: int = 1,
) -> torch.Tensor:
    """Tail-mean (CVaR-like) over the worst `tail_frac` of samples.

    Here `tail_frac` in (0, 1] means: average of the largest ceil(tail_frac * n) losses.
    """
    x = x.flatten()
    n = int(x.numel())
    if n <= 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    try:
        tf = float(tail_frac)
    except Exception:
        tf = 1.0
    if not (0.0 < tf <= 1.0):
        tf = 1.0
    k = int(math.ceil(tf * n))
    k = max(int(min_k), k)
    x2 = _apply_cap_or_trim(x, cap_p=cap_p, trim=bool(trim), required_k=k)
    n2 = int(x2.numel())
    if n2 <= 0:
        # fallback
        return x.mean()
    k = min(k, n2)
    return _topk_mean(x2, k)


def _groupwise_tail_mean(
    per_loss: torch.Tensor,
    gid: torch.Tensor,
    ng: int,
    tail_frac: float,
    cap_p: float | None,
    trim: bool,
    min_k: int,
) -> torch.Tensor:
    out = torch.zeros(int(ng), device=per_loss.device, dtype=per_loss.dtype)
    for g in range(int(ng)):
        m = gid == int(g)
        if m.any():
            out[g] = _cvar_tail_mean(per_loss[m], tail_frac=tail_frac, cap_p=cap_p, trim=trim, min_k=min_k)
    return out


def _cvar_diag_1d(
    x: torch.Tensor,
    tail_frac: float,
    cap_p: float | None,
    trim: bool,
    min_k: int,
) -> dict:
    """Diagnostics for within-group CVaR behavior on a 1D loss tensor."""
    x = x.flatten()
    n = int(x.numel())
    if n <= 0:
        return {
            "n": 0,
            "k": 0,
            "tail_frac": float("nan"),
            "cap_value": float("nan"),
            "clipped_rate": float("nan"),
        }

    try:
        tf = float(tail_frac)
    except Exception:
        tf = 1.0
    if not (0.0 < tf <= 1.0):
        tf = 1.0
    k = max(int(min_k), int(math.ceil(tf * n)))
    k = min(k, n)

    thr = None
    x2 = x
    clipped = 0
    if cap_p is not None:
        try:
            cp = float(cap_p)
        except Exception:
            cp = None
        if cp is not None and (0.0 < cp < 1.0) and n > 1:
            thr = _quantile_threshold(x, cp)
            if bool(trim):
                kept = x[x <= thr]
                if int(kept.numel()) >= max(1, k):
                    x2 = kept
                    clipped = int(n - int(kept.numel()))
                else:
                    x2 = torch.minimum(x, thr)
                    clipped = int((x > thr).sum().item())
            else:
                x2 = torch.minimum(x, thr)
                clipped = int((x > thr).sum().item())

    n2 = int(x2.numel())
    if n2 <= 0:
        n2 = n
    k2 = min(k, n2)
    return {
        "n": n,
        "k": int(k2),
        "tail_frac": float(k2 / max(1, n)),
        "cap_value": float(thr.item()) if thr is not None else float("nan"),
        "clipped_rate": float(clipped / max(1, n)),
    }



class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = self._select_device(opt)
        self.model = get_model(opt)
        self.model.to(self.device)
        self.residual_params, self.other_params = self._split_params()
        self.optimizer = self._build_optimizer()
        self.use_amp = bool(getattr(opt, "use_amp", False)) and self.device.type == "cuda"
        self.scaler = _make_grad_scaler(enabled=self.use_amp)
        self.grad_accum_steps = max(1, int(getattr(opt, "grad_accum_steps", 1) or 1))
        self._grad_accum_counter = 0
        self.scheduler = self._build_scheduler()
        self.total_steps = 0
        self.resume_epoch = -1
        self.resume_best_metric = None
        self._maybe_resume()

        # ---------------- GroupDRO (optional) ----------------
        # Goal: robust optimization over nuisance-defined groups (e.g. quality_bin),
        # to reduce shortcuts driven by image quality artifacts.
        self.use_groupdro = bool(getattr(opt, "use_groupdro", False))
        self.groupdro_group_key = str(getattr(opt, "groupdro_group_key", "quality_bin"))
        self.groupdro_class_balance = bool(getattr(opt, "groupdro_class_balance", False))
        self.groupdro_eta = float(getattr(opt, "groupdro_eta", 0.10))
        self.groupdro_ema_beta = float(getattr(opt, "groupdro_ema_beta", 0.90))
        self.groupdro_q_floor = float(getattr(opt, "groupdro_q_floor", 0.00))
        self.groupdro_logq_clip = getattr(opt, "groupdro_logq_clip", None)
        self.groupdro_min_count_group = int(getattr(opt, "groupdro_min_count_group", 1) or 1)
        self.groupdro_min_count_class = int(getattr(opt, "groupdro_min_count_class", 1) or 1)
        self.groupdro_warmup_steps = getattr(opt, "groupdro_warmup_steps", 0)
        self.groupdro_update_every = int(getattr(opt, "groupdro_update_every", 1) or 1)

        # Stabilize GroupDRO q updates (reduce seed variance / q collapse)
        self.groupdro_q_temp = float(getattr(opt, "groupdro_q_temp", 1.0) or 1.0)
        self.groupdro_q_mix = float(getattr(opt, "groupdro_q_mix", 0.0) or 0.0)
        if self.groupdro_q_temp <= 0:
            self.groupdro_q_temp = 1.0
        # clamp to [0,1]
        self.groupdro_q_mix = float(max(0.0, min(1.0, self.groupdro_q_mix)))

        qbins = int(getattr(opt, "quality_bins", 0) or 0)
        dbins = int(getattr(opt, "degrade_bins", 0) or 0)
        explicit_ng = getattr(opt, "groupdro_num_groups", None)
        if explicit_ng is not None:
            ng = int(explicit_ng)
        else:
            if self.groupdro_group_key == "quality_bin":
                ng = qbins
            elif self.groupdro_group_key == "stratum_id":
                # stratum_id is typically label * quality_bins + quality_bin
                ng = 2 * qbins
            elif self.groupdro_group_key == "degrade_bin":
                ng = dbins
            else:
                ng = 0
        self.groupdro_num_groups = max(1, int(ng or 1))

        self.groupdro_q = None
        self.groupdro_ema_loss = None
        self._groupdro_seen = None
        if self.use_groupdro:
            self._init_groupdro_state()

        # ---------------- Noise-robust fine-tune (optional) ----------------
        self.noise_robust_enable = bool(getattr(opt, "noise_robust_enable", False))
        self.noise_robust_warmup_steps = int(getattr(opt, "noise_robust_warmup_steps", 0) or 0)
        self.noise_robust_conf_thresh = float(getattr(opt, "noise_robust_conf_thresh", 0.65) or 0.65)
        self.noise_robust_min_weight = float(getattr(opt, "noise_robust_min_weight", 0.2) or 0.2)
        self.noise_robust_soft_weight = float(getattr(opt, "noise_robust_soft_weight", 0.5) or 0.5)
        self.noise_robust_label_smooth = float(getattr(opt, "noise_robust_label_smooth", 0.0) or 0.0)
        self.noise_robust_group_key = str(getattr(opt, "noise_robust_group_key", "quality_bin") or "quality_bin")
        self.noise_robust_group_momentum = float(getattr(opt, "noise_robust_group_momentum", 0.9) or 0.9)
        self.noise_robust_group_fake_prob = None
        if self.noise_robust_enable:
            self._init_noise_robust_state()

        # ---------------- Inverse-MEv1 (optional) ----------------
        # Keep behavior close to MEv1 teacher while forcing adapter params to
        # deviate from an anchor checkpoint for complementary diversity.
        self.inv_mev1_enable = bool(getattr(opt, "inv_mev1_enable", False))
        self.inv_mev1_div_lambda = float(getattr(opt, "inv_mev1_div_lambda", 0.0) or 0.0)
        self.inv_mev1_div_target_cos = float(getattr(opt, "inv_mev1_div_target_cos", 0.995) or 0.995)
        self.inv_mev1_kd_lambda = float(getattr(opt, "inv_mev1_kd_lambda", 0.0) or 0.0)
        self.inv_mev1_kd_temp = float(getattr(opt, "inv_mev1_kd_temp", 2.0) or 2.0)
        self.inv_mev1_teacher_every = int(max(1, int(getattr(opt, "inv_mev1_teacher_every", 1) or 1)))
        self.inv_mev1_warmup_steps = float(getattr(opt, "inv_mev1_warmup_steps", 0.0) or 0.0)
        self.inv_mev1_ramp_steps = float(getattr(opt, "inv_mev1_ramp_steps", 0.0) or 0.0)
        self.inv_mev1_anchor_ckpt = str(
            getattr(opt, "inv_mev1_anchor_ckpt", None)
            or getattr(opt, "resume_path", None)
            or ""
        )
        self.inv_mev1_teacher_ckpt = str(
            getattr(opt, "inv_mev1_teacher_ckpt", None)
            or getattr(opt, "resume_path", None)
            or ""
        )
        self.inv_anchor_state = {}
        self.inv_teacher = None
        if self.inv_mev1_enable:
            if self.inv_mev1_div_lambda > 0.0:
                self.inv_anchor_state = self._load_inv_anchor_state(self.inv_mev1_anchor_ckpt)
            if self.inv_mev1_kd_lambda > 0.0:
                self.inv_teacher = self._build_inv_teacher(self.inv_mev1_teacher_ckpt)

    def _maybe_resume(self):
        resume_path = getattr(self.opt, "resume_path", None)
        if resume_path in {None, ""}:
            return
        path = str(resume_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"resume_path not found: {path}")

        missing, unexpected = load_ckpt(path, self.model, strict=False)
        logging.info(
            "[resume] loaded model from %s | missing=%d unexpected=%d",
            path,
            len(list(missing or [])),
            len(list(unexpected or [])),
        )

        if not bool(getattr(self.opt, "load_whole_model", False)):
            return

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            return
        if ckpt.get("optimizer", None) is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                logging.info("[resume] optimizer state restored")
            except Exception as e:
                logging.warning("[resume] optimizer restore failed: %s", e)
        if (self.scaler is not None) and (ckpt.get("scaler", None) is not None):
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
                logging.info("[resume] scaler state restored")
            except Exception as e:
                logging.warning("[resume] scaler restore failed: %s", e)
        self.resume_epoch = int(ckpt.get("epoch", -1))
        self.resume_best_metric = ckpt.get("best_metric", None)
        logging.info(
            "[resume] metadata: epoch=%s best_metric=%s",
            self.resume_epoch,
            self.resume_best_metric,
        )

    def _select_device(self, opt):
        if getattr(opt, "gpu_ids", None) and torch.cuda.is_available():
            return torch.device(f"cuda:{opt.gpu_ids[0]}")
        return torch.device("cpu")

    def _split_params(self):
        residual_params = []
        for m in self.model.modules():
            if isinstance(m, SVDResidualLinear):
                if hasattr(m, "iter_residual_params"):
                    for p in m.iter_residual_params():
                        if p is not None and p.requires_grad:
                            residual_params.append(p)
                else:
                    for p in (getattr(m, "S_residual", None), getattr(m, "U_residual", None), getattr(m, "V_residual", None)):
                        if p is not None and p.requires_grad:
                            residual_params.append(p)
        residual_ids = {id(p) for p in residual_params}
        other_params = []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in residual_ids:
                continue
            other_params.append(p)
        return residual_params, other_params

    def _build_optimizer(self):
        opt = self.opt
        params = []
        if self.residual_params:
            residual_lr = opt.lr * 0.5 if opt.residual_lr is None else opt.residual_lr
            params = [
                {"params": self.other_params, "lr": opt.lr, "weight_decay": opt.weight_decay},
                {"params": self.residual_params, "lr": residual_lr, "weight_decay": opt.weight_decay},
            ]
        else:
            params = [{"params": self.other_params, "lr": opt.lr, "weight_decay": opt.weight_decay}]
        betas = (float(getattr(opt, "beta1", 0.9)), 0.999)
        optim = (opt.optim or "adam").lower()
        if optim == "sgd":
            return torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        if optim == "adamw":
            return torch.optim.AdamW(params, lr=opt.lr, betas=betas, weight_decay=opt.weight_decay)
        return torch.optim.Adam(params, lr=opt.lr, betas=betas, weight_decay=opt.weight_decay)

    def _set_lr_scale(self, scale: float):
        scale = float(max(0.0, scale))
        if not hasattr(self, "_base_lrs"):
            self._base_lrs = [float(pg.get("lr", 0.0)) for pg in self.optimizer.param_groups]
        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            pg["lr"] = float(base_lr) * scale

    def _apply_lr_warmup(self, global_step: int):
        warmup_steps = int(getattr(self.opt, "lr_warmup_steps", 0) or 0)
        if warmup_steps <= 0:
            return 1.0
        step_idx = int(max(0, global_step)) + 1
        scale = min(1.0, step_idx / float(max(1, warmup_steps)))
        self._set_lr_scale(scale)
        return float(scale)

    def _build_scheduler(self):
        opt = self.opt
        if getattr(opt, "lr_scheduler", "none") != "cosine":
            return None
        t_max = opt.cosine_t_max if opt.cosine_t_max is not None else int(opt.niter)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(t_max)),
            eta_min=float(opt.cosine_min_lr),
            last_epoch=int(getattr(opt, "last_epoch", -1)),
        )

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()
            self._base_lrs = [float(pg.get("lr", 0.0)) for pg in self.optimizer.param_groups]

    def _optimizer_step(self):
        clip_params = self.residual_params if self.residual_params else self.other_params
        if self.use_amp:
            if self.opt.grad_clip is not None and clip_params:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, float(self.opt.grad_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.opt.grad_clip is not None and clip_params:
                torch.nn.utils.clip_grad_norm_(clip_params, float(self.opt.grad_clip))
            self.optimizer.step()

    def flush_grad_accum(self):
        if self._grad_accum_counter <= 0:
            return False
        self._optimizer_step()
        self._grad_accum_counter = 0
        return True

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def save_checkpoint(self, path, epoch, best_metric=None):
        save_checkpoint(path, self.model, self.optimizer, self.scaler, epoch, best_metric=best_metric)

    # ---------------- GroupDRO helpers ----------------
    def _init_groupdro_state(self):
        ng = max(1, int(self.groupdro_num_groups))
        self.groupdro_q = torch.ones(ng, device=self.device, dtype=torch.float32) / float(ng)
        self.groupdro_ema_loss = torch.zeros(ng, device=self.device, dtype=torch.float32)
        self._groupdro_seen = torch.zeros(ng, device=self.device, dtype=torch.bool)

    def _init_noise_robust_state(self):
        key = str(self.noise_robust_group_key).lower()
        if key in {"none", "", "off"}:
            self.noise_robust_group_fake_prob = None
            return
        qbins = int(getattr(self.opt, "quality_bins", 0) or 0)
        dbins = int(getattr(self.opt, "degrade_bins", 0) or 0)
        if (qbins <= 0) and (dbins <= 0):
            self.noise_robust_group_fake_prob = None
            return
        if key == "quality_bin":
            ng = qbins
        elif key == "stratum_id":
            ng = 2 * qbins
        elif key == "degrade_bin":
            ng = dbins
        else:
            self.noise_robust_group_fake_prob = None
            return
        self.noise_robust_group_fake_prob = torch.full(
            (int(ng),),
            0.5,
            device=self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def _noise_robust_update_group_ema(self, gid: torch.Tensor, pred_fake: torch.Tensor):
        if self.noise_robust_group_fake_prob is None:
            return
        if gid.numel() <= 0:
            return
        ng = int(self.noise_robust_group_fake_prob.numel())
        gid = gid.long().clamp_min(0).clamp_max(max(0, ng - 1))
        mom = float(max(0.0, min(0.9999, self.noise_robust_group_momentum)))
        for g in torch.unique(gid):
            m = gid == g
            if not m.any():
                continue
            gidx = int(g.item())
            cur = self.noise_robust_group_fake_prob[gidx]
            new = pred_fake[m].mean().to(dtype=cur.dtype)
            self.noise_robust_group_fake_prob[gidx] = mom * cur + (1.0 - mom) * new

    def _noise_robust_cls_loss(self, logits: torch.Tensor, labels: torch.Tensor, batch: dict, global_step: int):
        """Confidence-gated bootstrapping loss for noisy-label fine-tune.

        High-confidence samples keep hard CE weight.
        Low-confidence samples interpolate toward soft targets (group EMA or self bootstrap).
        """
        labels = labels.long()
        per_ce = F.cross_entropy(logits, labels, reduction="none")

        probs_det = torch.softmax(logits.detach().float(), dim=1).clamp_min(1e-6)
        conf_gt = probs_det.gather(1, labels[:, None]).squeeze(1)

        if global_step < int(self.noise_robust_warmup_steps):
            alpha = torch.zeros_like(conf_gt)
        else:
            thr = float(max(0.0, min(0.99, self.noise_robust_conf_thresh)))
            denom = max(1e-6, 1.0 - thr)
            w_hard = ((conf_gt - thr) / denom).clamp(0.0, 1.0)
            min_w = float(max(0.0, min(1.0, self.noise_robust_min_weight)))
            w_hard = min_w + (1.0 - min_w) * w_hard
            alpha = 1.0 - w_hard

        ncls = int(logits.shape[1])
        onehot = F.one_hot(labels, num_classes=ncls).to(dtype=logits.dtype)
        eps = float(max(0.0, min(0.2, self.noise_robust_label_smooth)))
        if eps > 0.0:
            onehot = onehot * (1.0 - eps) + (eps / float(ncls))

        soft_w = float(max(0.0, min(1.0, self.noise_robust_soft_weight)))
        soft_t = (1.0 - soft_w) * onehot + soft_w * probs_det.to(dtype=logits.dtype)

        key = str(self.noise_robust_group_key).lower()
        if (
            self.noise_robust_group_fake_prob is not None
            and key in batch
            and int(logits.shape[1]) == 2
        ):
            gid = batch[key].long()
            ng = int(self.noise_robust_group_fake_prob.numel())
            gid = gid.clamp_min(0).clamp_max(max(0, ng - 1))
            p_fake_g = self.noise_robust_group_fake_prob[gid].detach().to(dtype=logits.dtype)
            tgt_g = torch.stack([1.0 - p_fake_g, p_fake_g], dim=1)
            soft_t = (1.0 - soft_w) * onehot + soft_w * tgt_g
            self._noise_robust_update_group_ema(gid=gid, pred_fake=probs_det[:, 1])

        per_soft = -(soft_t * F.log_softmax(logits, dim=1)).sum(dim=1)
        per_mix = (1.0 - alpha) * per_ce + alpha * per_soft
        loss_cls = per_mix.mean()

        stats = {
            "nr_conf_gt_mean": float(conf_gt.mean().item()),
            "nr_alpha_mean": float(alpha.mean().item()),
            "nr_soft_ce": float(per_soft.mean().item()),
        }
        if self.noise_robust_group_fake_prob is not None:
            p = self.noise_robust_group_fake_prob.detach().float()
            stats["nr_group_fake_std"] = float(p.std().item())
            stats["nr_group_fake_min"] = float(p.min().item())
            stats["nr_group_fake_max"] = float(p.max().item())
        else:
            stats["nr_group_fake_std"] = float("nan")
            stats["nr_group_fake_min"] = float("nan")
            stats["nr_group_fake_max"] = float("nan")
        return loss_cls, stats

    def _build_inv_teacher(self, ckpt_path: str):
        ckpt = str(ckpt_path or "").strip()
        if not ckpt:
            logging.warning("[inv_mev1] teacher ckpt is empty; disable teacher guidance")
            return None
        if not os.path.exists(ckpt):
            logging.warning("[inv_mev1] teacher ckpt not found: %s", ckpt)
            return None
        teacher = get_model(self.opt)
        miss, unexp = load_ckpt(ckpt, teacher, strict=False)
        teacher.to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        logging.info(
            "[inv_mev1] teacher loaded: %s | missing=%d unexpected=%d",
            ckpt,
            len(list(miss or [])),
            len(list(unexp or [])),
        )
        return teacher

    def _load_inv_anchor_state(self, ckpt_path: str):
        ckpt = str(ckpt_path or "").strip()
        if not ckpt:
            logging.warning("[inv_mev1] anchor ckpt is empty; disable adapter-divergence loss")
            return {}
        if not os.path.exists(ckpt):
            logging.warning("[inv_mev1] anchor ckpt not found: %s", ckpt)
            return {}
        anchor_model = get_model(self.opt)
        miss, unexp = load_ckpt(ckpt, anchor_model, strict=False)
        state = {}
        for name, p in anchor_model.named_parameters():
            if not _is_residual_param_name(name):
                continue
            state[name] = p.detach().to(device=self.device, dtype=torch.float32)
        logging.info(
            "[inv_mev1] anchor loaded: %s | residual_tensors=%d missing=%d unexpected=%d",
            ckpt,
            len(state),
            len(list(miss or [])),
            len(list(unexp or [])),
        )
        return state

    def _inv_mev1_adapter_div_loss(self, device: torch.device):
        if not self.inv_anchor_state:
            z = torch.tensor(0.0, device=device)
            return z, float("nan"), 0.0

        penalties = []
        cos_vals = []
        covered = 0
        target = float(self.inv_mev1_div_target_cos)
        target = max(-1.0, min(1.0, target))

        for name, p in self.model.named_parameters():
            ref = self.inv_anchor_state.get(name, None)
            if ref is None:
                continue
            covered += 1
            cur = p.float().reshape(-1)
            refv = ref.reshape(-1)
            cur_n = cur.norm()
            ref_n = refv.norm()
            if (cur_n <= 1e-12) or (ref_n <= 1e-12):
                continue
            cos = torch.dot(cur, refv) / (cur_n * ref_n + 1e-12)
            penalties.append(F.relu(cos - target))
            cos_vals.append(cos)

        if not penalties:
            z = torch.tensor(0.0, device=device)
            return z, float("nan"), float(covered)
        loss_div = torch.stack(penalties).mean()
        cos_mean = float(torch.stack(cos_vals).mean().item()) if cos_vals else float("nan")
        return loss_div, cos_mean, float(covered)

    def _inv_mev1_kd_loss(self, batch: dict, student_logits: torch.Tensor):
        if self.inv_teacher is None:
            return torch.tensor(0.0, device=student_logits.device)
        with torch.no_grad():
            t_pred = self.inv_teacher(batch)
            t_logits = t_pred["cls"] if isinstance(t_pred, dict) else t_pred
        temp = max(1e-6, float(self.inv_mev1_kd_temp))
        logp_s = F.log_softmax(student_logits.float() / temp, dim=1)
        p_t = F.softmax(t_logits.detach().float() / temp, dim=1)
        return F.kl_div(logp_s, p_t, reduction="batchmean") * (temp * temp)

    @torch.no_grad()
    def _groupdro_update_q(self, group_loss_ema: torch.Tensor):
        """Exponentiated-gradient update on q using EMA group losses.

        q <- softmax(log q + eta * L_ema), optionally add q_floor then renormalize.
        """
        if self.groupdro_q is None or self.groupdro_q.numel() != group_loss_ema.numel():
            self.groupdro_num_groups = int(group_loss_ema.numel())
            self._init_groupdro_state()

        old_q = self.groupdro_q
        log_q = torch.log(self.groupdro_q.clamp_min(1e-12)) + (self.groupdro_eta * group_loss_ema)
        log_q = log_q - log_q.max()
        if self.groupdro_logq_clip is not None:
            try:
                c = float(self.groupdro_logq_clip)
                if c > 0:
                    log_q = log_q.clamp(min=-c, max=c)
            except Exception:
                pass
        # Temperature (>1 flattens q) to avoid over-concentration
        try:
            temp = float(getattr(self, "groupdro_q_temp", 1.0) or 1.0)
        except Exception:
            temp = 1.0
        if temp <= 0:
            temp = 1.0
        if abs(temp - 1.0) > 1e-6:
            log_q = log_q / max(1e-6, temp)
        q = torch.exp(log_q)
        if self.groupdro_q_floor > 0:
            q = q + float(self.groupdro_q_floor)
        q = q / q.sum().clamp_min(1e-12)
        # Mix with previous q (EMA in q-space) to reduce update variance
        try:
            mix = float(getattr(self, "groupdro_q_mix", 0.0) or 0.0)
        except Exception:
            mix = 0.0
        mix = float(max(0.0, min(1.0, mix)))
        if mix > 0:
            q = (1.0 - mix) * old_q + mix * q
            q = q / q.sum().clamp_min(1e-12)
        self.groupdro_q = q

    def train_step(self, batch, global_step):
        lr_warmup_scale = self._apply_lr_warmup(global_step)
        self.model.train()
        # Keep non-tensor metadata as-is (e.g., paths). Dataset returns dict batches.
        batch = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if self._grad_accum_counter == 0:
            self.optimizer.zero_grad(set_to_none=True)

        if hasattr(self.model, "set_patch_scale"):
            if float(getattr(self.opt, "patch_pool_gamma", 0.0)) != 0.0:
                pscale = float(self.opt.patch_pool_gamma) * _ramp_scale(
                    global_step, int(self.opt.patch_warmup_steps), int(self.opt.patch_ramp_steps)
                )
            else:
                pscale = 0.0
            self.model.set_patch_scale(pscale)

        # Tri-view consistency: original + mild degradation + light restoration.
        cons_enable = bool(getattr(self.opt, "tri_consistency_enable", False))
        cons_prob = float(getattr(self.opt, "tri_consistency_apply_prob", 0.0) or 0.0)
        cons_prob = max(0.0, min(1.0, cons_prob))
        cons_applied = False
        cons_img_aux = None
        cons_aux_kind = "none"
        if cons_enable and ("image" in batch) and (cons_prob > 0.0):
            if float(torch.rand((), device=self.device).item()) < cons_prob:
                cons_applied = True
                with torch.no_grad():
                    img_deg = _tri_view_degrade(
                        batch["image"],
                        blur_sig=getattr(self.opt, "tri_consistency_blur_sig", [0.2, 1.2]),
                        jpeg_qual=getattr(self.opt, "tri_consistency_jpeg_qual", [70, 95]),
                        resize_jitter=int(getattr(self.opt, "tri_consistency_resize_jitter", 4) or 0),
                        noise_std=float(getattr(self.opt, "tri_consistency_noise_std", 0.0) or 0.0),
                    )
                    if float(torch.rand((), device=self.device).item()) < 0.5:
                        cons_img_aux = img_deg
                        cons_aux_kind = "deg"
                    else:
                        cons_img_aux = _tri_view_restore(
                            img_deg,
                            original_norm=batch["image"],
                            unsharp_amount=float(getattr(self.opt, "tri_consistency_unsharp_amount", 0.35) or 0.0),
                            restore_mix=float(getattr(self.opt, "tri_consistency_restore_mix", 0.25) or 0.0),
                        )
                        cons_aux_kind = "rst"

        with _autocast_ctx(device_type=self.device.type, enabled=self.use_amp):
            pred = self.model(batch)
            logits_diag = pred["cls"] if (isinstance(pred, dict) and ("cls" in pred)) else None
            if logits_diag is not None:
                ltmp = logits_diag.float()
                ptmp = torch.softmax(ltmp, dim=1).clamp_min(1e-12)
                logit_abs_mean = float(ltmp.abs().mean().item())
                prob_entropy_mean = float((-(ptmp * ptmp.log()).sum(dim=1)).mean().item())
            else:
                logit_abs_mean = float("nan")
                prob_entropy_mean = float("nan")
            loss_dict = self.model.get_losses(batch, pred)
            loss = loss_dict.get("cls", loss_dict.get("overall"))
            loss_cls = loss_dict.get("cls", loss)
            loss_real = loss_dict.get("real_loss", loss)
            loss_fake = loss_dict.get("fake_loss", loss)
            loss_orth = loss_dict.get("osd_orth", 0.0)
            loss_ksv = loss_dict.get("osd_ksv", 0.0)
            loss_route_sup = loss_dict.get("route_sup", 0.0)
            loss_route_bal = loss_dict.get("route_bal", 0.0)
            loss_expert_div = loss_dict.get("expert_div", 0.0)
            reg_term = torch.tensor(0.0, device=loss.device)
            me_term = torch.tensor(0.0, device=loss.device)
            inv_term = torch.tensor(0.0, device=loss.device)
            loss_inv_kd = torch.tensor(0.0, device=loss.device)
            loss_inv_div = torch.tensor(0.0, device=loss.device)
            inv_scale = 0.0
            inv_anchor_cos_mean = float("nan")
            inv_anchor_covered = 0.0
            nr_stats = {
                "nr_conf_gt_mean": float("nan"),
                "nr_alpha_mean": float("nan"),
                "nr_soft_ce": float("nan"),
                "nr_group_fake_std": float("nan"),
                "nr_group_fake_min": float("nan"),
                "nr_group_fake_max": float("nan"),
                "lr_warmup_scale": float(lr_warmup_scale),
            }

            if self.noise_robust_enable and (not self.use_groupdro):
                loss_cls_nr, nr_stats = self._noise_robust_cls_loss(
                    logits=pred["cls"],
                    labels=batch["label"],
                    batch=batch,
                    global_step=global_step,
                )
                loss_cls = loss_cls_nr
                loss = loss_cls_nr

            if getattr(self.opt, "osd_regs", False) and getattr(self.model, "enable_osd_regs", False):
                if self.opt.lambda_orth > 0 or self.opt.lambda_ksv > 0:
                    scale = _ramp_scale(global_step, self.opt.reg_warmup_steps, self.opt.reg_ramp_steps)
                    loss_orth_f = loss_orth.float() if hasattr(loss_orth, "float") else torch.tensor(loss_orth, device=loss.device)
                    loss_ksv_f = loss_ksv.float() if hasattr(loss_ksv, "float") else torch.tensor(loss_ksv, device=loss.device)
                    reg_term = scale * (self.opt.lambda_orth * loss_orth_f + self.opt.lambda_ksv * loss_ksv_f)
                    loss = loss + reg_term

            # ---------------- Reweighted ERM (optional) ----------------
            if (
                (not self.use_groupdro)
                and bool(getattr(self.opt, "use_reweighted_erm", False))
                and ("stratum_id" in batch)
            ):
                gid = batch["stratum_id"].long()
                logits = pred["cls"]
                labels = batch["label"].long()
                per_loss = F.cross_entropy(logits, labels, reduction="none")
                wmap = getattr(self.opt, "stratum_weight_map", {}) or {}
                w = torch.tensor(
                    [float(wmap.get(int(g.item()), 1.0)) for g in gid],
                    device=per_loss.device,
                    dtype=per_loss.dtype,
                )
                # normalized weighted mean keeps loss scale stable
                loss_cls = (per_loss * w).sum() / w.sum().clamp_min(1e-6)
                loss = loss_cls + reg_term
            # ------------------------------------------------------------

            # ---------------- GroupDRO (optional) ----------------
            gd_present_frac = float("nan")
            gd_present0_frac = float("nan")
            gd_present1_frac = float("nan")
            gd_both_class_frac = float("nan")
            gd_cvar_alpha = float("nan")
            gd_cvar_lambda = float("nan")
            gd_cvar_cap_p = float("nan")
            gd_cvar_trim = False
            gd_cvar_min_k = 1
            use_cvar = False
            gd_tail_k_mean = float("nan")
            gd_tail_frac_mean = float("nan")
            gd_tail_coverage = float("nan")
            gd_cap_value_mean = float("nan")
            gd_clipped_rate_mean = float("nan")
            gd_n_eff_samples = float("nan")
            if self.use_groupdro and (self.groupdro_group_key in batch):
                warm_steps = int(getattr(self.opt, "groupdro_warmup_steps", self.groupdro_warmup_steps) or 0)
                warm = bool(global_step < warm_steps)
                do_update = bool((global_step % max(1, self.groupdro_update_every)) == 0)

                gid = batch[self.groupdro_group_key].long()
                logits = pred["cls"]
                labels = batch["label"].long()
                per_loss = F.cross_entropy(logits, labels, reduction="none")

                ng = max(1, int(self.groupdro_num_groups))
                gid = gid.clamp_min(0).clamp_max(ng - 1)

                # ---- Within-group CVaR (tail-mean) options (optional) ----
                # Goal: focus on hard-tail cases *inside each group* without exploding group capacity.
                # Enable by setting `groupdro_use_cvar=True` OR `groupdro_cvar_alpha>0`.
                use_cvar = bool(getattr(self.opt, "groupdro_use_cvar", False))
                cvar_alpha = getattr(self.opt, "groupdro_cvar_alpha", None)
                if (not use_cvar) and (cvar_alpha is not None):
                    try:
                        use_cvar = float(cvar_alpha) > 0.0
                    except Exception:
                        use_cvar = False

                if use_cvar:
                    gd_cvar_alpha = float(getattr(self.opt, "groupdro_cvar_alpha", 0.30))
                    gd_cvar_cap_p = float(getattr(self.opt, "groupdro_cvar_cap_p", 0.99))
                    gd_cvar_trim = bool(getattr(self.opt, "groupdro_cvar_trim", False))
                    gd_cvar_min_k = int(getattr(self.opt, "groupdro_cvar_min_k", 1) or 1)

                    lam_max = float(getattr(self.opt, "groupdro_cvar_lambda", 1.0))
                    # By default, share warmup with GroupDRO warmup (keeps early training ERM-ish)
                    cvar_warm = getattr(self.opt, "groupdro_cvar_warmup_steps", warm_steps)
                    cvar_ramp = getattr(self.opt, "groupdro_cvar_ramp_steps", 0)
                    gd_cvar_lambda = float(lam_max) * _ramp_scale(global_step, cvar_warm, cvar_ramp)
                else:
                    gd_cvar_lambda = 0.0

                # CVaR diagnostics (tail size / clipping / effective sample proxy)
                if use_cvar and gd_cvar_lambda > 0.0:
                    diag_parts = []
                    if self.groupdro_class_balance:
                        for m in (labels == 0, labels == 1):
                            if not m.any():
                                continue
                            gid_sub = gid[m]
                            loss_sub = per_loss[m]
                            for g in range(int(ng)):
                                mg = gid_sub == g
                                if mg.any():
                                    diag_parts.append(
                                        _cvar_diag_1d(
                                            x=loss_sub[mg],
                                            tail_frac=gd_cvar_alpha,
                                            cap_p=gd_cvar_cap_p,
                                            trim=gd_cvar_trim,
                                            min_k=gd_cvar_min_k,
                                        )
                                    )
                    else:
                        for g in range(int(ng)):
                            mg = gid == g
                            if mg.any():
                                diag_parts.append(
                                    _cvar_diag_1d(
                                        x=per_loss[mg],
                                        tail_frac=gd_cvar_alpha,
                                        cap_p=gd_cvar_cap_p,
                                        trim=gd_cvar_trim,
                                        min_k=gd_cvar_min_k,
                                    )
                                )
                    if diag_parts:
                        ks = [d["k"] for d in diag_parts if d["n"] > 0]
                        frs = [d["tail_frac"] for d in diag_parts if d["n"] > 0]
                        cps = [d["cap_value"] for d in diag_parts if not math.isnan(d["cap_value"])]
                        total_n = float(sum(d["n"] for d in diag_parts if d["n"] > 0))
                        total_k = float(sum(d["k"] for d in diag_parts if d["n"] > 0))
                        total_clip = float(sum((d["clipped_rate"] * d["n"]) for d in diag_parts if d["n"] > 0))
                        if ks:
                            gd_tail_k_mean = float(sum(ks) / len(ks))
                        if frs:
                            gd_tail_frac_mean = float(sum(frs) / len(frs))
                        if cps:
                            gd_cap_value_mean = float(sum(cps) / len(cps))
                        if total_n > 0:
                            gd_tail_coverage = float(total_k / total_n)
                            gd_clipped_rate_mean = float(total_clip / total_n)

                if self.groupdro_class_balance:
                    # Compute per-group loss as average of (real-agg, fake-agg) within each group.
                    # This avoids "label imbalance inside group" turning quality_bin into a proxy label-group.
                    group_sum0 = torch.zeros(ng, device=per_loss.device, dtype=per_loss.dtype)
                    group_cnt0 = torch.zeros(ng, device=per_loss.device, dtype=per_loss.dtype)
                    group_sum1 = torch.zeros(ng, device=per_loss.device, dtype=per_loss.dtype)
                    group_cnt1 = torch.zeros(ng, device=per_loss.device, dtype=per_loss.dtype)

                    mask0 = labels == 0
                    if mask0.any():
                        gid0 = gid[mask0]
                        l0 = per_loss[mask0]
                        group_sum0.index_add_(0, gid0, l0)
                        group_cnt0.index_add_(0, gid0, torch.ones_like(l0))

                    mask1 = labels == 1
                    if mask1.any():
                        gid1 = gid[mask1]
                        l1 = per_loss[mask1]
                        group_sum1.index_add_(0, gid1, l1)
                        group_cnt1.index_add_(0, gid1, torch.ones_like(l1))

                    mean0 = group_sum0 / group_cnt0.clamp_min(1.0)
                    mean1 = group_sum1 / group_cnt1.clamp_min(1.0)

                    # Optional tail focus per class within each group
                    if use_cvar and gd_cvar_lambda > 0.0:
                        tail0 = _groupwise_tail_mean(
                            per_loss=per_loss[mask0],
                            gid=gid[mask0],
                            ng=ng,
                            tail_frac=gd_cvar_alpha,
                            cap_p=gd_cvar_cap_p,
                            trim=gd_cvar_trim,
                            min_k=gd_cvar_min_k,
                        ) if mask0.any() else torch.zeros_like(mean0)
                        tail1 = _groupwise_tail_mean(
                            per_loss=per_loss[mask1],
                            gid=gid[mask1],
                            ng=ng,
                            tail_frac=gd_cvar_alpha,
                            cap_p=gd_cvar_cap_p,
                            trim=gd_cvar_trim,
                            min_k=gd_cvar_min_k,
                        ) if mask1.any() else torch.zeros_like(mean1)

                        agg0 = (1.0 - gd_cvar_lambda) * mean0 + gd_cvar_lambda * tail0
                        agg1 = (1.0 - gd_cvar_lambda) * mean1 + gd_cvar_lambda * tail1
                    else:
                        agg0, agg1 = mean0, mean1

                    present0 = group_cnt0 >= float(self.groupdro_min_count_class)
                    present1 = group_cnt1 >= float(self.groupdro_min_count_class)
                    total_cnt = group_cnt0 + group_cnt1
                    present_total = total_cnt >= float(self.groupdro_min_count_group)
                    present = (present0 | present1) & present_total

                    denom = present0.to(per_loss.dtype) + present1.to(per_loss.dtype)
                    group_mean = (agg0 * present0.to(per_loss.dtype) + agg1 * present1.to(per_loss.dtype)) / denom.clamp_min(1.0)
                else:
                    group_sum = torch.zeros(ng, device=per_loss.device, dtype=per_loss.dtype)
                    group_cnt = torch.zeros(ng, device=per_loss.device, dtype=per_loss.dtype)
                    group_sum.index_add_(0, gid, per_loss)
                    group_cnt.index_add_(0, gid, torch.ones_like(per_loss))

                    group_mean_raw = group_sum / group_cnt.clamp_min(1.0)
                    present = group_cnt >= float(self.groupdro_min_count_group)

                    if use_cvar and gd_cvar_lambda > 0.0:
                        group_tail = _groupwise_tail_mean(
                            per_loss=per_loss,
                            gid=gid,
                            ng=ng,
                            tail_frac=gd_cvar_alpha,
                            cap_p=gd_cvar_cap_p,
                            trim=gd_cvar_trim,
                            min_k=gd_cvar_min_k,
                        )
                        group_mean = (1.0 - gd_cvar_lambda) * group_mean_raw + gd_cvar_lambda * group_tail
                    else:
                        group_mean = group_mean_raw

                # ---- GroupDRO batch diagnostics (no grad) ---- (no grad) ----
                with torch.no_grad():
                    gd_present_frac = float(present.float().mean().item())
                    if self.groupdro_class_balance:
                        both = present0 & present1
                        gd_present0_frac = float(present0.float().mean().item())
                        gd_present1_frac = float(present1.float().mean().item())
                        gd_both_class_frac = float(both.float().mean().item())

                # Update EMA + q on observed groups only (no grad).
                with torch.no_grad():
                    if self.groupdro_ema_loss is None or self.groupdro_ema_loss.numel() != ng:
                        self.groupdro_num_groups = ng
                        self._init_groupdro_state()
                    beta = float(self.groupdro_ema_beta)

                    newly = present & (~self._groupdro_seen)
                    if newly.any():
                        self.groupdro_ema_loss[newly] = group_mean[newly].detach().float()
                        self._groupdro_seen[newly] = True

                    upd = present & self._groupdro_seen
                    if upd.any():
                        self.groupdro_ema_loss[upd] = beta * self.groupdro_ema_loss[upd] + (1.0 - beta) * group_mean[upd].detach().float()

                    if do_update and (not warm):
                        self._groupdro_update_q(self.groupdro_ema_loss)

                # Robust cls loss; renormalize q over groups present in this batch to keep scale stable.
                if present.any():
                    if warm or (self.groupdro_q is None):
                        q = present.to(per_loss.dtype)
                        q = q / q.sum().clamp_min(1.0)
                    else:
                        q = self.groupdro_q.detach().to(device=per_loss.device, dtype=per_loss.dtype) * present.to(per_loss.dtype)
                        q = q / q.sum().clamp_min(1e-12)
                    loss_cls = (q * group_mean).sum()
                    loss = loss_cls + reg_term
            # ------------------------------------------------------------

            # ---------------- Multi-expert routing losses (optional) ----------------
            if bool(getattr(self.opt, "multi_expert_enable", False)):
                l_sup = float(getattr(self.opt, "multi_expert_route_sup_lambda", 0.0) or 0.0)
                l_bal = float(getattr(self.opt, "multi_expert_balance_lambda", 0.0) or 0.0)
                l_div = float(getattr(self.opt, "multi_expert_div_lambda", 0.0) or 0.0)
                if (l_sup > 0.0) or (l_bal > 0.0) or (l_div > 0.0):
                    sup_f = (
                        loss_route_sup.float()
                        if hasattr(loss_route_sup, "float")
                        else torch.tensor(loss_route_sup, device=loss.device)
                    )
                    bal_f = (
                        loss_route_bal.float()
                        if hasattr(loss_route_bal, "float")
                        else torch.tensor(loss_route_bal, device=loss.device)
                    )
                    div_f = (
                        loss_expert_div.float()
                        if hasattr(loss_expert_div, "float")
                        else torch.tensor(loss_expert_div, device=loss.device)
                    )
                    me_term = l_sup * sup_f + l_bal * bal_f + l_div * div_f
                    loss = loss + me_term
            # -----------------------------------------------------------------------

            # ---------------- Tri-view consistency loss (optional) ----------------
            loss_cons_cls = torch.tensor(0.0, device=loss.device)
            loss_cons_align = torch.tensor(0.0, device=loss.device)
            loss_cons_total = torch.tensor(0.0, device=loss.device)
            if cons_applied and (cons_img_aux is not None):
                batch_aux = dict(batch)
                batch_aux["image"] = cons_img_aux
                pred_aux = self.model(batch_aux)
                label = batch["label"].long()

                # Keep classification objective valid on an auxiliary view.
                loss_cons_cls = F.cross_entropy(pred_aux["cls"], label)

                # Distill auxiliary-view logits toward original-view logits.
                anchor = pred["cls"].detach()
                loss_cons_align = F.mse_loss(pred_aux["cls"], anchor)

                w_cls = float(getattr(self.opt, "tri_consistency_ce_weight", 0.25) or 0.0)
                w_align = float(getattr(self.opt, "tri_consistency_align_weight", 0.15) or 0.0)
                loss_cons_total = w_cls * loss_cons_cls + w_align * loss_cons_align
                loss = loss + loss_cons_total
            # ----------------------------------------------------------------------

            # ---------------- Learned restore consistency (optional) ----------------
            loss_rst_cls = torch.tensor(0.0, device=loss.device)
            loss_rst_align = torch.tensor(0.0, device=loss.device)
            loss_rst_img = torch.tensor(0.0, device=loss.device)
            loss_rst_deg = torch.tensor(0.0, device=loss.device)
            loss_rst_total = torch.tensor(0.0, device=loss.device)
            rst_applied = False
            rst_prob = float(getattr(self.opt, "restore_consistency_apply_prob", 0.0) or 0.0)
            rst_prob = max(0.0, min(1.0, rst_prob))
            if (
                bool(getattr(self.opt, "restore_consistency_enable", False))
                and ("image" in batch)
                and (rst_prob > 0.0)
                and hasattr(self.model, "restore_image")
            ):
                if float(torch.rand((), device=self.device).item()) < rst_prob:
                    rst_applied = True
                    img_rst = self.model.restore_image(batch["image"])
                    batch_rst = dict(batch)
                    batch_rst["image"] = img_rst
                    pred_rst = self.model(batch_rst)
                    label = batch["label"].long()

                    loss_rst_cls = F.cross_entropy(pred_rst["cls"], label)
                    loss_rst_align = F.mse_loss(pred_rst["cls"], pred["cls"].detach())
                    loss_rst_img = F.l1_loss(img_rst, batch["image"])

                    if hasattr(self.model, "encode_degradation"):
                        feat_org = self.model.encode_degradation(batch["image"])
                        feat_rst = self.model.encode_degradation(img_rst)
                        if (feat_org is not None) and (feat_rst is not None):
                            f0 = F.normalize(feat_org.detach().float(), dim=1)
                            f1 = F.normalize(feat_rst.float(), dim=1)
                            loss_rst_deg = F.mse_loss(f1, f0)

                    wr_cls = float(getattr(self.opt, "restore_consistency_ce_weight", 0.25) or 0.0)
                    wr_align = float(getattr(self.opt, "restore_consistency_align_weight", 0.15) or 0.0)
                    wr_img = float(getattr(self.opt, "restore_consistency_img_weight", 0.05) or 0.0)
                    wr_deg = float(getattr(self.opt, "restore_consistency_degfeat_weight", 0.05) or 0.0)
                    loss_rst_total = (
                        wr_cls * loss_rst_cls
                        + wr_align * loss_rst_align
                        + wr_img * loss_rst_img
                        + wr_deg * loss_rst_deg
                    )
                    loss = loss + loss_rst_total
            # -----------------------------------------------------------------------

            # ---------------- Patch aux loss (warmup/ramp only) ----------------
            loss_aux = torch.tensor(0.0, device=loss.device)
            patch_aux_lambda = float(getattr(self.opt, "patch_aux_lambda", 0.0))

            if patch_aux_lambda > 0.0 and isinstance(pred, dict) and ("patch_evidence" in pred):
                patch_ramp = _ramp_scale(global_step, int(self.opt.patch_warmup_steps), int(self.opt.patch_ramp_steps))
                lam = patch_aux_lambda * (1.0 - float(patch_ramp))

                if lam > 0.0:
                    evi = pred["patch_evidence"].float()
                    aux_pred = torch.stack([-evi, evi], dim=1)
                    loss_aux = self.model.loss_func(aux_pred, batch["label"])
                    loss = loss + lam * loss_aux
            # -------------------------------------------------------------------

            # ---------------- Inverse-MEv1 guidance (optional) ----------------
            if self.inv_mev1_enable:
                inv_scale = _ramp_scale(global_step, self.inv_mev1_warmup_steps, self.inv_mev1_ramp_steps)
                if float(inv_scale) > 0.0:
                    if self.inv_mev1_kd_lambda > 0.0:
                        if int(global_step % self.inv_mev1_teacher_every) == 0:
                            loss_inv_kd = self._inv_mev1_kd_loss(batch=batch, student_logits=pred["cls"])
                            inv_term = inv_term + (self.inv_mev1_kd_lambda * float(inv_scale) * loss_inv_kd)
                    if self.inv_mev1_div_lambda > 0.0:
                        loss_inv_div, inv_anchor_cos_mean, inv_anchor_covered = self._inv_mev1_adapter_div_loss(
                            device=loss.device
                        )
                        inv_term = inv_term + (self.inv_mev1_div_lambda * float(inv_scale) * loss_inv_div)
                loss = loss + inv_term
            # -------------------------------------------------------------------

        loss_for_backward = loss / float(max(1, self.grad_accum_steps))
        if not torch.isfinite(loss_for_backward):
            # Guardrail: skip non-finite updates to avoid irreversible NaN cascade.
            self.optimizer.zero_grad(set_to_none=True)
            self._grad_accum_counter = 0
            logging.warning(
                "[train] non-finite loss at global_step=%s (loss=%s). skipped update.",
                global_step,
                float(loss.detach().float().cpu().item()) if torch.is_tensor(loss) else loss,
            )
            z = 0.0
            return {
                "loss": float("nan"),
                "loss_cls": float("nan"),
                "loss_reg": z,
                "loss_me": z,
                "loss_route_sup": z,
                "loss_route_bal": z,
                "loss_expert_div": z,
                "route_alpha_max": float("nan"),
                "route_alpha_entropy": float("nan"),
                "loss_aux": z,
                "loss_cons_cls": z,
                "loss_cons_align": z,
                "loss_cons_total": z,
                "tri_consistency_applied": z,
                "tri_consistency_kind": "none",
                "loss_rst_cls": z,
                "loss_rst_align": z,
                "loss_rst_img": z,
                "loss_rst_deg": z,
                "loss_rst_total": z,
                "restore_consistency_applied": z,
                "patch_aux_lambda": z,
                "loss_inv": z,
                "loss_inv_kd": z,
                "loss_inv_div": z,
                "inv_scale": z,
                "inv_anchor_cos_mean": float("nan"),
                "inv_anchor_covered": z,
                "loss_real": float("nan"),
                "loss_fake": float("nan"),
                "loss_orth": z,
                "loss_ksv": z,
                "patch_evidence": z,
                "patch_delta": z,
                "optim_step": z,
                "grad_accum_steps": float(self.grad_accum_steps),
                "groupdro_use_cvar": False,
                "groupdro_q_max": float("nan"),
                "groupdro_q_entropy": float("nan"),
                "groupdro_q_entropy_norm": float("nan"),
                "groupdro_q_eff": float("nan"),
                "groupdro_effective_groups": float("nan"),
                "groupdro_tail_k_mean": float("nan"),
                "groupdro_tail_frac_mean": float("nan"),
                "groupdro_tail_coverage": float("nan"),
                "groupdro_cap_value_mean": float("nan"),
                "groupdro_clipped_rate_mean": float("nan"),
                "groupdro_n_eff_samples": float("nan"),
                "groupdro_cvar_lambda_t": float("nan"),
                "logit_abs_mean": float("nan"),
                "prob_entropy_mean": float("nan"),
                "nr_conf_gt_mean": float("nan"),
                "nr_alpha_mean": float("nan"),
                "nr_soft_ce": float("nan"),
                "nr_group_fake_std": float("nan"),
                "nr_group_fake_min": float("nan"),
                "nr_group_fake_max": float("nan"),
            }
        if self.use_amp:
            self.scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        self._grad_accum_counter += 1
        did_step = bool(self._grad_accum_counter >= self.grad_accum_steps)
        if did_step:
            self._optimizer_step()
            self._grad_accum_counter = 0

        patch_evi = 0.0
        patch_delta = 0.0
        route_alpha_max = float("nan")
        route_alpha_entropy = float("nan")
        if isinstance(pred, dict) and ("patch_evidence" in pred) and ("patch_delta" in pred):
            try:
                patch_evi = float(pred["patch_evidence"].mean().item())
                patch_delta = float(pred["patch_delta"].abs().mean().item())
            except Exception:
                patch_evi = 0.0
                patch_delta = 0.0
        if isinstance(pred, dict) and ("route_alpha" in pred):
            try:
                pa = pred["route_alpha"].float()
                if pa.dim() == 1:
                    p_bar = pa
                else:
                    p_bar = pa.reshape(-1, pa.shape[-1]).mean(dim=0)
                p_bar = p_bar / p_bar.sum().clamp_min(1e-12)
                route_alpha_max = float(p_bar.max().item())
                route_alpha_entropy = float((-(p_bar * p_bar.clamp_min(1e-12).log())).sum().item())
            except Exception:
                route_alpha_max = float("nan")
                route_alpha_entropy = float("nan")

        stats = {
            "loss": float(loss.item()) if hasattr(loss, "item") else float(loss),
            "loss_cls": float(loss_cls.item()) if hasattr(loss_cls, "item") else float(loss_cls),
            "loss_reg": float(reg_term.item()) if hasattr(reg_term, "item") else float(reg_term),
            "loss_me": float(me_term.item()) if hasattr(me_term, "item") else float(me_term),
            "loss_route_sup": float(loss_route_sup.item()) if hasattr(loss_route_sup, "item") else float(loss_route_sup),
            "loss_route_bal": float(loss_route_bal.item()) if hasattr(loss_route_bal, "item") else float(loss_route_bal),
            "loss_expert_div": float(loss_expert_div.item()) if hasattr(loss_expert_div, "item") else float(loss_expert_div),
            "route_alpha_max": float(route_alpha_max),
            "route_alpha_entropy": float(route_alpha_entropy),
            "loss_aux": float(loss_aux.item()) if hasattr(loss_aux, "item") else float(loss_aux),
            "loss_cons_cls": float(loss_cons_cls.item()) if hasattr(loss_cons_cls, "item") else float(loss_cons_cls),
            "loss_cons_align": float(loss_cons_align.item()) if hasattr(loss_cons_align, "item") else float(loss_cons_align),
            "loss_cons_total": float(loss_cons_total.item()) if hasattr(loss_cons_total, "item") else float(loss_cons_total),
            "tri_consistency_applied": float(1.0 if cons_applied else 0.0),
            "tri_consistency_kind": cons_aux_kind,
            "loss_rst_cls": float(loss_rst_cls.item()) if hasattr(loss_rst_cls, "item") else float(loss_rst_cls),
            "loss_rst_align": float(loss_rst_align.item()) if hasattr(loss_rst_align, "item") else float(loss_rst_align),
            "loss_rst_img": float(loss_rst_img.item()) if hasattr(loss_rst_img, "item") else float(loss_rst_img),
            "loss_rst_deg": float(loss_rst_deg.item()) if hasattr(loss_rst_deg, "item") else float(loss_rst_deg),
            "loss_rst_total": float(loss_rst_total.item()) if hasattr(loss_rst_total, "item") else float(loss_rst_total),
            "restore_consistency_applied": float(1.0 if rst_applied else 0.0),
            "patch_aux_lambda": float(lam) if "lam" in locals() else 0.0,
            "loss_inv": float(inv_term.item()) if hasattr(inv_term, "item") else float(inv_term),
            "loss_inv_kd": float(loss_inv_kd.item()) if hasattr(loss_inv_kd, "item") else float(loss_inv_kd),
            "loss_inv_div": float(loss_inv_div.item()) if hasattr(loss_inv_div, "item") else float(loss_inv_div),
            "inv_scale": float(inv_scale),
            "inv_anchor_cos_mean": float(inv_anchor_cos_mean),
            "inv_anchor_covered": float(inv_anchor_covered),
            "loss_real": float(loss_real.item()) if hasattr(loss_real, "item") else float(loss_real),
            "loss_fake": float(loss_fake.item()) if hasattr(loss_fake, "item") else float(loss_fake),
            "loss_orth": float(loss_orth.item()) if hasattr(loss_orth, "item") else float(loss_orth),
            "loss_ksv": float(loss_ksv.item()) if hasattr(loss_ksv, "item") else float(loss_ksv),
            "patch_evidence": patch_evi,
            "patch_delta": patch_delta,
            "optim_step": float(1.0 if did_step else 0.0),
            "grad_accum_steps": float(self.grad_accum_steps),
            "lr_warmup_scale": float(lr_warmup_scale),
        }

        if self.use_groupdro and (self.groupdro_q is not None):
            try:
                stats["groupdro_q"] = self.groupdro_q.detach().float().cpu().numpy().tolist()
                stats["groupdro_ema_loss"] = self.groupdro_ema_loss.detach().float().cpu().numpy().tolist()
            except Exception:
                stats["groupdro_q"] = None
                stats["groupdro_ema_loss"] = None
            try:
                q = self.groupdro_q.detach().float()
                q = q / q.sum().clamp_min(1e-12)
                q_max = float(q.max().item())
                q_argmax = int(q.argmax().item())
                q_entropy = float(-(q * (q.clamp_min(1e-12).log())).sum().item())
                denom = float(math.log(float(max(2, int(q.numel())))))
                q_entropy_norm = float(q_entropy / max(1e-12, denom))
                q_eff = float(1.0 / (q.pow(2).sum().clamp_min(1e-12)).item())

                if self.groupdro_ema_loss is not None and self.groupdro_ema_loss.numel() > 0:
                    worst_gid = int(self.groupdro_ema_loss.detach().float().argmax().item())
                    worst_ema = float(self.groupdro_ema_loss.detach().float()[worst_gid].item())
                else:
                    worst_gid = -1
                    worst_ema = float("nan")

                stats.update(
                    {
                        "groupdro_use_cvar": bool(use_cvar),
                        "groupdro_cvar_alpha": float(gd_cvar_alpha),
                        "groupdro_cvar_lambda": float(gd_cvar_lambda),
                        "groupdro_cvar_lambda_t": float(gd_cvar_lambda),
                        "groupdro_cvar_cap_p": float(gd_cvar_cap_p),
                        "groupdro_cvar_trim": bool(gd_cvar_trim),
                        "groupdro_cvar_min_k": int(gd_cvar_min_k),
                        "groupdro_tail_k_mean": float(gd_tail_k_mean),
                        "groupdro_tail_frac_mean": float(gd_tail_frac_mean),
                        "groupdro_tail_coverage": float(gd_tail_coverage),
                        "groupdro_cap_value_mean": float(gd_cap_value_mean),
                        "groupdro_clipped_rate_mean": float(gd_clipped_rate_mean),
                        "groupdro_present_frac": float(gd_present_frac),
                        "groupdro_present0_frac": float(gd_present0_frac),
                        "groupdro_present1_frac": float(gd_present1_frac),
                        "groupdro_both_class_frac": float(gd_both_class_frac),
                        "groupdro_q_max": q_max,
                        "groupdro_q_argmax": q_argmax,
                        "groupdro_q_entropy": q_entropy,
                        "groupdro_q_entropy_norm": q_entropy_norm,
                        "groupdro_q_eff": q_eff,
                        "groupdro_effective_groups": q_eff,
                        "groupdro_worst_gid": worst_gid,
                        "groupdro_worst_ema": worst_ema,
                    }
                )
                # n_eff_samples ~= k_bar / sum_g q_g^2
                if (not math.isnan(gd_tail_k_mean)) and q_eff > 0:
                    gd_n_eff_samples = float(gd_tail_k_mean * q_eff)
                stats["groupdro_n_eff_samples"] = float(gd_n_eff_samples)
            except Exception:
                pass
        stats["logit_abs_mean"] = float(logit_abs_mean)
        stats["prob_entropy_mean"] = float(prob_entropy_mean)
        stats["nr_conf_gt_mean"] = float(nr_stats["nr_conf_gt_mean"])
        stats["nr_alpha_mean"] = float(nr_stats["nr_alpha_mean"])
        stats["nr_soft_ce"] = float(nr_stats["nr_soft_ce"])
        stats["nr_group_fake_std"] = float(nr_stats["nr_group_fake_std"])
        stats["nr_group_fake_min"] = float(nr_stats["nr_group_fake_min"])
        stats["nr_group_fake_max"] = float(nr_stats["nr_group_fake_max"])
        return stats
