import os
import logging
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from .svd_residual import (
    apply_svd_residual_to_self_attn_ex,
    set_svd_residual_num_experts,
    set_svd_residual_mode,
    set_svd_residual_train_mode,
    SVDResidualLinear,
)

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _clip_stats(device: torch.device, dtype: torch.dtype):
    mean = torch.tensor(CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def _denorm_clip(images_norm: torch.Tensor) -> torch.Tensor:
    mean, std = _clip_stats(images_norm.device, images_norm.dtype)
    return (images_norm * std + mean).clamp(0.0, 1.0)


def _norm_clip(images: torch.Tensor) -> torch.Tensor:
    mean, std = _clip_stats(images.device, images.dtype)
    return (images - mean) / std


class _TimmVisionBackbone(nn.Module):
    """Adapter to expose timm ViT/EVA as a CLIP-vision-like interface."""

    def __init__(self, timm_model: nn.Module):
        super().__init__()
        self.model = timm_model
        hidden = int(getattr(timm_model, "num_features", 0) or 0)
        if hidden <= 0:
            raise ValueError("timm backbone missing valid num_features")
        self.config = SimpleNamespace(hidden_size=hidden)

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        # Accept CLIP-style `pixel_values=` for drop-in compatibility.
        if pixel_values is not None:
            x = pixel_values
        if x is None:
            raise ValueError("timm backbone forward requires input tensor via `x` or `pixel_values`.")
        tokens = self.model.forward_features(x)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if tokens.dim() == 2:
            last_hidden = tokens.unsqueeze(1)
        elif tokens.dim() == 3:
            last_hidden = tokens
        else:
            raise ValueError(f"Unexpected timm forward_features shape: {tuple(tokens.shape)}")

        # Prefer timm-native pooled pre-logits when available (EVA uses attn-pool),
        # and only fallback to CLS token to keep backward compatibility.
        cls_feat = None
        if hasattr(self.model, "forward_head"):
            try:
                cls_feat = self.model.forward_head(last_hidden, pre_logits=True)
            except Exception:
                cls_feat = None
        if cls_feat is None and hasattr(self.model, "pool"):
            try:
                pooled = self.model.pool(last_hidden)
                if hasattr(self.model, "fc_norm"):
                    pooled = self.model.fc_norm(pooled)
                if hasattr(self.model, "head_drop"):
                    pooled = self.model.head_drop(pooled)
                cls_feat = pooled
            except Exception:
                cls_feat = None
        if cls_feat is None:
            cls_feat = last_hidden[:, 0, :]
        if cls_feat.dim() == 3 and int(cls_feat.shape[1]) == 1:
            cls_feat = cls_feat[:, 0, :]
        if cls_feat.dim() != 2:
            raise ValueError(f"Unexpected pooled feature shape: {tuple(cls_feat.shape)}")

        if return_dict:
            return {"pooler_output": cls_feat, "last_hidden_state": last_hidden}
        return last_hidden, cls_feat


def _is_timm_backbone_spec(spec: str) -> bool:
    s = str(spec or "").strip().lower()
    return s.startswith("timm/") or s.startswith("hf-hub:timm/") or s.startswith("timm:")


def _canonical_timm_name(spec: str) -> str:
    s = str(spec).strip()
    if s.lower().startswith("hf-hub:"):
        return s
    if s.lower().startswith("timm:"):
        return s.split(":", 1)[1].strip()
    if s.lower().startswith("timm/"):
        return f"hf-hub:{s}"
    return s


class RelationMLPHead(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 2,
        use_absdiff: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()
        in_dim = 4 * feat_dim
        norm = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()
        self.use_absdiff = use_absdiff
        self.net = nn.Sequential(
            norm,
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        d = (f0 - f1).abs() if self.use_absdiff else (f0 - f1)
        p = f0 * f1
        z = torch.cat([f0, f1, d, p], dim=-1)
        return self.net(z)


class DegradationEncoder(nn.Module):
    """Lightweight artifact encoder for learned degradation-aware routing."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        d = int(max(32, out_dim))
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(96, d, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
        )

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        # Degradation cues are usually clearer in RGB [0,1] domain.
        x = _denorm_clip(x_norm).float()
        z = self.net(x)
        z = self.proj(z)
        return z.to(dtype=x_norm.dtype)


class EffortCLIP(nn.Module):
    """Effort backbone (OSD-style SVD residual) + linear classifier.

    Notes:
    - This is the original single-view baseline kept for backward compatibility.
    - Patch pooling is implemented as an optional add-on on top of the CLS head.
      When disabled (default), the model behavior and state_dict keys match the original.
    """

    def __init__(
        self,
        clip_model_or_path: Optional[str] = None,
        num_classes: int = 2,
        enable_osd_regs: bool = False,
        freeze_backbone: bool = False,
        # Optional robust patch pooling (add-on; disabled by default)
        patch_pool: bool = False,
        patch_pool_tau: float = 1.5,
        patch_pool_mode: str = "lse",  # {"lse"} only
        patch_trim_p: float = 0.2,  # kept for checkpoint/arg compatibility (unused)
        patch_quality: str = "none",  # {"none","cos","cos_norm"}
        svd_cache_enable: bool = True,
        svd_cache_dir: str = "cache",
        svd_init_device: str = "auto",  # auto|cpu|cuda
        svd_show_progress: bool = True,
        svd_rank_cap: Optional[int] = None,
        # Optional multi-expert residual routing
        multi_expert_enable: bool = False,
        multi_expert_k: int = 1,
        multi_expert_route: str = "quality_bin",  # quality_bin|quality_soft|uniform|router|hybrid|degrade_router|degrade_hybrid
        multi_expert_route_temp: float = 1.0,
        multi_expert_route_detach: bool = True,
        multi_expert_hybrid_mix: float = 0.0,
        quality_cuts=None,
        degrade_router_dim: int = 128,
        degrade_router_dropout: float = 0.1,
        restoration_enable: bool = False,
        restoration_strength: float = 0.20,
        route_sup_lambda: float = 0.0,
        route_bal_lambda: float = 0.0,
        expert_div_lambda: float = 0.0,
        rank_loss_enable: bool = False,
        rank_loss_mode: str = "logistic",
        rank_loss_margin: float = 0.2,
        rank_loss_max_pairs: int = 1024,
        # SVD adapter / PEFT control
        adapter_last_n_layers: int = 0,
        svd_residual_mode: str = "add",  # add|gated|softplus
        svd_residual_gate_init: float = 0.0,
        svd_residual_train_mode: str = "full",  # full|sigma_only|uv_only
        peft_unfreeze_last_n_blocks: int = 0,
        peft_unfreeze_layernorm: bool = True,
        peft_unfreeze_bias: bool = False,
    ):
        super().__init__()
        self.svd_cache_enable = bool(svd_cache_enable)
        self.svd_cache_dir = str(svd_cache_dir)
        self.svd_init_device = str(svd_init_device)
        self.svd_show_progress = bool(svd_show_progress)
        self.svd_rank_cap = int(svd_rank_cap) if (svd_rank_cap is not None and int(svd_rank_cap) > 0) else None
        self.adapter_last_n_layers = int(max(0, int(adapter_last_n_layers or 0)))
        self.svd_residual_mode = str(svd_residual_mode or "add").lower()
        if self.svd_residual_mode not in {"add", "gated", "softplus"}:
            self.svd_residual_mode = "add"
        self.svd_residual_gate_init = float(svd_residual_gate_init or 0.0)
        self.svd_residual_train_mode = str(svd_residual_train_mode or "full").lower()
        if self.svd_residual_train_mode not in {"full", "sigma_only", "uv_only"}:
            self.svd_residual_train_mode = "full"
        self.peft_unfreeze_last_n_blocks = int(max(0, int(peft_unfreeze_last_n_blocks or 0)))
        self.peft_unfreeze_layernorm = bool(peft_unfreeze_layernorm)
        self.peft_unfreeze_bias = bool(peft_unfreeze_bias)
        self.backbone = self._build_backbone(clip_model_or_path)
        set_svd_residual_mode(
            self.backbone,
            mode=self.svd_residual_mode,
            gate_logit_init=self.svd_residual_gate_init,
        )
        set_svd_residual_train_mode(self.backbone, mode=self.svd_residual_train_mode)
        self.multi_expert_enable = bool(multi_expert_enable) and int(multi_expert_k or 1) > 1
        self.multi_expert_k = max(1, int(multi_expert_k or 1))
        self.multi_expert_route = str(multi_expert_route or "quality_bin").lower()
        self.multi_expert_route_temp = max(1e-6, float(multi_expert_route_temp or 1.0))
        self.multi_expert_route_detach = bool(multi_expert_route_detach)
        self.multi_expert_hybrid_mix = float(max(0.0, min(1.0, float(multi_expert_hybrid_mix or 0.0))))
        self.degrade_router_dim = int(max(16, int(degrade_router_dim or 128)))
        self.degrade_router_dropout = float(max(0.0, min(0.9, float(degrade_router_dropout or 0.0))))
        self.quality_cuts = None
        if quality_cuts is not None:
            try:
                cuts = sorted([float(v) for v in list(quality_cuts)])
                if len(cuts) >= 1:
                    self.quality_cuts = cuts
            except Exception:
                self.quality_cuts = None
        if self.multi_expert_enable:
            set_svd_residual_num_experts(self.backbone, num_experts=self.multi_expert_k, init="copy")
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            self._apply_peft_freeze_policy()
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, num_classes)
        self.route_head = None
        self.degrade_encoder = None
        self.degrade_route_head = None
        if self.multi_expert_enable and self.multi_expert_route in {"router", "hybrid"}:
            self.route_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, self.multi_expert_k),
            )
        if self.multi_expert_enable and self.multi_expert_route in {"degrade_router", "degrade_hybrid"}:
            self.degrade_encoder = DegradationEncoder(out_dim=self.degrade_router_dim)
            self.degrade_route_head = nn.Sequential(
                nn.LayerNorm(self.degrade_router_dim),
                nn.Dropout(self.degrade_router_dropout),
                nn.Linear(self.degrade_router_dim, self.multi_expert_k),
            )
        self.restoration_enable = bool(restoration_enable)
        self.restoration_strength = float(max(0.0, min(1.0, float(restoration_strength or 0.0))))
        self.route_sup_lambda = float(max(0.0, float(route_sup_lambda or 0.0)))
        self.route_bal_lambda = float(max(0.0, float(route_bal_lambda or 0.0)))
        self.expert_div_lambda = float(max(0.0, float(expert_div_lambda or 0.0)))
        self.rank_loss_enable = bool(rank_loss_enable)
        self.rank_loss_mode = str(rank_loss_mode or "logistic").lower()
        if self.rank_loss_mode not in {"logistic", "hinge"}:
            self.rank_loss_mode = "logistic"
        self.rank_loss_margin = float(rank_loss_margin)
        self.rank_loss_max_pairs = int(max(1, int(rank_loss_max_pairs or 1)))
        self.restoration_net = None
        if self.restoration_enable:
            self.restoration_net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
            )
        self.loss_func = nn.CrossEntropyLoss()
        self.enable_osd_regs = bool(enable_osd_regs)

        # Patch pooling config (kept as a plug-in, not a redesign)
        self.patch_pool = bool(patch_pool)
        self.patch_pool_tau = float(patch_pool_tau)
        self.patch_pool_mode = str(patch_pool_mode).lower()
        self.patch_trim_p = float(patch_trim_p)
        self.patch_quality = str(patch_quality).lower()
        self.patch_scale = 0.0  # effective gamma, scheduled by train.py

        # IMPORTANT: create patch_head only when patch_pool is enabled,
        # so legacy checkpoints can load with strict=True (no new keys).
        self.patch_head = nn.Linear(hidden, 1) if self.patch_pool else None

    @staticmethod
    def _extract_block_index(name: str):
        parts = str(name).split(".")
        for i, p in enumerate(parts):
            if p in {"layers", "blocks"} and (i + 1) < len(parts):
                try:
                    return int(parts[i + 1])
                except Exception:
                    return None
        return None

    def _max_block_index(self):
        max_idx = None
        for name, _ in self.backbone.named_parameters():
            bi = self._extract_block_index(name)
            if bi is None:
                continue
            max_idx = bi if max_idx is None else max(max_idx, bi)
        return max_idx

    def _apply_peft_freeze_policy(self):
        max_idx = self._max_block_index()
        keep_from = None
        if (max_idx is not None) and (self.peft_unfreeze_last_n_blocks > 0):
            keep_from = max_idx - int(self.peft_unfreeze_last_n_blocks) + 1

        for name, p in self.backbone.named_parameters():
            trainable = False
            # Keep Effort residual adapters trainable.
            if any(k in name for k in ("S_residual", "U_residual", "V_residual", "residual_gate_logit")):
                trainable = True
            else:
                if self.peft_unfreeze_layernorm and ("norm" in name.lower() or "layernorm" in name.lower()):
                    trainable = True
                if self.peft_unfreeze_bias and name.endswith(".bias"):
                    trainable = True
                if keep_from is not None:
                    bi = self._extract_block_index(name)
                    if bi is not None and bi >= keep_from:
                        trainable = True
            p.requires_grad = bool(trainable)

    def set_patch_scale(self, scale: float) -> None:
        """Set effective patch contribution scale (gamma)."""
        self.patch_scale = float(scale)

    def _set_route_alpha(self, alpha: Optional[torch.Tensor]) -> None:
        if not self.multi_expert_enable:
            return
        for m in self.modules():
            if isinstance(m, SVDResidualLinear):
                m.set_route_alpha(alpha)

    def _clear_route_alpha(self) -> None:
        if not self.multi_expert_enable:
            return
        for m in self.modules():
            if isinstance(m, SVDResidualLinear):
                m.clear_route_alpha()

    def _uniform_route_alpha(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        k = int(self.multi_expert_k)
        return torch.full((int(batch_size), k), 1.0 / float(k), device=device, dtype=dtype)

    def _hard_route_from_quality(
        self,
        data_dict: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if "quality_bin" not in data_dict:
            if ("q_log" in data_dict) and (self.quality_cuts is not None):
                q = data_dict["q_log"].float().to(device=device)
                cuts_t = torch.tensor(self.quality_cuts, device=device, dtype=q.dtype)
                qb = torch.bucketize(q, cuts_t).long()
                qb = qb.clamp_min(0).clamp_max(int(self.multi_expert_k) - 1)
                return F.one_hot(qb, num_classes=int(self.multi_expert_k)).to(device=device, dtype=dtype)
            return self._uniform_route_alpha(batch_size, device=device, dtype=dtype)
        qb = data_dict["quality_bin"].long()
        qb = qb.clamp_min(0).clamp_max(int(self.multi_expert_k) - 1)
        return F.one_hot(qb, num_classes=int(self.multi_expert_k)).to(device=device, dtype=dtype)

    def _soft_route_from_quality(
        self,
        data_dict: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if ("q_log" not in data_dict) or (self.quality_cuts is None):
            return self._uniform_route_alpha(batch_size, device=device, dtype=dtype)

        q = data_dict["q_log"].float().to(device=device).reshape(-1)
        if int(q.numel()) != int(batch_size):
            if int(q.numel()) <= 0:
                return self._uniform_route_alpha(batch_size, device=device, dtype=dtype)
            if int(q.numel()) == 1:
                q = q.expand(int(batch_size))
            else:
                q = q[: int(batch_size)]

        k = int(self.multi_expert_k)
        cuts = torch.tensor(self.quality_cuts, device=device, dtype=q.dtype).reshape(-1)

        # Build k quality centers from (k-1) cut points.
        # Example k=3, cuts=[c1,c2] -> centers at [left-mid, mid, right-mid].
        if int(cuts.numel()) != max(0, k - 1):
            qmin = q.min()
            qmax = q.max()
            if float((qmax - qmin).abs().item()) < 1e-6:
                return self._uniform_route_alpha(batch_size, device=device, dtype=dtype)
            centers = torch.linspace(qmin, qmax, steps=k, device=device, dtype=q.dtype)
        else:
            if int(cuts.numel()) >= 2:
                width = (cuts[1:] - cuts[:-1]).median().clamp_min(1e-3)
            else:
                width = torch.tensor(1.0, device=device, dtype=q.dtype)
            left = cuts[0] - width
            right = cuts[-1] + width
            edges = torch.cat([left.view(1), cuts, right.view(1)], dim=0)
            centers = 0.5 * (edges[:-1] + edges[1:])

        temp = max(1e-6, float(self.multi_expert_route_temp))
        dist = (q[:, None] - centers[None, :]).abs()
        alpha = torch.softmax(-dist / temp, dim=1).to(dtype=dtype)
        return alpha

    def _route_logits_from_feat(self, feat: torch.Tensor) -> Optional[torch.Tensor]:
        if self.route_head is None:
            return None
        z = feat.detach() if self.multi_expert_route_detach else feat
        return self.route_head(z)

    def encode_degradation(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.degrade_encoder is None:
            return None
        return self.degrade_encoder(x)

    def _route_logits_from_degradation(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if (self.degrade_encoder is None) or (self.degrade_route_head is None):
            return None
        z = self.encode_degradation(x)
        if z is None:
            return None
        z = z.detach() if self.multi_expert_route_detach else z
        return self.degrade_route_head(z)

    def restore_image(self, x_norm: torch.Tensor) -> torch.Tensor:
        if (not self.restoration_enable) or (self.restoration_net is None):
            return x_norm
        x = _denorm_clip(x_norm).float()
        delta = torch.tanh(self.restoration_net(x)) * float(self.restoration_strength)
        y = (x + delta).clamp(0.0, 1.0)
        return _norm_clip(y.to(dtype=x_norm.dtype))

    def _prepare_route_alpha(self, x: torch.Tensor, data_dict: Dict[str, torch.Tensor]):
        if not self.multi_expert_enable:
            return None, None

        mode = self.multi_expert_route
        batch_size = int(x.shape[0])
        dtype = x.dtype
        device = x.device

        if mode == "uniform":
            return self._uniform_route_alpha(batch_size, device=device, dtype=dtype), None

        if mode == "quality_bin":
            alpha = self._hard_route_from_quality(data_dict, batch_size, device=device, dtype=dtype)
            return alpha, None

        if mode == "quality_soft":
            alpha = self._soft_route_from_quality(data_dict, batch_size, device=device, dtype=dtype)
            return alpha, None

        if mode in {"degrade_router", "degrade_hybrid"}:
            route_logits = self._route_logits_from_degradation(x)
            if route_logits is None:
                alpha = self._uniform_route_alpha(batch_size, device=device, dtype=dtype)
                return alpha, None
            alpha = torch.softmax(route_logits / float(self.multi_expert_route_temp), dim=1).to(dtype=dtype)
            if mode == "degrade_hybrid":
                q_alpha = self._soft_route_from_quality(data_dict, batch_size, device=device, dtype=dtype)
                mix = float(self.multi_expert_hybrid_mix)
                alpha = (1.0 - mix) * alpha + mix * q_alpha
                alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
            return alpha, route_logits

        # Router modes run a cheap provisional pass with uniform routing.
        self._set_route_alpha(self._uniform_route_alpha(batch_size, device=device, dtype=dtype))
        cls_seed, _ = self.encode_tokens(x)
        self._clear_route_alpha()

        route_logits = self._route_logits_from_feat(cls_seed)
        if route_logits is None:
            alpha = self._uniform_route_alpha(batch_size, device=device, dtype=dtype)
            return alpha, None

        alpha = torch.softmax(route_logits / float(self.multi_expert_route_temp), dim=1).to(dtype=dtype)
        if mode == "hybrid":
            hard = self._hard_route_from_quality(data_dict, batch_size, device=device, dtype=dtype)
            mix = float(self.multi_expert_hybrid_mix)
            alpha = (1.0 - mix) * alpha + mix * hard
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return alpha, route_logits

    @torch.no_grad()
    def _osd_reg_modules(self):
        for m in self.modules():
            if isinstance(m, SVDResidualLinear):
                yield m

    def compute_osd_regularizers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (orth_loss, keepsv_loss) averaged over SVDResidualLinear modules."""
        device = next(self.parameters()).device
        orth = torch.tensor(0.0, device=device)
        ksv = torch.tensor(0.0, device=device)
        count = 0
        for m in self._osd_reg_modules():
            orth = orth + (m.compute_orthogonal_loss() if hasattr(m, "compute_orthogonal_loss") else 0.0)
            ksv = ksv + (m.compute_keepsv_loss() if hasattr(m, "compute_keepsv_loss") else 0.0)
            count += 1
        if count > 0:
            orth = orth / count
            ksv = ksv / count
        return orth, ksv

    def compute_expert_div_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        count = 0
        for m in self._osd_reg_modules():
            if hasattr(m, "compute_expert_div_loss"):
                loss = loss + m.compute_expert_div_loss()
                count += 1
        if count > 0:
            loss = loss / count
        return loss

    def _build_backbone(self, clip_model_or_path: Optional[str]):
        if clip_model_or_path is None:
            clip_model_or_path = os.environ.get("CLIP_MODEL_PATH", "openai/clip-vit-large-patch14")

        clip_spec = str(clip_model_or_path)
        clip_path = Path(clip_spec).expanduser()

        if _is_timm_backbone_spec(clip_spec):
            try:
                import timm
            except Exception as e:
                raise RuntimeError(
                    "timm backbone requested but timm is not available in current environment"
                ) from e
            timm_name = _canonical_timm_name(clip_spec)
            logging.info("[backbone] loading timm model: %s", timm_name)
            timm_model = timm.create_model(timm_name, pretrained=True)
            vision_model = _TimmVisionBackbone(timm_model)
            model_key = f"timm::{timm_name}"
        else:
            if clip_path.exists():
                clip_model = CLIPModel.from_pretrained(str(clip_path), local_files_only=True)
            else:
                clip_model = CLIPModel.from_pretrained(str(clip_spec))
            vision_model = clip_model.vision_model
            model_key = str(clip_path if clip_path.exists() else clip_spec)

        hidden = vision_model.config.hidden_size
        r = int(hidden - 32)
        if self.svd_rank_cap is not None:
            r = min(r, int(self.svd_rank_cap))
        r = max(1, int(r))

        # Apply SVD residual to self-attn linear layers; only residual params are trainable.
        # Defaults are optimized for training loops with many runs:
        #   - show progress
        #   - use cuda for SVD init when available
        #   - cache SVD-converted backbone to avoid repeated decompositions
        key_material = (
            f"{model_key}|r={r}|lastn={self.adapter_last_n_layers}"
            f"|mode={self.svd_residual_mode}|g0={self.svd_residual_gate_init:.4f}"
            f"|tm={self.svd_residual_train_mode}"
        )
        key_hash = hashlib.md5(key_material.encode("utf-8")).hexdigest()[:12]
        cache_dir = Path(self.svd_cache_dir).expanduser()
        cache_path = cache_dir / f"svd_vision_{key_hash}_r{r}.pt"

        if self.svd_cache_enable and cache_path.exists():
            logging.info("[svd] loading cache: %s", str(cache_path))
            vision_model = apply_svd_residual_to_self_attn_ex(
                vision_model,
                r=r,
                svd_device=None,
                show_progress=False,
                fast_init=True,
                adapter_last_n_layers=self.adapter_last_n_layers,
            )
            set_svd_residual_mode(
                vision_model,
                mode=self.svd_residual_mode,
                gate_logit_init=self.svd_residual_gate_init,
            )
            set_svd_residual_train_mode(vision_model, mode=self.svd_residual_train_mode)
            try:
                payload = torch.load(str(cache_path), map_location="cpu", weights_only=True)
            except TypeError:
                # Backward compatibility for older torch versions without weights_only.
                payload = torch.load(str(cache_path), map_location="cpu")
            state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            missing, unexpected = vision_model.load_state_dict(state, strict=False)
            logging.info("[svd] cache loaded: missing=%d unexpected=%d", len(missing), len(unexpected))
            return vision_model

        init_dev = self.svd_init_device.lower()
        if init_dev == "auto":
            init_dev = "cuda" if torch.cuda.is_available() else "cpu"
        if init_dev == "cuda" and (not torch.cuda.is_available()):
            init_dev = "cpu"

        logging.info(
            "[svd] init start: device=%s progress=%s cache=%s",
            init_dev,
            self.svd_show_progress,
            str(cache_path) if self.svd_cache_enable else "disabled",
        )
        vision_model = apply_svd_residual_to_self_attn_ex(
            vision_model,
            r=r,
            svd_device=init_dev,
            show_progress=self.svd_show_progress,
            fast_init=False,
            adapter_last_n_layers=self.adapter_last_n_layers,
        )
        set_svd_residual_mode(
            vision_model,
            mode=self.svd_residual_mode,
            gate_logit_init=self.svd_residual_gate_init,
        )
        set_svd_residual_train_mode(vision_model, mode=self.svd_residual_train_mode)
        if self.svd_cache_enable:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": vision_model.state_dict(), "r": r, "model_key": model_key}, str(cache_path))
            logging.info("[svd] cache saved: %s", str(cache_path))
        return vision_model

    def encode_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cls_feat, patch_tokens).

        - cls_feat: (B, D) from pooler_output
        - patch_tokens: (B, N, D) from last_hidden_state[:, 1:, :]
        """
        try:
            out = self.backbone(pixel_values=x, return_dict=True)
        except TypeError:
            # Older transformers CLIPVisionTransformer may not accept return_dict.
            out = self.backbone(pixel_values=x)
        cls_feat = getattr(out, "pooler_output", None)
        if cls_feat is None:
            cls_feat = out["pooler_output"]
        last = getattr(out, "last_hidden_state", None)
        if last is None:
            last = out["last_hidden_state"]
        patch_tok = last[:, 1:, :]
        return cls_feat, patch_tok

    def features(self, x: torch.Tensor) -> torch.Tensor:
        cls_feat, _ = self.encode_tokens(x)
        return cls_feat

    def _patch_quality_weights(self, patch_tok: torch.Tensor, cls_feat: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute per-patch quality weights q_i in [0,1]."""
        mode = (self.patch_quality or "none").lower()
        if mode == "none":
            return None

        pt = patch_tok.float()
        cf = cls_feat.float().unsqueeze(1)  # (B,1,D)
        pt_n = F.normalize(pt, dim=-1)
        cf_n = F.normalize(cf, dim=-1)
        cos = (pt_n * cf_n).sum(dim=-1)  # (B,N) in [-1,1]
        q = (cos + 1.0) * 0.5  # -> [0,1]
        q = q.clamp(0.0, 1.0)

        if mode == "cos_norm":
            n = pt.norm(dim=-1)  # (B,N)
            med = n.median(dim=1, keepdim=True).values.clamp_min(1e-6)
            n_rel = (n / med).clamp(0.0, 3.0) / 3.0  # -> [0,1]
            q = q * n_rel

        return q

    def _pool_patch_logits(self, patch_logits: torch.Tensor, q: Optional[torch.Tensor]) -> torch.Tensor:
        """Pool per-patch logits to a single evidence scalar per image (LSE only)."""

        tau = float(self.patch_pool_tau)
        tau = max(1e-6, tau)
        s = patch_logits / tau
            # --- LME / Weighted-LME：移除 log(N) 或 log(sum(q)) 的常數偏壓 ---
        if q is not None:
            w = q.clamp_min(1e-6)
            wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            # log(w_norm) = log(w) - log(sum(w))
            s = s + (torch.log(w) - torch.log(wsum))
            return (tau * torch.logsumexp(s, dim=1))

        # unweighted: logmeanexp = logsumexp - log(N)
        n = s.shape[1]
        log_n = s.new_tensor(float(n)).log()
        return (tau * (torch.logsumexp(s, dim=1) - log_n))

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = data_dict["image"]
        route_alpha, route_logits = self._prepare_route_alpha(x, data_dict)

        if self.multi_expert_enable:
            self._set_route_alpha(route_alpha)
        try:
            cls_feat, patch_tok = self.encode_tokens(x)
        finally:
            if self.multi_expert_enable:
                self._clear_route_alpha()
        logits = self.head(cls_feat)

        out: Dict[str, torch.Tensor] = {"cls": logits, "feat": cls_feat}
        if route_alpha is not None:
            out["route_alpha"] = route_alpha
        if route_logits is not None:
            out["route_logits"] = route_logits

        if self.patch_pool and (self.patch_head is not None):
            patch_logits = self.patch_head(patch_tok).squeeze(-1)  # (B,N)
            q = self._patch_quality_weights(patch_tok, cls_feat)
            evidence = self._pool_patch_logits(patch_logits.float(), q.float() if q is not None else None)  # (B,)
            delta = evidence.to(logits.dtype) * float(self.patch_scale)

            # 這些 key 名稱完全沿用你原本的
            out["patch_evidence"] = evidence                 # <-- 不要 detach：trainer 會用它算 aux loss
            out["patch_delta"] = delta.detach()
            out["patch_logit_min"] = patch_logits.min().detach()
            out["patch_logit_mean"] = patch_logits.mean().detach()
            out["patch_logit_max"] = patch_logits.max().detach()

        prob = torch.softmax(out["cls"], dim=1)[:, 1]
        out["prob"] = prob
        return out

    def get_losses(self, data_dict: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        label = data_dict["label"]
        pred = pred_dict["cls"]
        loss = self.loss_func(pred, label)

        mask_real = label == 0
        mask_fake = label == 1

        if mask_real.sum() > 0:
            loss_real = self.loss_func(pred[mask_real], label[mask_real])
        else:
            loss_real = torch.tensor(0.0, device=pred.device)

        if mask_fake.sum() > 0:
            loss_fake = self.loss_func(pred[mask_fake], label[mask_fake])
        else:
            loss_fake = torch.tensor(0.0, device=pred.device)

        orth, ksv = (torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device))
        if self.enable_osd_regs:
            orth, ksv = self.compute_osd_regularizers()

        route_sup = torch.tensor(0.0, device=pred.device)
        route_bal = torch.tensor(0.0, device=pred.device)
        expert_div = torch.tensor(0.0, device=pred.device)
        rank_loss = torch.tensor(0.0, device=pred.device)
        rank_pairs = torch.tensor(0.0, device=pred.device)
        if self.multi_expert_enable:
            if (
                self.route_sup_lambda > 0.0
                and ("route_logits" in pred_dict)
                and (pred_dict["route_logits"] is not None)
                and ("quality_bin" in data_dict)
            ):
                target = data_dict["quality_bin"].long().clamp_min(0).clamp_max(int(self.multi_expert_k) - 1)
                route_sup = F.cross_entropy(pred_dict["route_logits"], target)

            if self.route_bal_lambda > 0.0 and ("route_alpha" in pred_dict) and (pred_dict["route_alpha"] is not None):
                pa = pred_dict["route_alpha"].float()
                if pa.dim() == 1:
                    p_bar = pa
                else:
                    p_bar = pa.reshape(-1, pa.shape[-1]).mean(dim=0)
                p_bar = p_bar / p_bar.sum().clamp_min(1e-12)
                uni = torch.full_like(p_bar, 1.0 / float(max(1, p_bar.numel())))
                route_bal = (p_bar * (p_bar.clamp_min(1e-12).log() - uni.log())).sum()

            if self.expert_div_lambda > 0.0:
                expert_div = self.compute_expert_div_loss()

        if self.rank_loss_enable:
            score = pred[:, 1] - pred[:, 0]
            pos = score[mask_fake]
            neg = score[mask_real]
            if pos.numel() > 0 and neg.numel() > 0:
                total_pairs = int(pos.numel() * neg.numel())
                if total_pairs <= int(self.rank_loss_max_pairs):
                    diff = (pos[:, None] - neg[None, :]).reshape(-1)
                else:
                    n_pair = int(self.rank_loss_max_pairs)
                    pos_idx = torch.randint(
                        low=0,
                        high=int(pos.numel()),
                        size=(n_pair,),
                        device=pred.device,
                    )
                    neg_idx = torch.randint(
                        low=0,
                        high=int(neg.numel()),
                        size=(n_pair,),
                        device=pred.device,
                    )
                    diff = pos[pos_idx] - neg[neg_idx]
                margin = float(self.rank_loss_margin)
                if self.rank_loss_mode == "hinge":
                    rank_loss = F.relu(margin - diff).mean()
                else:
                    rank_loss = F.softplus(-(diff - margin)).mean()
                rank_pairs = torch.tensor(float(diff.numel()), device=pred.device)

        return {
            "cls": loss,
            "overall": loss,
            "real_loss": loss_real,
            "fake_loss": loss_fake,
            "osd_orth": orth,
            "osd_ksv": ksv,
            "route_sup": route_sup,
            "route_bal": route_bal,
            "expert_div": expert_div,
            "rank_loss": rank_loss,
            "rank_pairs": rank_pairs,
        }


def get_model(opt):
    return EffortCLIP(
        clip_model_or_path=getattr(opt, "clip_model", None),
        num_classes=int(getattr(opt, "num_classes", 2)),
        enable_osd_regs=bool(getattr(opt, "osd_regs", False)),
        freeze_backbone=bool(getattr(opt, "fix_backbone", False)),
        patch_pool=True,
        patch_pool_tau=float(getattr(opt, "patch_pool_tau", 1.5)),
        patch_pool_mode=str(getattr(opt, "patch_pool_mode", "lse")),
        patch_trim_p=float(getattr(opt, "patch_trim_p", 0.2)),
        patch_quality=str(getattr(opt, "patch_quality", "none")),
        svd_cache_enable=bool(getattr(opt, "svd_cache_enable", True)),
        svd_cache_dir=str(getattr(opt, "svd_cache_dir", "cache")),
        svd_init_device=str(getattr(opt, "svd_init_device", "auto")),
        svd_show_progress=bool(getattr(opt, "svd_show_progress", True)),
        svd_rank_cap=getattr(opt, "svd_rank_cap", None),
        multi_expert_enable=bool(getattr(opt, "multi_expert_enable", False)),
        multi_expert_k=int(getattr(opt, "multi_expert_k", 1)),
        multi_expert_route=str(getattr(opt, "multi_expert_route", "quality_bin")),
        multi_expert_route_temp=float(getattr(opt, "multi_expert_route_temp", 1.0)),
        multi_expert_route_detach=bool(getattr(opt, "multi_expert_route_detach", True)),
        multi_expert_hybrid_mix=float(getattr(opt, "multi_expert_hybrid_mix", 0.0)),
        route_sup_lambda=float(getattr(opt, "multi_expert_route_sup_lambda", 0.0)),
        route_bal_lambda=float(getattr(opt, "multi_expert_balance_lambda", 0.0)),
        expert_div_lambda=float(getattr(opt, "multi_expert_div_lambda", 0.0)),
        rank_loss_enable=bool(getattr(opt, "rank_loss_enable", False)),
        rank_loss_mode=str(getattr(opt, "rank_loss_mode", "logistic")),
        rank_loss_margin=float(getattr(opt, "rank_loss_margin", 0.2)),
        rank_loss_max_pairs=int(getattr(opt, "rank_loss_max_pairs", 1024)),
        quality_cuts=getattr(opt, "quality_cuts", None),
        degrade_router_dim=int(getattr(opt, "degrade_router_dim", 128)),
        degrade_router_dropout=float(getattr(opt, "degrade_router_dropout", 0.1)),
        restoration_enable=bool(getattr(opt, "restoration_enable", False)),
        restoration_strength=float(getattr(opt, "restoration_strength", 0.20)),
        adapter_last_n_layers=int(getattr(opt, "adapter_last_n_layers", 0)),
        svd_residual_mode=str(getattr(opt, "svd_residual_mode", "add")),
        svd_residual_gate_init=float(getattr(opt, "svd_residual_gate_init", 0.0)),
        svd_residual_train_mode=str(getattr(opt, "svd_residual_train_mode", "full")),
        peft_unfreeze_last_n_blocks=int(getattr(opt, "peft_unfreeze_last_n_blocks", 0)),
        peft_unfreeze_layernorm=bool(getattr(opt, "peft_unfreeze_layernorm", True)),
        peft_unfreeze_bias=bool(getattr(opt, "peft_unfreeze_bias", False)),
    )


__all__ = [
    "EffortCLIP",
    "get_model",
]
