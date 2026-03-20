import logging
import hashlib
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from effort.svd_residual import (
    SVDResidualLinear,
    apply_svd_residual_to_linears_ex,
    set_svd_residual_num_experts,
)

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _stats(device: torch.device, dtype: torch.dtype, mean_vals, std_vals):
    mean = torch.tensor(mean_vals, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(std_vals, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def _clip_to_imagenet(x: torch.Tensor) -> torch.Tensor:
    c_mean, c_std = _stats(x.device, x.dtype, CLIP_MEAN, CLIP_STD)
    i_mean, i_std = _stats(x.device, x.dtype, IMAGENET_MEAN, IMAGENET_STD)
    rgb = (x * c_std + c_mean).clamp(0.0, 1.0)
    return (rgb - i_mean) / i_std


class DinoV3HardRouteModel(nn.Module):
    """DINOv3-based classifier with handcrafted routing and patch evidence head."""

    def __init__(
        self,
        *,
        num_classes: int = 2,
        freeze_backbone: bool = False,
        multi_expert_enable: bool = False,
        multi_expert_k: int = 1,
        multi_expert_route: str = "quality_bin",
        multi_expert_route_temp: float = 1.0,
        quality_cuts=None,
        dino_source: str = "timm",
        dino_repo: str = "facebookresearch/dinov3",
        dino_model_name: str = "hf-hub:timm/convnext_base.dinov3_lvd1689m",
        dino_pretrained: bool = True,
        dino_local_weights: Optional[str] = None,
        dino_force_imagenet_norm: bool = True,
        enable_osd_regs: bool = False,
        dino_svd_residual_enable: bool = True,
        dino_svd_residual_dim: int = 32,
        dino_svd_target: str = "mlp_fc",
        svd_cache_enable: bool = True,
        svd_cache_dir: str = "cache",
        svd_init_device: str = "auto",
        svd_show_progress: bool = True,
        patch_pool: bool = True,
        patch_pool_tau: float = 1.5,
        patch_pool_mode: str = "lse",
        patch_trim_p: float = 0.2,
        patch_quality: str = "none",
    ):
        super().__init__()
        self.dino_source = str(dino_source or "timm").lower()
        self.dino_repo = str(dino_repo or "facebookresearch/dinov3")
        self.dino_model_name = str(dino_model_name or "hf-hub:timm/convnext_base.dinov3_lvd1689m")
        self.dino_pretrained = bool(dino_pretrained)
        self.dino_force_imagenet_norm = bool(dino_force_imagenet_norm)
        self.freeze_backbone = bool(freeze_backbone)
        self.enable_osd_regs = bool(enable_osd_regs)
        self.dino_svd_residual_enable = bool(dino_svd_residual_enable)
        self.dino_svd_residual_dim = int(max(1, int(dino_svd_residual_dim or 32)))
        self.dino_svd_target = str(dino_svd_target or "mlp_fc").lower()
        self.svd_cache_enable = bool(svd_cache_enable)
        self.svd_cache_dir = str(svd_cache_dir)
        self.svd_init_device = str(svd_init_device or "auto")
        self.svd_show_progress = bool(svd_show_progress)
        self.multi_expert_enable = bool(multi_expert_enable) and int(multi_expert_k or 1) > 1
        self.multi_expert_k = max(1, int(multi_expert_k or 1))
        self.multi_expert_route = str(multi_expert_route or "quality_bin").lower()
        self.multi_expert_route_temp = max(1e-6, float(multi_expert_route_temp or 1.0))
        self.quality_cuts = None
        if quality_cuts is not None:
            try:
                cuts = sorted([float(v) for v in list(quality_cuts)])
                if len(cuts) > 0:
                    self.quality_cuts = cuts
            except Exception:
                self.quality_cuts = None

        self.backbone = self._build_backbone()
        self._maybe_load_local_weights(dino_local_weights)
        if self.dino_svd_residual_enable and self.multi_expert_enable:
            set_svd_residual_num_experts(self.backbone, num_experts=self.multi_expert_k, init="copy")
        feat_dim = self._infer_feat_dim()

        if self.freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if any(k in name for k in ("S_residual", "U_residual", "V_residual")):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        self.head = nn.Linear(feat_dim, int(num_classes))
        if self.multi_expert_enable and (not self.dino_svd_residual_enable):
            self.expert_heads = nn.ModuleList([nn.Linear(feat_dim, int(num_classes)) for _ in range(self.multi_expert_k)])
        else:
            self.expert_heads = None
        self.patch_pool = bool(patch_pool)
        self.patch_pool_tau = float(patch_pool_tau)
        self.patch_pool_mode = str(patch_pool_mode).lower()
        self.patch_trim_p = float(patch_trim_p)
        self.patch_quality = str(patch_quality).lower()
        self.patch_scale = 0.0
        self.patch_head = nn.Linear(feat_dim, 1) if self.patch_pool else None
        self.loss_func = nn.CrossEntropyLoss()

    def set_patch_scale(self, scale: float) -> None:
        self.patch_scale = float(scale)

    def _svd_target_predicate(self, name, module) -> bool:
        _ = module
        target = self.dino_svd_target
        if target in {"all", "all_linear"}:
            return True
        if target in {"attn", "attn_linear", "self_attn"}:
            n = str(name)
            # HF CLIP style: *.self_attn.{q_proj,k_proj,v_proj,out_proj}
            # timm ViT style: blocks.*.attn.{qkv,proj}
            return ("self_attn" in n) or (".attn.qkv" in n) or (".attn.proj" in n)
        if target in {"attn_qkv"}:
            n = str(name)
            return ("self_attn" in n and any(x in n for x in (".q_proj", ".k_proj", ".v_proj"))) or (".attn.qkv" in n)
        if target in {"attn_out", "attn_proj"}:
            n = str(name)
            return ("self_attn" in n and ".out_proj" in n) or (".attn.proj" in n)
        if target in {"mlp_fc", "mlp", "fc"}:
            return ".mlp.fc" in str(name)
        if target in {"mlp_fc1", "fc1"}:
            return ".mlp.fc1" in str(name)
        if target in {"mlp_fc2", "fc2"}:
            return ".mlp.fc2" in str(name)
        return ".mlp.fc" in str(name)

    def _dino_svd_cache_path(self) -> Path:
        cache_dir = Path(self.svd_cache_dir).expanduser()
        key = (
            f"dino|{self.dino_source}|{self.dino_model_name}|"
            f"rd{self.dino_svd_residual_dim}|target={self.dino_svd_target}"
        )
        key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        return cache_dir / f"svd_dino_{key_hash}.pt"

    def _build_backbone(self):
        if self.dino_source == "torchhub":
            model = torch.hub.load(self.dino_repo, self.dino_model_name, pretrained=self.dino_pretrained)
        elif self.dino_source == "timm":
            try:
                import timm
            except Exception as e:
                raise RuntimeError("timm is required for dino_source=timm") from e
            create_kwargs = {
                "pretrained": self.dino_pretrained,
                "num_classes": 0,
            }
            name_l = self.dino_model_name.lower()
            is_vit_like = ("vit" in name_l) or ("deit" in name_l)
            if not is_vit_like:
                create_kwargs["global_pool"] = "avg"
            else:
                # Allow training on the project's 224x224 pipeline.
                create_kwargs["dynamic_img_size"] = True
            model = timm.create_model(self.dino_model_name, **create_kwargs)
        else:
            raise ValueError(f"Unsupported dino_source: {self.dino_source}")

        if not self.dino_svd_residual_enable:
            return model

        cache_path = self._dino_svd_cache_path()
        if self.svd_cache_enable and cache_path.exists():
            logging.info("[svd-dino] loading cache: %s", str(cache_path))
            model = apply_svd_residual_to_linears_ex(
                model=model,
                residual_dim=self.dino_svd_residual_dim,
                name_predicate=self._svd_target_predicate,
                svd_device=None,
                show_progress=False,
                fast_init=True,
                freeze_non_residual=False,
            )
            try:
                payload = torch.load(str(cache_path), map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(str(cache_path), map_location="cpu")
            state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            missing, unexpected = model.load_state_dict(state, strict=False)
            logging.info("[svd-dino] cache loaded: missing=%d unexpected=%d", len(missing), len(unexpected))
            return model

        init_dev = self.svd_init_device.lower()
        if init_dev == "auto":
            init_dev = "cuda" if torch.cuda.is_available() else "cpu"
        if init_dev == "cuda" and (not torch.cuda.is_available()):
            init_dev = "cpu"
        logging.info(
            "[svd-dino] init start: device=%s progress=%s target=%s cache=%s",
            init_dev,
            self.svd_show_progress,
            self.dino_svd_target,
            str(cache_path) if self.svd_cache_enable else "disabled",
        )
        model = apply_svd_residual_to_linears_ex(
            model=model,
            residual_dim=self.dino_svd_residual_dim,
            name_predicate=self._svd_target_predicate,
            svd_device=init_dev,
            show_progress=self.svd_show_progress,
            fast_init=False,
            freeze_non_residual=False,
        )
        if self.svd_cache_enable:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_key": self.dino_model_name,
                    "target": self.dino_svd_target,
                    "residual_dim": self.dino_svd_residual_dim,
                },
                str(cache_path),
            )
            logging.info("[svd-dino] cache saved: %s", str(cache_path))
        return model

    def _maybe_load_local_weights(self, weight_path: Optional[str]) -> None:
        if not weight_path:
            return
        path = str(weight_path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model", "backbone"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported local weight format: {path}")
        stripped = {}
        for k, v in state.items():
            nk = str(k)
            for pref in ("module.", "backbone."):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            stripped[nk] = v
        missing, unexpected = self.backbone.load_state_dict(stripped, strict=False)
        logging.info(
            "[dinov3] loaded local backbone weights from %s | missing=%d unexpected=%d",
            path,
            len(list(missing or [])),
            len(list(unexpected or [])),
        )

    def _extract_feat_and_tokens(self, x: torch.Tensor):
        if self.dino_force_imagenet_norm:
            x = _clip_to_imagenet(x)
        if self.dino_source == "timm" and hasattr(self.backbone, "forward_features"):
            y = self.backbone.forward_features(x)
        else:
            y = self.backbone(x)
        if isinstance(y, dict):
            for key in ("x_norm_clstoken", "pooler_output", "last_hidden_state"):
                if key in y and torch.is_tensor(y[key]):
                    y = y[key]
                    break
            else:
                tvals = [v for v in y.values() if torch.is_tensor(v)]
                if not tvals:
                    raise RuntimeError("Backbone dict output has no tensor values.")
                y = tvals[0]
        elif isinstance(y, (list, tuple)):
            tvals = [v for v in y if torch.is_tensor(v)]
            if not tvals:
                raise RuntimeError("Backbone tuple output has no tensor values.")
            y = tvals[0]
        if not torch.is_tensor(y):
            raise RuntimeError(f"Unsupported backbone output type: {type(y)}")
        if y.dim() == 4:
            feat_dim_hint = None
            for attr in ("num_features", "embed_dim", "feature_dim"):
                if hasattr(self.backbone, attr):
                    try:
                        feat_dim_hint = int(getattr(self.backbone, attr))
                        break
                    except Exception:
                        feat_dim_hint = None
            # timm backbones such as Swin may emit NHWC features from forward_features.
            if (feat_dim_hint is not None) and int(y.shape[-1]) == int(feat_dim_hint) and int(y.shape[1]) != int(feat_dim_hint):
                feat = y.mean(dim=(1, 2))
                patch_tok = y.reshape(y.shape[0], -1, y.shape[-1])
                return feat, patch_tok
            feat = F.adaptive_avg_pool2d(y, output_size=1).flatten(1)
            patch_tok = y.flatten(2).transpose(1, 2)
            return feat, patch_tok
        if y.dim() == 3:
            if y.shape[1] > 1:
                feat = y[:, 0, :]
                patch_tok = y[:, 1:, :]
                if patch_tok.shape[1] == 0:
                    patch_tok = y
            else:
                feat = y.mean(dim=1)
                patch_tok = y
            return feat, patch_tok
        if y.dim() == 2:
            return y, y.unsqueeze(1)
        if y.dim() > 4:
            feat = y.flatten(1)
            return feat, feat.unsqueeze(1)
        raise RuntimeError(f"Unsupported feature shape: {tuple(y.shape)}")

    def _infer_feat_dim(self) -> int:
        for attr in ("num_features", "embed_dim", "feature_dim"):
            if hasattr(self.backbone, attr):
                try:
                    return int(getattr(self.backbone, attr))
                except Exception:
                    pass
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            f, _ = self._extract_feat_and_tokens(x)
        if f.dim() != 2:
            raise RuntimeError(f"Expected 2D feature tensor, got shape {tuple(f.shape)}")
        return int(f.shape[1])

    def _uniform_route(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        k = int(self.multi_expert_k)
        return torch.full((int(batch_size), k), 1.0 / float(k), device=device, dtype=dtype)

    def _set_route_alpha(self, alpha: Optional[torch.Tensor]) -> None:
        if not self.dino_svd_residual_enable:
            return
        for m in self.backbone.modules():
            if isinstance(m, SVDResidualLinear):
                m.set_route_alpha(alpha)

    def _clear_route_alpha(self) -> None:
        if not self.dino_svd_residual_enable:
            return
        for m in self.backbone.modules():
            if isinstance(m, SVDResidualLinear):
                m.clear_route_alpha()

    def _hard_route_quality(
        self,
        data_dict: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if "quality_bin" in data_dict:
            qb = data_dict["quality_bin"].long()
            qb = qb.clamp_min(0).clamp_max(int(self.multi_expert_k) - 1)
            return F.one_hot(qb, num_classes=int(self.multi_expert_k)).to(device=device, dtype=dtype)
        if ("q_log" in data_dict) and (self.quality_cuts is not None):
            q = data_dict["q_log"].float().to(device=device).reshape(-1)
            cuts_t = torch.tensor(self.quality_cuts, device=device, dtype=q.dtype)
            qb = torch.bucketize(q, cuts_t).long().clamp_min(0).clamp_max(int(self.multi_expert_k) - 1)
            return F.one_hot(qb, num_classes=int(self.multi_expert_k)).to(device=device, dtype=dtype)
        return self._uniform_route(batch_size=batch_size, device=device, dtype=dtype)

    def _soft_route_quality(
        self,
        data_dict: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if ("q_log" not in data_dict) or (self.quality_cuts is None):
            return self._hard_route_quality(data_dict, batch_size, device, dtype)
        q = data_dict["q_log"].float().to(device=device).reshape(-1)
        if int(q.numel()) != int(batch_size):
            if int(q.numel()) <= 0:
                return self._uniform_route(batch_size, device, dtype)
            if int(q.numel()) == 1:
                q = q.expand(int(batch_size))
            else:
                q = q[: int(batch_size)]
        k = int(self.multi_expert_k)
        cuts = torch.tensor(self.quality_cuts, device=device, dtype=q.dtype).reshape(-1)
        if int(cuts.numel()) == max(0, k - 1):
            if int(cuts.numel()) >= 2:
                width = (cuts[1:] - cuts[:-1]).median().clamp_min(1e-3)
            else:
                width = torch.tensor(1.0, device=device, dtype=q.dtype)
            left = cuts[0] - width
            right = cuts[-1] + width
            edges = torch.cat([left.view(1), cuts, right.view(1)], dim=0)
            centers = 0.5 * (edges[:-1] + edges[1:])
        else:
            qmin = q.min()
            qmax = q.max()
            if float((qmax - qmin).abs().item()) < 1e-6:
                return self._uniform_route(batch_size, device, dtype)
            centers = torch.linspace(qmin, qmax, steps=k, device=device, dtype=q.dtype)
        dist = (q[:, None] - centers[None, :]).abs()
        alpha = torch.softmax(-dist / float(self.multi_expert_route_temp), dim=1).to(dtype=dtype)
        return alpha

    def _hard_route_degrade(
        self,
        data_dict: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if "degrade_bin" not in data_dict:
            return self._uniform_route(batch_size=batch_size, device=device, dtype=dtype)
        db = data_dict["degrade_bin"].long()
        db = db.clamp_min(0).clamp_max(int(self.multi_expert_k) - 1)
        return F.one_hot(db, num_classes=int(self.multi_expert_k)).to(device=device, dtype=dtype)

    def _prepare_route(
        self,
        data_dict: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.multi_expert_enable:
            return None
        route = self.multi_expert_route
        if route == "quality_bin":
            return self._hard_route_quality(data_dict, batch_size, device, dtype)
        if route == "quality_soft":
            return self._soft_route_quality(data_dict, batch_size, device, dtype)
        if route == "degrade_bin":
            return self._hard_route_degrade(data_dict, batch_size, device, dtype)
        if route == "uniform":
            return self._uniform_route(batch_size, device, dtype)
        # Learned/dynamic routes are intentionally not used here.
        return self._hard_route_quality(data_dict, batch_size, device, dtype)

    def _expert_div_loss(self) -> torch.Tensor:
        if (self.expert_heads is None) or (not self.multi_expert_enable) or int(self.multi_expert_k) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        norms = []
        for head in self.expert_heads:
            w = head.weight.float().flatten()
            w = F.normalize(w, dim=0)
            norms.append(w)
        if len(norms) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        sims = []
        for i in range(len(norms)):
            for j in range(i + 1, len(norms)):
                sims.append((norms[i] * norms[j]).sum().abs())
        if not sims:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return torch.stack(sims).mean()

    @torch.no_grad()
    def _osd_reg_modules(self):
        for m in self.backbone.modules():
            if isinstance(m, SVDResidualLinear):
                yield m

    def compute_osd_regularizers(self):
        device = next(self.parameters()).device
        orth = torch.tensor(0.0, device=device)
        ksv = torch.tensor(0.0, device=device)
        count = 0
        for m in self._osd_reg_modules():
            if hasattr(m, "compute_orthogonal_loss"):
                orth = orth + m.compute_orthogonal_loss()
            if hasattr(m, "compute_keepsv_loss"):
                ksv = ksv + m.compute_keepsv_loss()
            count += 1
        if count > 0:
            orth = orth / count
            ksv = ksv / count
        return orth, ksv

    def compute_residual_expert_div_loss(self):
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        count = 0
        for m in self._osd_reg_modules():
            if hasattr(m, "compute_expert_div_loss"):
                v = m.compute_expert_div_loss()
                if not torch.is_tensor(v):
                    v = torch.tensor(float(v), device=device)
                loss = loss + v
                count += 1
        if count > 0:
            loss = loss / count
        return loss

    def _patch_quality_weights(self, patch_tok: torch.Tensor, cls_feat: torch.Tensor):
        mode = (self.patch_quality or "none").lower()
        if mode == "none":
            return None

        pt = patch_tok.float()
        cf = cls_feat.float().unsqueeze(1)
        pt_n = F.normalize(pt, dim=-1)
        cf_n = F.normalize(cf, dim=-1)
        cos = (pt_n * cf_n).sum(dim=-1)
        q = (cos + 1.0) * 0.5
        q = q.clamp(0.0, 1.0)

        if mode == "cos_norm":
            n = pt.norm(dim=-1)
            med = n.median(dim=1, keepdim=True).values.clamp_min(1e-6)
            n_rel = (n / med).clamp(0.0, 3.0) / 3.0
            q = q * n_rel

        return q

    def _pool_patch_logits(self, patch_logits: torch.Tensor, q):
        tau = max(1e-6, float(self.patch_pool_tau))
        s = patch_logits / tau

        if q is not None:
            w = q.clamp_min(1e-6)
            wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            s = s + (torch.log(w) - torch.log(wsum))
            return tau * torch.logsumexp(s, dim=1)

        n = s.shape[1]
        log_n = s.new_tensor(float(n)).log()
        return tau * (torch.logsumexp(s, dim=1) - log_n)

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = data_dict["image"]
        out = {}
        if self.multi_expert_enable:
            route_alpha = self._prepare_route(
                data_dict=data_dict,
                batch_size=int(x.shape[0]),
                device=x.device,
                dtype=x.dtype,
            )
            out["route_alpha"] = route_alpha
            if self.dino_svd_residual_enable:
                self._set_route_alpha(route_alpha)
                try:
                    feat, patch_tok = self._extract_feat_and_tokens(x)
                finally:
                    self._clear_route_alpha()
                logits = self.head(feat)
            else:
                feat, patch_tok = self._extract_feat_and_tokens(x)
                logits_each = torch.stack([head(feat) for head in self.expert_heads], dim=1)
                logits = (logits_each * route_alpha.unsqueeze(-1)).sum(dim=1)
                out["expert_logits"] = logits_each
        else:
            feat, patch_tok = self._extract_feat_and_tokens(x)
            logits = self.head(feat)
        out["feat"] = feat
        out["cls"] = logits

        if self.patch_pool and (self.patch_head is not None):
            patch_logits = self.patch_head(patch_tok).squeeze(-1)
            q = self._patch_quality_weights(patch_tok, feat)
            evidence = self._pool_patch_logits(
                patch_logits.float(),
                q.float() if q is not None else None,
            )
            delta = evidence.to(logits.dtype) * float(self.patch_scale)
            out["patch_evidence"] = evidence
            out["patch_delta"] = delta.detach()
            out["patch_logit_min"] = patch_logits.min().detach()
            out["patch_logit_mean"] = patch_logits.mean().detach()
            out["patch_logit_max"] = patch_logits.max().detach()

        out["prob"] = torch.softmax(logits, dim=1)[:, 1]
        return out

    def get_losses(self, data_dict: Dict[str, torch.Tensor], pred_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        label = data_dict["label"].long()
        pred = pred_dict["cls"]
        loss = self.loss_func(pred, label)

        mask_real = label == 0
        mask_fake = label == 1
        if mask_real.any():
            loss_real = self.loss_func(pred[mask_real], label[mask_real])
        else:
            loss_real = torch.tensor(0.0, device=pred.device)
        if mask_fake.any():
            loss_fake = self.loss_func(pred[mask_fake], label[mask_fake])
        else:
            loss_fake = torch.tensor(0.0, device=pred.device)

        if self.enable_osd_regs and self.dino_svd_residual_enable:
            orth, ksv = self.compute_osd_regularizers()
        else:
            orth = torch.tensor(0.0, device=pred.device)
            ksv = torch.tensor(0.0, device=pred.device)

        route_sup = torch.tensor(0.0, device=pred.device)
        route_bal = torch.tensor(0.0, device=pred.device)
        if self.multi_expert_enable and ("route_alpha" in pred_dict):
            pa = pred_dict["route_alpha"].float()
            p_bar = pa.reshape(-1, pa.shape[-1]).mean(dim=0)
            p_bar = p_bar / p_bar.sum().clamp_min(1e-12)
            uni = torch.full_like(p_bar, 1.0 / float(max(1, p_bar.numel())))
            route_bal = (p_bar * (p_bar.clamp_min(1e-12).log() - uni.log())).sum()
        if self.multi_expert_enable and self.dino_svd_residual_enable:
            expert_div = self.compute_residual_expert_div_loss()
        else:
            expert_div = self._expert_div_loss()

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
        }
