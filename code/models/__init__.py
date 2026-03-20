from effort.model import EffortCLIP

from .dinov3_hardroute import DinoV3HardRouteModel


def _build_effort(opt):
    return EffortCLIP(
        clip_model_or_path=getattr(opt, "clip_model", None),
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
        multi_expert_enable=bool(getattr(opt, "multi_expert_enable", False)),
        multi_expert_k=int(getattr(opt, "multi_expert_k", 1)),
        multi_expert_route=str(getattr(opt, "multi_expert_route", "quality_bin")),
        multi_expert_route_temp=float(getattr(opt, "multi_expert_route_temp", 1.0)),
        multi_expert_route_detach=bool(getattr(opt, "multi_expert_route_detach", True)),
        multi_expert_hybrid_mix=float(getattr(opt, "multi_expert_hybrid_mix", 0.0)),
        quality_cuts=getattr(opt, "quality_cuts", None),
        degrade_router_dim=int(getattr(opt, "degrade_router_dim", 128)),
        degrade_router_dropout=float(getattr(opt, "degrade_router_dropout", 0.1)),
        restoration_enable=bool(getattr(opt, "restoration_enable", False)),
        restoration_strength=float(getattr(opt, "restoration_strength", 0.20)),
    )


def _build_dinov3(opt):
    return DinoV3HardRouteModel(
        num_classes=int(getattr(opt, "num_classes", 2)),
        freeze_backbone=bool(getattr(opt, "fix_backbone", False)),
        multi_expert_enable=bool(getattr(opt, "multi_expert_enable", False)),
        multi_expert_k=int(getattr(opt, "multi_expert_k", 1)),
        multi_expert_route=str(getattr(opt, "multi_expert_route", "quality_bin")),
        multi_expert_route_temp=float(getattr(opt, "multi_expert_route_temp", 1.0)),
        quality_cuts=getattr(opt, "quality_cuts", None),
        dino_source=str(getattr(opt, "dino_source", "timm")),
        dino_repo=str(getattr(opt, "dino_repo", "facebookresearch/dinov3")),
        dino_model_name=str(
            getattr(
                opt,
                "dino_model_name",
                "hf-hub:timm/convnext_base.dinov3_lvd1689m",
            )
        ),
        dino_pretrained=bool(getattr(opt, "dino_pretrained", True)),
        dino_local_weights=getattr(opt, "dino_local_weights", None),
        dino_force_imagenet_norm=bool(getattr(opt, "dino_force_imagenet_norm", True)),
        enable_osd_regs=bool(getattr(opt, "osd_regs", False)),
        dino_svd_residual_enable=bool(getattr(opt, "dino_svd_residual_enable", True)),
        dino_svd_residual_dim=int(getattr(opt, "dino_svd_residual_dim", 32)),
        dino_svd_target=str(getattr(opt, "dino_svd_target", "mlp_fc")),
        svd_cache_enable=bool(getattr(opt, "svd_cache_enable", True)),
        svd_cache_dir=str(getattr(opt, "svd_cache_dir", "cache")),
        svd_init_device=str(getattr(opt, "svd_init_device", "auto")),
        svd_show_progress=bool(getattr(opt, "svd_show_progress", True)),
        patch_pool=True,
        patch_pool_tau=float(getattr(opt, "patch_pool_tau", 1.5)),
        patch_pool_mode=str(getattr(opt, "patch_pool_mode", "lse")),
        patch_trim_p=float(getattr(opt, "patch_trim_p", 0.2)),
        patch_quality=str(getattr(opt, "patch_quality", "none")),
    )


def get_model(opt):
    family = str(getattr(opt, "model_family", "effort") or "effort").lower()
    if family in {"dinov3", "dino3", "dino_v3"}:
        return _build_dinov3(opt)
    return _build_effort(opt)


__all__ = ["EffortCLIP", "DinoV3HardRouteModel", "get_model"]
