import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_cuda_conv_runtime(allow_cudnn_fallback: bool = True):
    """
    Probe a tiny CUDA conv2d op and fallback by disabling cuDNN on known engine errors.
    The probe uses fixed tensors (no random init), so it does not perturb RNG state.
    """
    status = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_enabled": bool(getattr(torch.backends.cudnn, "enabled", False)),
        "fallback_applied": False,
        "probe_ok": False,
    }
    if not status["cuda_available"]:
        return status

    def _probe_once():
        x = torch.zeros((1, 3, 32, 32), device="cuda", dtype=torch.float32)
        w = torch.zeros((8, 3, 3, 3), device="cuda", dtype=torch.float32)
        y = F.conv2d(x, w, bias=None, stride=1, padding=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return y

    try:
        _probe_once()
        status["probe_ok"] = True
        status["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
        return status
    except RuntimeError as e:
        msg = str(e).lower()
        is_known = "unable to find an engine to execute this computation" in msg
        if (not allow_cudnn_fallback) or (not is_known):
            raise

    torch.backends.cudnn.enabled = False
    _probe_once()
    status["fallback_applied"] = True
    status["probe_ok"] = True
    status["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
    return status


def save_checkpoint(path, model, optimizer, scaler, epoch, best_metric=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
    }
    torch.save(payload, str(path))


def load_ckpt(path, model, strict=False, weights_only=False):
    ckpt = torch.load(path, map_location="cpu", weights_only=weights_only)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a valid state dict")

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    return missing, unexpected


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
