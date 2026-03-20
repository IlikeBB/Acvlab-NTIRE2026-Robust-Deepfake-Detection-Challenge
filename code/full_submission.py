#!/usr/bin/env python3
"""
Generate submission probabilities for Effort/AIGI models.

Example:
  python full_submission.py\
      --ckpt /work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_full6_8f_s2048_local/best_auc.pt\
          --txt_path /work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final\
              --root_dir /work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final\
                  --out_csv ./submission.txt
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image, ImageFile

from models import get_model
from effort.utils import load_ckpt, ensure_cuda_conv_runtime
from Config import (
    MODEL_FAMILY,
    CLIP_MODEL,
    DINO_SOURCE,
    DINO_REPO,
    DINO_MODEL_NAME,
    DINO_PRETRAINED,
    DINO_FORCE_IMAGENET_NORM,
    IMAGE_SIZE,
    NUM_WORKERS,
    PATCH_POOL_TAU,
    PATCH_POOL_MODE,
    PATCH_TRIM_P,
    PATCH_QUALITY,
    PATCH_POOL_GAMMA,
)

DEFAULT_ROOT = Path("/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final")
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_image(path: str):
    with Image.open(path) as img:
        return img.convert("RGB")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_label(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"0", "real"}:
        return 0
    if v in {"1", "fake"}:
        return 1
    try:
        return int(v)
    except Exception:
        return None


def _infer_label_from_name(path: Path) -> Optional[int]:
    name = path.name.lower()
    if "fake" in name:
        return 1
    if "real" in name:
        return 0
    return None


def _sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def _read_txt(txt_path: Path, root_dir: Path) -> List[Tuple[str, int, bool]]:
    records: List[Tuple[str, int, bool]] = []
    missing = 0
    with txt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                parts = [p.strip() for p in line.split(",") if p.strip()]
            else:
                parts = line.split()
            if not parts:
                continue

            path_str = parts[0]
            label = _parse_label(parts[1]) if len(parts) > 1 else None
            path = Path(path_str)
            if not path.is_absolute():
                path = root_dir / path

            if label is None:
                label = _infer_label_from_name(path)

            label_valid = label is not None
            if label is None:
                label = -1

            if not path.exists():
                missing += 1
                continue

            records.append((str(path), int(label), label_valid))

    if missing:
        print(f"[warn] skipped {missing} missing files from txt.")
    if not records:
        raise RuntimeError("No valid records found in txt.")
    return records


def _read_dir(root_dir: Path) -> List[Tuple[str, int, bool]]:
    root_dir = root_dir.resolve()
    records: List[Tuple[str, int, bool]] = []
    paths = [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    paths.sort(key=_sort_key)
    for p in paths:
        label = _infer_label_from_name(p)
        label_valid = label is not None
        if label is None:
            label = -1
        records.append((str(p), int(label), label_valid))
    if not records:
        raise RuntimeError(f"No images found under {root_dir}")
    return records


def _write_txt_from_dir(root_dir: Path, out_path: Path) -> None:
    root_dir = root_dir.resolve()
    records = []
    skipped = 0
    paths = [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    paths.sort(key=_sort_key)
    for p in paths:
        label = _infer_label_from_name(p)
        if label is None:
            skipped += 1
            continue
        rel = p.relative_to(root_dir)
        records.append((str(rel), label))

    if not records:
        raise RuntimeError(f"No labeled images found under {root_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rel, label in records:
            f.write(f"{rel} {label}\n")

    print(f"[txt] saved {len(records)} entries -> {out_path}")
    if skipped:
        print(f"[txt] skipped {skipped} files without real/fake in name")


def _laplacian_var_log(img: Image.Image, max_side: int = 256, eps: float = 1e-6) -> float:
    if int(max_side or 0) > 0:
        w, h = img.size
        m = max(w, h)
        if m > int(max_side):
            scale = float(max_side) / float(m)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            img = img.resize((nw, nh), resample=Image.BILINEAR)

    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (
        -4.0 * g
        + np.roll(g, 1, axis=0)
        + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1)
        + np.roll(g, -1, axis=1)
    )
    v = float(np.var(lap))
    return float(np.log(v + float(eps)))


def _normalize_quality_metric(metric: Optional[str]) -> str:
    m = str(metric or "laplacian").strip().lower()
    if m in {"clipiqa", "clip-iqa", "clip_iqa"}:
        return "clipiqa"
    return "laplacian"


def _parse_clipiqa_weights(weights):
    default = (0.5, 0.3, 0.2)
    if weights is None:
        return default
    try:
        vals = [float(v) for v in list(weights)]
    except Exception:
        return default
    if len(vals) != 3:
        return default
    s = float(sum(vals))
    if s <= 0:
        return default
    vals = [v / s for v in vals]
    return (float(vals[0]), float(vals[1]), float(vals[2]))


class _CLIPIQABatchScorer:
    def __init__(self, *, model_device: torch.device, clipiqa_device: Optional[str] = None, weights=None):
        try:
            from torchmetrics.multimodal import CLIPImageQualityAssessment
        except Exception as e:
            raise RuntimeError(
                "CLIP-IQA requires torchmetrics multimodal extras (and piq). "
                "Install in current env, e.g. `pip install torchmetrics[multimodal] piq`."
            ) from e

        self.model_device = model_device
        if clipiqa_device:
            dev = torch.device(str(clipiqa_device))
        else:
            dev = model_device
        if dev.type == "cuda" and (not torch.cuda.is_available()):
            dev = torch.device("cpu")
        self.device = dev
        self.weights = _parse_clipiqa_weights(weights)
        self.metric = CLIPImageQualityAssessment(prompts=("quality", "sharpness", "noisiness")).to(self.device)
        self.metric.eval()
        self.to_tensor = T.ToTensor()

    @staticmethod
    def _resize_for_quality(img: Image.Image, max_side: int = 256) -> Image.Image:
        if int(max_side or 0) > 0:
            w, h = img.size
            m = max(w, h)
            if m > int(max_side):
                scale = float(max_side) / float(m)
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                img = img.resize((nw, nh), resample=Image.BILINEAR)
        return img

    @torch.no_grad()
    def score_from_normalized(self, images_norm: torch.Tensor) -> torch.Tensor:
        x = _denorm_clip(images_norm).float().clamp(0.0, 1.0).to(self.device)
        out = self.metric(x)
        q = out.get("quality", None)
        s = out.get("sharpness", None)
        n = out.get("noisiness", None)
        if q is None:
            score = torch.full((x.shape[0],), 0.5, device=self.device, dtype=torch.float32)
        else:
            q = q.float()
            s = s.float() if s is not None else q
            n = n.float() if n is not None else torch.zeros_like(q)
            wq, ws, wn = self.weights
            score = float(wq) * q + float(ws) * s + float(wn) * (1.0 - n)
        return score.to(device=self.model_device, dtype=torch.float32)

    @torch.no_grad()
    def score_from_paths(self, paths: List[str], *, max_side: int = 256) -> torch.Tensor:
        tensors = []
        for p in paths:
            img = _load_image(p)
            img = self._resize_for_quality(img, max_side=max_side)
            tensors.append(self.to_tensor(img))
        x = torch.stack(tensors, dim=0).to(self.device)
        out = self.metric(x)
        q = out.get("quality", None)
        s = out.get("sharpness", None)
        n = out.get("noisiness", None)
        if q is None:
            score = torch.full((x.shape[0],), 0.5, device=self.device, dtype=torch.float32)
        else:
            q = q.float()
            s = s.float() if s is not None else q
            n = n.float() if n is not None else torch.zeros_like(q)
            wq, ws, wn = self.weights
            score = float(wq) * q + float(ws) * s + float(wn) * (1.0 - n)
        return score.to(device=self.model_device, dtype=torch.float32)


def _degrade_metrics(img: Image.Image, max_side: int = 256, eps: float = 1e-6):
    if int(max_side or 0) > 0:
        w, h = img.size
        m = max(w, h)
        if m > int(max_side):
            scale = float(max_side) / float(m)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            img = img.resize((nw, nh), resample=Image.BILINEAR)

    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (
        -4.0 * g
        + np.roll(g, 1, axis=0)
        + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1)
        + np.roll(g, -1, axis=1)
    )
    q_log = float(np.log(float(np.var(lap)) + float(eps)))

    g_mean = (
        g
        + np.roll(g, 1, axis=0)
        + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1)
        + np.roll(g, -1, axis=1)
        + np.roll(np.roll(g, 1, axis=0), 1, axis=1)
        + np.roll(np.roll(g, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(g, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(g, -1, axis=0), -1, axis=1)
    ) / 9.0
    resid = g - g_mean
    noise_log = float(np.log(float(np.var(resid)) + float(eps)))

    gv = np.abs(g[:, 1:] - g[:, :-1])
    gh = np.abs(g[1:, :] - g[:-1, :])
    if gv.shape[1] > 8:
        cols = np.arange(7, gv.shape[1], 8, dtype=np.int64)
        bnd_v = float(gv[:, cols].mean())
    else:
        bnd_v = float(gv.mean())
    if gh.shape[0] > 8:
        rows = np.arange(7, gh.shape[0], 8, dtype=np.int64)
        bnd_h = float(gh[rows, :].mean())
    else:
        bnd_h = float(gh.mean())
    bnd = 0.5 * (bnd_v + bnd_h)
    base = 0.5 * (float(gv.mean()) + float(gh.mean()))
    block_log = float(np.log((bnd / (base + float(eps))) + float(eps)))

    return q_log, noise_log, block_log


def _infer_degrade_bin(img: Image.Image, degrade_bins: int = 3, max_side: int = 256) -> int:
    q_log, noise_log, block_log = _degrade_metrics(img, max_side=max_side)
    scores = np.asarray([-q_log, noise_log, block_log], dtype=np.float32)
    dom = int(np.argmax(scores))
    n_bins = int(max(1, degrade_bins))
    if n_bins <= 1:
        return 0
    if n_bins == 2:
        return int(dom != 0)  # blur vs non-blur
    if n_bins == 3:
        return int(dom)
    # Fallback when requesting >3 bins.
    dom = max(0, min(2, dom))
    return int(round(dom * float(n_bins - 1) / 2.0))


class TxtImageDataset(Dataset):
    def __init__(
        self,
        records: List[Tuple[str, int, bool]],
        image_size: int,
        on_error: str = "skip",
        emit_q_log: bool = False,
        quality_metric: str = "laplacian",
        quality_max_side: int = 256,
        emit_degrade_bin: bool = False,
        degrade_bins: int = 3,
        degrade_max_side: int = 256,
    ):
        self.records = records
        self.on_error = on_error
        self.emit_q_log = bool(emit_q_log)
        self.quality_metric = _normalize_quality_metric(quality_metric)
        self.quality_max_side = int(quality_max_side)
        self.emit_degrade_bin = bool(emit_degrade_bin)
        self.degrade_bins = int(max(1, int(degrade_bins)))
        self.degrade_max_side = int(degrade_max_side)
        self.transform = T.Compose(
            [
                T.Resize((int(image_size), int(image_size))),
                T.ToTensor(),
                T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        path, label, label_valid = self.records[idx]
        try:
            img = _load_image(path)
            q_log = None
            degrade_bin = None
            if self.emit_q_log and self.quality_metric == "laplacian":
                q_log = _laplacian_var_log(img, max_side=self.quality_max_side)
            if self.emit_degrade_bin:
                degrade_bin = _infer_degrade_bin(
                    img,
                    degrade_bins=self.degrade_bins,
                    max_side=self.degrade_max_side,
                )
            img = self.transform(img)
            out = {
                "image": img,
                "label": torch.tensor(label, dtype=torch.long),
                "path": path,
                "label_valid": label_valid,
            }
            if q_log is not None:
                out["q_log"] = torch.tensor(float(q_log), dtype=torch.float32)
            if degrade_bin is not None:
                out["degrade_bin"] = torch.tensor(int(degrade_bin), dtype=torch.long)
            return out
        except Exception:
            if self.on_error == "skip":
                return None
            raise


def _collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    paths = [b["path"] for b in batch]
    label_valid = [bool(b["label_valid"]) for b in batch]
    out = {
        "image": images,
        "label": labels,
        "path": paths,
        "label_valid": label_valid,
    }
    if "q_log" in batch[0]:
        out["q_log"] = torch.stack([b["q_log"] for b in batch], dim=0)
    if "degrade_bin" in batch[0]:
        out["degrade_bin"] = torch.stack([b["degrade_bin"] for b in batch], dim=0)
    return out


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


def _print_key_summary(tag: str, keys, max_show: int = 8) -> None:
    if not keys:
        return
    klist = list(keys)
    print(f"[load] {tag}: {len(klist)}")
    show = min(int(max_show), len(klist))
    for k in klist[:show]:
        print(f"  - {k}")
    if len(klist) > show:
        print(f"  ... (+{len(klist) - show} more)")


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
    x = torch.nn.functional.conv2d(x, ky, padding=(pad, 0), groups=3)
    x = torch.nn.functional.conv2d(x, kx, padding=(0, pad), groups=3)
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
    blk = torch.nn.functional.avg_pool2d(x, kernel_size=8, stride=8)
    blk = torch.nn.functional.interpolate(blk, size=x.shape[-2:], mode="nearest")
    return ((1.0 - strength) * x + strength * blk).clamp(0.0, 1.0)


def _tri_view_degrade(
    images_norm: torch.Tensor,
    *,
    blur_sig=(0.2, 1.2),
    jpeg_qual=(70, 95),
    resize_jitter: int = 4,
) -> torch.Tensor:
    x = _denorm_clip(images_norm).float()

    b_lo, b_hi = _to_float_range(blur_sig, (0.2, 1.2))
    if b_hi > 0.0:
        sigma = float(torch.empty((), device=x.device).uniform_(b_lo, b_hi).item())
        x = _gaussian_blur_rgb(x, sigma=sigma)

    rj = int(resize_jitter or 0)
    if rj > 0:
        _, _, h, w = x.shape
        delta = int(torch.randint(low=-rj, high=rj + 1, size=(), device=x.device).item())
        nh = max(8, h + delta)
        nw = max(8, w + delta)
        x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    q_lo, q_hi = _to_int_range(jpeg_qual, (70, 95))
    x = _jpegish(x, q_lo, q_hi)
    return _norm_clip(x.to(dtype=images_norm.dtype))


def _tri_view_restore(
    images_norm_deg: torch.Tensor,
    *,
    original_norm: Optional[torch.Tensor] = None,
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


def _tta_resize_back(images_norm: torch.Tensor, scale: float) -> torch.Tensor:
    scale = float(scale)
    if abs(scale - 1.0) < 1e-6:
        return images_norm
    x = _denorm_clip(images_norm).float()
    _, _, h, w = x.shape
    nh = max(8, int(round(float(h) * scale)))
    nw = max(8, int(round(float(w) * scale)))
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return _norm_clip(x.to(dtype=images_norm.dtype))


def _tta_logits(
    model,
    images: torch.Tensor,
    q_log: Optional[torch.Tensor],
    degrade_bin: Optional[torch.Tensor],
    *,
    mode: str = "none",
    scales=None,
    logit_agg: str = "mean",
) -> torch.Tensor:
    mode = str(mode or "none").lower()
    agg = str(logit_agg or "mean").lower()
    views = [images]

    if mode in {"hflip", "hflip_resize"}:
        views.append(torch.flip(images, dims=[3]))

    if mode == "hflip_resize":
        svals = []
        try:
            for s in list(scales or []):
                sf = float(s)
                if sf <= 0.0:
                    continue
                if abs(sf - 1.0) < 1e-6:
                    continue
                svals.append(sf)
        except Exception:
            svals = []
        svals = sorted(set(svals))
        for sf in svals:
            v = _tta_resize_back(images, scale=sf)
            views.append(v)
            views.append(torch.flip(v, dims=[3]))

    logits_all = []
    for v in views:
        model_in = {"image": v}
        if q_log is not None:
            model_in["q_log"] = q_log
        if degrade_bin is not None:
            model_in["degrade_bin"] = degrade_bin
        logits_all.append(model(model_in)["cls"].float())

    if len(logits_all) == 1:
        return logits_all[0]
    x = torch.stack(logits_all, dim=0)
    if agg == "median":
        return torch.median(x, dim=0).values
    return x.mean(dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path (best_auc.pt / last.pt)")
    parser.add_argument("--weights_path", type=str, default=None, help="alias for --ckpt")
    parser.add_argument("--clip_model", type=str, default=CLIP_MODEL)
    parser.add_argument("--model_family", type=str, default=None, choices=["effort", "dinov3"])
    parser.add_argument("--dino_source", type=str, default=None, choices=["timm", "torchhub"])
    parser.add_argument("--dino_repo", type=str, default=None)
    parser.add_argument("--dino_model_name", type=str, default=None)
    parser.add_argument("--dino_pretrained", action="store_true", default=None)
    parser.add_argument("--no_dino_pretrained", dest="dino_pretrained", action="store_false")
    parser.add_argument("--dino_local_weights", type=str, default=None)
    parser.add_argument("--dino_force_imagenet_norm", action="store_true", default=None)
    parser.add_argument("--no_dino_force_imagenet_norm", dest="dino_force_imagenet_norm", action="store_false")
    parser.add_argument("--dino_svd_residual_enable", action="store_true", default=None)
    parser.add_argument("--no_dino_svd_residual_enable", dest="dino_svd_residual_enable", action="store_false")
    parser.add_argument("--dino_svd_residual_dim", type=int, default=None)
    parser.add_argument("--dino_svd_target", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--image_size", type=int, default=int(IMAGE_SIZE))
    parser.add_argument("--num_workers", type=int, default=int(NUM_WORKERS))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--on_error", type=str, default="skip", choices=["skip", "raise"])

    # Patch pooling options (single-view)
    parser.add_argument("--patch_pool_tau", type=float, default=float(PATCH_POOL_TAU))
    parser.add_argument("--patch_pool_mode", type=str, default=str(PATCH_POOL_MODE))
    parser.add_argument("--patch_trim_p", type=float, default=float(PATCH_TRIM_P))
    parser.add_argument("--patch_quality", type=str, default=str(PATCH_QUALITY))
    parser.add_argument("--patch_pool_gamma", type=float, default=float(PATCH_POOL_GAMMA))
    parser.add_argument("--patch_warmup_steps", type=float, default=0.0)
    parser.add_argument("--patch_ramp_steps", type=float, default=0.0)
    parser.add_argument("--multi_expert_enable", action="store_true", default=False)
    parser.add_argument("--multi_expert_k", type=int, default=1)
    parser.add_argument(
        "--multi_expert_route",
        type=str,
        default="quality_bin",
        choices=[
            "quality_bin",
            "quality_soft",
            "degrade_bin",
            "uniform",
            "router",
            "hybrid",
            "degrade_router",
            "degrade_hybrid",
        ],
    )
    parser.add_argument("--multi_expert_route_temp", type=float, default=1.0)
    parser.add_argument("--multi_expert_route_detach", action="store_true", default=True)
    parser.add_argument("--no_multi_expert_route_detach", dest="multi_expert_route_detach", action="store_false")
    parser.add_argument("--multi_expert_hybrid_mix", type=float, default=0.0)
    parser.add_argument(
        "--restoration_enable",
        action="store_true",
        default=False,
        help="Enable restoration branch (if checkpoint/config supports it).",
    )
    parser.add_argument(
        "--restoration_strength",
        type=float,
        default=None,
        help="Restoration blend strength. If omitted, use value from ckpt config when available.",
    )
    parser.add_argument(
        "--infer_quality_bin",
        action="store_true",
        default=False,
        help="Compute q_log per image and route quality-based multi-expert by quality_cuts during inference.",
    )
    parser.add_argument(
        "--infer_degrade_bin",
        action="store_true",
        default=False,
        help="Compute hard degradation bin per image and route ExpertV2 (degrade_bin) during inference.",
    )
    parser.add_argument(
        "--quality_cuts",
        type=float,
        nargs="+",
        default=None,
        help="Quality cutpoints used for q_log->quality_bin (e.g., from training config quality_cuts).",
    )
    parser.add_argument(
        "--quality_metric",
        type=str,
        default=None,
        choices=["laplacian", "clipiqa"],
        help="Quality proxy used for infer_quality_bin. If omitted, read from checkpoint config or fallback to laplacian.",
    )
    parser.add_argument("--quality_clipiqa_device", type=str, default=None)
    parser.add_argument(
        "--quality_clipiqa_weights",
        type=float,
        nargs="+",
        default=None,
        help="3 weights for (quality, sharpness, 1-noisiness), e.g. 0.5 0.3 0.2",
    )
    parser.add_argument("--quality_max_side", type=int, default=256)
    parser.add_argument("--degrade_bins", type=int, default=3)
    parser.add_argument("--degrade_max_side", type=int, default=256)
    parser.add_argument("--tri_view", action="store_true", help="Use 3-view averaging: original + degraded + restored")
    parser.add_argument("--tri_blur_sig", type=float, nargs="+", default=[0.2, 1.2])
    parser.add_argument("--tri_jpeg_qual", type=int, nargs="+", default=[70, 95])
    parser.add_argument("--tri_resize_jitter", type=int, default=4)
    parser.add_argument("--tri_unsharp_amount", type=float, default=0.35)
    parser.add_argument("--tri_restore_mix", type=float, default=0.25)
    parser.add_argument(
        "--tta_mode",
        type=str,
        default="none",
        choices=["none", "hflip", "hflip_resize"],
        help="Standard test-time augmentation without degradation.",
    )
    parser.add_argument(
        "--tta_scales",
        type=float,
        nargs="+",
        default=[0.94, 1.06],
        help="Resize factors used when --tta_mode=hflip_resize.",
    )
    parser.add_argument(
        "--tta_logit_agg",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="How to aggregate logits across TTA views.",
    )

    parser.add_argument("--strict", action="store_true", help="strict load checkpoint")
    parser.add_argument("--txt_path", type=str, default=None, help="TXT with path [label] or a directory")
    parser.add_argument("--root_dir", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument(
        "--zip_submission",
        action="store_true",
        help="Also write Codabench-ready zip containing a single file named submission.txt",
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default=None,
        help="Output zip path for --zip_submission (default: <out_csv_stem>_codabench.zip)",
    )
    parser.add_argument(
        "--out_path_score",
        type=str,
        default=None,
        help="Optional CSV output with two columns: absolute_path,score",
    )
    parser.add_argument("--make_txt_from_dir", type=str, default=None, help="Generate txt from a folder (real/fake in filename)")
    parser.add_argument("--txt_out", type=str, default=None, help="Output txt path for --make_txt_from_dir")
    parser.add_argument("--only_make_txt", action="store_true", help="Only generate txt and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        s = int(args.seed)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    if torch.cuda.is_available():
        rt = ensure_cuda_conv_runtime(allow_cudnn_fallback=True)
        print(
            "[runtime] cuda probe_ok={} cudnn_enabled={} fallback_applied={}".format(
                rt.get("probe_ok"),
                rt.get("cudnn_enabled"),
                rt.get("fallback_applied"),
            )
        )
    ckpt_path = args.ckpt or args.weights_path
    if ckpt_path is None:
        raise ValueError("Provide --ckpt (or --weights_path)")

    if args.make_txt_from_dir:
        out_txt = Path(args.txt_out) if args.txt_out else Path(args.make_txt_from_dir) / "train_list.txt"
        _write_txt_from_dir(Path(args.make_txt_from_dir), out_txt)
        if args.only_make_txt:
            return
        if args.txt_path is None:
            args.txt_path = str(out_txt)

    if args.txt_path is None:
        raise ValueError("txt_path is required unless --only_make_txt is set.")

    txt_path = Path(args.txt_path)
    if txt_path.is_dir():
        root_dir = txt_path
        records = _read_dir(root_dir)
    else:
        root_dir = Path(args.root_dir)
        records = _read_txt(txt_path, root_dir)

    out_csv = Path(args.out_csv) if args.out_csv else txt_path.with_suffix(".probs.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quality_cuts = None
    cfg = {}
    if args.quality_cuts is not None:
        try:
            quality_cuts = sorted([float(v) for v in list(args.quality_cuts)])
        except Exception:
            quality_cuts = None
    cfg_path = Path(ckpt_path).resolve().parent / "config.json"
    if cfg_path.exists():
        try:
            import json
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    if quality_cuts is None:
        try:
            cuts = cfg.get("quality_cuts", None)
            if cuts is not None:
                quality_cuts = sorted([float(v) for v in list(cuts)])
        except Exception:
            quality_cuts = None
    quality_metric = _normalize_quality_metric(args.quality_metric or cfg.get("quality_metric", "laplacian"))
    quality_clipiqa_device = args.quality_clipiqa_device or cfg.get("quality_clipiqa_device", None)
    quality_clipiqa_weights = args.quality_clipiqa_weights
    if quality_clipiqa_weights is None:
        quality_clipiqa_weights = cfg.get("quality_clipiqa_weights", None)

    restoration_enable = bool(args.restoration_enable or bool(cfg.get("restoration_enable", False)))
    if args.restoration_strength is None:
        restoration_strength = float(cfg.get("restoration_strength", 0.20))
    else:
        restoration_strength = float(args.restoration_strength)
    model_family = str(args.model_family or cfg.get("model_family", MODEL_FAMILY))
    dino_source = str(args.dino_source or cfg.get("dino_source", DINO_SOURCE))
    dino_repo = str(args.dino_repo or cfg.get("dino_repo", DINO_REPO))
    dino_model_name = str(args.dino_model_name or cfg.get("dino_model_name", DINO_MODEL_NAME))
    if args.dino_pretrained is None:
        dino_pretrained = bool(cfg.get("dino_pretrained", DINO_PRETRAINED))
    else:
        dino_pretrained = bool(args.dino_pretrained)
    dino_local_weights = args.dino_local_weights or cfg.get("dino_local_weights", None)
    if args.dino_force_imagenet_norm is None:
        dino_force_imagenet_norm = bool(cfg.get("dino_force_imagenet_norm", DINO_FORCE_IMAGENET_NORM))
    else:
        dino_force_imagenet_norm = bool(args.dino_force_imagenet_norm)
    if args.dino_svd_residual_enable is None:
        dino_svd_residual_enable = bool(cfg.get("dino_svd_residual_enable", True))
    else:
        dino_svd_residual_enable = bool(args.dino_svd_residual_enable)
    if args.dino_svd_residual_dim is None:
        dino_svd_residual_dim = int(cfg.get("dino_svd_residual_dim", 32))
    else:
        dino_svd_residual_dim = int(args.dino_svd_residual_dim)
    dino_svd_target = str(args.dino_svd_target or cfg.get("dino_svd_target", "mlp_fc"))

    opt = SimpleNamespace(
        model_family=model_family,
        clip_model=args.clip_model,
        patch_pool_tau=args.patch_pool_tau,
        patch_pool_mode=args.patch_pool_mode,
        patch_trim_p=args.patch_trim_p,
        patch_quality=args.patch_quality,
        multi_expert_enable=bool(args.multi_expert_enable),
        multi_expert_k=int(args.multi_expert_k),
        multi_expert_route=str(args.multi_expert_route),
        multi_expert_route_temp=float(args.multi_expert_route_temp),
        multi_expert_route_detach=bool(args.multi_expert_route_detach),
        multi_expert_hybrid_mix=float(args.multi_expert_hybrid_mix),
        quality_cuts=quality_cuts,
        restoration_enable=restoration_enable,
        restoration_strength=restoration_strength,
        dino_source=dino_source,
        dino_repo=dino_repo,
        dino_model_name=dino_model_name,
        dino_pretrained=dino_pretrained,
        dino_local_weights=dino_local_weights,
        dino_force_imagenet_norm=dino_force_imagenet_norm,
        dino_svd_residual_enable=dino_svd_residual_enable,
        dino_svd_residual_dim=dino_svd_residual_dim,
        dino_svd_target=dino_svd_target,
        osd_regs=False,
    )

    model = get_model(opt).to(device)
    if hasattr(model, "set_patch_scale"):
        model.set_patch_scale(float(args.patch_pool_gamma))

    missing, unexpected = load_ckpt(ckpt_path, model, strict=bool(args.strict))
    _print_key_summary("missing keys", missing)
    _print_key_summary("unexpected keys", unexpected)

    model.eval()

    emit_q_log = bool(args.infer_quality_bin) and bool(args.multi_expert_enable) and (
        str(args.multi_expert_route).lower() in {"quality_bin", "quality_soft", "hybrid", "degrade_hybrid"}
    )
    emit_degrade_bin = bool(args.infer_degrade_bin) and bool(args.multi_expert_enable) and (
        str(args.multi_expert_route).lower() in {"degrade_bin"}
    )
    if emit_q_log and (quality_cuts is None):
        print("[warn] --infer_quality_bin enabled but no quality_cuts found; route may fallback to uniform.")
    if emit_q_log:
        print(f"[quality] infer_quality_bin metric={quality_metric}")
    if args.tri_view and str(args.tta_mode).lower() != "none":
        print("[warn] both tri_view and tta_mode enabled; tri_view takes precedence.")

    dataset_emit_q_log = bool(emit_q_log and quality_metric == "laplacian")
    dataset = TxtImageDataset(
        records=records,
        image_size=int(args.image_size),
        on_error=args.on_error,
        emit_q_log=dataset_emit_q_log,
        quality_metric=quality_metric,
        quality_max_side=int(args.quality_max_side),
        emit_degrade_bin=emit_degrade_bin,
        degrade_bins=int(args.degrade_bins),
        degrade_max_side=int(args.degrade_max_side),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=_collate_fn,
    )

    all_probs: List[float] = []
    all_paths: List[str] = []
    clipiqa_scorer = None
    if emit_q_log and quality_metric == "clipiqa":
        clipiqa_scorer = _CLIPIQABatchScorer(
            model_device=device,
            clipiqa_device=quality_clipiqa_device,
            weights=quality_clipiqa_weights,
        )
        print(f"[quality] CLIP-IQA scorer ready on {clipiqa_scorer.device}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="infer", ncols=80):
            if batch is None:
                continue
            images = batch["image"].to(device, non_blocking=True)
            q_log = batch.get("q_log", None)
            if q_log is not None:
                q_log = q_log.to(device, non_blocking=True)
            elif emit_q_log and (clipiqa_scorer is not None):
                # Keep CLIP-IQA routing scale aligned with training: score raw images (max_side), not 224-resized tensors.
                q_log = clipiqa_scorer.score_from_paths(
                    batch["path"],
                    max_side=int(args.quality_max_side),
                )
            degrade_bin = batch.get("degrade_bin", None)
            if degrade_bin is not None:
                degrade_bin = degrade_bin.to(device, non_blocking=True)
            if args.tri_view:
                img_deg = _tri_view_degrade(
                    images,
                    blur_sig=args.tri_blur_sig,
                    jpeg_qual=args.tri_jpeg_qual,
                    resize_jitter=int(args.tri_resize_jitter),
                )
                img_rst = _tri_view_restore(
                    img_deg,
                    original_norm=images,
                    unsharp_amount=float(args.tri_unsharp_amount),
                    restore_mix=float(args.tri_restore_mix),
                )
                in0 = {"image": images}
                in1 = {"image": img_deg}
                in2 = {"image": img_rst}
                if q_log is not None:
                    in0["q_log"] = q_log
                    in1["q_log"] = q_log
                    in2["q_log"] = q_log
                if degrade_bin is not None:
                    in0["degrade_bin"] = degrade_bin
                    in1["degrade_bin"] = degrade_bin
                    in2["degrade_bin"] = degrade_bin
                logits_0 = model(in0)["cls"].float()
                logits_1 = model(in1)["cls"].float()
                logits_2 = model(in2)["cls"].float()
                logits = (logits_0 + logits_1 + logits_2) / 3.0
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().reshape(-1).tolist()
            elif str(args.tta_mode).lower() != "none":
                logits = _tta_logits(
                    model,
                    images,
                    q_log,
                    degrade_bin,
                    mode=str(args.tta_mode),
                    scales=args.tta_scales,
                    logit_agg=str(args.tta_logit_agg),
                )
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().reshape(-1).tolist()
            else:
                model_in = {"image": images}
                if q_log is not None:
                    model_in["q_log"] = q_log
                if degrade_bin is not None:
                    model_in["degrade_bin"] = degrade_bin
                pred = model(model_in)
                probs = pred["prob"].detach().cpu().numpy().reshape(-1).tolist()
            all_probs.extend(probs)
            all_paths.extend(batch["path"])

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        for prob in all_probs:
            writer.writerow([f"{prob:.6f}"])

    print(f"[done] saved probs -> {out_csv}")

    if bool(args.zip_submission) or (args.zip_path is not None):
        # Codabench requires a .zip with exactly one member named submission.txt.
        with out_csv.open("r", encoding="utf-8") as f:
            for idx, raw in enumerate(f, start=1):
                s = raw.strip()
                if not s:
                    raise ValueError(f"Submission has empty line at {idx}")
                try:
                    v = float(s)
                except Exception as e:
                    raise ValueError(f"Submission line {idx} is not a float: {s!r}") from e
                if (v < 0.0) or (v > 1.0):
                    raise ValueError(f"Submission line {idx} out of range [0,1]: {v}")

        zip_path = Path(args.zip_path) if args.zip_path is not None else out_csv.with_name(f"{out_csv.stem}_codabench.zip")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_csv, arcname="submission.txt")
        print(f"[done] saved codabench zip -> {zip_path}")

    if args.out_path_score is not None:
        out_path_score = Path(args.out_path_score)
        out_path_score.parent.mkdir(parents=True, exist_ok=True)
        with out_path_score.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "score"])
            for path, prob in zip(all_paths, all_probs):
                writer.writerow([str(Path(path).resolve()), f"{prob:.6f}"])
        print(f"[done] saved path+score -> {out_path_score}")


if __name__ == "__main__":
    main()
