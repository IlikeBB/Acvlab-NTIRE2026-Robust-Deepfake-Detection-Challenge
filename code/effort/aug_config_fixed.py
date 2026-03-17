from __future__ import annotations

from typing import Sequence, Optional
from random import random, choice

import numpy as np
from importlib import util as _importlib_util
from pathlib import Path


# Augmentation defaults (shared by Config.py)
AUG_DEFAULTS = {
    # --- mild/hard mixture (mixture of severities) ---
    "jpg_qual_mild": None,
    "jpg_qual_hard": None,
    "jpg_hard_prob": 0.0,

    "blur_sig_mild": None,
    "blur_sig_hard": None,
    "blur_hard_prob": 0.0,
    "lowres_scale_mild": None,
    "lowres_scale_hard": None,
    "lowres_hard_prob": 0.0,
    "rz_interp": ["bilinear"],  # resize 內插方法
    "blur_prob": 0.1,   # 模糊增強機率
    "blur_sig": [0.0, 3.0],  # 模糊 sigma 範圍
    "motion_blur_prob": 0.1,   # 動態模糊機率
    "motion_blur_ksize": [7, 15],  # 動態模糊 kernel 尺寸範圍（奇數）
    "motion_blur_angle": [0, 180],  # 動態模糊角度範圍
    "jpg_prob": 0.1,   # JPEG 壓縮增強機率
    "jpg_method": ["cv2", "pil"],  # JPEG 方法清單
    "jpg_qual": list(range(5, 31)),  # JPEG 品質範圍（更狠）
    "lowres_prob": 0.1,   # 低解析度縮放機率（downsample -> upsample）
    "lowres_scale": [0.15, 0.4],  # downsample 比例（更低解析）
    "lowres_interp": ["nearest", "box", "bilinear"],  # resize 插值方法
    "noise_prob": 0.1,   # 加性高斯雜訊機率（改用 ISO/shot noise）
    "noise_std": [5.0, 25.0],  # 雜訊標準差 (pixel scale 0-255)
    "iso_noise_prob": 0.1,   # ISO/shot noise 機率
    "iso_shot": [1.5, 6.0],  # shot noise 強度 (越小越吵)
    "iso_gauss_std": [12.0, 35.0],  # ISO 高斯噪聲 std (pixel scale 0-255)
    "sp_prob": 0.1,   # salt & pepper 機率（少量點狀）
    "max_aug_per_image": 1,  # 每張圖最多套用幾個 augmentation（單獨增強）
    "sp_amount": 0.006,  # salt & pepper 比例
    # grainy group (multi-augment combo)
    "grainy_group_prob": 0.1,   # 觸發「顆粒感組合」的機率
    "grainy_lowres_scale": [0.70, 0.90],
    "grainy_lowres_interp": ["nearest", "box", "bilinear"],
    "grainy_blur_prob": 0.1,
    "grainy_blur_sig": [0.2, 0.6],
    "grainy_jpg_prob": 0.1,
    "grainy_jpg_qual": list(range(28, 42)),
    "grainy_iso_prob": 0.1,
    "grainy_iso_shot": [6.0, 14.0],
    "grainy_iso_gauss_std": [4.0, 10.0],
    "grainy_sp_prob": 0.1,
    "grainy_sp_amount": [0.0004, 0.0010],
    # mono/low-contrast group (down->up first, then other effects)
    "mono_group_prob": 0.1,
    "mono_lowres_scale": [0.18, 0.35],
    "mono_lowres_interp": ["box", "nearest", "bilinear"],
    "mono_gray_prob": 0.1,
    "mono_contrast": [0.45, 0.70],
    "mono_brightness": [0.92, 1.02],
    "mono_blur_prob": 0.1,
    "mono_blur_sig": [0.8, 1.6],
    "mono_jpg_prob": 0.1,
    "mono_jpg_qual": list(range(12, 28)),
    # motion+grainy group
    "motion_grainy_group_prob": 0.1,
    "motion_grainy_lowres_scale": [0.65, 0.85],
    "motion_grainy_motion_prob": 0.1,
    "motion_grainy_motion_ksize": [9, 17],
    "motion_grainy_motion_angle": [0, 180],
    "motion_grainy_iso_prob": 0.1,
    "motion_grainy_iso_shot": [6.5, 13.0],
    "motion_grainy_iso_gauss_std": [3.5, 8.4],
    "motion_grainy_gray_noise": True,  # use grayscale noise (black/white grain)
    "motion_grainy_jpg_prob": 0.1,
    "motion_grainy_jpg_qual": list(range(18, 34)),
    "motion_grainy_sp_prob": 0.1,
    "motion_grainy_sp_amount": [0.00028, 0.00063],
    # low-light group (under-exposure + sensor artifacts)
    "lowlight_group_prob": 0.1,
    "lowlight_strength": ["light", "medium", "hard"],
    "lowlight_p_cast": 0.8,
    "lowlight_p_jpeg": 0.7,
    "lowlight_p_denoise": 0.5,
    # PMM (exclusive group): if hit, return immediately without stacking classic degradations.
    "use_pmm_aug": False,
    "pmm_group_prob": 0.0,
    "pmm_pdm_type": "ours",
    "pmm_strength": 0.5,
    "pmm_use_beta": False,
    "pmm_beta_a": 0.5,
    "pmm_beta_b": 0.5,
    "pmm_distractor_p": 0.0,
}


def _lazy_cv2():
    import cv2  # local import to avoid hard dependency at Config import time
    return cv2


def _lazy_pil_image():
    from PIL import Image  # local import
    return Image


def _pil_interp(name: str):
    Image = _lazy_pil_image()
    mapping = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
        "box": Image.BOX,
    }
    return mapping.get(str(name).lower(), Image.BILINEAR)


def _lazy_gaussian_filter():
    from scipy.ndimage.filters import gaussian_filter  # local import
    return gaussian_filter


_PMM_MODULE = None
_PMM_AUGMENTER = None
_PMM_AUGMENTER_KEY = None


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _lazy_pmm_module():
    global _PMM_MODULE
    if _PMM_MODULE is not None:
        return _PMM_MODULE if _PMM_MODULE is not False else None

    repo_root = Path(__file__).resolve().parents[1]
    pmm_dir = repo_root / "fix_gpt2" / "pmm_integration_pack"
    pmm_path = pmm_dir / "pmm_augment.py"
    if not pmm_path.exists():
        raise FileNotFoundError(f"[PMM-AUG] pmm_augment.py not found: {pmm_path}")

    import sys as _sys
    pmm_dir_str = str(pmm_dir)
    if pmm_dir_str not in _sys.path:
        _sys.path.insert(0, pmm_dir_str)

    spec = _importlib_util.spec_from_file_location("pmm_augment", str(pmm_path))
    if spec is None or spec.loader is None:
        raise ImportError("[PMM-AUG] failed to create import spec")
    mod = _importlib_util.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _PMM_MODULE = mod
    return mod


def sample_continuous(s: Sequence[float]):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def _clamp01(p: float) -> float:
    try:
        p = float(p)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, p))


def _mixture_cont(opt, base_key, mild_key, hard_key, hard_prob_key):
    base = getattr(opt, base_key, None)
    mild = getattr(opt, mild_key, None)
    hard = getattr(opt, hard_key, None)
    p = _clamp01(getattr(opt, hard_prob_key, 0.0) or 0.0)
    if mild is None or hard is None:
        return sample_continuous(base if base is not None else [0.0, 0.0])
    return sample_continuous(hard if random() < p else mild)


def _mixture_disc(opt, base_key, mild_key, hard_key, hard_prob_key):
    base = getattr(opt, base_key, None)
    mild = getattr(opt, mild_key, None)
    hard = getattr(opt, hard_key, None)
    p = _clamp01(getattr(opt, hard_prob_key, 0.0) or 0.0)
    if mild is None or hard is None:
        return sample_discrete(base if base is not None else [0])
    return sample_discrete(hard if random() < p else mild)

def sample_discrete(s: Sequence):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma: float) -> None:
    gaussian_filter = _lazy_gaussian_filter()
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def motion_blur(img, ksize: int, angle: float) -> None:
    cv2 = _lazy_cv2()
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    center = (ksize / 2.0, ksize / 2.0)
    rot = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (ksize, ksize))
    kernel = kernel / max(kernel.sum(), 1e-8)
    for c in range(3):
        img[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel)


def cv2_jpg(img, compress_val: int):
    cv2 = _lazy_cv2()
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(compress_val)]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val: int):
    Image = _lazy_pil_image()
    from io import BytesIO
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=int(compress_val))
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img, compress_val: int, key: str):
    method = jpeg_dict[key]
    return method(img, compress_val)


def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    max_aug = int(getattr(opt, "max_aug_per_image", 0) or 0)
    aug_count = 0

    def _can_apply():
        return max_aug <= 0 or aug_count < max_aug

    def _mark_applied():
        nonlocal aug_count
        aug_count += 1

    # --- group selection without order bias (order bias 消除) ---
    pmm_prob = float(getattr(opt, "pmm_group_prob", 0.0) or 0.0)
    if not bool(getattr(opt, "use_pmm_aug", False)):
        pmm_prob = 0.0
    p_groups = [
        ("pmm",           pmm_prob),
        ("motion_grainy", float(getattr(opt, "motion_grainy_group_prob", 0.0) or 0.0)),
        ("lowlight",      float(getattr(opt, "lowlight_group_prob", 0.0) or 0.0)),
        ("mono",          float(getattr(opt, "mono_group_prob", 0.0) or 0.0)),
        ("grainy",        float(getattr(opt, "grainy_group_prob", 0.0) or 0.0)),
    ]

    # 總觸發率沿用原本的語意：p_total = 1 - Π(1-p_i)
    p_total = 1.0
    for _, p in p_groups:
        p = max(0.0, min(1.0, p))
        p_total *= (1.0 - p)
    p_total = 1.0 - p_total

    if p_total > 0.0 and random() < p_total:
        names = [n for n, p in p_groups if p > 0.0]
        weights = [p for n, p in p_groups if p > 0.0]
        s = sum(weights)
        r = random() * s
        acc = 0.0
        chosen = names[-1]
        for n, w in zip(names, weights):
            acc += w
            if r <= acc:
                chosen = n
                break

        if chosen == "pmm":
            pmm_aug = build_pmm_augmenter(opt, image_size=img.shape[:2])
            out = pmm_aug(img)
            if isinstance(out, np.ndarray):
                arr = out
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
                Image = _lazy_pil_image()
                return Image.fromarray(arr)
            return out
        if chosen == "motion_grainy":
            img = _apply_motion_grainy_group(img, opt)
        elif chosen == "lowlight":
            img = _apply_lowlight_group(img, opt)
        elif chosen == "mono":
            img = _apply_mono_group(img, opt)
        else:
            img = _apply_grainy_group(img, opt)

        Image = _lazy_pil_image()
        return Image.fromarray(img)

    # low-resolution downsample + upsample
    if _can_apply() and random() < float(getattr(opt, "lowres_prob", 0.0)):
        scale = _mixture_cont(opt, "lowres_scale", "lowres_scale_mild", "lowres_scale_hard", "lowres_hard_prob")
        if 0.0 < scale < 1.0:
            Image = _lazy_pil_image()
            h, w = img.shape[:2]
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            interp = sample_discrete(getattr(opt, "lowres_interp", ["bilinear"]))
            pil = Image.fromarray(img)
            pil = pil.resize((new_w, new_h), resample=_pil_interp(interp))
            pil = pil.resize((w, h), resample=_pil_interp("bilinear"))
            img = np.array(pil)
            _mark_applied()

    # motion blur
    if _can_apply() and random() < float(getattr(opt, "motion_blur_prob", 0.0)):
        ksize = sample_continuous(getattr(opt, "motion_blur_ksize", [7, 7]))
        angle = sample_continuous(getattr(opt, "motion_blur_angle", [0, 0]))
        motion_blur(img, int(ksize), float(angle))
        _mark_applied()

    if _can_apply() and random() < float(getattr(opt, "blur_prob", 0.0)):
        sig = sample_continuous(getattr(opt, "blur_sig", [0.0, 0.0]))
        gaussian_blur(img, sig)
        _mark_applied()

    if _can_apply() and random() < float(getattr(opt, "jpg_prob", 0.0)):
        method = sample_discrete(getattr(opt, "jpg_method", ["cv2"]))
        qual = sample_discrete(getattr(opt, "jpg_qual", [95]))
        img = jpeg_from_key(img, qual, method)
        _mark_applied()

    # additive gaussian noise (pixel scale 0-255)
    if _can_apply() and random() < float(getattr(opt, "noise_prob", 0.0)):
        std = sample_continuous(getattr(opt, "noise_std", [0.0, 0.0]))
        if std > 0:
            noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
            _mark_applied()

    # ISO-like shot + gaussian noise (grainy)
    if _can_apply() and random() < float(getattr(opt, "iso_noise_prob", 0.0)):
        shot = sample_continuous(getattr(opt, "iso_shot", [0.0, 0.0]))
        gauss_std = sample_continuous(getattr(opt, "iso_gauss_std", [0.0, 0.0])) / 255.0
        img_f = img.astype(np.float32) / 255.0
        if shot > 0:
            img_f = np.random.poisson(img_f * shot) / shot
        if gauss_std > 0:
            img_f = img_f + np.random.normal(0.0, gauss_std, img_f.shape).astype(np.float32)
        img_f = np.clip(img_f, 0.0, 1.0)
        img = (img_f * 255.0).astype(np.uint8)
        _mark_applied()

    # salt & pepper noise
    if _can_apply() and random() < float(getattr(opt, "sp_prob", 0.0)):
        amount = float(getattr(opt, "sp_amount", 0.0))
        if amount > 0:
            h, w, c = img.shape
            num = int(amount * h * w)
            if num > 0:
                ys = np.random.randint(0, h, num)
                xs = np.random.randint(0, w, num)
                img[ys, xs, :] = 255
                ys = np.random.randint(0, h, num)
                xs = np.random.randint(0, w, num)
                img[ys, xs, :] = 0
            _mark_applied()

    Image = _lazy_pil_image()
    return Image.fromarray(img)


def _apply_grainy_group(img, opt):
    # low-resolution downsample + upsample (strong)
    scale = sample_continuous(getattr(opt, "grainy_lowres_scale", [0.2, 0.2]))
    if 0.0 < scale < 1.0:
        Image = _lazy_pil_image()
        h, w = img.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        interp = sample_discrete(getattr(opt, "grainy_lowres_interp", ["bilinear"]))
        pil = Image.fromarray(img)
        pil = pil.resize((new_w, new_h), resample=_pil_interp(interp))
        pil = pil.resize((w, h), resample=_pil_interp("bilinear"))
        img = np.array(pil)

    # optional blur
    if random() < float(getattr(opt, "grainy_blur_prob", 0.0)):
        sig = _mixture_cont(opt, "blur_sig", "blur_sig_mild", "blur_sig_hard", "blur_hard_prob")
        gaussian_blur(img, sig)

    # strong JPEG artifacts
    if random() < float(getattr(opt, "grainy_jpg_prob", 0.0)):
        qual = _mixture_disc(opt, "jpg_qual", "jpg_qual_mild", "jpg_qual_hard", "jpg_hard_prob")
        img = jpeg_from_key(img, qual, "pil")

    # ISO-like shot + gaussian noise (grainy)
    if random() < float(getattr(opt, "grainy_iso_prob", 0.0)):
        shot = sample_continuous(getattr(opt, "grainy_iso_shot", [0.0, 0.0]))
        gauss_std = sample_continuous(getattr(opt, "grainy_iso_gauss_std", [0.0, 0.0])) / 255.0
        img_f = img.astype(np.float32) / 255.0
        if shot > 0:
            img_f = np.random.poisson(img_f * shot) / shot
        if gauss_std > 0:
            img_f = img_f + np.random.normal(0.0, gauss_std, img_f.shape).astype(np.float32)
        img_f = np.clip(img_f, 0.0, 1.0)
        img = (img_f * 255.0).astype(np.uint8)

    # salt & pepper noise
    if random() < float(getattr(opt, "grainy_sp_prob", 0.0)):
        amount = sample_continuous(getattr(opt, "grainy_sp_amount", [0.0, 0.0]))
        if amount > 0:
            h, w, c = img.shape
            num = int(amount * h * w)
            if num > 0:
                ys = np.random.randint(0, h, num)
                xs = np.random.randint(0, w, num)
                img[ys, xs, :] = 255
                ys = np.random.randint(0, h, num)
                xs = np.random.randint(0, w, num)
                img[ys, xs, :] = 0
    return img


def _apply_motion_grainy_group(img, opt):
    # mild low-res down/up
    scale = sample_continuous(getattr(opt, "motion_grainy_lowres_scale", [0.8, 0.8]))
    if 0.0 < scale < 1.0:
        Image = _lazy_pil_image()
        h, w = img.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        pil = Image.fromarray(img)
        pil = pil.resize((new_w, new_h), resample=_pil_interp("box"))
        pil = pil.resize((w, h), resample=_pil_interp("bilinear"))
        img = np.array(pil)

    # motion blur
    if random() < float(getattr(opt, "motion_grainy_motion_prob", 0.0)):
        ksize = sample_continuous(getattr(opt, "motion_grainy_motion_ksize", [9, 9]))
        angle = sample_continuous(getattr(opt, "motion_grainy_motion_angle", [0, 0]))
        motion_blur(img, int(ksize), float(angle))

    # ISO-like grain
    if random() < float(getattr(opt, "motion_grainy_iso_prob", 0.0)):
        shot = sample_continuous(getattr(opt, "motion_grainy_iso_shot", [0.0, 0.0]))
        gauss_std = sample_continuous(getattr(opt, "motion_grainy_iso_gauss_std", [0.0, 0.0])) / 255.0
        img_f = img.astype(np.float32) / 255.0
        use_gray = bool(getattr(opt, "motion_grainy_gray_noise", True))
        if use_gray:
            base = img_f.mean(axis=2, keepdims=True)
        else:
            base = img_f
        if shot > 0:
            base = np.random.poisson(base * shot) / shot
        if gauss_std > 0:
            base = base + np.random.normal(0.0, gauss_std, base.shape).astype(np.float32)
        base = np.clip(base, 0.0, 1.0)
        if use_gray:
            img_f = np.clip(img_f + (base - img_f.mean(axis=2, keepdims=True)), 0.0, 1.0)
        else:
            img_f = base
        img = (img_f * 255.0).astype(np.uint8)

    # optional JPEG artifacts
    if random() < float(getattr(opt, "motion_grainy_jpg_prob", 0.0)):
        qual = sample_discrete(getattr(opt, "motion_grainy_jpg_qual", [28]))
        img = jpeg_from_key(img, qual, "pil")

    # tiny salt & pepper (optional)
    if random() < float(getattr(opt, "motion_grainy_sp_prob", 0.0)):
        amount = sample_continuous(getattr(opt, "motion_grainy_sp_amount", [0.0, 0.0]))
        if amount > 0:
            h, w, c = img.shape
            num = int(amount * h * w)
            if num > 0:
                ys = np.random.randint(0, h, num)
                xs = np.random.randint(0, w, num)
                img[ys, xs, :] = 255
                ys = np.random.randint(0, h, num)
                xs = np.random.randint(0, w, num)
                img[ys, xs, :] = 0
    return img


def _apply_mono_group(img, opt):
    # downsample -> upsample first
    scale = sample_continuous(getattr(opt, "mono_lowres_scale", [0.3, 0.3]))
    if 0.0 < scale < 1.0:
        Image = _lazy_pil_image()
        h, w = img.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        interp = sample_discrete(getattr(opt, "mono_lowres_interp", ["bilinear"]))
        pil = Image.fromarray(img)
        pil = pil.resize((new_w, new_h), resample=_pil_interp(interp))
        pil = pil.resize((w, h), resample=_pil_interp("bilinear"))
        img = np.array(pil)

    # grayscale + contrast/brightness (after upsample)
    if random() < float(getattr(opt, "mono_gray_prob", 0.0)):
        from PIL import ImageOps, ImageEnhance
        pil = Image.fromarray(img)
        pil = ImageOps.grayscale(pil).convert("RGB")
        contrast = sample_continuous(getattr(opt, "mono_contrast", [1.0, 1.0]))
        if contrast != 1.0:
            pil = ImageEnhance.Contrast(pil).enhance(float(contrast))
        brightness = sample_continuous(getattr(opt, "mono_brightness", [1.0, 1.0]))
        if brightness != 1.0:
            pil = ImageEnhance.Brightness(pil).enhance(float(brightness))
        img = np.array(pil)

    # slight blur
    if random() < float(getattr(opt, "mono_blur_prob", 0.0)):
        sig = sample_continuous(getattr(opt, "mono_blur_sig", [0.0, 0.0]))
        gaussian_blur(img, sig)

    # JPEG compression at the end
    if random() < float(getattr(opt, "mono_jpg_prob", 0.0)):
        qual = sample_discrete(getattr(opt, "mono_jpg_qual", [22]))
        img = jpeg_from_key(img, qual, "pil")

    return img


def _apply_lowlight_group(img, opt):
    cv2 = _lazy_cv2()
    strength = sample_discrete(getattr(opt, "lowlight_strength", ["medium"]))
    p_cast = float(getattr(opt, "lowlight_p_cast", 0.8))
    p_jpeg = float(getattr(opt, "lowlight_p_jpeg", 0.7))
    p_denoise = float(getattr(opt, "lowlight_p_denoise", 0.5))

    # RGB -> BGR
    x = img[:, :, ::-1].astype(np.float32) / 255.0

    if strength == "light":
        gamma = np.random.uniform(1.05, 1.45)
        exposure = np.random.uniform(0.8, 1.05)
        noise_sigma = np.random.uniform(1, 4) / 255.0
        qmin, qmax = 70, 98
    elif strength == "hard":
        gamma = np.random.uniform(1.8, 2.6)
        exposure = np.random.uniform(0.55, 0.8)
        noise_sigma = np.random.uniform(7, 15) / 255.0
        qmin, qmax = 28, 65
    else:
        # brighter "medium" low-light
        gamma = np.random.uniform(1.2, 1.8)
        exposure = np.random.uniform(0.75, 1.0)
        noise_sigma = np.random.uniform(2, 7) / 255.0
        qmin, qmax = 55, 90

    # (A) Under-exposure + gamma
    x = _clip01(x * exposure)
    x = _clip01(x ** gamma)

    # (A) Color cast (green/yellow bias)
    if np.random.rand() < p_cast:
        b = np.random.uniform(0.85, 1.05)
        g = np.random.uniform(0.95, 1.20)
        r = np.random.uniform(0.90, 1.10)
        x[..., 0] *= b
        x[..., 1] *= g
        x[..., 2] *= r
        x = _clip01(x)

    # (B) Dark-region weighted noise (simulate sensor noise)
    gray = (0.114 * x[..., 0] + 0.587 * x[..., 1] + 0.299 * x[..., 2])
    w = (1.0 - gray) ** 2
    noise = np.random.normal(0.0, noise_sigma, size=x.shape).astype(np.float32)
    x = _clip01(x + noise * w[..., None])

    # (B) weak denoise + weak sharpen artifacts
    if np.random.rand() < p_denoise:
        x8 = (x * 255).astype(np.uint8)
        den = cv2.bilateralFilter(x8, d=5, sigmaColor=25, sigmaSpace=3)
        den = den.astype(np.float32) / 255.0
        blur = cv2.GaussianBlur((den * 255).astype(np.uint8), (0, 0), 1.0).astype(np.float32) / 255.0
        x = _clip01(den + 0.35 * (den - blur))

    # (B) JPEG compression
    if np.random.rand() < p_jpeg:
        q = int(np.random.uniform(qmin, qmax))
        x8 = (x * 255).astype(np.uint8)
        _, enc = cv2.imencode(".jpg", x8, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        x8 = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return x8[:, :, ::-1]

    return (x * 255).astype(np.uint8)[:, :, ::-1]


def build_pmm_augmenter(opt, image_size=None) -> Optional[object]:
    global _PMM_AUGMENTER, _PMM_AUGMENTER_KEY
    if not bool(getattr(opt, "use_pmm_aug", False)):
        return None

    mod = _lazy_pmm_module()

    key = (
        float(getattr(opt, "pmm_group_prob", 0.0) or 0.0),
        str(getattr(opt, "pmm_pdm_type", "ours")),
        float(getattr(opt, "pmm_strength", 0.5) or 0.5),
        bool(getattr(opt, "pmm_use_beta", False)),
        float(getattr(opt, "pmm_beta_a", 0.5) or 0.5),
        float(getattr(opt, "pmm_beta_b", 0.5) or 0.5),
        float(getattr(opt, "pmm_distractor_p", 0.0) or 0.0),
    )
    if _PMM_AUGMENTER is not None and _PMM_AUGMENTER_KEY == key:
        return _PMM_AUGMENTER

    pmm_cfg = mod.PMMAugConfig(
        p=float(getattr(opt, "pmm_group_prob", 0.0) or 0.0),
        pdm_type=str(getattr(opt, "pmm_pdm_type", "ours")),
        strength=float(getattr(opt, "pmm_strength", 0.5) or 0.5),
        use_beta=bool(getattr(opt, "pmm_use_beta", False)),
        a=float(getattr(opt, "pmm_beta_a", 0.5) or 0.5),
        b=float(getattr(opt, "pmm_beta_b", 0.5) or 0.5),
        distractor_p=float(getattr(opt, "pmm_distractor_p", 0.0) or 0.0),
    )
    _PMM_AUGMENTER = mod.PMMAugmenter(pmm_cfg)
    _PMM_AUGMENTER_KEY = key
    return _PMM_AUGMENTER
