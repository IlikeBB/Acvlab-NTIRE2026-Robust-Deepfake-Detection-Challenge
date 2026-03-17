#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Model-based diagnostics on user-selected images:
- TTA sensitivity (std/var/range over blur/jpeg/hflip views)
- Blur/JPEG ladder monotonicity (Spearman + sign changes)

This script is adapted for the current repo layout:
  repo_root/
    src/effort/...
"""

import argparse
import csv
import io
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    from PIL import Image, ImageFilter
except Exception:
    Image = None
    ImageFilter = None

import torch
from torch.amp import autocast


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_exists(path: str, what: str) -> None:
    if not path:
        raise ValueError(f"Empty {what}")
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def read_image_list(path: str) -> List[Dict[str, Any]]:
    ensure_exists(path, "image_list")
    p = Path(path)
    items: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".csv":
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "path" not in reader.fieldnames:
                raise ValueError("CSV must contain column 'path'")
            for row in reader:
                ip = (row.get("path") or "").strip()
                if not ip:
                    continue
                items.append(
                    {
                        "path": ip,
                        "label": (row.get("label") or "").strip(),
                        "note": (row.get("note") or "").strip(),
                    }
                )
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = [x.strip() for x in s.split(",")]
                ip = parts[0]
                label = parts[1] if len(parts) >= 2 else ""
                note = parts[2] if len(parts) >= 3 else ""
                items.append({"path": ip, "label": label, "note": note})
    if not items:
        raise RuntimeError(f"No items found in {path}")
    return items


def robust_trimmed_mean(values: List[float], trim_ratio: float = 0.2) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    k = int(len(vs) * trim_ratio)
    core = vs[k : len(vs) - k] if len(vs) - 2 * k > 0 else vs
    return float(np.mean(core))


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks


def spearman_corr(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx = rankdata(np.asarray(x, dtype=float))
    ry = rankdata(np.asarray(y, dtype=float))
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = np.linalg.norm(vx) * np.linalg.norm(vy)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(vx, vy) / denom)


def monotonicity_stats(levels: List[float], scores: List[float]) -> Dict[str, float]:
    if len(scores) < 2:
        return {"spearman": float("nan"), "mean_abs_step": float("nan"), "sign_changes": float("nan")}
    diffs = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
    mean_abs_step = float(np.mean(np.abs(diffs)))
    signs = [0 if abs(d) < 1e-9 else (1 if d > 0 else -1) for d in diffs]
    filtered = [s for s in signs if s != 0]
    sign_changes = 0
    for i in range(len(filtered) - 1):
        if filtered[i] != filtered[i + 1]:
            sign_changes += 1
    return {
        "spearman": spearman_corr(levels, scores),
        "mean_abs_step": mean_abs_step,
        "sign_changes": float(sign_changes),
    }


@dataclass
class RepoImports:
    repo_root: str


def resolve_imports(repo_root: str) -> RepoImports:
    rr = Path(repo_root).resolve()
    src_effort_model = rr / "src" / "effort" / "model.py"
    root_effort_model = rr / "effort" / "model.py"
    if src_effort_model.exists():
        src_root = rr / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        return RepoImports(repo_root=str(rr))
    if root_effort_model.exists():
        if str(rr) not in sys.path:
            sys.path.insert(0, str(rr))
        return RepoImports(repo_root=str(rr))
    raise FileNotFoundError(f"Cannot find effort model under {rr} (tried src/effort and effort)")


def apply_hflip(img: Image.Image) -> Image.Image:
    if Image is None:
        raise ModuleNotFoundError("Pillow is required. Please install with: pip install pillow")
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def apply_gaussian_blur_pil(img: Image.Image, sigma: float) -> Image.Image:
    if ImageFilter is None:
        raise ModuleNotFoundError("Pillow is required. Please install with: pip install pillow")
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def apply_jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    if Image is None:
        raise ModuleNotFoundError("Pillow is required. Please install with: pip install pillow")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    return out.copy()


@dataclass
class TTASummary:
    k: int
    scores: List[float]
    mean: float
    std: float
    var: float
    median: float
    trimmed_mean: float
    min: float
    max: float
    range: float
    patch_delta_std: Optional[float] = None
    patch_delta_range: Optional[float] = None


@dataclass
class LadderSummary:
    name: str
    levels: List[float]
    scores: List[float]
    spearman: float
    mean_abs_step: float
    sign_changes: float
    min: float
    max: float
    range: float


def load_model_and_transform(args):
    ensure_exists(args.ckpt, "ckpt")
    from effort.model import EffortCLIP
    from effort.data import build_transform
    from effort.utils import load_ckpt

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.use_d3:
        from effort.model import EffortCLIPD3
        model = EffortCLIPD3(
            clip_model_or_path=args.clip_model,
            use_d3=True,
            patch_shuffle_prob=float(args.d3_patch_shuffle_prob),
            patch_size=int(args.d3_patch_size),
            degrade_prob=float(args.d3_degrade_prob),
            n_heads=int(args.d3_heads),
            n_layers=int(args.d3_layers),
            dropout=float(args.d3_dropout),
            enable_osd_regs=False,
        )
    else:
        model = EffortCLIP(
            clip_model_or_path=args.clip_model,
            patch_pool=bool(args.patch_pool),
            patch_pool_tau=float(args.patch_pool_tau),
            patch_pool_mode=str(args.patch_pool_mode),
            patch_trim_p=float(args.patch_trim_p),
            patch_quality=str(args.patch_quality),
        )
        if args.patch_pool and hasattr(model, "set_patch_scale"):
            model.set_patch_scale(float(args.patch_pool_gamma))

    missing, unexpected = load_ckpt(args.ckpt, model, strict=args.strict)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    transform = build_transform(image_size=int(args.image_size), augment=False)
    return model, transform, device


@torch.no_grad()
def score_one(model, device, transform, pil_img: Image.Image, amp: bool) -> Dict[str, Any]:
    x = transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with autocast(device_type=device.type, enabled=bool(amp)):
        out = model({"image": x})
    ret = {"prob": float(out["prob"].detach().cpu().item())}
    if "patch_delta" in out:
        ret["patch_delta"] = float(out["patch_delta"].detach().cpu().item())
    return ret


def run_tta(
    model,
    device,
    transform,
    pil_img: Image.Image,
    amp: bool,
    k: int,
    hflip_p: float,
    blur_sig: Tuple[float, float],
    jpeg_q: Tuple[int, int],
    tta_disable_blur: bool,
    tta_disable_jpeg: bool,
    trim_ratio: float,
) -> TTASummary:
    scores: List[float] = []
    deltas: List[float] = []
    for i in range(k):
        v = pil_img.convert("RGB")
        if i > 0:
            if hflip_p > 0 and random.random() < hflip_p:
                v = apply_hflip(v)
            if not tta_disable_blur:
                lo, hi = float(blur_sig[0]), float(blur_sig[1])
                sigma = random.random() * (hi - lo) + lo
                v = apply_gaussian_blur_pil(v, sigma=sigma)
            if not tta_disable_jpeg:
                qlo, qhi = int(jpeg_q[0]), int(jpeg_q[1])
                q = random.randint(min(qlo, qhi), max(qlo, qhi))
                v = apply_jpeg_recompress(v, quality=q)
        out = score_one(model, device, transform, v, amp=amp)
        scores.append(out["prob"])
        if "patch_delta" in out:
            deltas.append(out["patch_delta"])

    arr = np.asarray(scores, dtype=np.float32)
    summ = TTASummary(
        k=len(scores),
        scores=[float(x) for x in scores],
        mean=float(arr.mean()),
        std=float(arr.std()),
        var=float(arr.var()),
        median=float(np.median(arr)),
        trimmed_mean=float(robust_trimmed_mean(scores, trim_ratio=trim_ratio)),
        min=float(arr.min()),
        max=float(arr.max()),
        range=float(arr.max() - arr.min()),
    )
    if deltas:
        darr = np.asarray(deltas, dtype=np.float32)
        summ.patch_delta_std = float(darr.std())
        summ.patch_delta_range = float(darr.max() - darr.min())
    return summ


def run_blur_ladder(model, device, transform, pil_img: Image.Image, amp: bool, sigmas: List[float]) -> LadderSummary:
    scores = []
    for s in sigmas:
        v = apply_gaussian_blur_pil(pil_img.convert("RGB"), sigma=float(s))
        out = score_one(model, device, transform, v, amp=amp)
        scores.append(out["prob"])
    mono = monotonicity_stats(sigmas, scores)
    arr = np.asarray(scores, dtype=np.float32)
    return LadderSummary(
        name="gaussian_blur_sigma",
        levels=[float(x) for x in sigmas],
        scores=[float(x) for x in scores],
        spearman=float(mono["spearman"]),
        mean_abs_step=float(mono["mean_abs_step"]),
        sign_changes=float(mono["sign_changes"]),
        min=float(arr.min()),
        max=float(arr.max()),
        range=float(arr.max() - arr.min()),
    )


def run_jpeg_ladder(
    model, device, transform, pil_img: Image.Image, amp: bool, qualities: List[int]
) -> LadderSummary:
    scores = []
    for q in qualities:
        v = apply_jpeg_recompress(pil_img.convert("RGB"), quality=int(q))
        out = score_one(model, device, transform, v, amp=amp)
        scores.append(out["prob"])
    levels = [float(q) for q in qualities]
    mono = monotonicity_stats(levels, scores)
    arr = np.asarray(scores, dtype=np.float32)
    return LadderSummary(
        name="jpeg_quality",
        levels=levels,
        scores=[float(x) for x in scores],
        spearman=float(mono["spearman"]),
        mean_abs_step=float(mono["mean_abs_step"]),
        sign_changes=float(mono["sign_changes"]),
        min=float(arr.min()),
        max=float(arr.max()),
        range=float(arr.max() - arr.min()),
    )


def _sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", type=str, default=".")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--clip_model", type=str, default=None)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--strict", action="store_true")

    p.add_argument("--image_list", type=str, default="")
    p.add_argument("--image_dir", type=str, default="")
    p.add_argument("--image_size", type=int, default=224)

    p.add_argument("--use_d3", action="store_true")
    p.add_argument("--d3_patch_shuffle_prob", type=float, default=0.5)
    p.add_argument("--d3_patch_size", type=int, default=16)
    p.add_argument("--d3_degrade_prob", type=float, default=1.0)
    p.add_argument("--d3_heads", type=int, default=8)
    p.add_argument("--d3_layers", type=int, default=1)
    p.add_argument("--d3_dropout", type=float, default=0.1)

    p.add_argument("--patch_pool", action="store_true")
    p.add_argument("--patch_pool_tau", type=float, default=1.5)
    p.add_argument("--patch_pool_mode", type=str, default="lse", choices=["lse"])
    p.add_argument("--patch_trim_p", type=float, default=0.2)
    p.add_argument("--patch_quality", type=str, default="none", choices=["none", "cos", "cos_norm"])
    p.add_argument("--patch_pool_gamma", type=float, default=0.0)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tta_k", type=int, default=9)
    p.add_argument("--tta_trim_ratio", type=float, default=0.2)
    p.add_argument("--tta_hflip_p", type=float, default=0.5)
    p.add_argument("--tta_blur_sig", type=float, nargs=2, default=[0.0, 3.0])
    p.add_argument("--tta_jpeg_q", type=int, nargs=2, default=[30, 95])
    p.add_argument("--tta_disable_blur", action="store_true")
    p.add_argument("--tta_disable_jpeg", action="store_true")

    p.add_argument("--blur_sigmas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0])
    p.add_argument("--jpeg_qualities", type=int, nargs="+", default=[95, 80, 65, 50, 35, 25, 15])
    p.add_argument("--outdir", type=str, default="diag_out_comp")
    return p.parse_args()


def main():
    args = parse_args()
    if Image is None:
        raise ModuleNotFoundError("Pillow is required. Please install with: pip install pillow")
    resolve_imports(args.repo_root)

    ensure_exists(args.ckpt, "ckpt")
    if not args.image_list and not args.image_dir:
        raise ValueError("Provide --image_list or --image_dir")
    if args.image_list:
        ensure_exists(args.image_list, "image_list")
    if args.image_dir:
        ensure_exists(args.image_dir, "image_dir")

    set_seed(int(args.seed))
    model, transform, device = load_model_and_transform(args)

    items: List[Dict[str, Any]] = []
    if args.image_list:
        items.extend(read_image_list(args.image_list))
    if args.image_dir:
        root = Path(args.image_dir)
        exts = {".jpg", ".jpeg", ".png"}
        for p in sorted([x for x in root.glob("*") if x.is_file() and x.suffix.lower() in exts], key=_sort_key):
            items.append({"path": str(p), "label": "", "note": ""})

    seen = set()
    uniq = []
    for it in items:
        ip = it["path"]
        if ip in seen:
            continue
        seen.add(ip)
        uniq.append(it)
    items = uniq

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    full_report = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(Path(args.repo_root).resolve()),
        "ckpt": args.ckpt,
        "model_mode": "d3" if args.use_d3 else "single_view",
        "tta": {
            "k": args.tta_k,
            "hflip_p": args.tta_hflip_p,
            "blur_sig": args.tta_blur_sig,
            "jpeg_q": args.tta_jpeg_q,
            "trim_ratio": args.tta_trim_ratio,
        },
        "ladders": {
            "blur_sigmas": args.blur_sigmas,
            "jpeg_qualities": args.jpeg_qualities,
        },
        "items": [],
    }

    for it in items:
        img_path = it["path"]
        if not Path(img_path).exists():
            print(f"[skip] missing image: {img_path}")
            continue
        pil = Image.open(img_path).convert("RGB")

        base = score_one(model, device, transform, pil, amp=args.amp)
        tta = run_tta(
            model,
            device,
            transform,
            pil,
            amp=args.amp,
            k=int(args.tta_k),
            hflip_p=float(args.tta_hflip_p),
            blur_sig=(float(args.tta_blur_sig[0]), float(args.tta_blur_sig[1])),
            jpeg_q=(int(args.tta_jpeg_q[0]), int(args.tta_jpeg_q[1])),
            tta_disable_blur=bool(args.tta_disable_blur),
            tta_disable_jpeg=bool(args.tta_disable_jpeg),
            trim_ratio=float(args.tta_trim_ratio),
        )
        blur_ladder = run_blur_ladder(
            model, device, transform, pil, amp=args.amp, sigmas=[float(x) for x in args.blur_sigmas]
        )
        jpeg_ladder = run_jpeg_ladder(
            model, device, transform, pil, amp=args.amp, qualities=[int(x) for x in args.jpeg_qualities]
        )

        dice_like = (blur_ladder.sign_changes >= 2 or jpeg_ladder.sign_changes >= 2) and (tta.range >= 0.25)
        wiped_out_candidate = dice_like or (tta.var >= 0.02 and 0.35 < tta.median < 0.65)

        record = {
            "path": img_path,
            "label": it.get("label", ""),
            "note": it.get("note", ""),
            "base_prob": base["prob"],
            "tta": asdict(tta),
            "blur_ladder": asdict(blur_ladder),
            "jpeg_ladder": asdict(jpeg_ladder),
            "flags": {
                "dice_like": bool(dice_like),
                "wiped_out_candidate": bool(wiped_out_candidate),
            },
        }
        full_report["items"].append(record)

        rows.append(
            {
                "path": img_path,
                "label": it.get("label", ""),
                "note": it.get("note", ""),
                "base_prob": base["prob"],
                "tta_mean": tta.mean,
                "tta_std": tta.std,
                "tta_var": tta.var,
                "tta_range": tta.range,
                "tta_median": tta.median,
                "tta_trimmed": tta.trimmed_mean,
                "tta_patch_delta_std": "" if tta.patch_delta_std is None else tta.patch_delta_std,
                "tta_patch_delta_range": "" if tta.patch_delta_range is None else tta.patch_delta_range,
                "blur_spearman": blur_ladder.spearman,
                "blur_sign_changes": blur_ladder.sign_changes,
                "jpeg_spearman": jpeg_ladder.spearman,
                "jpeg_sign_changes": jpeg_ladder.sign_changes,
                "dice_like": int(dice_like),
                "wiped_out_candidate": int(wiped_out_candidate),
            }
        )

        print(
            f"[ok] {Path(img_path).name} base={base['prob']:.3f} "
            f"tta_std={tta.std:.3f} range={tta.range:.3f} "
            f"blur_wiggle={blur_ladder.sign_changes} jpeg_wiggle={jpeg_ladder.sign_changes} "
            f"wiped={int(wiped_out_candidate)}"
        )

    report_json = outdir / "report.json"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    summary_csv = outdir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["path"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[DONE] wrote {report_json}")
    print(f"[DONE] wrote {summary_csv}")


if __name__ == "__main__":
    main()
