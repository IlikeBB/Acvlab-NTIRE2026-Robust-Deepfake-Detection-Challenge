import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from effort.metrics import (
    compute_metrics,
    fpr_at_tpr,
    threshold_at_tpr,
    precision_at_recall,
    spearman_corr,
    slice_fpr_real_bottom,
    per_group_metrics,
    worst_real_fpr_over_bins,
    worst_fake_fnr_over_bins,
    nll_binary,
    brier_score,
    ece_binary,
    pred_entropy_binary,
    tpr_at_fpr,
    tnr_at_fnr,
)

from effort.data import CLIP_MEAN, CLIP_STD


def _as_pair(x, default):
    if x is None:
        return default
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return float(x[0]), float(x[1])
    return default


def _micro_tta_batch(
    images_norm: torch.Tensor,
    *,
    seed: int,
    blur_max_sig: float,
    jpeg_qual,
    contrast,
    gamma,
    resize_jitter: int,
):
    """Tensor-space micro-TTA on normalized CLIP images (B,C,H,W).

    This stays in-memory (no file I/O). JPEG is approximated via quantization + blockiness.
    """
    device = images_norm.device
    dtype = images_norm.dtype

    mean = torch.tensor(CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    x = images_norm * std + mean
    x = x.clamp(0.0, 1.0)

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    # Blur
    if blur_max_sig and float(blur_max_sig) > 0:
        sig = float(torch.rand((), generator=g, device=device).item()) * float(blur_max_sig)
        if sig > 1e-6:
            k = int(2 * int(np.ceil(3.0 * sig)) + 1)
            k = max(3, min(k, 31))
            t = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
            kern1d = torch.exp(-(t ** 2) / (2.0 * (sig ** 2)))
            kern1d = kern1d / kern1d.sum().clamp_min(1e-12)
            ky = kern1d.view(1, 1, k, 1)
            kx = kern1d.view(1, 1, 1, k)
            w_y = ky.expand(3, 1, k, 1)
            w_x = kx.expand(3, 1, 1, k)
            pad = k // 2
            x = F.conv2d(x, w_y, padding=(pad, 0), groups=3)
            x = F.conv2d(x, w_x, padding=(0, pad), groups=3)

    # Resize jitter (down/up)
    rj = int(resize_jitter or 0)
    if rj > 0:
        _, _, h, w = x.shape
        delta = int(torch.randint(low=-rj, high=rj + 1, size=(), generator=g, device=device).item())
        nh = max(8, h + delta)
        nw = max(8, w + delta)
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    # Contrast jitter
    c_lo, c_hi = _as_pair(contrast, (1.0, 1.0))
    if c_lo != 1.0 or c_hi != 1.0:
        c = float(torch.rand((), generator=g, device=device).item()) * (c_hi - c_lo) + c_lo
        m = x.mean(dim=(2, 3), keepdim=True)
        x = (x - m) * c + m

    # Gamma jitter
    g_lo, g_hi = _as_pair(gamma, (1.0, 1.0))
    if g_lo != 1.0 or g_hi != 1.0:
        gg = float(torch.rand((), generator=g, device=device).item()) * (g_hi - g_lo) + g_lo
        x = x.clamp(0.0, 1.0) ** gg

    # JPEG-ish blockiness proxy (fast tensor approximation)
    q_lo, q_hi = 60, 95
    if isinstance(jpeg_qual, (list, tuple)) and len(jpeg_qual) >= 2:
        q_lo, q_hi = int(jpeg_qual[0]), int(jpeg_qual[1])
    q_lo = max(1, min(100, q_lo))
    q_hi = max(1, min(100, q_hi))
    if q_hi < q_lo:
        q_lo, q_hi = q_hi, q_lo
    qv = int(torch.randint(low=q_lo, high=q_hi + 1, size=(), generator=g, device=device).item())
    strength = float(100 - qv) / 100.0
    if strength > 0:
        step = 1.0 + strength * 10.0
        x8 = torch.round(x * 255.0)
        xq = torch.round(x8 / step) * step
        x = (xq / 255.0).clamp(0.0, 1.0)
        blk = F.avg_pool2d(x, kernel_size=8, stride=8)
        blk = F.interpolate(blk, size=x.shape[-2:], mode="nearest")
        x = (1.0 - strength) * x + strength * blk

    x = x.clamp(0.0, 1.0)
    return (x - mean) / std


def _quality_proxy_from_tensor(images_norm: torch.Tensor, proxy: str):
    """Compute quality proxy from normalized tensor images (B,C,H,W)."""
    proxy = str(proxy or "blur").lower()
    device = images_norm.device
    dtype = images_norm.dtype
    mean = torch.tensor(CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    x = (images_norm * std + mean).clamp(0.0, 1.0)
    g = (0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]).unsqueeze(1)  # (B,1,H,W)

    if proxy == "brightness":
        q = g.mean(dim=(2, 3)).squeeze(1)
        return q.detach().cpu().numpy().astype(float)

    if proxy in {"jpeg", "jpegish", "blockiness"}:
        # Blockiness proxy via 8x8 boundary jumps normalized by overall local gradients.
        gv = torch.abs(g[:, :, :, 1:] - g[:, :, :, :-1])  # (B,1,H,W-1)
        gh = torch.abs(g[:, :, 1:, :] - g[:, :, :-1, :])  # (B,1,H-1,W)

        wv = gv.shape[-1]
        hv = gh.shape[-2]
        if wv > 8:
            cols = torch.arange(7, wv, 8, device=device)
            bnd_v = gv.index_select(dim=3, index=cols).mean(dim=(1, 2, 3))
        else:
            bnd_v = gv.mean(dim=(1, 2, 3))
        if hv > 8:
            rows = torch.arange(7, hv, 8, device=device)
            bnd_h = gh.index_select(dim=2, index=rows).mean(dim=(1, 2, 3))
        else:
            bnd_h = gh.mean(dim=(1, 2, 3))
        bnd = 0.5 * (bnd_v + bnd_h)
        base = 0.5 * (gv.mean(dim=(1, 2, 3)) + gh.mean(dim=(1, 2, 3)))
        q = torch.log((bnd / (base + 1e-6)).clamp_min(1e-6))
        return q.detach().cpu().numpy().astype(float)

    # default: blur proxy via Laplacian variance
    lap_k = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(g, lap_k, padding=1)
    q = torch.log(lap.var(dim=(2, 3), unbiased=False).squeeze(1).clamp_min(1e-6))
    return q.detach().cpu().numpy().astype(float)


def _assign_bins(values: np.ndarray, cuts):
    cuts = [float(x) for x in (cuts or [])]
    out = []
    for v in values.tolist():
        b = 0
        while b < len(cuts) and v > cuts[b]:
            b += 1
        out.append(int(b))
    return out


def validate(
    model,
    loader,
    amp=False,
    seed=None,
    device=None,
    *,
    tta_micro: bool = False,
    tta_k: int = 0,
    tta_seed: int = 0,
    tta_blur_max_sig: float = 1.5,
    tta_jpeg_qual=(60, 95),
    tta_contrast=(0.9, 1.1),
    tta_gamma=(0.9, 1.1),
    tta_resize_jitter: int = 8,
    swap_quality_proxy: str = None,
    swap_quality_cuts=None,
):
    model.eval()
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    all_probs = []
    all_labels = []
    losses = []
    all_q = []
    all_qbin = []
    all_group = []
    all_qswap = []
    tta_ranges = []
    tta_ranges_by_qbin = {}

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            with autocast(device_type=device.type, enabled=bool(amp) and device.type == "cuda"):
                pred = model(batch)
                loss_dict = model.get_losses(batch, pred)
                loss = loss_dict.get("overall", loss_dict.get("cls"))

            probs = pred["prob"].detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

            # Micro-TTA stability (tensor-space)
            if tta_micro and int(tta_k or 0) > 0 and ("image" in batch):
                images = batch["image"]
                bsz = int(images.shape[0])
                pmin = torch.full((bsz,), float("inf"), device=device)
                pmax = torch.full((bsz,), float("-inf"), device=device)
                for k in range(int(tta_k)):
                    seed_k = int(tta_seed) + 1000003 * int(k) + 1009 * int(len(all_probs))
                    img_aug = _micro_tta_batch(
                        images,
                        seed=seed_k,
                        blur_max_sig=float(tta_blur_max_sig),
                        jpeg_qual=tta_jpeg_qual,
                        contrast=tta_contrast,
                        gamma=tta_gamma,
                        resize_jitter=int(tta_resize_jitter),
                    )
                    b2 = {"image": img_aug}
                    with autocast(device_type=device.type, enabled=bool(amp) and device.type == "cuda"):
                        pred2 = model(b2)
                    p = pred2["prob"].detach().view(-1)
                    pmin = torch.minimum(pmin, p)
                    pmax = torch.maximum(pmax, p)
                rng = (pmax - pmin).detach().cpu().numpy().astype(float)
                tta_ranges.extend(rng.tolist())
                if "quality_bin" in batch:
                    qb_np = (
                        batch["quality_bin"].detach().cpu().numpy()
                        if torch.is_tensor(batch["quality_bin"])
                        else np.asarray(batch["quality_bin"])
                    ).astype(int)
                    for r, qb in zip(rng.tolist(), qb_np.tolist()):
                        tta_ranges_by_qbin.setdefault(int(qb), []).append(float(r))
            if "q_log" in batch:
                q = batch["q_log"].detach().cpu().numpy() if torch.is_tensor(batch["q_log"]) else np.asarray(batch["q_log"])
                all_q.extend(q.tolist())
            if "quality_bin" in batch:
                qb = (
                    batch["quality_bin"].detach().cpu().numpy()
                    if torch.is_tensor(batch["quality_bin"])
                    else np.asarray(batch["quality_bin"])
                )
                all_qbin.extend(qb.tolist())
            if "stratum_id" in batch:
                g = (
                    batch["stratum_id"].detach().cpu().numpy()
                    if torch.is_tensor(batch["stratum_id"])
                    else np.asarray(batch["stratum_id"])
                )
                all_group.extend(g.tolist())
            if loss is not None:
                losses.append(float(loss.item()) if hasattr(loss, "item") else float(loss))

            if swap_quality_proxy:
                q_swap = _quality_proxy_from_tensor(batch["image"], proxy=swap_quality_proxy)
                all_qswap.extend(np.asarray(q_swap, dtype=float).tolist())

    if all_labels:
        metrics = compute_metrics(all_labels, all_probs)
    else:
        metrics = {"acc": 0.0, "auc": 0.0, "eer": 0.0, "ap": 0.0, "precision": 0.0, "recall": 0.0}
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["fpr_at_tpr95"] = fpr_at_tpr(all_labels, all_probs, tpr_target=0.95)
    metrics["threshold_at_tpr95"] = threshold_at_tpr(all_labels, all_probs, tpr_target=0.95)
    metrics["precision_at_recall80"] = precision_at_recall(all_labels, all_probs, recall_target=0.80)
    # Calibration / confidence
    metrics["nll"] = nll_binary(all_labels, all_probs)
    metrics["brier"] = brier_score(all_labels, all_probs)
    metrics["ece15"] = ece_binary(all_labels, all_probs, n_bins=15)
    metrics["pred_entropy"] = pred_entropy_binary(all_probs)
    # Operating points (TPR@FPR)
    metrics["tpr_at_fpr1"] = tpr_at_fpr(all_labels, all_probs, fpr_target=0.01)
    metrics["tpr_at_fpr0.5"] = tpr_at_fpr(all_labels, all_probs, fpr_target=0.005)
    # Reverse operating points (TNR@FNR)
    metrics["tnr_at_fnr1"] = tnr_at_fnr(all_labels, all_probs, fnr_target=0.01)
    metrics["tnr_at_fnr0.5"] = tnr_at_fnr(all_labels, all_probs, fnr_target=0.005)
    metrics["blur_spearman_all"] = float("nan")
    metrics["blur_spearman_real_only"] = float("nan")
    metrics["blur_spearman_fake_only"] = float("nan")

    # Optional E10 swap-test: override quality proxy / bins using tensor-space proxy.
    if swap_quality_proxy and all_qswap and len(all_qswap) == len(all_probs):
        qv = np.asarray(all_qswap, dtype=float)
        all_q = qv.tolist()
        if swap_quality_cuts is not None:
            all_qbin = _assign_bins(qv, swap_quality_cuts)
        elif all_qbin:
            # keep same #bins as dataset but recompute on current val if cuts absent
            n_bins = int(max(all_qbin) + 1)
            qs = [i / n_bins for i in range(1, n_bins)] if n_bins > 1 else []
            cuts = [float(np.quantile(qv, x)) for x in qs]
            all_qbin = _assign_bins(qv, cuts)
        else:
            n_bins = 3
            cuts = [float(np.quantile(qv, 1 / n_bins)), float(np.quantile(qv, 2 / n_bins))]
            all_qbin = _assign_bins(qv, cuts)
        metrics["quality_proxy"] = str(swap_quality_proxy)
    else:
        metrics["quality_proxy"] = "blur"

    if all_q and len(all_q) == len(all_probs):
        y = np.asarray(all_labels).astype(int)
        s = np.asarray(all_probs).astype(float)
        q = np.asarray(all_q).astype(float)
        metrics["spearman_q_score_overall"] = spearman_corr(q, s)
        metrics["spearman_q_score_real"] = spearman_corr(q[y == 0], s[y == 0]) if (y == 0).sum() > 1 else float("nan")
        metrics["spearman_q_score_fake"] = spearman_corr(q[y == 1], s[y == 1]) if (y == 1).sum() > 1 else float("nan")
        metrics["blur_spearman_all"] = metrics["spearman_q_score_overall"]
        metrics["blur_spearman_real_only"] = metrics["spearman_q_score_real"]
        metrics["blur_spearman_fake_only"] = metrics["spearman_q_score_fake"]
        metrics["proxy_spearman_all"] = metrics["spearman_q_score_overall"]
        metrics["proxy_spearman_real_only"] = metrics["spearman_q_score_real"]
        metrics["proxy_spearman_fake_only"] = metrics["spearman_q_score_fake"]
        thr = metrics.get("threshold_at_tpr95", float("nan"))
        if np.isfinite(thr):
            metrics["fpr_real_bottom10"] = slice_fpr_real_bottom(y, s, q, pct=0.10, threshold=thr)
            metrics["fpr_real_bottom20"] = slice_fpr_real_bottom(y, s, q, pct=0.20, threshold=thr)
        else:
            metrics["fpr_real_bottom10"] = slice_fpr_real_bottom(y, s, q, pct=0.10)
            metrics["fpr_real_bottom20"] = slice_fpr_real_bottom(y, s, q, pct=0.20)

    if all_group and len(all_group) == len(all_probs):
        grp = per_group_metrics(all_labels, all_probs, all_group)
        metrics["per_stratum"] = grp
        valid_err = [v["error"] for v in grp.values() if np.isfinite(v["error"])]
        metrics["worst_stratum_error"] = float(max(valid_err)) if valid_err else float("nan")

    # quality-bin diagnostics: can support per-bin AUC and class-conditional worst risk.
    if all_qbin and len(all_qbin) == len(all_probs):
        qgrp = per_group_metrics(all_labels, all_probs, all_qbin)
        metrics["per_quality_bin"] = qgrp
        valid_q_auc = [v["auc"] for v in qgrp.values() if np.isfinite(v["auc"])]
        metrics["worst_quality_bin_auc"] = float(min(valid_q_auc)) if valid_q_auc else float("nan")
        thr = metrics.get("threshold_at_tpr95", 0.5)
        worst_real, detail_real = worst_real_fpr_over_bins(all_labels, all_probs, all_qbin, threshold=thr)
        worst_fake, detail_fake = worst_fake_fnr_over_bins(all_labels, all_probs, all_qbin, threshold=thr)
        metrics["worst_real_fpr_over_bins"] = worst_real
        metrics["worst_fake_fnr_over_bins"] = worst_fake
        metrics["fpr_real_per_quality_bin"] = detail_real
        metrics["fnr_fake_per_quality_bin"] = detail_fake

        # Fixed global threshold from real-only scores (target FPR=1%)
        y = np.asarray(all_labels).astype(int)
        s = np.asarray(all_probs).astype(float)
        qb = np.asarray(all_qbin).astype(int)
        real = y == 0
        if real.sum() > 0:
            thr1 = float(np.quantile(s[real], 0.99))
            perbin = {}
            for b in sorted(np.unique(qb).tolist()):
                m = (qb == int(b)) & real
                perbin[int(b)] = float((s[m] >= thr1).mean()) if m.sum() > 0 else float("nan")
            metrics["thr_real_fpr1"] = thr1
            metrics["perbin_real_fpr1"] = perbin
            metrics["worstbin_real_fpr1"] = float(perbin.get(0, float("nan")))
            finite = [v for v in perbin.values() if np.isfinite(v)]
            metrics["gap_real_fpr1"] = float((max(finite) - min(finite)) if finite else float("nan"))
        else:
            metrics["thr_real_fpr1"] = float("nan")
            metrics["perbin_real_fpr1"] = {}
            metrics["worstbin_real_fpr1"] = float("nan")
            metrics["gap_real_fpr1"] = float("nan")

        # Fixed global threshold from fake-only scores (target FNR=1%)
        fake = y == 1
        if fake.sum() > 0:
            thr_fake = float(np.quantile(s[fake], 0.01))
            perbin_fake = {}
            for b in sorted(np.unique(qb).tolist()):
                m = (qb == int(b)) & fake
                perbin_fake[int(b)] = float((s[m] <= thr_fake).mean()) if m.sum() > 0 else float("nan")
            metrics["thr_fake_fnr1"] = thr_fake
            metrics["perbin_fake_fnr1"] = perbin_fake
            metrics["worstbin_fake_fnr1"] = float(perbin_fake.get(0, float("nan")))
            finite_fake = [v for v in perbin_fake.values() if np.isfinite(v)]
            metrics["gap_fake_fnr1"] = float((max(finite_fake) - min(finite_fake)) if finite_fake else float("nan"))
        else:
            metrics["thr_fake_fnr1"] = float("nan")
            metrics["perbin_fake_fnr1"] = {}
            metrics["worstbin_fake_fnr1"] = float("nan")
            metrics["gap_fake_fnr1"] = float("nan")

        aucs = [v.get("auc", float("nan")) for v in qgrp.values()]
        aucs = [float(a) for a in aucs if np.isfinite(a)]
        metrics["group_gap_auc"] = float((max(aucs) - min(aucs)) if aucs else float("nan"))

    if tta_ranges:
        tr = np.asarray(tta_ranges, dtype=float)
        metrics["tta_range_micro_mean"] = float(tr.mean())
        metrics["tta_range_micro_p90"] = float(np.quantile(tr, 0.90))
        by_bin = {}
        for b, vals in sorted(tta_ranges_by_qbin.items()):
            v = np.asarray(vals, dtype=float)
            by_bin[int(b)] = {
                "mean": float(v.mean()) if len(v) else float("nan"),
                "p90": float(np.quantile(v, 0.90)) if len(v) else float("nan"),
                "count": int(len(v)),
            }
        metrics["tta_range_micro_by_qbin"] = by_bin
    else:
        metrics["tta_range_micro_mean"] = float("nan")
        metrics["tta_range_micro_p90"] = float("nan")
        metrics["tta_range_micro_by_qbin"] = {}
    return metrics
