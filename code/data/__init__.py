import csv
import logging
from collections import Counter

import torch
from torch.utils.data import DataLoader

from effort.data import (
    FolderDataset,
    FramesDirCSVDataset,
    ManifestDataset,
    MixedFolderDataset,
    StratifiedBatchSampler,
    collate_skip_none,
)


def _as_roots(value):
    if value is None:
        return []
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",") if v.strip()]
        return parts if parts else [value]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def _csv_has_frames_dir(csv_path):
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
        return "frames_dir" in fields
    except Exception:
        return False


def _quality_route_needs_meta(opt):
    route = str(getattr(opt, "multi_expert_route", "quality_bin") or "quality_bin").lower()
    return bool(getattr(opt, "multi_expert_enable", False)) and route in {
        "quality_bin",
        "quality_soft",
        "hybrid",
        "degrade_hybrid",
    }


def _train_needs_quality_meta(opt):
    return any(
        [
            bool(getattr(opt, "quality_balance", False)),
            bool(getattr(opt, "use_groupdro", False)),
            bool(getattr(opt, "use_reweighted_erm", False)),
            bool(getattr(opt, "use_stratified_sampler", False)),
            _quality_route_needs_meta(opt),
        ]
    )


def _val_needs_quality_meta(opt):
    return any(
        [
            bool(getattr(opt, "val_quality_diag", False)),
            _train_needs_quality_meta(opt),
            _quality_route_needs_meta(opt),
        ]
    )


def _dataset_from_csv(
    csv_path,
    *,
    split,
    frame_num,
    image_size,
    on_error,
    augment,
    opt,
    quality_balance,
    quality_bins,
    quality_max_side,
    quality_cache_path,
    return_path=False,
):
    common = dict(
        image_size=image_size,
        on_error=on_error,
        augment=augment,
        hflip_prob=float(getattr(opt, "hflip_prob", 0.0)),
        blur_prob=float(getattr(opt, "blur_prob", 0.0)),
        blur_sig=getattr(opt, "blur_sig", (0.0, 3.0)),
        jpg_prob=float(getattr(opt, "jpg_prob", 0.0)),
        jpg_qual=getattr(opt, "jpg_qual", (30, 100)),
        opt=opt,
        quality_balance=bool(quality_balance),
        quality_bins=int(quality_bins),
        quality_max_side=int(quality_max_side),
        quality_cache_path=quality_cache_path,
        return_path=bool(return_path),
    )
    if _csv_has_frames_dir(csv_path):
        return FramesDirCSVDataset(
            csv_path,
            split=split,
            frame_num=frame_num,
            **common,
        )
    return ManifestDataset(csv_path, **common)


def _build_train_dataset(opt):
    quality_balance = _train_needs_quality_meta(opt)
    quality_bins = int(getattr(opt, "quality_bins", 3) or 3)
    quality_max_side = int(getattr(opt, "quality_max_side", 256) or 256)
    quality_cache_path = getattr(opt, "quality_cache_train_path", None) or getattr(opt, "quality_cache_path", None)
    image_size = int(getattr(opt, "image_size", 224) or 224)
    on_error = str(getattr(opt, "on_error", "skip"))
    train_split = str(getattr(opt, "train_split", "train"))
    train_frame_num = getattr(opt, "train_frame_num", None)

    if getattr(opt, "train_csv", None):
        return _dataset_from_csv(
            getattr(opt, "train_csv"),
            split=train_split,
            frame_num=train_frame_num,
            image_size=image_size,
            on_error=on_error,
            augment=True,
            opt=opt,
            quality_balance=quality_balance,
            quality_bins=quality_bins,
            quality_max_side=quality_max_side,
            quality_cache_path=quality_cache_path,
            return_path=False,
        )

    if getattr(opt, "csv", None):
        return _dataset_from_csv(
            getattr(opt, "csv"),
            split=train_split,
            frame_num=train_frame_num,
            image_size=image_size,
            on_error=on_error,
            augment=True,
            opt=opt,
            quality_balance=quality_balance,
            quality_bins=quality_bins,
            quality_max_side=quality_max_side,
            quality_cache_path=quality_cache_path,
            return_path=False,
        )

    if getattr(opt, "folder_root", None):
        return FolderDataset(
            getattr(opt, "folder_root"),
            split=train_split,
            image_size=image_size,
            on_error=on_error,
            augment=True,
            hflip_prob=float(getattr(opt, "hflip_prob", 0.0)),
            blur_prob=float(getattr(opt, "blur_prob", 0.0)),
            blur_sig=getattr(opt, "blur_sig", (0.0, 3.0)),
            jpg_prob=float(getattr(opt, "jpg_prob", 0.0)),
            jpg_qual=getattr(opt, "jpg_qual", (30, 100)),
            opt=opt,
            quality_balance=quality_balance,
            quality_bins=quality_bins,
            quality_max_side=quality_max_side,
            quality_cache_path=quality_cache_path,
            return_path=False,
        )

    raise ValueError("No training source set. Provide `train_csv`, `csv`, or `folder_root`.")


def _build_val_dataset(opt):
    quality_balance = _val_needs_quality_meta(opt)
    quality_bins = int(getattr(opt, "quality_bins", 3) or 3)
    quality_max_side = int(getattr(opt, "quality_max_side", 256) or 256)
    quality_cache_path = getattr(opt, "quality_cache_val_path", None) or getattr(opt, "quality_cache_path", None)
    image_size = int(getattr(opt, "image_size", 224) or 224)
    on_error = str(getattr(opt, "on_error", "skip"))
    val_split = str(getattr(opt, "val_split", "val"))
    val_frame_num = getattr(opt, "val_frame_num", None)

    if getattr(opt, "val_csv", None):
        return _dataset_from_csv(
            getattr(opt, "val_csv"),
            split=val_split,
            frame_num=val_frame_num,
            image_size=image_size,
            on_error=on_error,
            augment=False,
            opt=opt,
            quality_balance=quality_balance,
            quality_bins=quality_bins,
            quality_max_side=quality_max_side,
            quality_cache_path=quality_cache_path,
            return_path=False,
        )

    if getattr(opt, "val_data_root", None):
        roots = _as_roots(getattr(opt, "val_data_root"))
        if roots:
            return MixedFolderDataset(
                roots,
                image_size=image_size,
                on_error=on_error,
                augment=False,
                hflip_prob=0.0,
                blur_prob=0.0,
                blur_sig=(0.0, 0.0),
                jpg_prob=0.0,
                jpg_qual=(100, 100),
                opt=opt,
                quality_balance=quality_balance,
                quality_bins=quality_bins,
                quality_max_side=quality_max_side,
                quality_cache_path=quality_cache_path,
                return_path=False,
            )

    if getattr(opt, "folder_root", None):
        return FolderDataset(
            getattr(opt, "folder_root"),
            split=val_split,
            image_size=image_size,
            on_error=on_error,
            augment=False,
            hflip_prob=0.0,
            blur_prob=0.0,
            blur_sig=(0.0, 0.0),
            jpg_prob=0.0,
            jpg_qual=(100, 100),
            opt=opt,
            quality_balance=quality_balance,
            quality_bins=quality_bins,
            quality_max_side=quality_max_side,
            quality_cache_path=quality_cache_path,
            return_path=False,
        )

    if getattr(opt, "csv", None):
        return _dataset_from_csv(
            getattr(opt, "csv"),
            split=val_split,
            frame_num=val_frame_num,
            image_size=image_size,
            on_error=on_error,
            augment=False,
            opt=opt,
            quality_balance=quality_balance,
            quality_bins=quality_bins,
            quality_max_side=quality_max_side,
            quality_cache_path=quality_cache_path,
            return_path=False,
        )

    return None


def _build_stratum_weight_map(opt, train_dataset):
    if (not bool(getattr(opt, "use_reweighted_erm", False))) or (not hasattr(train_dataset, "stratum_ids")):
        return
    gids = getattr(train_dataset, "stratum_ids", None)
    if gids is None:
        logging.warning("[data] use_reweighted_erm=True but train dataset has no stratum_ids; skip reweight map.")
        return

    counts = Counter(int(g) for g in gids)
    if not counts:
        return
    use_sqrt = bool(getattr(opt, "reweight_use_sqrt", False))
    raw = {}
    for g, c in counts.items():
        c = max(1, int(c))
        raw[g] = (1.0 / (c ** 0.5)) if use_sqrt else (1.0 / float(c))

    n = float(sum(counts.values()))
    mean_w = sum(raw[g] * counts[g] for g in counts) / max(1.0, n)
    if mean_w <= 0:
        mean_w = 1.0
    w = {g: (raw[g] / mean_w) for g in raw}

    wmin = float(getattr(opt, "reweight_clip_min", 0.2) or 0.2)
    wmax = float(getattr(opt, "reweight_clip_max", 5.0) or 5.0)
    if wmax < wmin:
        wmin, wmax = wmax, wmin
    w = {g: float(max(wmin, min(wmax, ww))) for g, ww in w.items()}
    opt.stratum_weight_map = {int(g): float(v) for g, v in w.items()}
    logging.info("[data] built stratum_weight_map for %d strata (sqrt=%s)", len(w), use_sqrt)


def create_dataloaders(opt):
    train_ds = _build_train_dataset(opt)
    _build_stratum_weight_map(opt, train_ds)
    val_ds = _build_val_dataset(opt)

    batch_size = int(getattr(opt, "batch_size", 32) or 32)
    num_workers = int(getattr(opt, "num_threads", 4) or 0)
    pin_memory = bool(torch.cuda.is_available())
    collate_fn = collate_skip_none if str(getattr(opt, "on_error", "skip")) == "skip" else None

    if bool(getattr(opt, "use_stratified_sampler", False)):
        sampler_key = str(getattr(opt, "stratified_sampler_key", "stratum_id"))
        batch_sampler = StratifiedBatchSampler(
            train_ds,
            batch_size=batch_size,
            key=sampler_key,
            shuffle=not bool(getattr(opt, "serial_batches", False)),
            drop_last=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=(num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=not bool(getattr(opt, "serial_batches", False)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=(num_workers > 0),
        )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=(num_workers > 0),
        )

    logging.info(
        "[data] train=%d val=%s q_cuts=%s",
        len(train_ds),
        ("none" if val_ds is None else str(len(val_ds))),
        getattr(opt, "quality_cuts", None),
    )
    return train_loader, val_loader


__all__ = ["create_dataloaders"]
