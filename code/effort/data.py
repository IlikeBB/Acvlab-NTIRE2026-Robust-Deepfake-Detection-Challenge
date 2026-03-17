import csv
import os
import io
import random
from types import SimpleNamespace
from pathlib import Path

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import aug_config_fixed as aug_cfg

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _parse_rgb_triplet(value, default_triplet):
    if value is None:
        return [float(x) for x in default_triplet]
    vals = value
    if isinstance(vals, str):
        vals = [v.strip() for v in vals.split(",") if v.strip()]
    try:
        vals = [float(v) for v in list(vals)]
    except Exception:
        return [float(x) for x in default_triplet]
    if len(vals) == 1:
        vals = vals * 3
    if len(vals) != 3:
        return [float(x) for x in default_triplet]
    return vals


def _resolve_norm_stats(opt):
    mean = _parse_rgb_triplet(getattr(opt, "normalize_mean", None) if opt is not None else None, CLIP_MEAN)
    std = _parse_rgb_triplet(getattr(opt, "normalize_std", None) if opt is not None else None, CLIP_STD)
    std = [max(1e-6, float(v)) for v in std]
    return mean, std


def _load_image(path):
    with Image.open(path) as img:
        return img.convert("RGB")



# ---------------------------------------------------------------------------
# Quality-balanced training utilities
# ---------------------------------------------------------------------------

def _to_gray_np(img: Image.Image) -> "np.ndarray":
    """Convert PIL image to grayscale numpy float32 array (H,W)."""
    import numpy as np
    g = img.convert("L")
    return np.asarray(g, dtype=np.float32)


def laplacian_variance(img: Image.Image, max_side: int = 256) -> float:
    """Laplacian variance sharpness proxy.

    English: Laplacian variance = sharpness proxy.
    We compute on a deterministic resized copy (max_side) for speed + consistency.
    """
    import numpy as np

    if max_side is not None and max_side > 0:
        w, h = img.size
        m = max(w, h)
        if m > max_side:
            scale = float(max_side) / float(m)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            img = img.resize((nw, nh), resample=Image.BILINEAR)

    g = _to_gray_np(img)  # (H,W)
    # 3x3 Laplacian kernel (4-neighborhood)
    # [[0, 1, 0],
    #  [1,-4, 1],
    #  [0, 1, 0]]
    lap = (
        -4.0 * g
        + np.roll(g, 1, axis=0)
        + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1)
        + np.roll(g, -1, axis=1)
    )
    return float(lap.var())


def _compute_quality_metadata(
    records,
    n_bins: int = 3,
    max_side: int = 256,
    cache_path: str = None,
    cutpoints=None,
    eps: float = 1e-6,
):
    """
    Precompute q_sharp (log-laplacian-var) and quality_bin for each record index.

    Returns:
        q_log_list: List[float] length=len(records)
        bin_list:   List[int]   length=len(records)
        thresholds: (q33, q66) when n_bins==3 else tuple of quantile cutpoints
    """
    import numpy as np
    from pathlib import Path

    paths = [p for (p, _) in records]
    labels = [y for (_, y) in records]

    q_by_path = {}
    if cache_path:
        cp = Path(cache_path)
        if cp.exists():
            try:
                import csv
                with cp.open(newline="") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        q_by_path[row["path"]] = float(row["q_log"])
            except Exception:
                q_by_path = {}

    # Compute missing paths
    missing = [p for p in paths if p not in q_by_path]
    if missing:
        computed_rows = []
        iterator = missing
        try:
            from tqdm import tqdm  # optional dependency
            # Low-refresh progress bar to reduce update overhead.
            show_bar = len(missing) >= 1000
            iterator = tqdm(
                missing,
                desc="quality-cache",
                leave=False,
                disable=(not show_bar),
                mininterval=3.0,
                maxinterval=10.0,
                miniters=max(1, len(missing) // 200),
            )
        except Exception:
            iterator = missing
        for p in iterator:
            try:
                img = _load_image(p)
                q = laplacian_variance(img, max_side=max_side)
                q_by_path[p] = float(np.log(q + eps))
                computed_rows.append((p, q_by_path[p]))
            except Exception:
                # fallback to NaN; we will fill with median later
                q_by_path[p] = float("nan")
                computed_rows.append((p, q_by_path[p]))

        if cache_path and computed_rows:
            cp = Path(cache_path)
            cp.parent.mkdir(parents=True, exist_ok=True)
            try:
                import csv
                exists = cp.exists()
                mode = "a" if exists else "w"
                with cp.open(mode, newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["path", "q_log"])
                    if (not exists) or cp.stat().st_size == 0:
                        w.writeheader()
                    for p, q in computed_rows:
                        w.writerow({"path": p, "q_log": q})
            except Exception:
                pass

    q_list = np.array([q_by_path[p] for p in paths], dtype=np.float32)
    # Fill NaNs with median for stable binning
    if np.isnan(q_list).any():
        med = np.nanmedian(q_list)
        q_list = np.where(np.isnan(q_list), med, q_list)

    # Cutpoints (fixed or estimated on this split)
    if cutpoints is not None:
        cuts = [float(x) for x in list(cutpoints)]
    else:
        # Quantile cutpoints
        if n_bins < 2:
            cuts = []
        else:
            qs = [i / n_bins for i in range(1, n_bins)]
            cuts = [float(np.quantile(q_list, q)) for q in qs]

    # Assign bins
    bin_list = []
    for qv in q_list.tolist():
        b = 0
        while b < len(cuts) and qv > cuts[b]:
            b += 1
        bin_list.append(int(b))

    return q_list.tolist(), bin_list, tuple(cuts)


class StratifiedBatchSampler(torch.utils.data.Sampler):
    """Balanced batch sampler over a discrete grouping key.

    Supported keys:
      - key="stratum_id"  -> dataset.stratum_ids (label x quality_bin)
      - key="quality_bin" -> dataset.quality_bin (label-free)

    It produces (approximately) balanced batches across groups by sampling with replacement.
    """

    def __init__(self, dataset, batch_size: int, key: str = "stratum_id", shuffle: bool = True, drop_last: bool = True):
        key = str(key or "stratum_id")
        if key in {"stratum_id", "stratum_ids"}:
            attr = "stratum_ids"
        elif key in {"quality_bin", "quality_bins"}:
            attr = "quality_bin"
        else:
            raise ValueError(f"Unknown stratified key: {key}")

        if not hasattr(dataset, attr) or getattr(dataset, attr) is None:
            raise ValueError(
                f"Dataset must have attribute `{attr}` (enable quality_balance=True, and ensure the key is computed)."
            )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.key = key
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        self.group_ids = list(getattr(dataset, attr))
        self.n = len(self.group_ids)
        self.strata = {}
        for i, gid in enumerate(self.group_ids):
            self.strata.setdefault(int(gid), []).append(i)
        self.stratum_keys = sorted(self.strata.keys())
        if not self.stratum_keys:
            raise RuntimeError("No strata found.")

        # Remove empty strata (shouldn't happen, but just in case)
        self.stratum_keys = [k for k in self.stratum_keys if len(self.strata.get(k, [])) > 0]
        self.n_strata = len(self.stratum_keys)

        # Target per-stratum counts per batch (near-uniform)
        base = self.batch_size // self.n_strata
        rem = self.batch_size % self.n_strata
        self.per_stratum = {k: base for k in self.stratum_keys}
        for k in self.stratum_keys[:rem]:
            self.per_stratum[k] += 1

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        import random
        # make local copies
        pools = {k: list(v) for k, v in self.strata.items() if k in self.per_stratum}
        if self.shuffle:
            for k in pools:
                random.shuffle(pools[k])
        ptr = {k: 0 for k in pools}

        n_batches = len(self)
        for _ in range(n_batches):
            batch = []
            for k in self.stratum_keys:
                need = self.per_stratum[k]
                if need <= 0:
                    continue
                pool = pools[k]
                if not pool:
                    continue
                p = ptr[k]
                # If not enough, reshuffle + wrap (sampling with replacement)
                if p + need > len(pool):
                    if self.shuffle:
                        random.shuffle(pool)
                    p = 0
                batch.extend(pool[p : p + need])
                ptr[k] = p + need

            if len(batch) < self.batch_size:
                # pad from any stratum (rare)
                k0 = self.stratum_keys[0]
                pool = pools[k0]
                if pool:
                    while len(batch) < self.batch_size:
                        batch.append(random.choice(pool))
            if self.shuffle:
                random.shuffle(batch)
            yield batch


class RandomGaussianBlur:
    def __init__(self, prob=0.0, sigma=(0.0, 3.0), kernel_size=3):
        self.prob = float(prob)
        self.sigma = sigma
        self.kernel_size = int(kernel_size)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

    def __call__(self, img):
        if self.prob <= 0 or random.random() >= self.prob:
            return img
        sigma = self._sample_sigma()
        blur = T.GaussianBlur(kernel_size=self.kernel_size, sigma=sigma)
        return blur(img)

    def _sample_sigma(self):
        if isinstance(self.sigma, (list, tuple)):
            if len(self.sigma) == 1:
                return float(self.sigma[0])
            if len(self.sigma) >= 2:
                lo, hi = float(self.sigma[0]), float(self.sigma[1])
                if lo > hi:
                    lo, hi = hi, lo
                return random.random() * (hi - lo) + lo
        return float(self.sigma)


class RandomJPEG:
    def __init__(self, prob=0.0, quality=(30, 100)):
        self.prob = float(prob)
        self.quality = quality

    def _sample_quality(self):
        q = self.quality
        if isinstance(q, (list, tuple)):
            if len(q) == 1:
                return int(q[0])
            if len(q) == 2 and all(isinstance(v, (int, float)) for v in q):
                lo, hi = int(q[0]), int(q[1])
                if lo > hi:
                    lo, hi = hi, lo
                return random.randint(lo, hi)
            return int(random.choice(list(q)))
        return int(q)

    def __call__(self, img):
        if self.prob <= 0 or random.random() >= self.prob:
            return img
        quality = self._sample_quality()
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        out = Image.open(buffer).convert("RGB")
        return out.copy()


def _make_aug_opt(opt, **overrides):
    params = dict(getattr(aug_cfg, "AUG_DEFAULTS", {}))
    if opt is not None:
        for key in list(params.keys()):
            if hasattr(opt, key):
                value = getattr(opt, key)
                # Normalize comma-separated strings when defaults are list-like.
                if isinstance(value, str) and isinstance(params.get(key), (list, tuple)):
                    value = [v.strip() for v in value.split(",") if v.strip()]
                params[key] = value
    for key, value in overrides.items():
        if value is not None:
            params[key] = value
    return SimpleNamespace(**params)


def build_transform(
    image_size=224,
    augment=False,
    hflip_prob=0.0,
    blur_prob=0.0,
    blur_sig=(0.0, 3.0),
    jpg_prob=0.0,
    jpg_qual=(30, 100),
    opt=None,
):
    transforms = []
    aug_before_resize = bool(getattr(opt, "aug_before_resize", False))
    norm_mean, norm_std = _resolve_norm_stats(opt)

    # E0/E1/E2: False -> old behavior (resize then augment)
    # E3/E4: True  -> augment then resize
    if not aug_before_resize:
        transforms.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC))
    if augment:
        if hflip_prob and hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(p=float(hflip_prob)))
        aug_opt = _make_aug_opt(
            opt,
            blur_prob=blur_prob,
            blur_sig=blur_sig,
            jpg_prob=jpg_prob,
            jpg_qual=jpg_qual,
        )
        transforms.append(T.Lambda(lambda img: aug_cfg.data_augment(img, aug_opt)))
    if aug_before_resize:
        transforms.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC))

    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
    return T.Compose(transforms)




def _read_manifest(csv_path):
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "path" not in fields or "label" not in fields:
            raise ValueError("CSV must contain path and label columns")
        for row in reader:
            path = row["path"].strip()
            label = row["label"].strip()
            if not path:
                continue
            records.append((path, int(label)))
    if not records:
        raise RuntimeError("No records found in CSV")
    return records


def _parse_label(value, label_name=None):
    if value is not None and value != "":
        return int(value)
    if label_name:
        v = str(label_name).strip().lower()
        if v == "real":
            return 0
        if v == "fake":
            return 1
    raise ValueError("Cannot parse label")


def _sample_frames(frame_paths, frame_num):
    if frame_num is None:
        return frame_paths
    total = len(frame_paths)
    if total <= frame_num:
        return frame_paths
    step = max(total // frame_num, 1)
    return [frame_paths[i] for i in range(0, total, step)][:frame_num]


def _read_frames_csv(csv_path, split=None, frame_num=None):
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "frames_dir" not in fields:
            raise ValueError("CSV must contain frames_dir column")
        for row in reader:
            row_split = (row.get("split") or "").strip().lower()
            if split and row_split != split:
                continue
            frames_dir = row["frames_dir"]
            if not frames_dir:
                continue
            label = _parse_label(row.get("label"), row.get("label_name"))
            frames_path = Path(frames_dir)
            if frames_path.is_dir():
                frames = sorted(
                    [
                        p
                        for p in frames_path.glob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    ]
                )
            else:
                frames = [frames_path]
            if not frames:
                continue
            frames = _sample_frames(frames, frame_num)
            records.extend([(str(p), label) for p in frames])
    if not records:
        raise RuntimeError("No records found in frames_dir CSV")
    return records


def _read_folder(root_dir):
    root = Path(root_dir)
    records = []
    for split in ["train", "val", "test"]:
        for cls_name, label in [("real", 0), ("fake", 1)]:
            for p in sorted((root / split / cls_name).rglob("*")):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    records.append((str(p), label, split))
    if not records:
        raise RuntimeError(f"No images found under {root}")
    return records



def _infer_label_from_path(path):
    lower = path.lower()
    base = Path(lower).name
    if f"{os.sep}real{os.sep}" in lower or "_real" in base or base.startswith("real") or "true" in base:
        return 0
    if f"{os.sep}fake{os.sep}" in lower or "_fake" in base or base.startswith("fake") or "false" in base:
        return 1
    return None


def _read_mixed_folder(root_dir):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    records = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            label = _infer_label_from_path(str(p))
            if label is None:
                continue
            records.append((str(p), label))
    if not records:
        raise RuntimeError(f"No labeled images found under {root}")
    return records

class ManifestDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_size=224,
        on_error="skip",
        augment=False,
        hflip_prob=0.0,
        blur_prob=0.0,
        blur_sig=(0.0, 3.0),
        jpg_prob=0.0,
        jpg_qual=(30, 100),
        opt=None,
        # Quality-balanced training (optional)
        quality_balance: bool = False,
        quality_bins: int = 3,
        quality_max_side: int = 256,
        quality_cache_path: str = None,
        return_path: bool = False,
    ):
        self.records = _read_manifest(csv_path)
        self.return_path = bool(return_path)
        self.quality_balance = bool(quality_balance)
        self.quality_bins = int(quality_bins)
        self.quality_max_side = int(quality_max_side)
        self.quality_cache_path = quality_cache_path

        # Precompute quality metadata once (static proxy on original image)
        if self.quality_balance:
            fixed_cuts = getattr(opt, "quality_cuts", None) if opt is not None else None
            q_log, q_bin, cuts = _compute_quality_metadata(
                self.records,
                n_bins=self.quality_bins,
                max_side=self.quality_max_side,
                cache_path=self.quality_cache_path,
                cutpoints=fixed_cuts,
            )
            self.q_log = q_log
            self.quality_bin = q_bin
            self.quality_cuts = cuts
            if opt is not None and getattr(opt, "quality_cuts", None) is None and cuts:
                # Train split can populate fixed cutpoints for subsequent splits (val/test).
                opt.quality_cuts = tuple(float(x) for x in cuts)
            # strata: (label, quality_bin)
            self.stratum_ids = [int(y) * self.quality_bins + int(b) for (_, y), b in zip(self.records, self.quality_bin)]
        else:
            self.q_log = None
            self.quality_bin = None
            self.quality_cuts = None
            self.stratum_ids = None

        self.transform = build_transform(
            image_size=image_size,
            augment=augment,
            hflip_prob=hflip_prob,
            blur_prob=blur_prob,
            blur_sig=blur_sig,
            jpg_prob=jpg_prob,
            jpg_qual=jpg_qual,
            opt=opt,
        )
        self.on_error = on_error

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        try:
            img = _load_image(path)
            img = self.transform(img)
            out = {"image": img, "label": torch.tensor(label, dtype=torch.long)}
            if getattr(self, "quality_balance", False):
                out["q_log"] = torch.tensor(float(self.q_log[idx]), dtype=torch.float32)
                out["quality_bin"] = torch.tensor(int(self.quality_bin[idx]), dtype=torch.long)
                out["stratum_id"] = torch.tensor(int(self.stratum_ids[idx]), dtype=torch.long)
            if getattr(self, "return_path", False):
                out["path"] = path
            return out
        except Exception:
            if self.on_error == "skip":
                return None
            raise
class FramesDirCSVDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split=None,
        frame_num=None,
        image_size=224,
        on_error="skip",
        augment=False,
        hflip_prob=0.0,
        blur_prob=0.0,
        blur_sig=(0.0, 3.0),
        jpg_prob=0.0,
        jpg_qual=(30, 100),
        opt=None,
        # Quality-balanced training (optional)
        quality_balance: bool = False,
        quality_bins: int = 3,
        quality_max_side: int = 256,
        quality_cache_path: str = None,
        return_path: bool = False,
    ):
        self.records = _read_frames_csv(csv_path, split=split, frame_num=frame_num)
        self.return_path = bool(return_path)
        self.quality_balance = bool(quality_balance)
        self.quality_bins = int(quality_bins)
        self.quality_max_side = int(quality_max_side)
        self.quality_cache_path = quality_cache_path

        # Precompute quality metadata once (static proxy on original image)
        if self.quality_balance:
            fixed_cuts = getattr(opt, "quality_cuts", None) if opt is not None else None
            q_log, q_bin, cuts = _compute_quality_metadata(
                self.records,
                n_bins=self.quality_bins,
                max_side=self.quality_max_side,
                cache_path=self.quality_cache_path,
                cutpoints=fixed_cuts,
            )
            self.q_log = q_log
            self.quality_bin = q_bin
            self.quality_cuts = cuts
            if opt is not None and getattr(opt, "quality_cuts", None) is None and cuts:
                opt.quality_cuts = tuple(float(x) for x in cuts)
            # strata: (label, quality_bin)
            self.stratum_ids = [int(y) * self.quality_bins + int(b) for (_, y), b in zip(self.records, self.quality_bin)]
        else:
            self.q_log = None
            self.quality_bin = None
            self.quality_cuts = None
            self.stratum_ids = None

        self.transform = build_transform(
            image_size=image_size,
            augment=augment,
            hflip_prob=hflip_prob,
            blur_prob=blur_prob,
            blur_sig=blur_sig,
            jpg_prob=jpg_prob,
            jpg_qual=jpg_qual,
            opt=opt,
        )
        self.on_error = on_error

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        try:
            img = _load_image(path)
            img = self.transform(img)
            out = {"image": img, "label": torch.tensor(label, dtype=torch.long)}
            if getattr(self, "quality_balance", False):
                out["q_log"] = torch.tensor(float(self.q_log[idx]), dtype=torch.float32)
                out["quality_bin"] = torch.tensor(int(self.quality_bin[idx]), dtype=torch.long)
                out["stratum_id"] = torch.tensor(int(self.stratum_ids[idx]), dtype=torch.long)
            if getattr(self, "return_path", False):
                out["path"] = path
            return out
        except Exception:
            if self.on_error == "skip":
                return None
            raise
class MixedFolderDataset(Dataset):
    def __init__(
        self,
        root_dirs,
        image_size=224,
        on_error="skip",
        augment=False,
        hflip_prob=0.0,
        blur_prob=0.0,
        blur_sig=(0.0, 3.0),
        jpg_prob=0.0,
        jpg_qual=(30, 100),
        opt=None,
        # Quality-balanced training (optional)
        quality_balance: bool = False,
        quality_bins: int = 3,
        quality_max_side: int = 256,
        quality_cache_path: str = None,
        return_path: bool = False,
    ):
        if isinstance(root_dirs, (list, tuple)):
            roots = [str(r) for r in root_dirs]
        else:
            roots = [str(root_dirs)] if root_dirs is not None else []
        records = []
        for root in roots:
            if not root:
                continue
            records.extend(_read_mixed_folder(root))
        if not records:
            raise RuntimeError("No labeled images found in mixed folder dataset")
        self.records = records
        self.return_path = bool(return_path)
        self.quality_balance = bool(quality_balance)
        self.quality_bins = int(quality_bins)
        self.quality_max_side = int(quality_max_side)
        self.quality_cache_path = quality_cache_path

        # Precompute quality metadata once (static proxy on original image)
        if self.quality_balance:
            fixed_cuts = getattr(opt, "quality_cuts", None) if opt is not None else None
            q_log, q_bin, cuts = _compute_quality_metadata(
                self.records,
                n_bins=self.quality_bins,
                max_side=self.quality_max_side,
                cache_path=self.quality_cache_path,
                cutpoints=fixed_cuts,
            )
            self.q_log = q_log
            self.quality_bin = q_bin
            self.quality_cuts = cuts
            if opt is not None and getattr(opt, "quality_cuts", None) is None and cuts:
                opt.quality_cuts = tuple(float(x) for x in cuts)
            # strata: (label, quality_bin)
            self.stratum_ids = [int(y) * self.quality_bins + int(b) for (_, y), b in zip(self.records, self.quality_bin)]
        else:
            self.q_log = None
            self.quality_bin = None
            self.quality_cuts = None
            self.stratum_ids = None

        self.transform = build_transform(
            image_size=image_size,
            augment=augment,
            hflip_prob=hflip_prob,
            blur_prob=blur_prob,
            blur_sig=blur_sig,
            jpg_prob=jpg_prob,
            jpg_qual=jpg_qual,
            opt=opt,
        )
        self.on_error = on_error

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
            img = self.transform(img)
            out = {"image": img, "label": torch.tensor(label, dtype=torch.long)}
            if getattr(self, "quality_balance", False):
                out["q_log"] = torch.tensor(float(self.q_log[idx]), dtype=torch.float32)
                out["quality_bin"] = torch.tensor(int(self.quality_bin[idx]), dtype=torch.long)
                out["stratum_id"] = torch.tensor(int(self.stratum_ids[idx]), dtype=torch.long)
            if getattr(self, "return_path", False):
                out["path"] = path
            return out
        except Exception:
            if self.on_error == "skip":
                return None
            raise
class FolderDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        image_size=224,
        on_error="skip",
        augment=False,
        hflip_prob=0.0,
        blur_prob=0.0,
        blur_sig=(0.0, 3.0),
        jpg_prob=0.0,
        jpg_qual=(30, 100),
        opt=None,
        # Quality-balanced training (optional)
        quality_balance: bool = False,
        quality_bins: int = 3,
        quality_max_side: int = 256,
        quality_cache_path: str = None,
        return_path: bool = False,
    ):
        records = _read_folder(root_dir)
        self.records = [(p, label) for p, label, sp in records if sp == split]
        self.return_path = bool(return_path)
        self.quality_balance = bool(quality_balance)
        self.quality_bins = int(quality_bins)
        self.quality_max_side = int(quality_max_side)
        self.quality_cache_path = quality_cache_path

        # Precompute quality metadata once (static proxy on original image)
        if self.quality_balance:
            q_log, q_bin, cuts = _compute_quality_metadata(
                self.records,
                n_bins=self.quality_bins,
                max_side=self.quality_max_side,
                cache_path=self.quality_cache_path,
            )
            self.q_log = q_log
            self.quality_bin = q_bin
            self.quality_cuts = cuts
            # strata: (label, quality_bin)
            self.stratum_ids = [int(y) * self.quality_bins + int(b) for (_, y), b in zip(self.records, self.quality_bin)]
        else:
            self.q_log = None
            self.quality_bin = None
            self.quality_cuts = None
            self.stratum_ids = None

        self.transform = build_transform(
            image_size=image_size,
            augment=augment,
            hflip_prob=hflip_prob,
            blur_prob=blur_prob,
            blur_sig=blur_sig,
            jpg_prob=jpg_prob,
            jpg_qual=jpg_qual,
            opt=opt,
        )
        self.on_error = on_error

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        try:
            img = _load_image(path)
            img = self.transform(img)
            out = {"image": img, "label": torch.tensor(label, dtype=torch.long)}
            if getattr(self, "quality_balance", False):
                out["q_log"] = torch.tensor(float(self.q_log[idx]), dtype=torch.float32)
                out["quality_bin"] = torch.tensor(int(self.quality_bin[idx]), dtype=torch.long)
                out["stratum_id"] = torch.tensor(int(self.stratum_ids[idx]), dtype=torch.long)
            if getattr(self, "return_path", False):
                out["path"] = path
            return out
        except Exception:
            if self.on_error == "skip":
                return None
            raise
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    out = {"image": images, "label": labels}

    # Optional quality-balanced metadata
    if "q_log" in batch[0]:
        out["q_log"] = torch.stack([b["q_log"] for b in batch], dim=0)
        out["quality_bin"] = torch.stack([b["quality_bin"] for b in batch], dim=0)
        out["stratum_id"] = torch.stack([b["stratum_id"] for b in batch], dim=0)

    # Optional paths (kept as Python list)
    if "path" in batch[0]:
        out["path"] = [b["path"] for b in batch]

    return out


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--frames_dir_csv", action="store_true")
    parser.add_argument("--folder_root", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--on_error", type=str, default="raise", choices=["skip", "raise"])
    parser.add_argument("--max_items", type=int, default=50)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--quality_balance", action="store_true")
    parser.add_argument("--quality_bins", type=int, default=3)
    parser.add_argument("--quality_max_side", type=int, default=256)
    parser.add_argument("--quality_cache_path", type=str, default=None)
    parser.add_argument("--return_path", action="store_true")
    parser.add_argument("--hflip_prob", type=float, default=0.0)
    parser.add_argument("--blur_prob", type=float, default=0.0)
    parser.add_argument("--blur_sig", type=float, nargs="+", default=[0.0, 3.0])
    parser.add_argument("--jpg_prob", type=float, default=0.0)
    parser.add_argument("--jpg_qual", type=int, nargs="+", default=[30, 100])
    args = parser.parse_args()

    if args.csv:
        if args.frames_dir_csv:
            dataset = FramesDirCSVDataset(
                args.csv,
                split=args.split,
                frame_num=args.frame_num,
                image_size=args.image_size,
                on_error=args.on_error,
                augment=args.augment,
                hflip_prob=args.hflip_prob,
                blur_prob=args.blur_prob,
                blur_sig=args.blur_sig,
                jpg_prob=args.jpg_prob,
                jpg_qual=args.jpg_qual,
                quality_balance=args.quality_balance,
                quality_bins=args.quality_bins,
                quality_max_side=args.quality_max_side,
                quality_cache_path=args.quality_cache_path,
                return_path=args.return_path,
            )
        else:
            dataset = ManifestDataset(
                args.csv,
                image_size=args.image_size,
                on_error=args.on_error,
                augment=args.augment,
                hflip_prob=args.hflip_prob,
                blur_prob=args.blur_prob,
                blur_sig=args.blur_sig,
                jpg_prob=args.jpg_prob,
                jpg_qual=args.jpg_qual,
                quality_balance=args.quality_balance,
                quality_bins=args.quality_bins,
                quality_max_side=args.quality_max_side,
                quality_cache_path=args.quality_cache_path,
                return_path=args.return_path,
            )
    elif args.folder_root:
        dataset = FolderDataset(
            args.folder_root,
            split=args.split,
            image_size=args.image_size,
            on_error=args.on_error,
            augment=args.augment,
            hflip_prob=args.hflip_prob,
            blur_prob=args.blur_prob,
            blur_sig=args.blur_sig,
            jpg_prob=args.jpg_prob,
            jpg_qual=args.jpg_qual,
                quality_balance=args.quality_balance,
                quality_bins=args.quality_bins,
                quality_max_side=args.quality_max_side,
                quality_cache_path=args.quality_cache_path,
                return_path=args.return_path,
        )
    else:
        raise ValueError("Provide --csv or --folder_root")

    total = len(dataset)
    ok = 0
    failed = 0
    for i in range(min(args.max_items, total)):
        sample = dataset[i]
        if sample is None:
            print(f"idx {i} -> None")
            failed += 1
        else:
            ok += 1
    print(f"inspect: ok={ok} failed={failed} total_checked={min(args.max_items, total)} total_len={total}")

    collate_fn = collate_skip_none if args.on_error == "skip" else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    for i, batch in enumerate(loader):
        print(f"batch {i}:", None if batch is None else {"image": tuple(batch["image"].shape), "label": tuple(batch["label"].shape)})
        # if i >= 3:
        #     break
