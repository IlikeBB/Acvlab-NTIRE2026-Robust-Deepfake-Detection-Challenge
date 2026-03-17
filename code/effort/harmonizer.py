from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def to_clip_norm(x01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CLIP_MEAN, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


def highpass_3x3(x01: torch.Tensor) -> torch.Tensor:
    low = F.avg_pool2d(x01, kernel_size=3, stride=1, padding=1)
    return x01 - low


def lowpass_5x5(x01: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x01, kernel_size=5, stride=1, padding=2)


def compute_q_log_batch(x01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    g = (0.2989 * x01[:, 0] + 0.5870 * x01[:, 1] + 0.1140 * x01[:, 2]).float()
    lap = (
        -4.0 * g
        + torch.roll(g, shifts=1, dims=1)
        + torch.roll(g, shifts=-1, dims=1)
        + torch.roll(g, shifts=1, dims=2)
        + torch.roll(g, shifts=-1, dims=2)
    )
    var = lap.flatten(1).var(dim=1, unbiased=False)
    return torch.log(var + float(eps))


def compute_noise_std_batch(x01: torch.Tensor) -> torch.Tensor:
    g = (0.2989 * x01[:, 0] + 0.5870 * x01[:, 1] + 0.1140 * x01[:, 2]).float()
    low = F.avg_pool2d(g.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    return (g - low).flatten(1).std(dim=1, unbiased=False)


def ks_distance_np(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.sort(np.asarray(a, dtype=np.float64))
    bb = np.sort(np.asarray(b, dtype=np.float64))
    n = int(aa.size)
    m = int(bb.size)
    if n <= 0 or m <= 0:
        return 0.0
    i = 0
    j = 0
    d = 0.0
    while i < n and j < m:
        if aa[i] <= bb[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / float(n) - j / float(m)))
    d = max(d, abs(1.0 - j / float(m)), abs(i / float(n) - 1.0))
    return float(d)


def read_manifest_rows(path_csv: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(path_csv).open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p = str(row.get("path", "")).strip()
            if not p:
                continue
            if not Path(p).exists():
                continue
            rows.append(
                {
                    "path": p,
                    "label": str(row.get("label", "")).strip(),
                    "source": str(row.get("source", "")).strip(),
                }
            )
    return rows


def sample_manifest_rows(
    rows: Sequence[Dict[str, str]],
    max_n: int = 0,
    seed: int = 2048,
    source_includes: Optional[str] = None,
) -> List[Dict[str, str]]:
    out = list(rows)
    if source_includes:
        key = str(source_includes).lower()
        out = [x for x in out if key in str(x.get("source", "")).lower()]
    if int(max_n or 0) > 0 and len(out) > int(max_n):
        rng = random.Random(int(seed))
        out = rng.sample(out, int(max_n))
    return out


class ImagePathDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, str]], image_size: int = 256):
        self.rows = list(rows)
        self.transform = T.Compose(
            [
                T.Resize((int(image_size), int(image_size))),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = str(row["path"])
        try:
            img = _load_rgb(path)
            return {
                "image": self.transform(img),
                "path": path,
                "label": int(row["label"]) if str(row.get("label", "")).strip() != "" else -1,
                "source": str(row.get("source", "")),
            }
        except Exception:
            return None


def _collate(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return {
        "image": torch.stack([x["image"] for x in batch], dim=0),
        "path": [x["path"] for x in batch],
        "label": torch.tensor([int(x["label"]) for x in batch], dtype=torch.long),
        "source": [x["source"] for x in batch],
    }


def build_loader(
    rows: Sequence[Dict[str, str]],
    batch_size: int,
    image_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    ds = ImagePathDataset(rows=rows, image_size=int(image_size))
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=bool(drop_last),
        num_workers=int(num_workers),
        collate_fn=_collate,
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        c = int(channels)
        g = max(1, min(8, c // 16))
        self.block = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GroupNorm(g, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GroupNorm(g, c),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class HarmonizerNet(nn.Module):
    def __init__(self, channels: int = 64, n_blocks: int = 6, max_delta: float = 0.25):
        super().__init__()
        c = int(max(16, channels))
        self.max_delta = float(max(0.0, min(1.0, max_delta)))
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(c) for _ in range(int(max(1, n_blocks)))])
        self.head = nn.Conv2d(c, 3, kernel_size=3, padding=1)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        h = self.stem(x01)
        h = self.body(h)
        delta = torch.tanh(self.head(h)) * float(self.max_delta)
        return (x01 + delta).clamp(0.0, 1.0)


class PatchDiscriminator(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        c = int(max(16, channels))

        def blk(cin: int, cout: int, stride: int = 2):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=4, stride=stride, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            blk(3, c, stride=2),
            blk(c, c * 2, stride=2),
            blk(c * 2, c * 4, stride=2),
            nn.Conv2d(c * 4, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

