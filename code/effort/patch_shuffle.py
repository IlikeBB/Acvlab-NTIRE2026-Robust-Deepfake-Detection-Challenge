from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# CLIP normalization constants (same as effort.data)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def patch_shuffle(x: torch.Tensor, patch_size: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Shuffle non-overlapping patches (D3-style) on NCHW tensor.

    Note: permutation is shared across the batch (fast). For stronger randomness you can
    implement per-sample perms, but this is usually sufficient.
    """
    b, c, h, w = x.shape
    ps = int(patch_size)
    if ps <= 0 or (h % ps != 0) or (w % ps != 0):
        # fallback: do nothing
        return x
    patches = F.unfold(x, kernel_size=ps, stride=ps)  # (B, C*ps*ps, N)
    n = patches.shape[-1]
    perm = torch.randperm(n, device=x.device, generator=generator)
    patches = patches[:, :, perm]
    out = F.fold(patches, output_size=(h, w), kernel_size=ps, stride=ps)
    return out
