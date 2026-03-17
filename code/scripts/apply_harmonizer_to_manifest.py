#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm

from effort.harmonizer import HarmonizerNet, build_loader, read_manifest_rows, sample_manifest_rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--generator_ckpt",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/harmonizer/Exp_HarmU1_s2048/best_generator.pt",
    )
    p.add_argument(
        "--input_manifest",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/manifests/ext_df40_only_manifest.csv",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/harmonizer/Exp_HarmU1_s2048/harmonized_df40",
    )
    p.add_argument(
        "--output_manifest",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/manifests/ext_df40_harmonized_manifest.csv",
    )
    p.add_argument("--source_includes", type=str, default="df40")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=2048)
    p.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--merge_with_manifest", type=str, default=None)
    p.add_argument("--merge_out_manifest", type=str, default=None)
    p.add_argument("--merge_harm_max", type=int, default=0)
    return p.parse_args()


def _pick_device(arg: str) -> torch.device:
    key = str(arg or "auto").lower()
    if key == "auto":
        key = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(key)


def _to_pil(x: torch.Tensor) -> Image.Image:
    arr = x.detach().cpu().clamp(0.0, 1.0)
    arr = (arr * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


def _load_generator(path_ckpt: str, device: torch.device):
    ckpt_path = Path(path_ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"generator checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        cfg = ckpt.get("gen_cfg", {})
    else:
        state = ckpt
        cfg = {}
    gen = HarmonizerNet(
        channels=int(cfg.get("channels", 64)),
        n_blocks=int(cfg.get("n_blocks", 6)),
        max_delta=float(cfg.get("max_delta", 0.22)),
    ).to(device)
    miss, unexp = gen.load_state_dict(state, strict=False)
    gen.eval()
    print(f"[load] generator={ckpt_path} missing={len(miss)} unexpected={len(unexp)}")
    return gen


def _write_manifest(path_csv: Path, rows: List[Dict[str, str]]) -> None:
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    with path_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source"])
        for r in rows:
            w.writerow([r["path"], int(r["label"]), r["source"]])


def main():
    args = parse_args()
    random.seed(int(args.seed))

    device = _pick_device(args.device)
    gen = _load_generator(args.generator_ckpt, device=device)
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    in_rows = sample_manifest_rows(
        read_manifest_rows(args.input_manifest),
        max_n=int(args.max_samples),
        seed=int(args.seed),
        source_includes=str(args.source_includes or ""),
    )
    if not in_rows:
        raise RuntimeError("No rows selected from input manifest after filtering.")

    loader = build_loader(
        rows=in_rows,
        batch_size=int(args.batch_size),
        image_size=int(args.image_size),
        num_workers=int(args.num_workers),
        shuffle=False,
        drop_last=False,
    )

    out_rows: List[Dict[str, str]] = []
    idx = 0
    for batch in tqdm(loader, desc="harmonize", ncols=100):
        if batch is None:
            continue
        x = batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            y = gen(x)
        for i in range(int(y.shape[0])):
            src_path = batch["path"][i]
            src_name = Path(src_path).stem
            label = int(batch["label"][i].item())
            source = str(batch["source"][i]) if i < len(batch["source"]) else "harmonized"
            out_name = f"{idx:07d}_{src_name}.png"
            out_path = out_root / out_name
            _to_pil(y[i]).save(out_path)
            out_rows.append(
                {
                    "path": str(out_path.resolve()),
                    "label": int(label),
                    "source": f"harm_{source}",
                }
            )
            idx += 1

    out_manifest = Path(args.output_manifest).resolve()
    _write_manifest(out_manifest, out_rows)
    print(f"[done] harmonized rows={len(out_rows)} manifest={out_manifest}")

    if args.merge_with_manifest and args.merge_out_manifest:
        base_rows = read_manifest_rows(args.merge_with_manifest)
        harm_rows = list(out_rows)
        if int(args.merge_harm_max or 0) > 0 and len(harm_rows) > int(args.merge_harm_max):
            rng = random.Random(int(args.seed))
            harm_rows = rng.sample(harm_rows, int(args.merge_harm_max))
        merged = list(base_rows) + harm_rows
        out_merge = Path(args.merge_out_manifest).resolve()
        _write_manifest(out_merge, merged)
        print(
            f"[done] merged manifest={out_merge} total={len(merged)} base={len(base_rows)} harm={len(harm_rows)}"
        )


if __name__ == "__main__":
    main()

