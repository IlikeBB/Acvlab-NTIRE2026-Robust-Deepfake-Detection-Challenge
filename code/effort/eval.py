import argparse
import csv
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import torch
from torch.utils.data import DataLoader
from PIL import Image
from torch.amp import autocast
from tqdm import tqdm

from effort.model import EffortCLIP
from effort.data import ManifestDataset, FramesDirCSVDataset, FolderDataset, collate_skip_none, build_transform
from effort.metrics import compute_metrics
from effort.utils import load_ckpt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="CSV with path,label")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory with images for inference")
    parser.add_argument("--out_file", type=str, default=None, help="Write one prob per line")
    parser.add_argument("--submission_ref", type=str, default=None, help="Reference file to match output order/length")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--frame_num", type=int, default=1)
    parser.add_argument("--folder_root", type=str, default=None, help="Folder root with train/val/test")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clip_model", type=str, default=None)
    parser.add_argument("--svd_rank_cap", type=int, default=None)

    # Patch pooling (single-view; add-on on top of CLS head). Must match training if enabled.
    parser.add_argument("--patch_pool_tau", type=float, default=1.5)
    parser.add_argument("--patch_pool_mode", type=str, default="lse", choices=["lse", "trimmed"])
    parser.add_argument("--patch_trim_p", type=float, default=0.2)
    parser.add_argument("--patch_quality", type=str, default="none", choices=["none", "cos", "cos_norm"])
    parser.add_argument("--patch_pool_gamma", type=float, default=0.0, help="Patch delta scale (gamma) for inference")

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--on_error", type=str, default="raise", choices=["skip", "raise"])
    return parser.parse_args()


def _csv_has_frames_dir(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
    return "frames_dir" in fields


def build_loader(args):
    if args.image_dir:
        ds = ImageDirDataset(args.image_dir, image_size=args.image_size)
    elif args.csv:
        if _csv_has_frames_dir(args.csv):
            ds = FramesDirCSVDataset(
                args.csv, split=args.split, frame_num=args.frame_num, image_size=args.image_size, on_error="skip"
            )
        else:
            ds = ManifestDataset(args.csv, image_size=args.image_size, on_error="skip")
    elif args.folder_root:
        ds = FolderDataset(args.folder_root, split=args.split, image_size=args.image_size, on_error="skip")
    else:
        raise ValueError("Provide --image_dir or --csv or --folder_root")

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_skip_none if args.on_error == "skip" else None,
        persistent_workers=args.num_workers > 0,
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EffortCLIP(
        clip_model_or_path=args.clip_model,
        svd_rank_cap=args.svd_rank_cap,
        patch_pool=True,
        patch_pool_tau=float(args.patch_pool_tau),
        patch_pool_mode=str(args.patch_pool_mode),
        patch_trim_p=float(args.patch_trim_p),
        patch_quality=str(args.patch_quality),
        )
    if hasattr(model, "set_patch_scale"):
        model.set_patch_scale(float(getattr(args, "patch_pool_gamma", 0.0)))

    missing, unexpected = load_ckpt(args.ckpt, model, strict=args.strict)
    if missing:
        print(f"missing keys: {missing}")
    if unexpected:
        print(f"unexpected keys: {unexpected}")

    model.to(device)
    model.eval()

    loader = build_loader(args)
    all_probs = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", ncols=80):
            if batch is None:
                continue
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            with autocast(device_type=device.type, enabled=args.amp):
                pred = model(batch)
            probs = pred["prob"].detach().cpu().numpy()
            labels = batch.get("label")
            labels = labels.detach().cpu().numpy() if labels is not None else None
            all_probs.extend(probs.tolist())
            if labels is not None:
                all_labels.extend(labels.tolist())
            if "path" in batch:
                all_paths.extend(batch["path"])

    if args.image_dir:
        if args.submission_ref:
            with open(args.submission_ref, "r") as f:
                ref_lines = [line for line in f.readlines() if line.strip() != ""]
            if len(ref_lines) != len(all_probs):
                raise RuntimeError(
                    f"submission_ref lines ({len(ref_lines)}) != predictions ({len(all_probs)})"
                )
        if args.out_file:
            with open(args.out_file, "w") as f:
                f.write("".join(f"{prob:.6f}\n" for prob in all_probs))
            print(f"Wrote {len(all_probs)} lines to {args.out_file}")
        else:
            for path, prob in zip(all_paths, all_probs):
                print(f"{path},{prob:.6f}")
    else:
        metrics = compute_metrics(all_labels, all_probs)
        print(f"acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} eer={metrics['eer']:.4f} ap={metrics['ap']:.4f}")


class ImageDirDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, image_size=224):
        self.root = Path(image_dir)
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.paths = [
            p
            for p in sorted(self.root.glob("*"))
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        self.paths.sort(key=self._sort_key)
        if not self.paths:
            raise RuntimeError(f"No images found under {self.root}")
        self.transform = build_transform(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        img = self.transform(img)
        return {"image": img, "path": str(path)}

    @staticmethod
    def _sort_key(path):
        stem = path.stem
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem)


if __name__ == "__main__":
    main()
