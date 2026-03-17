#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _collect_rows(roots, max_per_class: int = 0):
    rows = []
    skipped = 0
    cap = int(max_per_class or 0)
    c0 = 0
    c1 = 0
    for raw_root in roots:
        root = Path(raw_root).resolve()
        if not root.exists():
            continue
        for label_name, label in (("real", 0), ("fake", 1)):
            cls_dir = root / label_name
            if not cls_dir.exists():
                continue
            for p in _iter_images(cls_dir):
                if cap > 0:
                    if label == 0 and c0 >= cap:
                        break
                    if label == 1 and c1 >= cap:
                        break
                rows.append((str(p.resolve()), int(label), root.name))
                if label == 0:
                    c0 += 1
                else:
                    c1 += 1
            if cap > 0 and c0 >= cap and c1 >= cap:
                break
        # Fallback for datasets not strictly split under /real and /fake
        if (root / "real").exists() or (root / "fake").exists():
            if cap > 0 and c0 >= cap and c1 >= cap:
                break
            continue
        for p in _iter_images(root):
            s = str(p).lower()
            if "/real/" in s:
                label = 0
            elif "/fake/" in s:
                label = 1
            else:
                skipped += 1
                continue
            if cap > 0:
                if label == 0 and c0 >= cap:
                    continue
                if label == 1 and c1 >= cap:
                    continue
            rows.append((str(p.resolve()), int(label), root.name))
            if label == 0:
                c0 += 1
            else:
                c1 += 1
            if cap > 0 and c0 >= cap and c1 >= cap:
                break
        if cap > 0 and c0 >= cap and c1 >= cap:
            break
    return rows, skipped


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--roots",
        type=str,
        nargs="+",
        default=[
            "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/DF40_frames/train/DF40",
            "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/nitra_frames_face/train/Celeb-DF",
            "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/nitra_frames_face/train/DeeperForensics-1.0",
        ],
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/manifests/ext_dfd_train_manifest.csv",
    )
    p.add_argument(
        "--max_per_class",
        type=int,
        default=0,
        help="0 means no cap. If >0, cap total samples per class.",
    )
    p.add_argument("--seed", type=int, default=2048)
    return p.parse_args()


def main():
    args = parse_args()
    rows, skipped = _collect_rows(args.roots, max_per_class=int(args.max_per_class or 0))
    if not rows:
        raise RuntimeError("No labeled images found from --roots")

    rng = random.Random(int(args.seed))
    rng.shuffle(rows)

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source"])
        for path, label, source in rows:
            w.writerow([path, label, source])

    n_real = sum(1 for _, y, _ in rows if y == 0)
    n_fake = sum(1 for _, y, _ in rows if y == 1)
    print(f"[done] manifest -> {out_csv}")
    print(f"[done] samples: total={len(rows)} real={n_real} fake={n_fake} skipped_unlabeled={skipped}")


if __name__ == "__main__":
    main()
