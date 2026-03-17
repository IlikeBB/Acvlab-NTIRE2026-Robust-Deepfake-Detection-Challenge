#!/usr/bin/env python3
import argparse
import csv
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path


def _as_json_value(v):
    if isinstance(v, (bool, dict, list)) or v is None:
        return json.dumps(v)
    return str(v)


def _infer_label_from_name(path_str: str):
    s = str(path_str).lower()
    name = Path(s).name
    if "_real." in name or name.startswith("real") or "/real/" in s:
        return 0
    if "_fake." in name or name.startswith("fake") or "/fake/" in s:
        return 1
    return None


def _build_comp_manifest(*, comp_root: str, out_csv: str, train_list: str = None):
    root = Path(comp_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"competition train root not found: {root}")

    names = []
    if train_list:
        lp = Path(train_list).resolve()
        if not lp.exists():
            raise FileNotFoundError(f"competition train list not found: {lp}")
        for raw in lp.read_text(encoding="utf-8").splitlines():
            item = raw.strip()
            if item:
                names.append(item)
    else:
        for p in sorted(root.glob("*")):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                names.append(p.name)

    rows = []
    for name in names:
        p = (root / name).resolve()
        if not p.exists():
            continue
        y = _infer_label_from_name(str(p))
        if y is None:
            continue
        rows.append((str(p), int(y), "competition"))
    if not rows:
        raise RuntimeError("No labeled competition rows found to build comp manifest")

    out = Path(out_csv).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source"])
        w.writerows(rows)
    n0 = sum(1 for _, y, _ in rows if y == 0)
    n1 = sum(1 for _, y, _ in rows if y == 1)
    print(f"[prep] competition manifest -> {out} total={len(rows)} real={n0} fake={n1}")
    return out


def _read_manifest_rows(path_csv: str):
    rows = []
    with Path(path_csv).open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "path" not in (r.fieldnames or []) or "label" not in (r.fieldnames or []):
            raise ValueError(f"manifest missing required columns path,label: {path_csv}")
        for row in r:
            p = str(row["path"]).strip()
            if not p:
                continue
            y = int(row["label"])
            src = str(row.get("source", "")).strip()
            rows.append((p, y, src))
    return rows


def _write_manifest_rows(path_csv: str, rows):
    out = Path(path_csv).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source"])
        for p, y, src in rows:
            w.writerow([p, int(y), src])
    return out


def _build_stagec_mix_manifest(
    *,
    comp_manifest_csv: str,
    ext_manifest_csv: str,
    out_csv: str,
    comp_ratio: float,
    seed: int,
):
    comp_ratio = float(comp_ratio)
    if comp_ratio <= 0.0 or comp_ratio > 1.0:
        raise ValueError(f"comp_ratio must be in (0,1], got {comp_ratio}")

    comp_rows_raw = _read_manifest_rows(comp_manifest_csv)
    ext_rows_raw = _read_manifest_rows(ext_manifest_csv)
    comp_rows = [(p, y, "competition") for (p, y, _) in comp_rows_raw]
    ext_rows = [(p, y, (s if s else "external")) for (p, y, s) in ext_rows_raw]
    if not comp_rows:
        raise RuntimeError("competition manifest is empty")
    if comp_ratio >= 0.999 or (not ext_rows):
        out = _write_manifest_rows(out_csv, comp_rows)
        print(f"[prep] stage C manifest (comp only) -> {out} total={len(comp_rows)}")
        return out

    rng = random.Random(int(seed))
    ext_need = int(round(len(comp_rows) * (1.0 - comp_ratio) / max(comp_ratio, 1e-6)))
    ext_need = max(1, ext_need)
    if ext_need <= len(ext_rows):
        sampled_ext = rng.sample(ext_rows, ext_need)
    else:
        sampled_ext = list(ext_rows)
        while len(sampled_ext) < ext_need:
            sampled_ext.append(rng.choice(ext_rows))

    rows = list(comp_rows) + sampled_ext
    rng.shuffle(rows)
    out = _write_manifest_rows(out_csv, rows)
    n_comp = sum(1 for _, _, s in rows if s == "competition")
    n_ext = len(rows) - n_comp
    print(
        f"[prep] stage C mixed manifest -> {out} total={len(rows)} comp={n_comp} ext={n_ext} comp_ratio_target={comp_ratio:.2f}"
    )
    return out


def _laplacian_var_log_from_path(path: str, max_side: int = 256, eps: float = 1e-6) -> float:
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("L")
        if int(max_side or 0) > 0:
            w, h = img.size
            m = max(w, h)
            if m > int(max_side):
                s = float(max_side) / float(m)
                nw = max(1, int(round(w * s)))
                nh = max(1, int(round(h * s)))
                img = img.resize((nw, nh), resample=Image.BILINEAR)
        g = np.asarray(img, dtype=np.float32)

    lap = (
        -4.0 * g
        + np.roll(g, 1, axis=0)
        + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1)
        + np.roll(g, -1, axis=1)
    )
    v = float(np.var(lap))
    return float(np.log(v + float(eps)))


def _build_comp_split_manifests(
    *,
    comp_manifest_csv: str,
    out_train_csv: str,
    out_val_csv: str,
    val_ratio: float,
    seed: int,
    quality_bins: int = 3,
    quality_max_side: int = 256,
):
    import numpy as np

    rows = _read_manifest_rows(comp_manifest_csv)
    if not rows:
        raise RuntimeError(f"Empty comp manifest: {comp_manifest_csv}")
    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    q = []
    for p, _, _ in rows:
        try:
            q.append(_laplacian_var_log_from_path(p, max_side=int(quality_max_side)))
        except Exception:
            q.append(float("nan"))
    q = np.asarray(q, dtype=np.float32)
    if np.isnan(q).any():
        med = float(np.nanmedian(q)) if np.isfinite(np.nanmedian(q)) else 0.0
        q = np.where(np.isnan(q), med, q)

    n_bins = max(1, int(quality_bins))
    if n_bins <= 1:
        cuts = []
    else:
        cuts = [float(np.quantile(q, i / n_bins)) for i in range(1, n_bins)]

    qbin = []
    for x in q.tolist():
        b = 0
        while b < len(cuts) and x > cuts[b]:
            b += 1
        qbin.append(int(b))

    buckets = defaultdict(list)
    for i, ((p, y, src), b) in enumerate(zip(rows, qbin)):
        buckets[(int(y), int(b))].append((i, p, int(y), src))

    rng = random.Random(int(seed))
    train_rows = []
    val_rows = []
    for key, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        if n <= 1:
            n_val = 0
        else:
            n_val = int(round(n * float(val_ratio)))
            n_val = max(1, min(n - 1, n_val))
        val_part = items[:n_val]
        train_part = items[n_val:]
        val_rows.extend([(p, y, src) for _, p, y, src in val_part])
        train_rows.extend([(p, y, src) for _, p, y, src in train_part])

    if not train_rows or not val_rows:
        raise RuntimeError(
            f"Invalid split produced empty set: train={len(train_rows)} val={len(val_rows)} from {len(rows)}"
        )

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    out_train = _write_manifest_rows(out_train_csv, train_rows)
    out_val = _write_manifest_rows(out_val_csv, val_rows)

    t0 = sum(1 for _, y, _ in train_rows if int(y) == 0)
    t1 = len(train_rows) - t0
    v0 = sum(1 for _, y, _ in val_rows if int(y) == 0)
    v1 = len(val_rows) - v0
    print(
        "[prep] comp split -> train=%s (n=%d real=%d fake=%d) | val=%s (n=%d real=%d fake=%d)"
        % (out_train, len(train_rows), t0, t1, out_val, len(val_rows), v0, v1)
    )
    return out_train, out_val, tuple(cuts)


def _stagea_comp_like_aug_overrides(strength: str = "strong"):
    s = str(strength or "strong").lower()
    if s == "mild":
        return {
            "aug_before_resize": True,
            "max_aug_per_image": 2,
            "blur_prob": 0.25,
            "blur_sig": [0.2, 1.8],
            "motion_blur_prob": 0.15,
            "motion_blur_ksize": [7, 13],
            "motion_blur_angle": [0, 180],
            "jpg_prob": 0.35,
            "jpg_method": ["cv2", "pil"],
            "jpg_qual": list(range(15, 70)),
            "lowres_prob": 0.30,
            "lowres_scale": [0.25, 0.65],
            "lowres_interp": ["nearest", "box", "bilinear"],
            "noise_prob": 0.22,
            "noise_std": [3.0, 18.0],
            "iso_noise_prob": 0.22,
            "iso_shot": [2.0, 9.0],
            "iso_gauss_std": [5.0, 20.0],
            "sp_prob": 0.04,
            "sp_amount": 0.002,
            "grainy_group_prob": 0.16,
            "mono_group_prob": 0.10,
            "motion_grainy_group_prob": 0.10,
            "lowlight_group_prob": 0.08,
            "hflip_prob": 0.5,
            "use_pmm_aug": False,
            "pmm_group_prob": 0.0,
        }
    return {
        "aug_before_resize": True,
        "max_aug_per_image": 2,
        "blur_prob": 0.35,
        "blur_sig": [0.2, 2.8],
        "motion_blur_prob": 0.25,
        "motion_blur_ksize": [7, 19],
        "motion_blur_angle": [0, 180],
        "jpg_prob": 0.45,
        "jpg_method": ["cv2", "pil"],
        "jpg_qual": list(range(8, 66)),
        "lowres_prob": 0.45,
        "lowres_scale": [0.14, 0.55],
        "lowres_interp": ["nearest", "box", "bilinear"],
        "noise_prob": 0.30,
        "noise_std": [4.0, 22.0],
        "iso_noise_prob": 0.35,
        "iso_shot": [1.5, 10.0],
        "iso_gauss_std": [6.0, 30.0],
        "sp_prob": 0.07,
        "sp_amount": 0.004,
        "grainy_group_prob": 0.24,
        "mono_group_prob": 0.16,
        "motion_grainy_group_prob": 0.16,
        "lowlight_group_prob": 0.12,
        "hflip_prob": 0.5,
        "use_pmm_aug": False,
        "pmm_group_prob": 0.0,
    }


def _run_stage(
    *,
    python_bin: str,
    run_script: str,
    repo: str,
    base_config: str,
    checkpoints_dir: str,
    name: str,
    seed: int,
    niter: int,
    batch_sizes,
    num_threads: int,
    train_frame_num: int,
    val_frame_num: int,
    multi_expert_k: int,
    multi_expert_route: str,
    resume_path: str = None,
    load_whole_model: bool = False,
    extra_overrides=None,
):
    extra_overrides = dict(extra_overrides or {})
    last_err = None
    tried = []
    for bs in list(batch_sizes):
        bs = int(bs)
        if bs <= 0:
            continue
        cmd = [
            str(python_bin),
            str(run_script),
            "--repo",
            str(repo),
            "--base_config",
            str(base_config),
            "--name",
            str(name),
            "--checkpoints_dir",
            str(checkpoints_dir),
            "--seed",
            str(int(seed)),
            "--niter",
            str(int(niter)),
            "--batch_size",
            str(int(bs)),
            "--num_threads",
            str(int(num_threads)),
            "--train_frame_num",
            str(int(train_frame_num)),
            "--val_frame_num",
            str(int(val_frame_num)),
            "--multi_expert_k",
            str(int(multi_expert_k)),
            "--multi_expert_route",
            str(multi_expert_route),
        ]
        if resume_path:
            cmd += ["--resume_path", str(resume_path)]
        if bool(load_whole_model):
            cmd += ["--load_whole_model"]
        for k, v in extra_overrides.items():
            cmd += ["--set", f"{k}={_as_json_value(v)}"]

        print(f"[stage:{name}] try batch_size={bs}")
        tried.append(bs)
        try:
            subprocess.run(cmd, check=True)
            return bs
        except subprocess.CalledProcessError as e:
            last_err = e
            print(f"[stage:{name}] failed with batch_size={bs}, trying smaller fallback...")

    raise RuntimeError(f"Stage {name} failed for batch sizes {tried}") from last_err


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--python_bin", type=str, default="/home/u1657859/miniconda3/envs/eamamba/bin/python")
    p.add_argument("--repo", type=str, default="/work/u1657859/Jess/app")
    p.add_argument("--run_script", type=str, default="/work/u1657859/Jess/app/scripts/run_multiexpert_local.py")
    p.add_argument(
        "--base_config",
        type=str,
        default="/work/u1657859/Jess/app/checkpoints/Exp_E1_DF40_tuneoff-AUXLoss_s2048/config.json",
    )
    p.add_argument("--checkpoints_dir", type=str, default="/work/u1657859/Jess/app/local_checkpoints")
    p.add_argument("--exp_prefix", type=str, default="Exp_MEv2_extcons_s2048")
    p.add_argument("--seed", type=int, default=2048)
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[32, 28, 24])
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--train_frame_num", type=int, default=8)
    p.add_argument("--val_frame_num", type=int, default=1)
    p.add_argument("--multi_expert_k", type=int, default=3)
    p.add_argument(
        "--multi_expert_route",
        type=str,
        default="quality_bin",
        choices=[
            "quality_bin",
            "quality_soft",
            "uniform",
            "router",
            "hybrid",
            "degrade_router",
            "degrade_hybrid",
        ],
    )
    p.add_argument("--stageA_niter", type=int, default=2)
    p.add_argument("--stageB_niter", type=int, default=3)
    p.add_argument("--stageC_niter", type=int, default=1)
    p.add_argument(
        "--external_roots",
        type=str,
        nargs="+",
        default=[
            "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/DF40_frames/train/DF40",
            "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/nitra_frames_face/train/Celeb-DF",
            "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/nitra_frames_face/train/DeeperForensics-1.0",
        ],
    )
    p.add_argument(
        "--external_manifest_csv",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/manifests/ext_dfd_train_manifest.csv",
    )
    p.add_argument("--skip_manifest_build", action="store_true", default=False)
    p.add_argument("--external_max_per_class", type=int, default=0)
    p.add_argument("--ext_quality_cache_path", type=str, default=None)
    p.add_argument("--comp_quality_cache_path", type=str, default=None)
    p.add_argument("--comp_train_csv", type=str, default=None)
    p.add_argument(
        "--comp_train_root",
        type=str,
        default="/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/training_data_final",
    )
    p.add_argument("--comp_train_list", type=str, default=None)
    p.add_argument(
        "--comp_manifest_csv",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/manifests/competition_train_manifest.csv",
    )
    p.add_argument("--skip_comp_manifest_build", action="store_true", default=False)
    p.add_argument(
        "--comp_val_ratio",
        type=float,
        default=0.2,
        help="Validation ratio used to build non-overlap competition train/val split.",
    )
    p.add_argument("--comp_split_seed", type=int, default=2048)
    p.add_argument("--skip_comp_split_build", action="store_true", default=False)
    p.add_argument("--comp_val_data_root", type=str, default=None)
    p.add_argument(
        "--stagec_comp_ratio",
        type=float,
        default=1.0,
        help="Stage C training mix ratio for competition samples. 1.0 means comp-only.",
    )
    p.add_argument("--stagea_comp_like_aug", action="store_true", default=True)
    p.add_argument("--no_stagea_comp_like_aug", dest="stagea_comp_like_aug", action="store_false")
    p.add_argument("--stagea_aug_strength", type=str, default="strong", choices=["mild", "strong"])
    p.add_argument("--cons_apply_prob", type=float, default=0.8)
    p.add_argument("--cons_ce_weight", type=float, default=0.25)
    p.add_argument("--cons_align_weight", type=float, default=0.15)
    p.add_argument("--cons_blur_sig", type=float, nargs="+", default=[0.2, 1.2])
    p.add_argument("--cons_jpeg_qual", type=int, nargs="+", default=[70, 95])
    p.add_argument("--cons_resize_jitter", type=int, default=4)
    p.add_argument("--cons_noise_std", type=float, default=0.003)
    p.add_argument("--restoration_enable", action="store_true", default=False)
    p.add_argument("--restoration_strength", type=float, default=0.20)
    p.add_argument("--degrade_router_dim", type=int, default=128)
    p.add_argument("--degrade_router_dropout", type=float, default=0.10)
    p.add_argument("--restore_consistency_enable", action="store_true", default=False)
    p.add_argument("--restore_apply_prob", type=float, default=0.80)
    p.add_argument("--restore_ce_weight", type=float, default=0.25)
    p.add_argument("--restore_align_weight", type=float, default=0.15)
    p.add_argument("--restore_img_weight", type=float, default=0.05)
    p.add_argument("--restore_degfeat_weight", type=float, default=0.05)
    p.add_argument("--stagec_use_tri_consistency", action="store_true", default=True)
    p.add_argument("--no_stagec_use_tri_consistency", dest="stagec_use_tri_consistency", action="store_false")
    p.add_argument("--dry_run", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    base_cfg = json.loads(Path(args.base_config).read_text(encoding="utf-8"))

    stage_a = f"{args.exp_prefix}_A_extwarm"
    stage_b = f"{args.exp_prefix}_B_compft"
    stage_c = f"{args.exp_prefix}_C_cons"
    ckpt_dir = Path(args.checkpoints_dir).resolve()
    cache_dir = ckpt_dir / "manifests"
    cache_dir.mkdir(parents=True, exist_ok=True)

    comp_manifest_csv = Path(args.comp_train_csv).resolve() if args.comp_train_csv else Path(args.comp_manifest_csv).resolve()
    if not args.comp_train_csv:
        default_train_list = Path(args.comp_train_root).resolve() / "train_list.txt"
        comp_train_list = args.comp_train_list
        if comp_train_list is None and default_train_list.exists():
            comp_train_list = str(default_train_list)
        if not bool(args.skip_comp_manifest_build):
            _build_comp_manifest(
                comp_root=str(args.comp_train_root),
                out_csv=str(comp_manifest_csv),
                train_list=comp_train_list,
            )
        elif not comp_manifest_csv.exists():
            raise FileNotFoundError(
                f"--skip_comp_manifest_build set but comp manifest not found: {comp_manifest_csv}"
            )
    elif not comp_manifest_csv.exists():
        raise FileNotFoundError(f"--comp_train_csv not found: {comp_manifest_csv}")

    manifest_csv = Path(args.external_manifest_csv).resolve()
    if not bool(args.skip_manifest_build):
        build_manifest_cmd = [
            str(args.python_bin),
            str(repo / "scripts" / "build_external_manifest.py"),
            "--out_csv",
            str(manifest_csv),
            "--max_per_class",
            str(int(args.external_max_per_class)),
            "--seed",
            str(int(args.seed)),
            "--roots",
            *[str(x) for x in list(args.external_roots)],
        ]
        print("[prep] building external manifest...")
        if not args.dry_run:
            subprocess.run(build_manifest_cmd, check=True)
    elif not manifest_csv.exists():
        raise FileNotFoundError(f"--skip_manifest_build set but manifest not found: {manifest_csv}")

    comp_train_split_csv = cache_dir / f"{args.exp_prefix}_comp_train_split.csv"
    comp_val_split_csv = cache_dir / f"{args.exp_prefix}_comp_val_split.csv"
    if not bool(args.skip_comp_split_build):
        _build_comp_split_manifests(
            comp_manifest_csv=str(comp_manifest_csv),
            out_train_csv=str(comp_train_split_csv),
            out_val_csv=str(comp_val_split_csv),
            val_ratio=float(args.comp_val_ratio),
            seed=int(args.comp_split_seed),
            quality_bins=int(base_cfg.get("quality_bins", 3)),
            quality_max_side=int(base_cfg.get("quality_max_side", 256)),
        )
    else:
        if (not comp_train_split_csv.exists()) or (not comp_val_split_csv.exists()):
            raise FileNotFoundError(
                f"--skip_comp_split_build set but split files missing: {comp_train_split_csv}, {comp_val_split_csv}"
            )

    ext_quality_cache = (
        Path(args.ext_quality_cache_path).resolve()
        if args.ext_quality_cache_path
        else (cache_dir / f"{args.exp_prefix}_ext_quality_cache.csv")
    )
    comp_quality_cache = (
        Path(args.comp_quality_cache_path).resolve()
        if args.comp_quality_cache_path
        else (cache_dir / f"{args.exp_prefix}_comp_quality_cache.csv")
    )
    ext_quality_cache.parent.mkdir(parents=True, exist_ok=True)
    comp_quality_cache.parent.mkdir(parents=True, exist_ok=True)
    stagec_mix_manifest = cache_dir / f"{args.exp_prefix}_stagec_mix_manifest.csv"

    common = {
        "deterministic": False,
        "train_csv": str(comp_train_split_csv),
        "val_data_root": None,
        "val_csv": str(comp_val_split_csv),
        "quality_cache_path": str(comp_quality_cache),
        "restoration_enable": bool(args.restoration_enable),
        "restoration_strength": float(args.restoration_strength),
        "degrade_router_dim": int(args.degrade_router_dim),
        "degrade_router_dropout": float(args.degrade_router_dropout),
        "restore_consistency_enable": bool(args.restore_consistency_enable),
        "restore_consistency_apply_prob": float(args.restore_apply_prob),
        "restore_consistency_ce_weight": float(args.restore_ce_weight),
        "restore_consistency_align_weight": float(args.restore_align_weight),
        "restore_consistency_img_weight": float(args.restore_img_weight),
        "restore_consistency_degfeat_weight": float(args.restore_degfeat_weight),
    }
    print(f"[prep] competition manifest(full): {comp_manifest_csv}")
    print(f"[prep] competition split train: {comp_train_split_csv}")
    print(f"[prep] competition split val: {comp_val_split_csv}")

    # Stage A: external warmup (no consistency)
    a_over = dict(common)
    a_over.update(
        {
            "train_csv": str(manifest_csv),
            "tri_consistency_enable": False,
            "tri_consistency_apply_prob": 0.0,
            "tri_consistency_ce_weight": 0.0,
            "tri_consistency_align_weight": 0.0,
            "quality_cache_path": str(ext_quality_cache),
            "resume_path": None,
            "load_whole_model": False,
        }
    )
    if bool(args.stagea_comp_like_aug):
        a_over.update(_stagea_comp_like_aug_overrides(args.stagea_aug_strength))
    if args.dry_run:
        print(f"[dry-run] stage A -> {stage_a}")
    else:
        _run_stage(
            python_bin=args.python_bin,
            run_script=args.run_script,
            repo=str(repo),
            base_config=args.base_config,
            checkpoints_dir=str(ckpt_dir),
            name=stage_a,
            seed=int(args.seed),
            niter=int(args.stageA_niter),
            batch_sizes=args.batch_sizes,
            num_threads=int(args.num_threads),
            train_frame_num=int(args.train_frame_num),
            val_frame_num=int(args.val_frame_num),
            multi_expert_k=int(args.multi_expert_k),
            multi_expert_route=str(args.multi_expert_route),
            extra_overrides=a_over,
        )

    a_best = ckpt_dir / stage_a / "best_auc.pt"

    # Stage B: competition finetune from stage A
    b_over = dict(common)
    b_over.update(
        {
            "tri_consistency_enable": False,
            "tri_consistency_apply_prob": 0.0,
            "tri_consistency_ce_weight": 0.0,
            "tri_consistency_align_weight": 0.0,
        }
    )
    if args.dry_run:
        print(f"[dry-run] stage B -> {stage_b} resume={a_best}")
    else:
        if not a_best.exists():
            raise FileNotFoundError(f"Stage A best ckpt not found: {a_best}")
        _run_stage(
            python_bin=args.python_bin,
            run_script=args.run_script,
            repo=str(repo),
            base_config=args.base_config,
            checkpoints_dir=str(ckpt_dir),
            name=stage_b,
            seed=int(args.seed),
            niter=int(args.stageB_niter),
            batch_sizes=args.batch_sizes,
            num_threads=int(args.num_threads),
            train_frame_num=int(args.train_frame_num),
            val_frame_num=int(args.val_frame_num),
            multi_expert_k=int(args.multi_expert_k),
            multi_expert_route=str(args.multi_expert_route),
            resume_path=str(a_best),
            load_whole_model=False,
            extra_overrides=b_over,
        )

    b_best = ckpt_dir / stage_b / "best_auc.pt"

    # Stage C: competition consistency finetune from stage B
    if float(args.stagec_comp_ratio) >= 0.999:
        c_train_csv = str(comp_train_split_csv)
        print(f"[prep] stage C train manifest (comp-only) -> {c_train_csv}")
    else:
        c_train_csv = str(
            _build_stagec_mix_manifest(
                comp_manifest_csv=str(comp_train_split_csv),
                ext_manifest_csv=str(manifest_csv),
                out_csv=str(stagec_mix_manifest),
                comp_ratio=float(args.stagec_comp_ratio),
                seed=int(args.seed),
            )
        )
    c_over = dict(common)
    c_over.update(
        {
            "train_csv": str(c_train_csv),
            "tri_consistency_enable": bool(args.stagec_use_tri_consistency),
            "tri_consistency_apply_prob": float(args.cons_apply_prob) if bool(args.stagec_use_tri_consistency) else 0.0,
            "tri_consistency_ce_weight": float(args.cons_ce_weight) if bool(args.stagec_use_tri_consistency) else 0.0,
            "tri_consistency_align_weight": float(args.cons_align_weight) if bool(args.stagec_use_tri_consistency) else 0.0,
            "tri_consistency_blur_sig": [float(v) for v in list(args.cons_blur_sig)],
            "tri_consistency_jpeg_qual": [int(v) for v in list(args.cons_jpeg_qual)],
            "tri_consistency_resize_jitter": int(args.cons_resize_jitter),
            "tri_consistency_noise_std": float(args.cons_noise_std),
        }
    )
    if args.dry_run:
        print(f"[dry-run] stage C -> {stage_c} resume={b_best}")
        return

    if not b_best.exists():
        raise FileNotFoundError(f"Stage B best ckpt not found: {b_best}")
    _run_stage(
        python_bin=args.python_bin,
        run_script=args.run_script,
        repo=str(repo),
        base_config=args.base_config,
        checkpoints_dir=str(ckpt_dir),
        name=stage_c,
        seed=int(args.seed),
        niter=int(args.stageC_niter),
        batch_sizes=args.batch_sizes,
        num_threads=int(args.num_threads),
        train_frame_num=int(args.train_frame_num),
        val_frame_num=int(args.val_frame_num),
        multi_expert_k=int(args.multi_expert_k),
        multi_expert_route=str(args.multi_expert_route),
        resume_path=str(b_best),
        load_whole_model=False,
        extra_overrides=c_over,
    )

    c_best = ckpt_dir / stage_c / "best_auc.pt"
    print(f"[done] staged training completed. final_best={c_best}")


if __name__ == "__main__":
    main()
