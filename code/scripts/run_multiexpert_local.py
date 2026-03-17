#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


def _parse_extra_value(raw: str):
    s = str(raw).strip()
    if s == "":
        return s
    try:
        return json.loads(s)
    except Exception:
        pass
    lo = s.lower()
    if lo in {"true", "false"}:
        return lo == "true"
    if lo in {"none", "null"}:
        return None
    try:
        if ("." in s) or ("e" in lo):
            return float(s)
        return int(s)
    except Exception:
        return s


def _parse_extra_overrides(items):
    out = {}
    for item in list(items or []):
        if "=" not in str(item):
            raise ValueError(f"--set expects key=value, got: {item!r}")
        key, raw = str(item).split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in --set item: {item!r}")
        out[key] = _parse_extra_value(raw)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default="/work/u1657859/Jess/app")
    p.add_argument(
        "--base_config",
        type=str,
        default="/work/u1657859/Jess/app/checkpoints/Exp_E1_DF40_tuneoff-AUXLoss_s2048/config.json",
    )
    p.add_argument("--name", type=str, default="Exp_MEv1_full6_8f_s2048_local")
    p.add_argument("--checkpoints_dir", type=str, default="/work/u1657859/Jess/app/local_checkpoints")
    p.add_argument("--seed", type=int, default=2048)
    p.add_argument("--niter", type=int, default=6)
    p.add_argument("--earlystop_epoch", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_threads", type=int, default=4)
    p.add_argument("--train_frame_num", type=int, default=8)
    p.add_argument("--val_frame_num", type=int, default=1)
    p.add_argument("--resume_path", type=str, default=None)
    p.add_argument("--load_whole_model", action="store_true", default=False)
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
    p.add_argument("--multi_expert_route_temp", type=float, default=1.0)
    p.add_argument("--multi_expert_hybrid_mix", type=float, default=0.0)
    p.add_argument("--multi_expert_route_sup_lambda", type=float, default=0.0)
    p.add_argument("--multi_expert_balance_lambda", type=float, default=0.0)
    p.add_argument("--multi_expert_div_lambda", type=float, default=0.002)
    p.add_argument(
        "--set",
        dest="extra_sets",
        action="append",
        default=[],
        help="Additional config overrides in key=value form; value may be JSON.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    os.chdir(str(repo))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import Config

    base_cfg = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    for k, v in base_cfg.items():
        Config.OPT_OVERRIDES[k] = v

    Config.OPT_OVERRIDES.update(
        {
            "name": args.name,
            "checkpoints_dir": str(Path(args.checkpoints_dir).resolve()),
            "seed": int(args.seed),
            "niter": int(args.niter),
            "earlystop_epoch": int(args.earlystop_epoch),
            "batch_size": int(args.batch_size),
            "num_threads": int(args.num_threads),
            "train_frame_num": int(args.train_frame_num),
            "val_frame_num": int(args.val_frame_num),
            "resume_path": args.resume_path,
            "load_whole_model": bool(args.load_whole_model),
            "tri_consistency_enable": False,
            "tri_consistency_apply_prob": 0.0,
            "tri_consistency_ce_weight": 0.0,
            "tri_consistency_align_weight": 0.0,
            "multi_expert_enable": True,
            "multi_expert_k": int(args.multi_expert_k),
            "multi_expert_route": str(args.multi_expert_route),
            "multi_expert_route_temp": float(args.multi_expert_route_temp),
            "multi_expert_hybrid_mix": float(args.multi_expert_hybrid_mix),
            "multi_expert_route_sup_lambda": float(args.multi_expert_route_sup_lambda),
            "multi_expert_balance_lambda": float(args.multi_expert_balance_lambda),
            "multi_expert_div_lambda": float(args.multi_expert_div_lambda),
        }
    )
    extra_overrides = _parse_extra_overrides(args.extra_sets)
    if extra_overrides:
        Config.OPT_OVERRIDES.update(extra_overrides)
        print(f"[run_multiexpert_local] applied extra overrides: {sorted(extra_overrides.keys())}")

    import train

    train.main()


if __name__ == "__main__":
    main()
