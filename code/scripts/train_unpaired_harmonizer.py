#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm

from effort.harmonizer import (
    HarmonizerNet,
    PatchDiscriminator,
    build_loader,
    compute_noise_std_batch,
    compute_q_log_batch,
    highpass_3x3,
    ks_distance_np,
    lowpass_5x5,
    read_manifest_rows,
    sample_manifest_rows,
    to_clip_norm,
)
from effort.utils import load_ckpt, set_seed
from models import get_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--df40_manifest",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/manifests/ext_df40_only_manifest.csv",
    )
    p.add_argument(
        "--comp_manifest",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/manifests/competition_train_manifest.csv",
    )
    p.add_argument(
        "--comp_val_manifest",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/manifests/Exp_MEv3_df40_drouter_rst_v3_comp_val_split.csv",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/harmonizer/Exp_HarmU1_s2048",
    )
    p.add_argument("--seed", type=int, default=2048)
    p.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--steps_per_epoch", type=int, default=0, help="0 -> auto by target loader length")
    p.add_argument("--max_df40", type=int, default=30000)
    p.add_argument("--max_comp", type=int, default=1000)
    p.add_argument("--max_comp_val", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=24)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--harm_channels", type=int, default=64)
    p.add_argument("--harm_blocks", type=int, default=6)
    p.add_argument("--harm_max_delta", type=float, default=0.22)
    p.add_argument("--disc_channels", type=int, default=32)
    p.add_argument("--w_adv", type=float, default=1.0)
    p.add_argument("--w_id", type=float, default=0.8)
    p.add_argument("--w_low", type=float, default=1.0)
    p.add_argument("--w_teacher", type=float, default=0.0)
    p.add_argument("--w_stat_q", type=float, default=2.0)
    p.add_argument("--w_stat_n", type=float, default=2.0)
    p.add_argument("--w_stat_hf", type=float, default=1.0)
    p.add_argument("--teacher_every", type=int, default=2)
    p.add_argument(
        "--teacher_ckpt",
        type=str,
        default="/work/u1657859/Jess/app/local_checkpoints/Exp_MEv1_full6_8f_s2048_local/best_auc.pt",
    )
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    return p.parse_args()


def _pick_device(arg: str) -> torch.device:
    key = str(arg or "auto").lower()
    if key == "auto":
        key = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(key)


def _next_batch(it, loader):
    while True:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if batch is not None:
            return batch, it


def _autocast_ctx(enabled: bool):
    if bool(enabled):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def _teacher_need_q(route: str) -> bool:
    return str(route or "").lower() in {"quality_bin", "quality_soft", "hybrid", "degrade_hybrid"}


def _build_teacher(ckpt_path: str, device: torch.device) -> Tuple[Optional[nn.Module], bool, int]:
    ckpt = Path(ckpt_path).resolve()
    if not ckpt.exists():
        logging.warning("[teacher] checkpoint not found, skip: %s", str(ckpt))
        return None, False, 224

    cfg_path = ckpt.parent / "config.json"
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

    opt = SimpleNamespace(
        clip_model=cfg.get("clip_model", "openai/clip-vit-large-patch14"),
        patch_pool_tau=float(cfg.get("patch_pool_tau", 1.8)),
        patch_pool_mode=str(cfg.get("patch_pool_mode", "lse")),
        patch_trim_p=float(cfg.get("patch_trim_p", 0.0)),
        patch_quality=str(cfg.get("patch_quality", "cos_norm")),
        multi_expert_enable=bool(cfg.get("multi_expert_enable", True)),
        multi_expert_k=int(cfg.get("multi_expert_k", 3)),
        multi_expert_route=str(cfg.get("multi_expert_route", "quality_bin")),
        multi_expert_route_temp=float(cfg.get("multi_expert_route_temp", 1.0)),
        multi_expert_route_detach=bool(cfg.get("multi_expert_route_detach", True)),
        multi_expert_hybrid_mix=float(cfg.get("multi_expert_hybrid_mix", 0.0)),
        quality_cuts=cfg.get("quality_cuts", None),
        osd_regs=False,
        degrade_router_dim=int(cfg.get("degrade_router_dim", 128)),
        degrade_router_dropout=float(cfg.get("degrade_router_dropout", 0.1)),
        restoration_enable=bool(cfg.get("restoration_enable", False)),
        restoration_strength=float(cfg.get("restoration_strength", 0.20)),
        fix_backbone=True,
    )
    model = get_model(opt).to(device)
    miss, unexp = load_ckpt(str(ckpt), model, strict=False)
    logging.info("[teacher] loaded from %s | missing=%d unexpected=%d", str(ckpt), len(miss), len(unexp))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    teacher_size = int(cfg.get("image_size", cfg.get("loadSize", 224)))
    teacher_size = max(64, teacher_size)
    return model, _teacher_need_q(opt.multi_expert_route), teacher_size


def _teacher_logits(model: nn.Module, x01: torch.Tensor, need_q: bool, teacher_size: int) -> torch.Tensor:
    x_in = x01
    if int(x_in.shape[-1]) != int(teacher_size) or int(x_in.shape[-2]) != int(teacher_size):
        x_in = F.interpolate(
            x_in,
            size=(int(teacher_size), int(teacher_size)),
            mode="bilinear",
            align_corners=False,
        )
    x_in = to_clip_norm(x_in)
    batch = {"image": x_in}
    if need_q:
        batch["q_log"] = compute_q_log_batch(x_in)
    pred = model(batch)
    return pred["cls"].float()


@torch.no_grad()
def _evaluate_alignment(
    gen: nn.Module,
    src_loader,
    tgt_loader,
    device: torch.device,
    eval_steps: int,
) -> Dict[str, float]:
    gen.eval()
    src_it = iter(src_loader)
    tgt_it = iter(tgt_loader)
    q_src = []
    q_tgt = []
    q_raw = []
    n_src = []
    n_tgt = []
    n_raw = []
    for _ in range(int(eval_steps)):
        sb, src_it = _next_batch(src_it, src_loader)
        tb, tgt_it = _next_batch(tgt_it, tgt_loader)
        xs = sb["image"].to(device, non_blocking=True)
        xt = tb["image"].to(device, non_blocking=True)
        xh = gen(xs)
        q_raw.extend(compute_q_log_batch(xs).detach().cpu().numpy().tolist())
        q_src.extend(compute_q_log_batch(xh).detach().cpu().numpy().tolist())
        q_tgt.extend(compute_q_log_batch(xt).detach().cpu().numpy().tolist())
        n_raw.extend(compute_noise_std_batch(xs).detach().cpu().numpy().tolist())
        n_src.extend(compute_noise_std_batch(xh).detach().cpu().numpy().tolist())
        n_tgt.extend(compute_noise_std_batch(xt).detach().cpu().numpy().tolist())

    q_raw_np = np.asarray(q_raw, dtype=np.float64)
    q_src_np = np.asarray(q_src, dtype=np.float64)
    q_tgt_np = np.asarray(q_tgt, dtype=np.float64)
    n_raw_np = np.asarray(n_raw, dtype=np.float64)
    n_src_np = np.asarray(n_src, dtype=np.float64)
    n_tgt_np = np.asarray(n_tgt, dtype=np.float64)

    out = {
        "q_raw_mean": float(q_raw_np.mean()) if q_raw_np.size else 0.0,
        "q_src_mean": float(q_src_np.mean()) if q_src_np.size else 0.0,
        "q_tgt_mean": float(q_tgt_np.mean()) if q_tgt_np.size else 0.0,
        "n_raw_mean": float(n_raw_np.mean()) if n_raw_np.size else 0.0,
        "n_src_mean": float(n_src_np.mean()) if n_src_np.size else 0.0,
        "n_tgt_mean": float(n_tgt_np.mean()) if n_tgt_np.size else 0.0,
        "ks_q_raw": ks_distance_np(q_raw_np, q_tgt_np),
        "ks_q": ks_distance_np(q_src_np, q_tgt_np),
        "ks_noise_raw": ks_distance_np(n_raw_np, n_tgt_np),
        "ks_noise": ks_distance_np(n_src_np, n_tgt_np),
    }
    out["score_raw"] = float(out["ks_q_raw"] + out["ks_noise_raw"])
    out["score"] = float(out["ks_q"] + out["ks_noise"])
    out["gain"] = float(out["score_raw"] - out["score"])
    return out


def _save_preview(gen: nn.Module, src_loader, tgt_loader, device: torch.device, out_path: Path) -> None:
    gen.eval()
    src_it = iter(src_loader)
    tgt_it = iter(tgt_loader)
    sb, _ = _next_batch(src_it, src_loader)
    tb, _ = _next_batch(tgt_it, tgt_loader)
    xs = sb["image"][:4].to(device, non_blocking=True)
    xt = tb["image"][:4].to(device, non_blocking=True)
    with torch.no_grad():
        xh = gen(xs)
    grid = torch.cat([xs, xh, xt], dim=0).detach().cpu()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path), nrow=4)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(out_dir / "train.log"), encoding="utf-8"),
        ],
    )

    set_seed(seed=int(args.seed), deterministic=False)
    device = _pick_device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logging.info("[setup] device=%s amp=%s", str(device), str(use_amp))

    df40_rows = sample_manifest_rows(
        read_manifest_rows(args.df40_manifest),
        max_n=int(args.max_df40),
        seed=int(args.seed),
    )
    comp_rows = sample_manifest_rows(
        read_manifest_rows(args.comp_manifest),
        max_n=int(args.max_comp),
        seed=int(args.seed),
    )
    comp_val_rows = sample_manifest_rows(
        read_manifest_rows(args.comp_val_manifest),
        max_n=int(args.max_comp_val),
        seed=int(args.seed),
    )
    if not df40_rows or not comp_rows:
        raise RuntimeError("Empty rows after sampling; check manifest paths and options.")
    if not comp_val_rows:
        comp_val_rows = list(comp_rows)
    logging.info(
        "[data] df40=%d comp_train=%d comp_val=%d",
        len(df40_rows),
        len(comp_rows),
        len(comp_val_rows),
    )

    src_loader = build_loader(
        rows=df40_rows,
        batch_size=int(args.batch_size),
        image_size=int(args.image_size),
        num_workers=int(args.num_workers),
        shuffle=True,
        drop_last=True,
    )
    tgt_loader = build_loader(
        rows=comp_rows,
        batch_size=int(args.batch_size),
        image_size=int(args.image_size),
        num_workers=int(args.num_workers),
        shuffle=True,
        drop_last=True,
    )
    eval_src_loader = build_loader(
        rows=sample_manifest_rows(df40_rows, max_n=1024, seed=int(args.seed)),
        batch_size=int(args.batch_size),
        image_size=int(args.image_size),
        num_workers=int(args.num_workers),
        shuffle=True,
        drop_last=True,
    )
    eval_tgt_loader = build_loader(
        rows=comp_val_rows,
        batch_size=int(args.batch_size),
        image_size=int(args.image_size),
        num_workers=int(args.num_workers),
        shuffle=True,
        drop_last=True,
    )

    gen_cfg = {
        "channels": int(args.harm_channels),
        "n_blocks": int(args.harm_blocks),
        "max_delta": float(args.harm_max_delta),
    }
    gen = HarmonizerNet(**gen_cfg).to(device)
    disc = PatchDiscriminator(channels=int(args.disc_channels)).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=float(args.lr_g), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(args.lr_d), betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    teacher = None
    teacher_need_q = False
    teacher_size = int(args.image_size)
    if float(args.w_teacher) > 0.0:
        teacher, teacher_need_q, teacher_size = _build_teacher(args.teacher_ckpt, device=device)
    if teacher is None:
        logging.info("[teacher] disabled")
    else:
        logging.info("[teacher] enabled route_need_q=%s", str(teacher_need_q))

    steps_per_epoch = int(args.steps_per_epoch or 0)
    if steps_per_epoch <= 0:
        steps_per_epoch = max(1, len(tgt_loader))
    logging.info("[setup] steps_per_epoch=%d", steps_per_epoch)

    best_score = float("inf")
    global_step = 0
    src_it = iter(src_loader)
    tgt_it = iter(tgt_loader)

    for epoch in range(int(args.epochs)):
        gen.train()
        disc.train()
        m_d = 0.0
        m_g = 0.0
        m_adv = 0.0
        m_id = 0.0
        m_low = 0.0
        m_t = 0.0
        m_sq = 0.0
        m_sn = 0.0
        m_shf = 0.0

        pbar = tqdm(range(steps_per_epoch), desc=f"harm-ep{epoch}", ncols=100)
        for step in pbar:
            sb, src_it = _next_batch(src_it, src_loader)
            tb, tgt_it = _next_batch(tgt_it, tgt_loader)
            xs = sb["image"].to(device, non_blocking=True)
            xt = tb["image"].to(device, non_blocking=True)

            # 1) Discriminator
            opt_d.zero_grad(set_to_none=True)
            with _autocast_ctx(use_amp):
                xh = gen(xs).detach()
                d_real = disc(highpass_3x3(xt))
                d_fake = disc(highpass_3x3(xh))
                loss_d_real = bce(d_real, torch.ones_like(d_real))
                loss_d_fake = bce(d_fake, torch.zeros_like(d_fake))
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
            scaler.scale(loss_d).backward()
            scaler.step(opt_d)

            # 2) Generator
            opt_g.zero_grad(set_to_none=True)
            with _autocast_ctx(use_amp):
                xh = gen(xs)
                d_fake_g = disc(highpass_3x3(xh))
                loss_adv = bce(d_fake_g, torch.ones_like(d_fake_g))
                loss_id = F.l1_loss(gen(xt), xt)
                loss_low = F.l1_loss(lowpass_5x5(xh), lowpass_5x5(xs))
                q_h = compute_q_log_batch(xh)
                q_t = compute_q_log_batch(xt).detach()
                n_h = compute_noise_std_batch(xh)
                n_t = compute_noise_std_batch(xt).detach()
                hf_h = highpass_3x3(xh).abs().flatten(1).mean(dim=1)
                hf_t = highpass_3x3(xt).abs().flatten(1).mean(dim=1).detach()
                loss_sq = F.l1_loss(torch.sort(q_h).values, torch.sort(q_t).values)
                loss_sn = F.l1_loss(torch.sort(n_h).values, torch.sort(n_t).values)
                loss_shf = F.l1_loss(torch.sort(hf_h).values, torch.sort(hf_t).values)

                loss_t = torch.zeros((), device=device, dtype=xh.dtype)
                if teacher is not None and (int(global_step) % max(1, int(args.teacher_every)) == 0):
                    with torch.no_grad():
                        logits_src = _teacher_logits(teacher, xs, teacher_need_q, teacher_size)
                    logits_h = _teacher_logits(teacher, xh, teacher_need_q, teacher_size)
                    loss_t = F.mse_loss(logits_h, logits_src)

                loss_g = (
                    float(args.w_adv) * loss_adv
                    + float(args.w_id) * loss_id
                    + float(args.w_low) * loss_low
                    + float(args.w_stat_q) * loss_sq
                    + float(args.w_stat_n) * loss_sn
                    + float(args.w_stat_hf) * loss_shf
                    + float(args.w_teacher) * loss_t
                )
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()

            global_step += 1
            m_d += float(loss_d.detach().cpu().item())
            m_g += float(loss_g.detach().cpu().item())
            m_adv += float(loss_adv.detach().cpu().item())
            m_id += float(loss_id.detach().cpu().item())
            m_low += float(loss_low.detach().cpu().item())
            m_t += float(loss_t.detach().cpu().item())
            m_sq += float(loss_sq.detach().cpu().item())
            m_sn += float(loss_sn.detach().cpu().item())
            m_shf += float(loss_shf.detach().cpu().item())
            pbar.set_postfix(
                d=f"{m_d/max(1, step + 1):.4f}",
                g=f"{m_g/max(1, step + 1):.4f}",
            )

        eval_stat = _evaluate_alignment(
            gen=gen,
            src_loader=eval_src_loader,
            tgt_loader=eval_tgt_loader,
            device=device,
            eval_steps=int(args.eval_steps),
        )
        _save_preview(gen, eval_src_loader, eval_tgt_loader, device=device, out_path=out_dir / f"preview_ep{epoch:02d}.png")

        epoch_stat = {
            "epoch": int(epoch),
            "loss_d": float(m_d / max(1, steps_per_epoch)),
            "loss_g": float(m_g / max(1, steps_per_epoch)),
            "loss_adv": float(m_adv / max(1, steps_per_epoch)),
            "loss_id": float(m_id / max(1, steps_per_epoch)),
            "loss_low": float(m_low / max(1, steps_per_epoch)),
            "loss_teacher": float(m_t / max(1, steps_per_epoch)),
            "loss_stat_q": float(m_sq / max(1, steps_per_epoch)),
            "loss_stat_n": float(m_sn / max(1, steps_per_epoch)),
            "loss_stat_hf": float(m_shf / max(1, steps_per_epoch)),
            **eval_stat,
        }

        payload = {
            "epoch": int(epoch),
            "generator": gen.state_dict(),
            "discriminator": disc.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "gen_cfg": gen_cfg,
            "args": vars(args),
            "stat": epoch_stat,
        }
        torch.save(payload, str(ckpt_dir / f"epoch_{epoch}.pt"))
        torch.save(
            {
                "model": gen.state_dict(),
                "gen_cfg": gen_cfg,
                "epoch": int(epoch),
                "stat": epoch_stat,
            },
            str(out_dir / "last_generator.pt"),
        )

        if float(eval_stat["score"]) <= float(best_score):
            best_score = float(eval_stat["score"])
            torch.save(
                {
                    "model": gen.state_dict(),
                    "gen_cfg": gen_cfg,
                    "epoch": int(epoch),
                    "stat": epoch_stat,
                },
                str(out_dir / "best_generator.pt"),
            )
            logging.info("[best] epoch=%d score=%.4f", epoch, best_score)

        logging.info(
            "[epoch %d] d=%.4f g=%.4f adv=%.4f id=%.4f low=%.4f sq=%.4f sn=%.4f shf=%.4f t=%.4f | ks_q_raw=%.4f ks_q=%.4f ks_n_raw=%.4f ks_n=%.4f score_raw=%.4f score=%.4f gain=%.4f",
            epoch,
            epoch_stat["loss_d"],
            epoch_stat["loss_g"],
            epoch_stat["loss_adv"],
            epoch_stat["loss_id"],
            epoch_stat["loss_low"],
            epoch_stat["loss_stat_q"],
            epoch_stat["loss_stat_n"],
            epoch_stat["loss_stat_hf"],
            epoch_stat["loss_teacher"],
            eval_stat["ks_q_raw"],
            eval_stat["ks_q"],
            eval_stat["ks_noise_raw"],
            eval_stat["ks_noise"],
            eval_stat["score_raw"],
            eval_stat["score"],
            eval_stat["gain"],
        )

    logging.info("[done] training finished. best_score=%.4f out_dir=%s", best_score, str(out_dir))


if __name__ == "__main__":
    main()
