import os
import time
import json
import logging
import gc
import sys
import subprocess
import numpy as np
from collections import Counter, deque

from Config import *

os.environ["CUDA_DEVICE_ORDER"] = CUDA_DEVICE_ORDER
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

import torch
from types import SimpleNamespace

from data import create_dataloaders
from networks.trainer import Trainer
from validate import validate
from effort.utils import set_seed, count_trainable_params, ensure_cuda_conv_runtime
from earlystop import EarlyStopping
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _setup_logging(opt):
    log_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"train_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return log_dir


def _normalize_steps(opt, total_steps):
    for name in [
        "lr_warmup_steps",
        "reg_warmup_steps",
        "reg_ramp_steps",
        "patch_warmup_steps",
        "patch_ramp_steps",
        "groupdro_warmup_steps",
        "groupdro_cvar_warmup_steps",
        "groupdro_cvar_ramp_steps",
        "inv_mev1_warmup_steps",
        "inv_mev1_ramp_steps",
    ]:
        val = getattr(opt, name, 0.0)
        if val is None:
            continue
        if 0 < val < 1:
            setattr(opt, name, int(total_steps * float(val)))
        else:
            setattr(opt, name, int(val))


def _maybe_submit_epoch(opt, log_dir, epoch):
    if not bool(getattr(opt, "submit_each_epoch", False)):
        return

    epoch = int(epoch)
    ckpt_path = os.path.join(log_dir, f"epoch_{epoch}.pt")
    if not os.path.isfile(ckpt_path):
        logging.warning("[submit_epoch] missing checkpoint: %s", ckpt_path)
        return

    builder = str(
        getattr(
            opt,
            "submit_builder_script",
            "/work/u1657859/Jess/app4/scripts/build_submission_mev1_strict.sh",
        )
    )
    if not os.path.isfile(builder):
        logging.error("[submit_epoch] builder script not found: %s", builder)
        return

    out_prefix = str(getattr(opt, "submit_out_prefix", f"app_{opt.name}"))
    out_stem = f"{out_prefix}_ep{epoch}"

    env = os.environ.copy()
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    env["APP"] = str(getattr(opt, "submit_app", repo_dir))
    env["PY"] = str(getattr(opt, "submit_python", sys.executable))
    env["OUT_DIR"] = str(
        getattr(opt, "submit_out_dir", "/work/u1657859/Jess/app/local_checkpoints/submissions")
    )
    env["VAL_DIR"] = str(
        getattr(
            opt,
            "submit_val_dir",
            "/work/u1657859/ntire_RDFC_2026_cvpr/Dataset/challange_data/validation_data_final",
        )
    )
    env["SEED"] = str(int(getattr(opt, "seed", 2048)))
    env["BATCH_SIZE"] = str(int(getattr(opt, "submit_batch_size", 64)))
    env["NUM_WORKERS"] = str(int(getattr(opt, "submit_num_workers", 4)))
    tta_mode_raw = getattr(opt, "submit_tta_mode", "none")
    tta_logit_agg_raw = getattr(opt, "submit_tta_logit_agg", "mean")
    tta_mode = "none" if tta_mode_raw is None else str(tta_mode_raw).strip().lower()
    tta_logit_agg = "mean" if tta_logit_agg_raw is None else str(tta_logit_agg_raw).strip().lower()
    if tta_mode not in {"none", "hflip", "hflip_resize"}:
        logging.warning("[submit_epoch] invalid submit_tta_mode=%r; fallback to none", tta_mode_raw)
        tta_mode = "none"
    if tta_logit_agg not in {"mean", "median"}:
        logging.warning("[submit_epoch] invalid submit_tta_logit_agg=%r; fallback to mean", tta_logit_agg_raw)
        tta_logit_agg = "mean"
    env["TTA_MODE"] = tta_mode
    env["TTA_LOGIT_AGG"] = tta_logit_agg

    cmd = [builder, ckpt_path, out_stem]
    logging.info("[submit_epoch] epoch=%d start -> %s", epoch, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception:
        logging.exception("[submit_epoch] failed to launch builder for epoch=%d", epoch)
        return

    if proc.stdout:
        tail = "\n".join(proc.stdout.strip().splitlines()[-30:])
        logging.info("[submit_epoch] epoch=%d output tail:\n%s", epoch, tail)
    if proc.returncode != 0:
        logging.error("[submit_epoch] epoch=%d failed with code=%d", epoch, int(proc.returncode))
        return
    logging.info("[submit_epoch] epoch=%d done -> %s/%s.zip", epoch, env["OUT_DIR"], out_stem)


def train_one_epoch(trainer, loader, global_step, log_every, audit_window):
    losses = []
    losses_cls = []
    losses_reg = []
    losses_real = []
    losses_fake = []
    losses_orth = []
    losses_ksv = []
    losses_inv = []
    losses_inv_kd = []
    losses_inv_div = []
    inv_anchor_cos = []
    patch_evis = []
    patch_deltas = []
    diag_acc = {
        "groupdro_q_max": [],
        "groupdro_q_entropy": [],
        "groupdro_q_entropy_norm": [],
        "groupdro_q_eff": [],
        "groupdro_effective_groups": [],
        "groupdro_tail_k_mean": [],
        "groupdro_tail_frac_mean": [],
        "groupdro_tail_coverage": [],
        "groupdro_cap_value_mean": [],
        "groupdro_clipped_rate_mean": [],
        "groupdro_n_eff_samples": [],
        "groupdro_cvar_lambda_t": [],
        "logit_abs_mean": [],
        "prob_entropy_mean": [],
    }
    cvar_enabled_seen = False
    cvar_trim_seen = False
    audit_recent = deque(maxlen=max(1, int(audit_window)))

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="train", ncols=80)

    for batch_idx, batch in enumerate(iterator, start=1):
        if batch is None:
            continue
        batch_stratum = Counter()
        batch_label_qbin = Counter()
        batch_degrade = Counter()
        batch_q_values = []
        if "stratum_id" in batch:
            sid = batch["stratum_id"]
            sid_np = sid.detach().cpu().numpy() if torch.is_tensor(sid) else np.asarray(sid)
            for x in sid_np.tolist():
                batch_stratum[int(x)] += 1
        if "quality_bin" in batch and "label" in batch:
            qbin = batch["quality_bin"]
            y = batch["label"]
            qbin_np = qbin.detach().cpu().numpy() if torch.is_tensor(qbin) else np.asarray(qbin)
            y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
            for qb, yy in zip(qbin_np.tolist(), y_np.tolist()):
                batch_label_qbin[(int(yy), int(qb))] += 1
        if "q_log" in batch:
            q = batch["q_log"]
            q_np = q.detach().cpu().numpy() if torch.is_tensor(q) else np.asarray(q)
            batch_q_values = np.asarray(q_np, dtype=float).tolist()
        if "degrade_bin" in batch:
            db = batch["degrade_bin"]
            db_np = db.detach().cpu().numpy() if torch.is_tensor(db) else np.asarray(db)
            for x in db_np.tolist():
                batch_degrade[int(x)] += 1

        audit_recent.append((batch_stratum, batch_label_qbin, batch_degrade, batch_q_values))

        stats = trainer.train_step(batch, global_step)
        global_step += 1

        losses.append(stats["loss"])
        losses_cls.append(stats["loss_cls"])
        losses_reg.append(stats["loss_reg"])
        losses_real.append(stats["loss_real"])
        losses_fake.append(stats["loss_fake"])
        losses_orth.append(stats["loss_orth"])
        losses_ksv.append(stats["loss_ksv"])
        losses_inv.append(stats.get("loss_inv", 0.0))
        losses_inv_kd.append(stats.get("loss_inv_kd", 0.0))
        losses_inv_div.append(stats.get("loss_inv_div", 0.0))
        inv_anchor_cos.append(stats.get("inv_anchor_cos_mean", float("nan")))
        patch_evis.append(stats["patch_evidence"])
        patch_deltas.append(stats["patch_delta"])
        cvar_enabled_seen = cvar_enabled_seen or bool(stats.get("groupdro_use_cvar", False))
        cvar_trim_seen = cvar_trim_seen or bool(stats.get("groupdro_cvar_trim", False))
        for k in diag_acc.keys():
            v = stats.get(k, None)
            if v is None:
                continue
            try:
                vf = float(v)
            except Exception:
                continue
            if np.isfinite(vf):
                diag_acc[k].append(vf)

        if tqdm is not None:
            iterator.set_postfix(loss=f"{stats['loss']:.4f}")

        if log_every and batch_idx % log_every == 0:
            mean_cls = float(np.mean(losses_cls[-100:])) if losses_cls else 0.0
            mean_reg = float(np.mean(losses_reg[-100:])) if losses_reg else 0.0
            mean_real = float(np.mean(losses_real[-100:])) if losses_real else 0.0
            mean_fake = float(np.mean(losses_fake[-100:])) if losses_fake else 0.0
            mean_orth = float(np.mean(losses_orth[-100:])) if losses_orth else 0.0
            mean_ksv = float(np.mean(losses_ksv[-100:])) if losses_ksv else 0.0
            mean_inv = float(np.mean(losses_inv[-100:])) if losses_inv else 0.0
            mean_inv_kd = float(np.mean(losses_inv_kd[-100:])) if losses_inv_kd else 0.0
            mean_inv_div = float(np.mean(losses_inv_div[-100:])) if losses_inv_div else 0.0
            cos_tail = [x for x in inv_anchor_cos[-100:] if np.isfinite(x)]
            mean_inv_cos = float(np.mean(cos_tail)) if cos_tail else float("nan")
            mean_patch_evi = float(np.mean(patch_evis[-100:])) if patch_evis else 0.0
            mean_patch_delta = float(np.mean(patch_deltas[-100:])) if patch_deltas else 0.0
            reg_ratio = mean_reg / mean_cls if mean_cls != 0 else 0.0
            logging.info(
                "[train] step %d mean_loss cls=%.4f reg=%.4f reg/cls=%.4f "
                "real=%.4f fake=%.4f osd_orth=%.4f osd_ksv=%.4f "
                "inv=%.4f inv_kd=%.4f inv_div=%.4f inv_cos=%.4f "
                "patch_evi=%.4f patch_d|mean=%.4f",
                batch_idx,
                mean_cls,
                mean_reg,
                reg_ratio,
                mean_real,
                mean_fake,
                mean_orth,
                mean_ksv,
                mean_inv,
                mean_inv_kd,
                mean_inv_div,
                mean_inv_cos,
                mean_patch_evi,
                mean_patch_delta,
            )
            audit_stratum = Counter()
            audit_label_qbin = Counter()
            audit_degrade = Counter()
            audit_q_values = []
            for rec_stratum, rec_label_qbin, rec_degrade, rec_q_values in audit_recent:
                audit_stratum.update(rec_stratum)
                audit_label_qbin.update(rec_label_qbin)
                audit_degrade.update(rec_degrade)
                audit_q_values.extend(rec_q_values)
            n_recent = len(audit_recent)
            if audit_stratum:
                logging.info(
                    "[audit] stratum_count(last_%d_batches)=%s",
                    n_recent,
                    dict(sorted(audit_stratum.items())),
                )
            if audit_label_qbin:
                # key format: y{label}_q{bin}
                payload = {f"y{k[0]}_q{k[1]}": v for k, v in sorted(audit_label_qbin.items())}
                logging.info("[audit] label_qbin_count(last_%d_batches)=%s", n_recent, payload)
            if audit_degrade:
                logging.info(
                    "[audit] degrade_bin_count(last_%d_batches)=%s",
                    n_recent,
                    dict(sorted(audit_degrade.items())),
                )
            if audit_q_values:
                qv = np.asarray(audit_q_values, dtype=float)
                logging.info(
                    "[audit] q_log(last_%d_batches): mean=%.4f std=%.4f min=%.4f max=%.4f p10=%.4f p50=%.4f p90=%.4f",
                    n_recent,
                    float(qv.mean()),
                    float(qv.std()),
                    float(qv.min()),
                    float(qv.max()),
                    float(np.quantile(qv, 0.10)),
                    float(np.quantile(qv, 0.50)),
                    float(np.quantile(qv, 0.90)),
                )

            if "groupdro_q_max" in stats:
                logging.info(
                    "[groupdro] q_max=%.3f q_eff=%.2f q_H=%.3f argmax=%s worst_gid=%s worst_ema=%.4f present=%.2f both=%.2f",
                    stats.get("groupdro_q_max", float("nan")),
                    stats.get("groupdro_q_eff", float("nan")),
                    stats.get("groupdro_q_entropy_norm", float("nan")),
                    stats.get("groupdro_q_argmax", -1),
                    stats.get("groupdro_worst_gid", -1),
                    stats.get("groupdro_worst_ema", float("nan")),
                    stats.get("groupdro_present_frac", float("nan")),
                    stats.get("groupdro_both_class_frac", float("nan")),
                )

    if hasattr(trainer, "flush_grad_accum"):
        flushed = bool(trainer.flush_grad_accum())
        if flushed:
            logging.info("[train] flushed pending gradients at epoch end")

    def _mean(xs):
        if not xs:
            return float("nan")
        return float(np.mean(xs))

    if diag_acc["groupdro_q_max"] or diag_acc["logit_abs_mean"]:
        logging.info(
            "[epoch_diag] groupdro cvar_on=%s trim=%s q_max=%.3f q_H=%.3f q_Hn=%.3f effG=%.2f n_eff=%.2f",
            cvar_enabled_seen,
            cvar_trim_seen,
            _mean(diag_acc["groupdro_q_max"]),
            _mean(diag_acc["groupdro_q_entropy"]),
            _mean(diag_acc["groupdro_q_entropy_norm"]),
            _mean(diag_acc["groupdro_effective_groups"] or diag_acc["groupdro_q_eff"]),
            _mean(diag_acc["groupdro_n_eff_samples"]),
        )
        logging.info(
            "[epoch_diag] cvar lambda_t=%.3f tail_k=%.2f tail_frac=%.3f tail_cov=%.3f cap=%.4f clip_rate=%.3f",
            _mean(diag_acc["groupdro_cvar_lambda_t"]),
            _mean(diag_acc["groupdro_tail_k_mean"]),
            _mean(diag_acc["groupdro_tail_frac_mean"]),
            _mean(diag_acc["groupdro_tail_coverage"]),
            _mean(diag_acc["groupdro_cap_value_mean"]),
            _mean(diag_acc["groupdro_clipped_rate_mean"]),
        )
        logging.info(
            "[epoch_diag] confidence logit_abs=%.4f prob_entropy=%.4f",
            _mean(diag_acc["logit_abs_mean"]),
            _mean(diag_acc["prob_entropy_mean"]),
        )

    return float(np.mean(losses)) if losses else 0.0, global_step


def main():
    opt = SimpleNamespace()
    apply_overrides(opt, OPT_OVERRIDES, PARSER_DEFAULTS)

    if getattr(opt, "no_clear_cache_each_epoch", False):
        opt.clear_cache_each_epoch = False
    if getattr(opt, "no_deterministic", False):
        opt.deterministic = False
    if getattr(opt, "no_osd_regs", False):
        opt.osd_regs = False

    det = bool(getattr(opt, "deterministic", True))
    set_seed(opt.seed, deterministic=det)

    # Determinism vs speed: don't silently override user intent
    torch.backends.cudnn.deterministic = det
    torch.backends.cudnn.benchmark = (not det)
    
    # Prefer new TF32 API when available (avoids deprecation warnings)
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    else:
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    log_dir = _setup_logging(opt)
    config_path = os.path.join(log_dir, "config.json")

    try:
        rt = ensure_cuda_conv_runtime(allow_cudnn_fallback=True)
        if rt.get("cuda_available", False):
            logging.info(
                "[runtime] cuda probe_ok=%s cudnn_enabled=%s fallback_applied=%s",
                rt.get("probe_ok"),
                rt.get("cudnn_enabled"),
                rt.get("fallback_applied"),
            )
    except Exception as e:
        logging.exception("[runtime] CUDA conv probe failed: %s", e)
        raise
    # Keep initialization RNG stream identical to legacy runs:
    # CUDA conv probe creates a temporary Conv2d whose random init consumes torch RNG.
    # Re-seed here so model/data-loader init remains seed-stable.
    set_seed(opt.seed, deterministic=det)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(opt), f, indent=2, sort_keys=True)

    # GroupDRO needs group ids (quality_bin or stratum_id). Those are only present when quality_balance is enabled.
    if bool(getattr(opt, "use_groupdro", False)) and (not bool(getattr(opt, "quality_balance", False))):
        logging.warning("[config] use_groupdro=True but quality_balance=False; forcing quality_balance=True for train.")
        opt.quality_balance = True

    train_loader, val_loader = create_dataloaders(opt)
    trainer = Trainer(opt)

    # Datasets may populate opt.quality_cuts at runtime (train-cutpoints) so val uses fixed bins.
    # Refresh config dump to persist those cutpoints into checkpoints/<exp>/config.json.
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(opt), f, indent=2, sort_keys=True)

    total_steps = max(1, len(train_loader) * int(opt.niter))
    _normalize_steps(opt, total_steps)

    logging.info("[config] total_steps=%d use_amp=%s", total_steps, opt.use_amp)
    eff_batch = int(opt.batch_size) * int(getattr(opt, "grad_accum_steps", 1) or 1)
    logging.info(
        "[config] batch_size=%d grad_accum_steps=%d effective_batch=%d lr=%.6g",
        int(opt.batch_size),
        int(getattr(opt, "grad_accum_steps", 1) or 1),
        int(eff_batch),
        float(getattr(opt, "lr", 0.0)),
    )
    patch_pool_on = getattr(trainer.model, "patch_pool", False)
    logging.info(
        "[config] patch_pool=%s mode=%s tau=%s quality=%s gamma=%s warmup=%s ramp=%s",
        patch_pool_on,
        getattr(opt, "patch_pool_mode", "lse"),
        getattr(opt, "patch_pool_tau", 1.5),
        getattr(opt, "patch_quality", "none"),
        getattr(opt, "patch_pool_gamma", 0.0),
        getattr(opt, "patch_warmup_steps", 0),
        getattr(opt, "patch_ramp_steps", 0),
    )
    logging.info(
        "[config] osd_regs=%s lambda_orth=%s lambda_ksv=%s reg_warmup=%s reg_ramp=%s",
        getattr(opt, "osd_regs", False),
        getattr(opt, "lambda_orth", 0.0),
        getattr(opt, "lambda_ksv", 0.0),
        getattr(opt, "reg_warmup_steps", 0),
        getattr(opt, "reg_ramp_steps", 0),
    )
    logging.info("trainable params: %d", count_trainable_params(trainer.model))

    best_auc = -1.0
    global_step = 0
    early_stopping = None
    if val_loader is not None and int(opt.earlystop_epoch) > 0:
        early_stopping = EarlyStopping(patience=int(opt.earlystop_epoch), delta=0.0, verbose=True)

    log_every = int(opt.loss_freq) if getattr(opt, "loss_freq", None) else 100
    log_every = max(1, log_every)
    audit_window = int(getattr(opt, "audit_window_batches", log_every) or log_every)
    audit_window = max(1, audit_window)
    logging.info("[config] log_every=%d audit_window_batches=%d", log_every, audit_window)

    for epoch in range(int(opt.niter)):
        train_loss, global_step = train_one_epoch(
            trainer,
            train_loader,
            global_step,
            log_every,
            audit_window,
        )
        logging.info(
            "epoch %d train loss: %.6f | trainable params: %d",
            epoch,
            train_loss,
            count_trainable_params(trainer.model),
        )

        if val_loader is not None:
            metrics = validate(
                trainer.model,
                val_loader,
                amp=opt.use_amp,
                seed=opt.seed + 1000 + epoch,
                tta_micro=bool(getattr(opt, "val_micro_tta", False)),
                tta_k=int(getattr(opt, "val_micro_tta_k", 0) or 0),
                tta_seed=int(getattr(opt, "val_micro_tta_seed", 0) or 0),
                tta_blur_max_sig=float(getattr(opt, "val_micro_tta_blur_max_sig", 1.5) or 1.5),
                tta_jpeg_qual=getattr(opt, "val_micro_tta_jpeg_qual", [60, 95]),
                tta_contrast=getattr(opt, "val_micro_tta_contrast", [0.9, 1.1]),
                tta_gamma=getattr(opt, "val_micro_tta_gamma", [0.9, 1.1]),
                tta_resize_jitter=int(getattr(opt, "val_micro_tta_resize_jitter", 0) or 0),
            )
            logging.info(
                "epoch %d val acc=%.4f prec=%.4f rec=%.4f auc=%.4f eer=%.4f ap=%.4f loss=%.6f",
                epoch,
                metrics.get("acc", 0.0),
                metrics.get("precision", 0.0),
                metrics.get("recall", 0.0),
                metrics.get("auc", 0.0),
                metrics.get("eer", 0.0),
                metrics.get("ap", 0.0),
                metrics.get("loss", 0.0),
            )
            logging.info(
                "epoch %d val fpr@tpr95=%.4f precision@recall80=%.4f "
                "fpr_real_bottom10=%.4f fpr_real_bottom20=%.4f "
                "spearman(q,score): overall=%.4f real=%.4f fake=%.4f "
                "worst_quality_bin_auc=%.4f worst_stratum_error=%.4f "
                "worst_real_fpr_over_bins=%.4f worst_fake_fnr_over_bins=%.4f",
                epoch,
                metrics.get("fpr_at_tpr95", float("nan")),
                metrics.get("precision_at_recall80", float("nan")),
                metrics.get("fpr_real_bottom10", float("nan")),
                metrics.get("fpr_real_bottom20", float("nan")),
                metrics.get("spearman_q_score_overall", float("nan")),
                metrics.get("spearman_q_score_real", float("nan")),
                metrics.get("spearman_q_score_fake", float("nan")),
                metrics.get("worst_quality_bin_auc", float("nan")),
                metrics.get("worst_stratum_error", float("nan")),
                metrics.get("worst_real_fpr_over_bins", float("nan")),
                metrics.get("worst_fake_fnr_over_bins", float("nan")),
            )
            logging.info(
                "epoch %d val shortcut: blur_sp_real=%.4f blur_sp_all=%.4f "
                "worstbin_real_fpr1=%.4f gap_real_fpr1=%.4f group_gap_auc=%.4f "
                "tta_micro_mean=%.4f tta_micro_p90=%.4f",
                epoch,
                metrics.get("blur_spearman_real_only", float("nan")),
                metrics.get("blur_spearman_all", float("nan")),
                metrics.get("worstbin_real_fpr1", float("nan")),
                metrics.get("gap_real_fpr1", float("nan")),
                metrics.get("group_gap_auc", float("nan")),
                metrics.get("tta_range_micro_mean", float("nan")),
                metrics.get("tta_range_micro_p90", float("nan")),
            )
            logging.info(
                "epoch %d val calib: nll=%.4f brier=%.4f ece15=%.4f pred_entropy=%.4f tpr@fpr1=%.4f tpr@fpr0.5=%.4f",
                epoch,
                metrics.get("nll", float("nan")),
                metrics.get("brier", float("nan")),
                metrics.get("ece15", float("nan")),
                metrics.get("pred_entropy", float("nan")),
                metrics.get("tpr_at_fpr1", float("nan")),
                metrics.get("tpr_at_fpr0.5", float("nan")),
            )

            if metrics.get("auc", 0.0) > best_auc:
                best_auc = metrics["auc"]
                trainer.save_checkpoint(os.path.join(log_dir, "best_auc.pt"), epoch, best_metric=best_auc)

            if early_stopping is not None:
                early_stopping(metrics.get("auc", 0.0), trainer)
                if early_stopping.early_stop:
                    cont_train = trainer.adjust_learning_rate()
                    if cont_train:
                        logging.info("Learning rate dropped by 10, continue training...")
                        early_stopping = EarlyStopping(patience=int(opt.earlystop_epoch), delta=0.0, verbose=True)
                    else:
                        logging.info("Early stopping.")
                        break

            trainer.model.train()

        trainer.save_checkpoint(os.path.join(log_dir, "last.pt"), epoch, best_metric=best_auc)
        save_every = int(getattr(opt, "save_epoch_freq", 0) or 0)
        need_epoch_ckpt = (save_every > 0 and epoch % save_every == 0) or bool(
            getattr(opt, "submit_each_epoch", False)
        )
        if need_epoch_ckpt:
            trainer.save_checkpoint(os.path.join(log_dir, f"epoch_{epoch}.pt"), epoch, best_metric=best_auc)
            _maybe_submit_epoch(opt, log_dir, epoch)

        trainer.step_scheduler()

        if opt.clear_cache_each_epoch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
