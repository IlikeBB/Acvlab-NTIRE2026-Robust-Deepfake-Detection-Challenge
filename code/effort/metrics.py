import numpy as np
from sklearn import metrics


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    ap = metrics.average_precision_score(y_true, y_prob)

    pred = (y_prob > 0.5).astype(int)
    acc = (pred == np.clip(y_true, 0, 1)).mean().item()
    precision = metrics.precision_score(y_true, pred, zero_division=0)
    recall = metrics.recall_score(y_true, pred, zero_division=0)

    return {"acc": acc, "auc": auc, "eer": eer, "ap": ap, "precision": precision, "recall": recall}


def fpr_at_tpr(y_true, y_score, tpr_target=0.95):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    idx = int(np.argmin(np.abs(tpr - float(tpr_target))))
    return float(fpr[idx])

def threshold_at_tpr(y_true, y_score, tpr_target=0.95):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    idx = int(np.argmin(np.abs(tpr - float(tpr_target))))
    return float(thresholds[idx])


def precision_at_recall(y_true, y_score, recall_target=0.80):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score, pos_label=1)
    idx = int(np.argmin(np.abs(recall - float(recall_target))))
    return float(precision[idx])


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks


def spearman_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    va = ra - ra.mean()
    vb = rb - rb.mean()
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def slice_fpr_real_bottom(y_true, y_score, q_log, pct=0.2, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    q_log = np.asarray(q_log).astype(float)
    real_mask = y_true == 0
    if real_mask.sum() == 0:
        return float("nan")
    q_real = q_log[real_mask]
    s_real = y_score[real_mask]
    n = len(q_real)
    k = max(1, int(np.ceil(float(pct) * n)))
    idx = np.argsort(q_real)[:k]
    # FPR on real-only slice under a fixed threshold.
    return float((s_real[idx] > float(threshold)).mean())


def per_group_metrics(y_true, y_score, group_id):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    group_id = np.asarray(group_id).astype(int)
    out = {}
    for gid in sorted(np.unique(group_id).tolist()):
        m = group_id == gid
        yt = y_true[m]
        ys = y_score[m]
        if len(yt) == 0:
            continue
        pred = (ys > 0.5).astype(int)
        err = float((pred != yt).mean())
        ap = float(metrics.average_precision_score(yt, ys)) if len(np.unique(yt)) >= 2 else float("nan")
        auc = float(metrics.roc_auc_score(yt, ys)) if len(np.unique(yt)) >= 2 else float("nan")
        fpr_real = float((ys[yt == 0] > 0.5).mean()) if (yt == 0).sum() > 0 else float("nan")
        out[int(gid)] = {"auc": auc, "ap": ap, "error": err, "fpr_real": fpr_real, "count": int(len(yt))}
    return out


def worst_real_fpr_over_bins(y_true, y_score, quality_bin, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    quality_bin = np.asarray(quality_bin).astype(int)
    vals = []
    detail = {}
    for b in sorted(np.unique(quality_bin).tolist()):
        m = quality_bin == b
        real = m & (y_true == 0)
        if real.sum() == 0:
            detail[int(b)] = float("nan")
            continue
        v = float((y_score[real] > float(threshold)).mean())
        detail[int(b)] = v
        vals.append(v)
    return (float(max(vals)) if vals else float("nan"), detail)


def worst_fake_fnr_over_bins(y_true, y_score, quality_bin, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    quality_bin = np.asarray(quality_bin).astype(int)
    vals = []
    detail = {}
    for b in sorted(np.unique(quality_bin).tolist()):
        m = quality_bin == b
        fake = m & (y_true == 1)
        if fake.sum() == 0:
            detail[int(b)] = float("nan")
            continue
        # FNR: fake predicted as real.
        v = float((y_score[fake] <= float(threshold)).mean())
        detail[int(b)] = v
        vals.append(v)
    return (float(max(vals)) if vals else float("nan"), detail)


# ---------------------------------------------------------------------------
# Calibration / confidence metrics (binary)
# ---------------------------------------------------------------------------

def nll_binary(y_true, y_prob):
    """Negative log-likelihood (log loss) for binary probability outputs."""
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1.0 - 1e-6)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())


def brier_score(y_true, y_prob):
    """Brier score for binary probability outputs."""
    y_true = np.asarray(y_true).astype(float)
    p = np.asarray(y_prob).astype(float)
    return float(((p - y_true) ** 2).mean())


def pred_entropy_binary(y_prob):
    """Mean predictive entropy for binary probability outputs."""
    p = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1.0 - 1e-6)
    return float((-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean())


def ece_binary(y_true, y_prob, n_bins=15):
    """Expected calibration error (ECE) using confidence bins on predicted class."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)

    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)  # confidence of predicted class
    acc = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(int(n_bins)):
        lo, hi = float(bins[i]), float(bins[i + 1])
        m = (conf >= lo) & (conf < hi) if i < int(n_bins) - 1 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += float(np.abs(acc[m].mean() - conf[m].mean()) * m.mean())
    return float(ece)


def tpr_at_fpr(y_true, y_score, fpr_target=0.01):
    """TPR at a target FPR operating point."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    idx = int(np.argmin(np.abs(fpr - float(fpr_target))))
    return float(tpr[idx])


def tnr_at_fnr(y_true, y_score, fnr_target=0.01):
    """TNR at a target FNR operating point (fake is positive class)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    target_tpr = 1.0 - float(fnr_target)  # FNR = 1 - TPR
    idx = int(np.argmin(np.abs(tpr - target_tpr)))
    return float(1.0 - fpr[idx])  # TNR = 1 - FPR
