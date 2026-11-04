"""Standard instance-segmentation metrics and measurement comparisons.

Objects are matched one-to-one by intersection over union (IoU) with the Hungarian
algorithm; a match counts as a true positive (TP) at threshold t when its IoU >= t.

* precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = 2 TP / (2 TP + FP + FN)
* average precision as used by Cellpose and the Data Science Bowl: AP = TP / (TP + FP + FN)
* mean matched IoU, mean Dice of the matched pairs
* panoptic quality PQ = SQ x RQ (SQ: mean IoU of the TPs, RQ: F1)
* mAP over IoU thresholds 0.50:0.05:0.95
* splits (one truth object covered by several predictions) and merges (one prediction
  covering several truth objects), each counted when the overlap is >= 50% of the
  smaller object involved
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))


def overlap(truth, pred):
    """(truth ids, pred ids, intersection matrix, truth sizes, pred sizes)."""
    truth = np.asarray(truth).ravel()
    pred = np.asarray(pred).ravel()
    if truth.shape != pred.shape:
        raise ValueError(f"truth {truth.shape} and prediction {pred.shape} differ in size")
    t_ids = np.unique(truth)
    t_ids = t_ids[t_ids != 0]
    p_ids = np.unique(pred)
    p_ids = p_ids[p_ids != 0]
    t_idx = np.searchsorted(t_ids, truth)
    p_idx = np.searchsorted(p_ids, pred)
    both = (truth != 0) & (pred != 0)
    inter = np.zeros((len(t_ids), len(p_ids)), np.int64)
    np.add.at(inter, (t_idx[both], p_idx[both]), 1)
    t_size = np.bincount(t_idx[truth != 0], minlength=len(t_ids))[: len(t_ids)]
    p_size = np.bincount(p_idx[pred != 0], minlength=len(p_ids))[: len(p_ids)]
    return t_ids, p_ids, inter, t_size, p_size


def match(truth, pred, thresholds=THRESHOLDS):
    """Match predicted objects to truth objects; returns a dict of metrics.

    Keys: n_true, n_pred, per-threshold lists (tp, fp, fn, precision, recall, f1, ap),
    'f1@0.5', 'ap@0.5', 'precision@0.5', 'recall@0.5', 'map' (mean AP over the
    thresholds), 'mean_iou' (matched pairs with IoU >= 0.5), 'mean_dice', 'pq', 'sq',
    'splits', 'merges', 'count_error' ((n_pred - n_true) / n_true) and 'pairs'
    (truth id, pred id, IoU) for every Hungarian pair with IoU > 0.
    """
    t_ids, p_ids, inter, t_size, p_size = overlap(truth, pred)
    nt, npred = len(t_ids), len(p_ids)
    union = t_size[:, None] + p_size[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    pairs = []
    if nt and npred:
        rows, cols = linear_sum_assignment(-iou)
        pairs = [(int(t_ids[r]), int(p_ids[c]), float(iou[r, c])) for r, c in zip(rows, cols) if iou[r, c] > 0]
    out = {"n_true": nt, "n_pred": npred, "thresholds": list(thresholds)}
    for key in ("tp", "fp", "fn", "precision", "recall", "f1", "ap"):
        out[key] = []
    for t in thresholds:
        tp = sum(1 for p in pairs if p[2] >= t - 1e-12)
        fp, fn = npred - tp, nt - tp
        out["tp"].append(tp)
        out["fp"].append(fp)
        out["fn"].append(fn)
        out["precision"].append(tp / npred if npred else (1.0 if nt == 0 else 0.0))
        out["recall"].append(tp / nt if nt else 1.0)
        out["f1"].append(2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0)
        out["ap"].append(tp / (tp + fp + fn) if (tp + fp + fn) else 1.0)
    k = list(thresholds).index(0.5) if 0.5 in thresholds else 0
    for key in ("precision", "recall", "f1", "ap"):
        out[f"{key}@0.5"] = out[key][k]
    out["map"] = float(np.mean(out["ap"]))
    good = [p[2] for p in pairs if p[2] >= 0.5]
    out["mean_iou"] = float(np.mean(good)) if good else 0.0
    out["mean_dice"] = float(np.mean([2 * i / (1 + i) for i in good])) if good else 0.0
    out["sq"] = out["mean_iou"]
    out["pq"] = out["sq"] * out["f1@0.5"]
    # Splits / merges: overlap of at least half of the smaller object.
    if nt and npred:
        smaller = np.minimum(t_size[:, None], p_size[None, :])
        strong = inter >= 0.5 * np.maximum(smaller, 1)
        out["splits"] = int(np.sum(strong.sum(axis=1) > 1))
        out["merges"] = int(np.sum(strong.sum(axis=0) > 1))
    else:
        out["splits"] = out["merges"] = 0
    out["count_error"] = (npred - nt) / nt if nt else 0.0
    out["pairs"] = pairs
    return out


def summary(m):
    """The headline numbers of `match` as a flat dict (for tables)."""
    keys = ("n_true", "n_pred", "precision@0.5", "recall@0.5", "f1@0.5", "ap@0.5", "map",
            "mean_iou", "mean_dice", "pq", "splits", "merges", "count_error")
    return {k: m[k] for k in keys}


def relative_error(measured, expected):
    measured = np.asarray(measured, float)
    expected = np.asarray(expected, float)
    return np.abs(measured - expected) / np.maximum(np.abs(expected), 1e-12)


def surface_distance(truth_mask, pred_mask, spacing):
    """Symmetric mean and 95th-percentile distance (µm) between the borders of two masks."""
    from scipy import ndimage as ndi

    def border(m):
        return m & ~ndi.binary_erosion(m)

    a, b = border(np.asarray(truth_mask, bool)), border(np.asarray(pred_mask, bool))
    if not a.any() or not b.any():
        return np.inf, np.inf
    da = ndi.distance_transform_edt(~a, sampling=spacing)
    db = ndi.distance_transform_edt(~b, sampling=spacing)
    d = np.concatenate([db[a], da[b]])
    return float(d.mean()), float(np.percentile(d, 95))
