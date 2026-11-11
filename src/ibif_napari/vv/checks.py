"""One check per requirement (see requirements.py).

Every check runs the plugin's own code (the functions its panels call) on the synthetic
scenes and compares the results with the truth or with an independent calculation.
A check returns a `Result`: PASS / FAIL per acceptance criterion, the numbers behind
them, tables and figures, and a short plain-language account of the evidence.
"""
from __future__ import annotations

import io
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import metrics, synth
from .requirements import BY_ID, LIMITS, REQUIREMENTS

PASS, FAIL, NOT_RUN, ERROR = "PASS", "FAIL", "NOT RUN", "ERROR"
CHECKS = {}


def check(req_id):
    def register(func):
        CHECKS[req_id] = func
        return func
    return register


# ---- results ---------------------------------------------------------------------------------


class Result:
    def __init__(self, req):
        self.req = req
        self.status = None
        self.criteria = []  # {name, measured, op, limit, passed, text}
        self.evidence = []  # sentences
        self.metrics = {}
        self.tables = {}  # name -> DataFrame
        self.figures = []  # (caption, png bytes)
        self.notes = []
        self.seconds = 0.0

    def check(self, name, measured, op, limit, fmt="{:.3g}"):
        measured_v = bool(measured) if op == "is" else float(measured)
        if op == ">=":
            passed = measured_v >= limit - 1e-12
        elif op == "<=":
            passed = measured_v <= limit + 1e-12
        elif op == "==":
            passed = measured_v == limit
        elif op == "is":
            passed = measured_v is bool(limit)
        else:
            raise ValueError(op)
        shown = ("yes" if measured_v else "no") if op == "is" else fmt.format(measured_v)
        want = ("yes" if limit else "no") if op == "is" else f"{op} {fmt.format(limit)}"
        self.criteria.append(dict(name=name, measured=measured_v, shown=shown, op=op,
                                  limit=limit if op != "is" else bool(limit), want=want,
                                  passed=bool(passed)))
        return passed

    def not_run(self, why):
        self.status = NOT_RUN
        self.notes.append(why)

    def finish(self):
        if self.status is None:
            self.status = PASS if self.criteria and all(c["passed"] for c in self.criteria) else FAIL
        return self

    def to_json(self):
        return {"id": self.req.id, "status": self.status, "criteria": self.criteria,
                "evidence": self.evidence, "metrics": _plain(self.metrics), "notes": self.notes,
                "seconds": round(self.seconds, 2), "tables": list(self.tables),
                "figures": [c for c, _ in self.figures]}


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---- context ---------------------------------------------------------------------------------


class Context:
    """What the checks share: the truth scenes, the back-end and its cached results."""

    def __init__(self, truth_dir, out_dir, backend, limits=None, log=print):
        self.truth_dir = Path(truth_dir)
        self.out = Path(out_dir)
        self.backend = backend
        self.limits = {**LIMITS, **(limits or {})}
        self.log = log
        self._scenes = {}
        self._pred = {}
        self.timings = {}
        self.backend_ok, self.backend_why = backend.available()

    def scene(self, name):
        if name not in self._scenes:
            folder = self.truth_dir / name
            if not (folder / "scene.json").exists():
                raise FileNotFoundError(f"scene {name!r} not found in {self.truth_dir} (run 'generate' first)")
            self._scenes[name] = synth.load(folder)
        return self._scenes[name]

    def predict(self, name, kind="nuclei", do_3D=True):
        """The back-end's segmentation of a scene (cached, and saved as a TIFF)."""
        key = (name, kind, do_3D)
        if key not in self._pred:
            sc = self.scene(name)
            if kind == "nuclei":
                image, nuc, truth = sc.acquired["nuclei"], None, sc.nuclei
            else:
                image, nuc, truth = sc.acquired["cytoplasm"], sc.acquired["nuclei"], sc.cells
            self.log(f"  segmenting {name} ({kind}{'' if do_3D or sc.spec.ndim == 2 else ', per plane + stitch'}) "
                     f"with {self.backend.name}…")
            t0 = time.perf_counter()
            pred = np.asarray(self.backend.segment(image, nuclei=nuc, spacing=sc.spec.spacing,
                                                   do_3D=do_3D, truth=truth, kind=kind))
            seconds = time.perf_counter() - t0
            import tifffile

            folder = self.out / "predictions"
            folder.mkdir(parents=True, exist_ok=True)
            suffix = "" if do_3D or sc.spec.ndim == 2 else "_stitched"
            tifffile.imwrite(folder / f"{name}_{kind}{suffix}.tif", pred.astype(np.uint32))
            self.timings[f"{name} {kind}{suffix}"] = seconds
            self._pred[key] = (pred, seconds)
        return self._pred[key]


def run_checks(ctx, only=None):
    results = []
    for req in REQUIREMENTS:
        if only and req.id not in only:
            continue
        res = Result(req)
        ctx.log(f"{req.id} {req.title}")
        t0 = time.perf_counter()
        try:
            missing = _missing(ctx, req)
            if missing:
                res.not_run(missing)
            else:
                CHECKS[req.id](ctx, res)
        except Exception as e:  # a crash is a failed check, with the traceback kept
            res.status = ERROR
            res.notes.append(f"{type(e).__name__}: {e}")
            res.notes.append(traceback.format_exc())
        res.seconds = time.perf_counter() - t0
        res.finish()
        ctx.log(f"  -> {res.status}")
        results.append(res)
    return results


def _missing(ctx, req):
    from importlib.util import find_spec

    for need in req.needs:
        if need == "segmentation" and not ctx.backend_ok:
            return f"segmentation back-end '{ctx.backend.name}' not available: {ctx.backend_why}"
        if need == "napari" and find_spec("napari") is None:
            return "napari is not installed"
        if need == "czi" and (find_spec("pylibCZIrw") is None or find_spec("bioio_czi") is None):
            return "writing/reading CZI needs pylibCZIrw and bioio-czi"
        if need == "lif" and find_spec("bioio_lif") is None:
            return "reading LIF needs bioio-lif"
    return None


# ---- figures -----------------------------------------------------------------------------------


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf.getvalue()


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _plane(arr, truth):
    """The Z plane with the most truth objects (2D arrays unchanged)."""
    if arr.ndim == 2:
        return arr
    z = int(np.argmax([len(np.unique(p)) for p in truth]))
    return arr[z]


def seg_figure(image, truth, pred, title):
    from skimage.segmentation import find_boundaries

    plt = _plt()
    img, t, p = _plane(image, truth), _plane(truth, truth), _plane(pred, truth)
    lo, hi = np.percentile(img, (1, 99.7))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for ax in axes:
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_axis_off()
    axes[0].set_title("image")
    rgba = np.zeros(img.shape + (4,))
    rgba[find_boundaries(t)] = (0.1, 0.9, 0.2, 1)
    axes[1].imshow(rgba)
    axes[1].set_title("truth (green)")
    rgba = np.zeros(img.shape + (4,))
    rgba[find_boundaries(t)] = (0.1, 0.9, 0.2, 0.8)
    rgba[find_boundaries(p)] = (1.0, 0.1, 0.9, 1)
    axes[2].imshow(rgba)
    axes[2].set_title("result (magenta) on truth")
    fig.suptitle(title + (" — middle plane" if image.ndim == 3 else ""), fontsize=10)
    return _png(fig)


def curve_figure(m, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(m["thresholds"], m["f1"], "o-", label="F1")
    ax.plot(m["thresholds"], m["ap"], "s-", label="AP")
    ax.axvline(0.5, color="0.7", lw=0.8)
    ax.set_xlabel("IoU threshold")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    return _png(fig)


def scatter_figure(expected, measured, xlabel, ylabel, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(expected, measured, s=14)
    lo = float(min(np.min(expected), np.min(measured)))
    hi = float(max(np.max(expected), np.max(measured)))
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    return _png(fig)


# ---- helpers --------------------------------------------------------------------------------------


def _renumber(labels, seed=0):
    ids = np.unique(labels)
    ids = ids[ids != 0]
    lut = np.zeros(int(labels.max()) + 1, np.int64)
    lut[ids] = np.random.default_rng(seed).permutation(len(ids)) + 1
    return lut[labels]


def _matching_table(m):
    return pd.DataFrame({k: m[k] for k in ("thresholds", "tp", "fp", "fn", "precision", "recall", "f1", "ap")})


def _seg_account(res, m, what, backend):
    tp, fp, fn = m["tp"][0], m["fp"][0], m["fn"][0]
    res.evidence.append(
        f"{backend} found {tp} of the {m['n_true']} true {what} (recall {m['recall@0.5']:.2f}) "
        f"with {fp} extra object(s) (precision {m['precision@0.5']:.2f}); F1 {m['f1@0.5']:.3f}, "
        f"AP {m['ap@0.5']:.3f} at IoU 0.5, mAP(0.5:0.95) {m['map']:.3f}.")
    res.evidence.append(
        f"The matched {what} overlap the truth with a mean IoU of {m['mean_iou']:.3f} "
        f"(Dice {m['mean_dice']:.3f}); {m['splits']} split and {m['merges']} merged; "
        f"{fn} missed.")


def _stub(data, spacing, origin=None, kind="image"):
    """What the panel's region functions read from a napari layer."""
    n = len(spacing)
    return SimpleNamespace(data=data, scale=np.asarray(spacing, float),
                           translate=np.asarray(origin if origin is not None else (0.0,) * n, float),
                           _type_string=kind)


def _distances(labels, spacing):
    """Distance (µm) of every pixel to each object: {label: array}."""
    out = {}
    for i in np.unique(labels):
        if i == 0:
            continue
        out[int(i)] = ndi.distance_transform_edt(labels != i, sampling=spacing)
    return out


def _nearest(dist):
    ids = np.array(list(dist))
    stack = np.stack([dist[i] for i in ids])
    order = np.argsort(stack, axis=0)
    best = np.take_along_axis(stack, order[:1], 0)[0]
    second = np.take_along_axis(stack, order[1:2], 0)[0] if len(ids) > 1 else np.full(best.shape, np.inf)
    return ids[order[0]], best, np.abs(second - best) < 1e-6


# ==================================================================================================
# Checks
# ==================================================================================================


@check("VV-H-01")
def _harness(ctx, res):
    sc = ctx.scene("nuclei_2d")
    t = sc.nuclei.astype(np.int64)
    perfect = metrics.match(t, _renumber(t))
    res.check("perfect: F1", perfect["f1@0.5"], "==", 1.0)
    res.check("perfect: AP over 0.5-0.95 (mAP)", perfect["map"], "==", 1.0)
    res.check("perfect: mean IoU", perfect["mean_iou"], "==", 1.0)
    res.check("perfect: splits + merges", perfect["splits"] + perfect["merges"], "==", 0, "{:.0f}")

    ids = np.unique(t)[1:]
    sizes = {i: int((t == i).sum()) for i in ids}
    by_size = sorted(ids, key=lambda i: sizes[i])
    gone, small, big, split = by_size[0], by_size[1], by_size[-1], by_size[-2]
    pred = _renumber(t, seed=1)
    lut = {int(i): int(pred[t == i][0]) for i in ids}
    pred[t == gone] = 0  # missed
    pred[t == small] = lut[int(big)]  # merged into the biggest (it keeps IoU >= 0.5)
    ys = np.nonzero(t == split)
    cut = np.percentile(ys[1], 30)  # split 30 / 70 along X
    part = (t == split) & (np.arange(t.shape[1])[None, :] < cut)
    pred[part] = pred.max() + 1
    dist = ndi.distance_transform_edt(t == 0)
    cy, cx = np.unravel_index(np.argmax(dist), t.shape)
    yy, xx = np.ogrid[: t.shape[0], : t.shape[1]]
    pred[(yy - cy) ** 2 + (xx - cx) ** 2 <= 16] = pred.max() + 1  # a false object in the background
    m = metrics.match(t, pred)
    n = len(ids)
    expect = {"recall@0.5": (n - 2) / n, "precision@0.5": (n - 2) / n, "ap@0.5": (n - 2) / (n + 2)}
    for key, value in expect.items():
        res.check(f"defects: {key} (expected {value:.4f})", abs(m[key] - value), "<=", 1e-12, "{:.1e}")
    res.check("defects: splits", m["splits"], "==", 1, "{:.0f}")
    res.check("defects: merges", m["merges"], "==", 1, "{:.0f}")
    res.metrics = {"perfect": metrics.summary(perfect), "defects": metrics.summary(m)}
    res.evidence.append(
        f"A renumbered copy of the {n} true nuclei scores F1 = mAP = mean IoU = 1. With one nucleus "
        f"removed, two merged, one split 30/70 and one false object added, the metrics give "
        f"recall = precision = {(n - 2) / n:.4f} and AP = {(n - 2) / (n + 2):.4f} exactly as defined, "
        f"and count 1 split and 1 merge: the suite measures what it claims to measure.")


@check("VV-SEG-01")
def _seg2d(ctx, res):
    _segmentation(ctx, res, "nuclei_2d", "nuclei", True,
                  [("F1 at IoU 0.5", "f1@0.5", ">=", "seg2d_f1"),
                   ("mean IoU of matched nuclei", "mean_iou", ">=", "seg2d_iou"),
                   ("|count error|", "abs_count", "<=", "seg2d_count")])


@check("VV-SEG-02")
def _seg3d(ctx, res):
    _segmentation(ctx, res, "nuclei_3d", "nuclei", True,
                  [("F1 at IoU 0.5", "f1@0.5", ">=", "seg3d_f1"),
                   ("mean IoU of matched nuclei", "mean_iou", ">=", "seg3d_iou")])


@check("VV-SEG-03")
def _stitch(ctx, res):
    _segmentation(ctx, res, "nuclei_3d", "nuclei", False,
                  [("F1 at IoU 0.5", "f1@0.5", ">=", "stitch_f1")])


@check("VV-SEG-04")
def _cells(ctx, res):
    _segmentation(ctx, res, "cells_2d", "cells", True,
                  [("F1 at IoU 0.5", "f1@0.5", ">=", "cells_f1"),
                   ("mean IoU of matched cells", "mean_iou", ">=", "cells_iou")])


def _segmentation(ctx, res, name, kind, do_3D, criteria):
    sc = ctx.scene(name)
    pred, seconds = ctx.predict(name, kind, do_3D)
    truth = sc.nuclei if kind == "nuclei" else sc.cells
    m = metrics.match(truth, pred)
    m["abs_count"] = abs(m["count_error"])
    for label, key, op, limit in criteria:
        res.check(label, m[key], op, ctx.limits[limit])
    res.metrics = {**metrics.summary(m), "seconds": seconds}
    res.tables["matching"] = _matching_table(m)
    _seg_account(res, m, kind, ctx.backend.name)
    res.evidence.append(f"Segmentation took {seconds:.1f} s.")
    image = sc.acquired["cytoplasm" if kind == "cells" else "nuclei"]
    res.figures.append((f"{name}: truth and result outlines", seg_figure(image, truth, pred, name)))
    res.figures.append((f"{name}: F1 and AP against the IoU threshold", curve_figure(m, name)))


@check("VV-SEG-05")
def _snr(ctx, res):
    rows = []
    for snr, limit in ((20, "snr_high_f1"), (8, "snr_high_f1"), (4, "snr_low_f1"), (2, None)):
        name = f"snr_{snr}"
        sc = ctx.scene(name)
        pred, seconds = ctx.predict(name)
        m = metrics.match(sc.nuclei, pred)
        rows.append({"SNR": snr, **metrics.summary(m), "seconds": seconds})
        if limit:
            res.check(f"F1 at IoU 0.5, SNR {snr}", m["f1@0.5"], ">=", ctx.limits[limit])
    table = pd.DataFrame(rows)
    res.tables["snr"] = table
    res.metrics = {f"snr_{r['SNR']}": {k: r[k] for k in ("f1@0.5", "mean_iou")} for r in rows}
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(table["SNR"], table["f1@0.5"], "o-", label="F1 at IoU 0.5")
    ax.plot(table["SNR"], table["mean_iou"], "s-", label="mean matched IoU")
    ax.set_xscale("log")
    ax.set_xticks(table["SNR"], [str(s) for s in table["SNR"]])
    ax.set_xlabel("signal-to-noise ratio")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    res.figures.append(("Segmentation quality against noise", _png(fig)))
    sc = ctx.scene("snr_4")
    res.figures.append(("SNR 4: truth and result outlines",
                        seg_figure(sc.acquired["nuclei"], sc.nuclei, ctx.predict("snr_4")[0], "snr_4")))
    res.evidence.append(
        "F1 at IoU 0.5 by SNR: " + ", ".join(f"SNR {r['SNR']}: {r['f1@0.5']:.3f}" for r in rows) + ".")


@check("VV-SEG-07")
def _classic_steps(ctx, res):
    from skimage.filters import threshold_otsu

    from .. import _classic

    rows, worst = [], {"thr": 0.0, "cc": True, "cover": True, "within": True, "numbers": True, "manual": True}
    for name in ("nuclei_2d", "nuclei_3d"):
        sc = ctx.scene(name)
        image = np.asarray(sc.acquired["nuclei"], float)
        sp = np.asarray(sc.spec.spacing, float)
        rel = sp / sp[-1]
        s = _classic.Settings()
        # Independent: the same normalisation, smoothing and Otsu, with numpy / scipy.
        lo, hi = np.percentile(image, (0.1, 99.9))
        work = np.clip((image - lo) / (hi - lo), 0, 1)
        work = ndi.gaussian_filter(work.astype(np.float32), sigma=list(s.smooth / rel))
        expected = float(threshold_otsu(work)) * (hi - lo) + lo
        fg, info, _ = _classic.foreground(image, s, rel)
        thr_err = abs(info["threshold"] - expected) / abs(expected)
        worst["thr"] = max(worst["thr"], thr_err)
        # Connected components of the same foreground.
        cc, _ = _classic.segment(image, sp, split="none")
        ref, _ = ndi.label(fg, structure=ndi.generate_binary_structure(fg.ndim, 1))
        sizes = np.bincount(ref.ravel())
        ref = np.where(sizes[ref] >= s.min_size, ref, 0)
        same_cc = _same_partition(cc, ref)
        worst["cc"] &= same_cc
        # The watershed: within the foreground, each object inside one component.
        ws, _ = _classic.segment(image, sp)
        kept = np.zeros(fg.shape, bool)
        cover = bool(np.all(ws[~fg] == 0))
        # Pixels lost only from pieces the minimum size removed.
        lost = fg & (ws == 0)
        if lost.any():
            pieces, _ = ndi.label(lost)
            cover &= bool(np.all(np.bincount(pieces.ravel())[1:] < s.min_size))
        kept |= ws > 0
        pairs = np.unique(np.stack([ws[kept], ref[kept]]), axis=1)
        within = len(np.unique(pairs[0])) == pairs.shape[1]
        n = int(ws.max())
        numbers = set(np.unique(ws)) == set(range(n + 1)) and \
            (np.bincount(ws.ravel())[1:] >= s.min_size).all()
        worst["cover"] &= cover
        worst["within"] &= within
        worst["numbers"] &= numbers
        # Manual threshold, nothing else: exactly image > value.
        value = float(np.percentile(image, 97))
        man, _ = _classic.segment(image, sp, threshold="manual", value=value, smooth=0, fill_holes=False,
                                  min_size=0, split="none")
        ref_man, _ = ndi.label(image > value, structure=ndi.generate_binary_structure(image.ndim, 1))
        same_man = _same_partition(man, ref_man)
        worst["manual"] &= same_man
        rows.append({"scene": name, "threshold": info["threshold"], "independent Otsu": expected,
                     "relative error": thr_err, "components identical": same_cc,
                     "objects (watershed)": n, "objects (components)": int(cc.max()),
                     "within the foreground": cover, "within one component": within,
                     "numbered 1..n": numbers, "manual = image > value": same_man})
    res.check("threshold vs independent Otsu (relative error)", worst["thr"], "<=", ctx.limits["exact_rel"], "{:.2g}")
    res.check("connected components identical", worst["cc"], "is", True)
    res.check("watershed objects cover the foreground", worst["cover"], "is", True)
    res.check("each watershed object within one component", worst["within"], "is", True)
    res.check("numbered 1..n, none below the minimum size", worst["numbers"], "is", True)
    res.check("manual threshold = components of image > value", worst["manual"], "is", True)
    res.tables["steps"] = pd.DataFrame(rows)
    res.evidence.append(
        "On both scenes the threshold equals an independent Otsu of the same processed image, "
        "'Split: none' is exactly SciPy's connected-components labelling of the same foreground, "
        "and the watershed only divides that foreground: "
        + "; ".join(f"{r['scene']}: {r['objects (components)']} components -> {r['objects (watershed)']} objects"
                    for r in rows) + ".")


def _same_partition(a, b):
    """True if two label images are the same objects (numbering aside)."""
    if a.shape != b.shape or not np.array_equal(a > 0, b > 0):
        return False
    both = a > 0
    if not both.any():
        return True
    pairs = np.unique(np.stack([a[both], b[both]]), axis=1)
    return len(np.unique(pairs[0])) == len(np.unique(pairs[1])) == pairs.shape[1]


@check("VV-SEG-08")
def _classic_nuclei(ctx, res):
    from .. import _classic

    rows = []
    cases = (("nuclei_2d", "2D", _classic.segment, "classic2d_f1", "classic2d_iou"),
             ("snr_4", "SNR 4", _classic.segment, "classic_snr4_f1", None),
             ("nuclei_3d", "3D", _classic.segment, "classic3d_f1", "classic3d_iou"),
             ("nuclei_3d", "3D by planes + stitch", _classic.segment_planes, "classic3d_f1", None))
    for name, label, func, f1_limit, iou_limit in cases:
        sc = ctx.scene(name)
        t0 = time.perf_counter()
        pred, info = func(sc.acquired["nuclei"], sc.spec.spacing)
        seconds = time.perf_counter() - t0
        m = metrics.match(sc.nuclei, pred)
        res.check(f"{label}: F1 at IoU 0.5", m["f1@0.5"], ">=", ctx.limits[f1_limit])
        if iou_limit:
            res.check(f"{label}: mean IoU of matched nuclei", m["mean_iou"], ">=", ctx.limits[iou_limit])
        rows.append({"case": label, "scene": name, **metrics.summary(m), "seconds": seconds,
                     "threshold": info.get("threshold")})
        if label in ("2D", "3D"):
            _seg_account(res, m, f"nuclei ({label})", "Threshold + watershed")
            res.figures.append((f"Threshold + watershed, {name}: truth and result outlines",
                                seg_figure(sc.acquired["nuclei"], sc.nuclei, pred, name)))
    res.tables["classic"] = pd.DataFrame(rows)
    res.metrics = {r["case"]: {k: r[k] for k in ("f1@0.5", "mean_iou", "seconds")} for r in rows}
    res.evidence.append("Times on this computer's CPU: " + ", ".join(
        f"{r['case']} {r['seconds']:.2f} s" for r in rows) + ".")


@check("VV-SEG-09")
def _touching(ctx, res):
    from .. import _classic

    rows = []
    for name in ("touching_2d", "touching_3d"):
        sc = ctx.scene(name)
        _, groups = ndi.label(sc.nuclei > 0)
        n_true = int(len(np.unique(sc.nuclei)) - 1)
        out = {}
        for split in ("distance", "none"):
            pred, _ = _classic.segment(sc.acquired["nuclei"], sc.spec.spacing, split=split)
            out[split] = (pred, metrics.match(sc.nuclei, pred))
        ws, cc = out["distance"][1], out["none"][1]
        dim = name.split("_")[1].upper()
        res.check(f"{dim}: F1 at IoU 0.5 with the watershed", ws["f1@0.5"], ">=", ctx.limits["touch_f1"])
        res.check(f"{dim}: F1 gained over connected components", ws["f1@0.5"] - cc["f1@0.5"], ">=",
                  ctx.limits["touch_gain"])
        res.check(f"{dim}: merges, watershed minus components", ws["merges"] - cc["merges"], "<=", -1, "{:.0f}")
        rows.append({"scene": name, "true nuclei": n_true, "touching contacts": n_true - groups,
                     "F1 watershed": ws["f1@0.5"], "F1 components": cc["f1@0.5"],
                     "merges watershed": ws["merges"], "merges components": cc["merges"],
                     "mean IoU watershed": ws["mean_iou"]})
        res.figures.append((f"{name}: truth and watershed outlines",
                            seg_figure(sc.acquired["nuclei"], sc.nuclei, out["distance"][0], name)))
        res.figures.append((f"{name}: truth and connected components (no split)",
                            seg_figure(sc.acquired["nuclei"], sc.nuclei, out["none"][0], name + " (components)")))
    res.tables["touching"] = pd.DataFrame(rows)
    res.metrics = {r["scene"]: {k: v for k, v in r.items() if k != "scene"} for r in rows}
    res.evidence.append("; ".join(
        f"{r['scene']}: {r['touching contacts']} contact(s) between nuclei; connected components merge "
        f"{r['merges components']} (F1 {r['F1 components']:.3f}), the watershed {r['merges watershed']} "
        f"(F1 {r['F1 watershed']:.3f})" for r in rows) + ".")


@check("VV-SEG-06")
def _region(ctx, res):
    from .._region import keep_inside, painted_mask, region_crop

    sc = ctx.scene("nuclei_2d")
    sp = sc.spec.spacing
    image = sc.acquired["nuclei"]
    ny, nx = image.shape
    yy, xx = np.ogrid[:ny, :nx]
    radius = min(ny, nx) * 0.3
    disc = ((yy - ny * 0.45) ** 2 + (xx - nx * 0.55) ** 2 <= radius ** 2).astype(np.uint8)
    img_layer = _stub(image, sp)
    region = _stub(disc, sp, kind="labels")
    crop = region_crop(img_layer, "YX", region, None)
    rows, cols = np.nonzero(disc)
    expected = {"Y": (rows.min(), rows.max() + 1), "X": (cols.min(), cols.max() + 1)}
    res.check("image read = region bounding box", crop == {k: tuple(map(int, v)) for k, v in expected.items()},
              "is", True)
    (y0, y1), (x0, x1) = crop["Y"], crop["X"]
    vol = image[y0:y1, x0:x1]
    t0 = time.perf_counter()
    masks = np.asarray(ctx.backend.segment(vol, spacing=sp, truth=sc.nuclei[y0:y1, x0:x1]))
    seconds = time.perf_counter() - t0
    mask = painted_mask(img_layer, region, crop)
    kept = keep_inside(masks, mask)
    # Placed back on the full image the way the panel places it (translate = crop start).
    full = np.zeros(image.shape, kept.dtype)
    full[y0:y1, x0:x1] = kept
    inside = disc.astype(bool)

    def mostly_inside(labels):
        ids = np.unique(labels)
        ids = ids[ids != 0]
        total = np.bincount(labels.ravel(), minlength=labels.max() + 1)
        within = np.bincount(labels[inside], minlength=labels.max() + 1)
        return ids, within[ids] / np.maximum(total[ids], 1)

    ids, frac = mostly_inside(full)
    res.check("objects mostly outside the painted area", int(np.sum(frac < 0.5)), "==", 0, "{:.0f}")
    # Truth the run can see: the true nuclei mostly inside the disc, clipped to the box.
    t_box = np.zeros_like(sc.nuclei)
    t_box[y0:y1, x0:x1] = sc.nuclei[y0:y1, x0:x1]
    t_ids, t_frac = mostly_inside(sc.nuclei)
    keep_t = t_ids[t_frac >= 0.5]
    truth_in = np.where(np.isin(t_box, keep_t), t_box, 0)
    m = metrics.match(truth_in, full)
    whole = metrics.match(sc.nuclei, ctx.predict("nuclei_2d")[0])
    res.check("F1 drop from the whole-image run", whole["f1@0.5"] - m["f1@0.5"], "<=",
              ctx.limits["region_f1_drop"])
    res.metrics = {"region": metrics.summary(m), "whole image": metrics.summary(whole),
                   "crop": {k: list(v) for k, v in crop.items()}, "seconds": seconds}
    frac_read = vol.size / image.size
    res.evidence.append(
        f"The run read a {y1 - y0}×{x1 - x0} box ({frac_read:.0%} of the image), the painted disc's "
        f"bounding box. It kept {len(ids)} objects, none mostly outside the disc, and found "
        f"{m['tp'][0]} of the {m['n_true']} true nuclei inside the disc (F1 {m['f1@0.5']:.3f}; "
        f"whole image {whole['f1@0.5']:.3f}).")
    plt = _plt()
    from skimage.segmentation import find_boundaries

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    lo, hi = np.percentile(image, (1, 99.7))
    ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
    rgba = np.zeros(image.shape + (4,))
    rgba[find_boundaries(disc)] = (1, 0.8, 0, 1)
    rgba[find_boundaries(sc.nuclei)] = (0.1, 0.9, 0.2, 0.7)
    rgba[find_boundaries(full)] = (1, 0.1, 0.9, 1)
    ax.imshow(rgba)
    ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="cyan", lw=1))
    ax.set_axis_off()
    ax.set_title("painted area (yellow), box read (cyan), truth (green), result (magenta)", fontsize=8)
    res.figures.append(("Region run", _png(fig)))


@check("VV-SEC-01")
def _expand(ctx, res):
    from .. import _secondary

    sc = ctx.scene("cells_3d")
    sp = sc.spec.spacing
    seeds = sc.nuclei
    d = 2.0
    out = _secondary.expand(seeds, d, sp)
    dist = _distances(seeds, sp)
    near, best, tie = _nearest(dist)
    assigned = out > 0
    own = np.zeros(out.shape)
    for i, di in dist.items():
        own[out == i] = di[out == i]
    res.check("pixels farther than the distance from their object", int(np.sum(assigned & (own > d + 1e-9))),
              "==", 0, "{:.0f}")
    res.check("pixels within the distance left unassigned", int(np.sum((best <= d - 1e-9) & ~assigned)),
              "==", 0, "{:.0f}")
    res.check("pixels given to a farther object (ties excluded)",
              int(np.sum(assigned & ~tie & (out != near))), "==", 0, "{:.0f}")
    res.check("object pixels changed", int(np.sum((seeds > 0) & (out != seeds))), "==", 0, "{:.0f}")
    grown = np.bincount(out.ravel())[1:].sum() / max(np.bincount(seeds.ravel())[1:].sum(), 1)
    res.metrics = {"distance_um": d, "spacing_um": list(sp), "size_ratio": grown}
    res.evidence.append(
        f"Each truth nucleus of cells_3d (voxels {sp[0]}×{sp[1]}×{sp[2]} µm) was expanded by {d} µm. "
        f"An independent distance map per object confirms every added voxel is within {d} µm of its "
        f"own nucleus, every voxel within {d} µm of a nucleus was assigned, contested voxels went to the "
        f"nearer nucleus, and the nuclei were unchanged (total size ×{grown:.2f}).")


@check("VV-SEC-02")
def _ring(ctx, res):
    from .. import _secondary

    sc = ctx.scene("cells_3d")
    sp = sc.spec.spacing
    seeds = sc.nuclei
    gap, width, rim_w = 1.0, 1.5, 1.0
    ring = _secondary.ring(seeds, width, gap, sp)
    dist = _distances(seeds, sp)
    near, best, tie = _nearest(dist)
    own = np.zeros(ring.shape)
    for i, di in dist.items():
        own[ring == i] = di[ring == i]
    in_ring = ring > 0
    res.check("ring pixels outside gap..gap+width",
              int(np.sum(in_ring & ((own <= gap - 1e-9) | (own > gap + width + 1e-9)))), "==", 0, "{:.0f}")
    want = (seeds == 0) & (best > gap + 1e-9) & (best <= gap + width - 1e-9)
    res.check("pixels in the band left out", int(np.sum(want & ~in_ring)), "==", 0, "{:.0f}")
    res.check("ring pixels given to a farther object (ties excluded)",
              int(np.sum(in_ring & ~tie & (ring != near))), "==", 0, "{:.0f}")
    rim = _secondary.ring(seeds, rim_w, spacing=sp, inside=True)
    # Independent rim: object pixels whose distance to the object's edge pixels (pixels with
    # a face neighbour outside the object) is below the width.
    edge = np.zeros(seeds.shape, bool)
    for ax in range(seeds.ndim):
        for shift in (1, -1):
            nb = np.roll(seeds, shift, axis=ax)
            sl = [slice(None)] * seeds.ndim
            sl[ax] = 0 if shift == 1 else -1
            nb[tuple(sl)] = seeds[tuple(sl)]
            edge |= (seeds > 0) & (nb != seeds)
    d_edge = ndi.distance_transform_edt(~edge, sampling=sp)
    expected_rim = np.where((seeds > 0) & (d_edge < rim_w), seeds, 0)
    res.check("rim pixels differing from the independent rim", int(np.sum(rim != expected_rim)),
              "==", 0, "{:.0f}")
    res.check("rim pixels outside their object", int(np.sum((rim > 0) & (rim != seeds))), "==", 0, "{:.0f}")
    # Mean thickness of the ring, from its volume and the nuclei's surface: informative.
    res.metrics = {"gap_um": gap, "width_um": width, "rim_um": rim_w,
                   "ring_voxels": int(in_ring.sum()), "rim_voxels": int((rim > 0).sum())}
    res.evidence.append(
        f"The ring ({gap} µm gap, {width} µm wide) and rim ({rim_w} µm) of each cells_3d nucleus were "
        f"compared with independent per-object distance maps in µm: every ring voxel lies {gap}–"
        f"{gap + width} µm from its own nucleus and the band is complete; the rim equals an "
        f"independently computed rim voxel for voxel; numbers are kept.")


@check("VV-SEC-03")
def _seeded(ctx, res):
    from .. import _secondary

    rows = []
    for name in ("cells_2d", "cells_3d"):
        sc = ctx.scene(name)
        sp = sc.spec.spacing
        border = synth._borders(sc.cells)  # each true cell's membrane voxels
        for kind in ("cytoplasm", "membrane"):
            # The panel's defaults: Otsu x 1.0, smoothing 0.5 µm, no maximum distance.
            cells, used = _secondary.seeded_cells(sc.nuclei, sc.acquired[kind], sp, kind=kind,
                                                  threshold="otsu", smooth=0.5)
            m = metrics.match(sc.cells, cells)
            same = all(p[0] == p[1] for p in m["pairs"] if p[2] >= 0.5)
            own_seed = all(np.all(cells[sc.nuclei == i] == i) for i in np.unique(sc.nuclei)[1:])
            stolen = 0  # true cells whose membrane mostly carries another seed's number
            for i in np.unique(sc.cells)[1:]:
                got = cells[(sc.cells == i) & border]
                stolen += int(np.mean((got != i) & (got != 0)) > 0.5)
            cyto = _secondary.cytoplasm(cells, sc.nuclei)
            cyto_ok = np.array_equal(cyto, np.where(sc.nuclei > 0, 0, cells))
            label = f"{name}, {kind}"
            res.check(f"{label}: mean IoU with the true cells", m["mean_iou"], ">=", ctx.limits["seeded_iou"])
            res.check(f"{label}: F1 at IoU 0.5", m["f1@0.5"], ">=", ctx.limits["seeded_f1"])
            res.check(f"{label}: every cell has its seed's number", same and own_seed, "is", True)
            res.check(f"{label}: membranes mostly given another seed's number", stolen, "==", 0)
            res.check(f"{label}: cytoplasm = cell minus nucleus", cyto_ok, "is", True)
            rows.append({"scene": name, "channel": kind, **metrics.summary(m), "threshold": used,
                         "membranes given to another cell": stolen})
            res.figures.append((f"{label}: cells grown from the true nuclei",
                                seg_figure(sc.acquired[kind], sc.cells, cells, name)))
            res.evidence.append(
                f"{label}: cells grown from the {m['n_true']} true nuclei match the true cells with "
                f"mean IoU {m['mean_iou']:.3f} (F1 {m['f1@0.5']:.3f}); each keeps its nucleus's number"
                f" and {stolen} true cells have their membrane mostly labelled as another cell.")
    res.tables["seeded cells"] = pd.DataFrame(rows)
    res.metrics = {f"{r['scene']} {r['channel']}": {k: r[k] for k in ("f1@0.5", "mean_iou")} for r in rows}


def _pair_iou(a, b):
    """Mean IoU between same-numbered objects of two label images (numbers of `a`)."""
    ious = []
    for i in np.unique(a)[1:]:
        x, y = a == i, b == i
        ious.append((x & y).sum() / max((x | y).sum(), 1))
    return float(np.mean(ious)) if ious else 1.0


@check("VV-SEC-04")
def _membrane(ctx, res):
    from .. import _secondary

    rows, sweep = [], []
    for name in ("cells_2d", "cells_3d"):
        sc = ctx.scene(name)
        sp = sc.spec.spacing
        image = sc.acquired["membrane"]
        border = synth._borders(sc.cells)  # each true cell's membrane voxels

        def grow(seeds, threshold="otsu", factor=1.0):
            return _secondary.seeded_cells(seeds, image, sp, kind="membrane", threshold=threshold,
                                           factor=factor, smooth=0.5)[0]

        # (1) the automatic thresholds agree, and each is accurate
        found = {t: grow(sc.nuclei, t) for t in ("otsu", "li", "triangle")}
        for t, cells in found.items():
            agree = _pair_iou(found["otsu"], cells)
            iou = metrics.match(sc.cells, cells)["mean_iou"]
            if t != "otsu":
                res.check(f"{name}: {t.title()} vs Otsu, mean IoU of the cells", agree, ">=",
                          ctx.limits["membrane_agree"])
            res.check(f"{name}: {t.title()}, mean IoU with the true cells", iou, ">=", ctx.limits["seeded_iou"])
            rows.append({"scene": name, "test": f"threshold {t}", "mean IoU with true cells": iou,
                         "mean IoU with Otsu's cells": agree})

        # (2) cells stop at their membrane (no maximum distance)
        cells = found["otsu"]
        outside = (cells > 0) & (cells != sc.cells) & ~border
        spill = outside.sum() / max(int((cells > 0).sum()), 1)
        ratio = max((cells == i).sum() / (sc.cells == i).sum() for i in np.unique(sc.cells)[1:])
        res.check(f"{name}: share of grown voxels outside their true cell", spill, "<=",
                  ctx.limits["membrane_spill"], "{:.1%}")
        res.check(f"{name}: largest grown cell / its true cell", ratio, "<=", ctx.limits["membrane_size_ratio"],
                  "{:.2f}")
        rows.append({"scene": name, "test": "spill (Otsu)", "share outside": spill, "largest size ratio": ratio})

        # (3) a cell without its nucleus stays empty
        taken = []
        for i in np.unique(sc.nuclei)[1:]:
            seeds = np.where(sc.nuclei == i, 0, sc.nuclei)
            inside = (sc.cells == i) & ~border
            taken.append(float(np.mean(grow(seeds)[inside] > 0)))
        res.check(f"{name}: most of a cell without its nucleus taken by others", max(taken), "<=",
                  ctx.limits["unseeded_taken"], "{:.1%}")
        rows.append({"scene": name, "test": "each nucleus left out in turn", "largest share taken": max(taken)})

        # evidence only: where the method holds as Threshold x changes
        for factor in (0.3, 0.5, 0.7, 1.0, 1.2, 1.5):
            cells = found["otsu"] if factor == 1.0 else grow(sc.nuclei, "otsu", factor)
            ratio = max((cells == i).sum() / (sc.cells == i).sum() for i in np.unique(sc.cells)[1:])
            sweep.append({"scene": name, "Threshold x": factor,
                          "mean IoU with true cells": metrics.match(sc.cells, cells)["mean_iou"],
                          "largest size ratio": ratio})
        res.figures.append((f"{name}: cells grown in the membrane channel (Otsu)",
                            seg_figure(image, sc.cells, found["otsu"], name)))
    res.tables["membrane cells"] = pd.DataFrame(rows)
    res.tables["Threshold x sweep (evidence, not a criterion)"] = sweep_df = pd.DataFrame(sweep)
    weak = sweep_df[(sweep_df["mean IoU with true cells"] < ctx.limits["seeded_iou"])
                    | (sweep_df["largest size ratio"] > ctx.limits["membrane_size_ratio"])]
    res.evidence.append(
        "Otsu, Li and Triangle give the same cells, which stop at their membranes without a "
        "maximum distance, and a cell without a nucleus stays empty rather than being taken by its "
        "neighbours.")
    res.evidence.append(
        "Threshold x sweep: " + ("the cells stay accurate from 0.3 to 1.5." if weak.empty else
        "accuracy drops at " + ", ".join(f"{r['scene']} x{r['Threshold x']:g} (mean IoU "
        f"{r['mean IoU with true cells']:.2f}, largest cell {r['largest size ratio']:.1f}x its true size)"
        for _, r in weak.iterrows())
        + ". Far below the automatic threshold, noise inside the dark cells counts as membrane; "
        "above it, gaps open in the membrane and a cell can run into the background. Keep "
        "Threshold x near 1 for membrane stains, or set a maximum distance."))
    res.metrics = {f"{r['scene']} {r['test']}": {k: v for k, v in r.items() if k not in ("scene", "test")}
                   for r in rows}


@check("VV-MEAS-01")
def _size(ctx, res):
    from .. import _measure

    rows = []
    for name in ("nuclei_2d", "cells_3d"):
        sc = ctx.scene(name)
        sp = sc.spec.spacing
        unit = "µm³" if sc.spec.ndim == 3 else "µm²"
        col = f"volume ({unit})" if sc.spec.ndim == 3 else f"area ({unit})"
        for what, lab in (("nucleus", sc.nuclei), ("cell", sc.cells)):
            table = _measure.measure_block(lab, None, sp).set_index("label")
            truth = sc.truth.set_index("label")
            exact = metrics.relative_error(table[col], truth.loc[table.index, f"{what} size ({unit})"])
            res.check(f"{name} {what}: size vs voxel count x voxel size", exact.max(), "<=",
                      ctx.limits["exact_rel"], "{:.1e}")
            eq = table[f"equivalent diameter (µm)"]
            eq_ref = np.cbrt(6 * table[col] / np.pi) if sc.spec.ndim == 3 else np.sqrt(4 * table[col] / np.pi)
            res.check(f"{name} {what}: equivalent diameter formula", metrics.relative_error(eq, eq_ref).max(),
                      "<=", ctx.limits["exact_rel"], "{:.1e}")
            if what == "nucleus":
                analytic = truth.loc[table.index, f"nucleus analytic size ({unit})"]
                rel = pd.Series(metrics.relative_error(table[col], analytic), index=table.index)
                res.check(f"{name}: median difference from the analytic size", float(np.median(rel)), "<=",
                          ctx.limits["analytic_size_rel"], "{:.1%}")
                rows += [{"scene": name, "label": i, "measured": table.loc[i, col], "analytic": analytic.loc[i],
                          "relative difference": rel.loc[i]} for i in table.index]
                res.figures.append((f"{name}: measured nucleus size vs the analytic size",
                                    scatter_figure(analytic, table[col], f"analytic ({unit})",
                                                   f"measured ({unit})", name)))
    res.tables["sizes"] = pd.DataFrame(rows)
    res.evidence.append(
        "Areas and volumes equal the voxel count times the voxel size to round-off. Against the "
        "analytic ellipse/ellipsoid sizes the measured sizes differ by a median of "
        + ", ".join(f"{g}: {d['relative difference'].median():.1%}" for g, d in pd.DataFrame(rows).groupby("scene"))
        + " — the expected discretisation error for objects a few µm across.")


def _timelapse_viewer(sc):
    """A napari viewer holding the timelapse scene as the panels would see it."""
    from napari.components import ViewerModel

    from .. import _secondary

    sp = sc.spec.spacing
    origin = sc.spec.origin_um
    T = sc.spec.timepoints
    meta = {"ibif_axes": "TZYX", "ibif_voxel_um": dict(zip("ZYX", sp))}
    v = ViewerModel()
    common = dict(scale=(1.0, *sp), translate=(0.0, *origin), metadata=meta)
    images = [v.add_image(sc.acquired[c], name=c, **common) for c in sc.acquired]
    nuclei = v.add_labels(np.stack([sc.nuclei] * T), name="nucleus objects", **common)
    cells = v.add_labels(np.stack([sc.cells] * T), name="cell objects", **common)
    cyto = v.add_labels(np.stack([_secondary.cytoplasm(sc.cells, sc.nuclei)] * T), name="cytoplasm objects",
                        **common)
    return v, images, nuclei, cells, cyto


@check("VV-MEAS-02")
def _centroid(ctx, res):
    from .. import _measure

    sc = ctx.scene("timelapse_3d")
    sp = np.asarray(sc.spec.spacing)
    origin = np.asarray(sc.spec.origin_um)
    v, images, nuclei, cells, _ = _timelapse_viewer(sc)
    table = _measure.measure_layer(v, nuclei, images)
    t0 = table[table["T"] == 0].set_index("label")
    truth = sc.truth[sc.truth["T"] == 0].set_index("label")
    coords = np.nonzero(sc.nuclei)
    lab = sc.nuclei[coords]
    ids = t0.index.to_numpy()
    worst_exact, worst_true = 0.0, 0.0
    errs = []
    for k, ax in enumerate("ZYX"):
        com = np.bincount(lab, weights=coords[k], minlength=ids.max() + 1)[ids] / np.bincount(lab)[ids]
        independent = com * sp[k] + origin[k]
        measured = t0[f"centroid {ax} (µm)"].to_numpy()
        worst_exact = max(worst_exact, float(np.max(np.abs(measured - independent) / np.maximum(np.abs(independent), 1))))
        errs.append(measured - truth.loc[ids, f"centre {ax} (µm)"].to_numpy())
    err = np.sqrt(np.sum(np.square(errs), axis=0))
    diag = float(np.linalg.norm(sp))
    res.check("centroid vs independent centre of mass (relative)", worst_exact, "<=", ctx.limits["exact_rel"], "{:.1e}")
    res.check("largest distance from the true centre (voxel diagonals)", err.max() / diag, "<=",
              ctx.limits["centroid_voxels"])
    res.tables["centroids"] = pd.DataFrame({"label": ids, "error (µm)": err})
    res.metrics = {"max_error_um": float(err.max()), "voxel_diagonal_um": diag}
    res.evidence.append(
        f"Through napari layers with an offset origin {tuple(origin)} µm, the measured centroids equal an "
        f"independent centre of mass to round-off and lie at most {err.max():.3f} µm "
        f"({err.max() / diag:.2f} voxel diagonals) from the true centres.")


@check("VV-MEAS-03")
def _intensity(ctx, res):
    from .. import _measure

    sc = ctx.scene("cells_3d")
    sp = sc.spec.spacing
    rows = []
    for what, lab in (("nucleus", sc.nuclei), ("cell", sc.cells)):
        table = _measure.measure_block(lab, sc.acquired, sp).set_index("label")
        truth = sc.truth.set_index("label").loc[table.index]
        worst = 0.0
        for ch in sc.acquired:
            for stat in ("mean", "median", "min", "max", "std", "sum"):
                worst = max(worst, float(metrics.relative_error(table[f"{ch} {stat}"],
                                                                truth[f"{what} {ch} acquired {stat}"]).max()))
            n = truth[f"{what} voxels"].to_numpy()
            se = sc.noise_sigma / np.sqrt(n)
            z = np.abs(table[f"{ch} mean"].to_numpy() - truth[f"{what} {ch} ideal mean"].to_numpy()) / se
            res.check(f"{what} {ch}: largest |mean - noise-free mean| (standard errors)", z.max(), "<=",
                      ctx.limits["noise_se"], "{:.2f}")
            rows += [{"object": what, "channel": ch, "label": i, "measured mean": a, "noise-free mean": b,
                      "standard errors": zz}
                     for i, a, b, zz in zip(table.index, table[f"{ch} mean"], truth[f"{what} {ch} ideal mean"], z)]
        res.check(f"{what}: all statistics vs independent values (relative)", worst, "<=",
                  ctx.limits["exact_rel"], "{:.1e}")
    df = pd.DataFrame(rows)
    res.tables["intensities"] = df
    res.figures.append(("Measured mean vs noise-free mean, every object and channel",
                        scatter_figure(df["noise-free mean"], df["measured mean"], "noise-free mean (counts)",
                                       "measured mean (counts)", "cells_3d")))
    res.evidence.append(
        f"For the {sc.truth.shape[0]} nuclei and cells of cells_3d in 3 channels, mean, median, min, max, "
        f"std and sum equal independent NumPy statistics to round-off, and every measured mean is within "
        f"{df['standard errors'].max():.2f} standard errors of the noise-free value (noise σ "
        f"{sc.noise_sigma:.0f} counts).")


@check("VV-MEAS-04")
def _timeseries(ctx, res):
    from .. import _measure

    sc = ctx.scene("timelapse_3d")
    T, n = sc.spec.timepoints, len(np.unique(sc.nuclei)) - 1
    v, images, nuclei, cells, cyto = _timelapse_viewer(sc)
    tables = {layer.name: _measure.measure_layer(v, layer, images) for layer in (nuclei, cells, cyto)}
    rows_ok = all(len(t) == T * n and set(zip(t["T"], t["label"])) ==
                  {(a, b) for a in range(T) for b in range(1, n + 1)} for t in tables.values())
    res.check("one row per object and time point", rows_ok, "is", True)
    t = tables["nucleus objects"].set_index(["T", "label"])
    bg = sc.spec.background
    worst = 0.0
    ratios = []
    for time_point in range(1, T):
        for label in range(1, n + 1):
            r = (t.loc[(time_point, label), "nuclei mean"] - bg) / (t.loc[(0, label), "nuclei mean"] - bg)
            true = 1 + sc.spec.intensity_drift * time_point
            ratios.append({"T": time_point, "label": label, "measured ratio": r, "true ratio": true})
            worst = max(worst, abs(r - true) / true)
    res.check("intensity ratio to time 0 (relative error)", worst, "<=", ctx.limits["drift_rel"], "{:.2%}")
    combined = _measure.combine(tables)
    full = combined.notna().all(axis=1).all() and len(combined) == T * n
    res.check("joined table: one complete row per (T, label)", bool(full), "is", True)
    res.tables["ratios"] = pd.DataFrame(ratios)
    res.tables["joined"] = combined
    res.evidence.append(
        f"Measuring the {T}-time-point series gave {T * n} rows per layer, the nuclear intensity rose by "
        f"the true 25% per step within {worst:.2%}, and the joined table has one complete row per "
        f"time point and label with nucleus, cell and cytoplasm columns.")


@check("VV-MEAS-05")
def _e2e(ctx, res):
    from .. import _measure

    sc = ctx.scene("nuclei_3d")
    sp = sc.spec.spacing
    pred, _ = ctx.predict("nuclei_3d")
    m = metrics.match(sc.nuclei, pred)
    image = {"nuclei": sc.acquired["nuclei"]}
    a = _measure.measure_block(sc.nuclei, image, sp).set_index("label")
    b = _measure.measure_block(pred, image, sp).set_index("label")
    rows = []
    for t_id, p_id, iou in m["pairs"]:
        if iou < 0.5:
            continue
        rows.append({"truth": t_id, "result": p_id, "IoU": iou,
                     "true volume": a.loc[t_id, "volume (µm³)"], "measured volume": b.loc[p_id, "volume (µm³)"],
                     "true mean": a.loc[t_id, "nuclei mean"], "measured mean": b.loc[p_id, "nuclei mean"]})
    df = pd.DataFrame(rows)
    if df.empty:
        res.check("matched nuclei", 0, ">=", 1, "{:.0f}")
        return
    vol = metrics.relative_error(df["measured volume"], df["true volume"])
    mean = metrics.relative_error(df["measured mean"], df["true mean"])
    res.check("median relative volume error", float(np.median(vol)), "<=", ctx.limits["e2e_size_rel"], "{:.1%}")
    res.check("median relative mean-intensity error", float(np.median(mean)), "<=", ctx.limits["e2e_mean_rel"],
              "{:.1%}")
    res.tables["matched nuclei"] = df
    res.figures.append(("Volume of matched nuclei: segmented vs true",
                        scatter_figure(df["true volume"], df["measured volume"], "true (µm³)", "segmented (µm³)",
                                       "nuclei_3d")))
    bias = float(np.median((df["measured volume"] - df["true volume"]) / df["true volume"]))
    res.evidence.append(
        f"For the {len(df)} matched nuclei, the segmented volume differs from the true volume by a median "
        f"{np.median(vol):.1%} (median bias {bias:+.1%}) and the mean intensity by {np.median(mean):.1%}.")


@check("VV-WS-01")
def _workspace(ctx, res):
    from napari.components import ViewerModel

    from .. import _measure, _workspace

    sc = ctx.scene("cells_3d")
    sp = sc.spec.spacing
    v = ViewerModel()
    meta = {"ibif_axes": "ZYX", "ibif_voxel_um": dict(zip("ZYX", sp))}
    for c, arr in sc.acquired.items():
        v.add_image(arr, name=c, scale=sp, translate=(2.0, -1.0, 3.5), metadata=meta)
    v.add_labels(sc.nuclei, name="nuclei", scale=sp, translate=(2.0, -1.0, 3.5), metadata=meta)
    v.add_labels(sc.cells, name="cells", scale=sp, translate=(2.0, -1.0, 3.5), metadata=meta)
    tables = {"nuclei": _measure.measure_layer(v, v.layers["nuclei"], [v.layers[c] for c in sc.acquired])}
    _measure.remember(v, tables)
    folder, notes = _workspace.save_workspace(v, ctx.out / "workspace" / "vv")
    w = ViewerModel()
    load_notes = _workspace.load_workspace(w, folder)
    same = [l.name for l in v.layers] == [l.name for l in w.layers]
    arrays = same and all(np.array_equal(np.asarray(a.data), np.asarray(b.data)) for a, b in zip(v.layers, w.layers))
    geometry = same and all(np.allclose(a.scale, b.scale) and np.allclose(a.translate, b.translate)
                            for a, b in zip(v.layers, w.layers))
    restored = _measure.tables_of(w).get("nuclei")
    tables_ok = restored is not None and np.allclose(restored.to_numpy(float), tables["nuclei"].to_numpy(float),
                                                     rtol=1e-12, equal_nan=True)
    res.check("arrays identical", arrays, "is", True)
    res.check("scale and translate identical", geometry, "is", True)
    res.check("measurement table identical", tables_ok, "is", True)
    res.notes += list(notes) + list(load_notes)
    res.evidence.append(
        f"A workspace with {len(v.layers)} layers (3 channels, nuclei and cells, voxel size and offset) and "
        f"its measurement table was saved to {folder.name} and reopened; every array, scale, offset and "
        f"table value came back identical.")


# ---- file reading -----------------------------------------------------------------------------


def _channels_of(layer_data):
    """{name: array} from a reader's (data, kwargs, kind) tuple."""
    data, kw, _ = layer_data
    names = kw["name"] if isinstance(kw["name"], list) else [kw["name"]]
    axis = kw.get("channel_axis")
    if axis is None:
        return {names[0]: np.asarray(data)}, kw
    return {n: np.asarray(np.take(data, i, axis=axis)) for i, n in enumerate(names)}, kw


@check("VV-IO-01")
def _czi(ctx, res):
    from pylibCZIrw import czi as pyczi

    from .._reader import napari_get_reader

    sc = ctx.scene("cells_3d")
    sp = sc.spec.spacing
    chans = {"DAPI": sc.acquired["nuclei"], "AF488": sc.acquired["cytoplasm"]}
    folder = ctx.out / "files"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "vv_cells.czi"
    path.unlink(missing_ok=True)
    with pyczi.create_czi(str(path)) as w:
        for c, arr in enumerate(chans.values()):
            for z in range(arr.shape[0]):
                w.write(arr[z][..., None], plane={"Z": z, "C": c})
        w.write_metadata(channel_names=dict(enumerate(chans)), scale_x=sp[2] * 1e-6, scale_y=sp[1] * 1e-6,
                         scale_z=sp[0] * 1e-6)
    layers = napari_get_reader(str(path))(str(path))
    got, kw = _channels_of(layers[0])
    names_ok = list(got) == [f"vv_cells {n}" for n in chans]
    values_ok = names_ok and all(np.array_equal(got[f"vv_cells {n}"], a) for n, a in chans.items())
    vox = kw["metadata"][0]["ibif_voxel_um"]
    vox_err = max(abs(vox[a] - s) for a, s in zip("ZYX", sp))
    res.check("pixel values identical", values_ok, "is", True)
    res.check("channel names identical", names_ok, "is", True)
    res.check("voxel size error (µm)", vox_err, "<=", 1e-6, "{:.1e}")
    res.metrics = {"names": list(got), "voxel_um": vox, "axes": kw["metadata"][0].get("ibif_axes")}
    res.evidence.append(
        f"A {len(chans)}-channel {sc.nuclei.shape} Z stack written as CZI opened as layers "
        f"{', '.join(got)} with identical pixel values and voxel size {vox} µm.")


@check("VV-IO-02")
def _lif(ctx, res):
    from .._lif import read_lif
    from ._lif_writer import write_lif

    sc = ctx.scene("timelapse_3d")
    sp = sc.spec.spacing
    folder = ctx.out / "files"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "vv_project.lif"
    series = np.stack([sc.acquired["nuclei"], sc.acquired["cytoplasm"]], axis=2)  # T Z C Y X
    single = sc.acquired["nuclei"][:1]  # T=1
    write_lif(path, [("Live", series, ["Blue", "Green"], sp[1], sp[0], 30.0),
                     ("Still", single[:, :, None], ["Gray"], sp[1], sp[0], 1)])
    listed = {}
    ok = True
    vox_err = 0.0
    interval = None
    for index, expected in ((0, {"vv_project Blue": series[:, :, 0], "vv_project Green": series[:, :, 1]}),
                            (1, {"vv_project Gray": single[0]})):
        def choose(title, scenes, index=index):
            listed[title] = [s.name for s in scenes]
            return [index], False

        layers = read_lif(path, choose=choose)
        got, kw = _channels_of(layers[0])
        ok &= list(got) == list(expected) and all(np.array_equal(got[k], expected[k]) for k in expected)
        meta = kw["metadata"][0]
        vox_err = max(vox_err, *(abs(meta["ibif_voxel_um"][a] - s) for a, s in zip("ZYX", sp)))
        if index == 0:
            interval = meta.get("ibif_time_interval_s")
    res.check("both images listed in the chooser", listed.get("vv_project") == ["Live", "Still"], "is", True)
    res.check("pixel values and channel names identical", bool(ok), "is", True)
    res.check("voxel size error (µm)", vox_err, "<=", 1e-6, "{:.1e}")
    res.check("time interval error (s)", abs((interval or 0) - 30.0), "<=", 1e-6, "{:.1e}")
    res.evidence.append(
        "A LIF file with a 3-time-point, 2-channel Z stack and a single image listed both images in the "
        "chooser; each opened with identical pixel values, channel names, voxel size and the 30 s interval.")
