"""Segmentation back-ends the V&V suite can test.

* ``cellpose``: the plugin's own segmentation (``ibif_napari._cellpose.segment``, the same
  function the Cellpose 3D panel runs in its worker process), with the panel's defaults:
  automatic diameter, flow threshold 0.4, cell probability 0, minimum size 15, and in 3D
  the anisotropy filled in from the voxel size. This is what the segmentation
  requirements are about.
* ``baseline``: a classical pipeline (smoothing, Otsu threshold, distance-transform
  watershed). It needs no model weights, so the whole suite can be exercised on any
  computer; its results say nothing about Cellpose.
* ``truth``: returns the truth labels renumbered (an oracle). Every segmentation metric
  must then be perfect: this checks the suite itself.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


class Backend:
    name = "?"
    verifies_cellpose = False

    def available(self):
        """(True, "") or (False, why not)."""
        return True, ""

    def describe(self):
        return {"backend": self.name}

    def segment(self, image, *, nuclei=None, spacing, do_3D=True, truth=None, kind="nuclei"):
        raise NotImplementedError


class CellposeBackend(Backend):
    name = "cellpose"
    verifies_cellpose = True

    def __init__(self, model=None, gpu=False, diameter=None):
        self.model = model
        self.gpu = gpu
        self.diameter = diameter

    def available(self):
        try:
            from .. import _cellpose

            import cellpose  # noqa: F401
        except Exception as e:  # pragma: no cover - depends on the computer
            return False, f"Cellpose is not installed ({e})"
        model = self.model or _cellpose.default_model()
        if model is None:
            return False, "no Cellpose model found"
        try:
            for _ in _cellpose.load_model(model, self.gpu):  # downloads the weights if needed
                pass
        except Exception as e:  # e.g. no internet access to download the weights
            return False, f"the Cellpose model {model!r} could not be loaded ({e})"
        return True, ""

    def describe(self):
        from .. import _cellpose

        try:
            import cellpose

            version = getattr(cellpose, "__version__", None) or _version("cellpose")
        except Exception:
            version = None
        try:
            import torch

            device = "cuda" if (self.gpu and torch.cuda.is_available()) else \
                ("mps" if self.gpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            torch_version = torch.__version__
        except Exception:
            device, torch_version = "cpu", None
        return {"backend": self.name, "model": self.model or _cellpose.default_model(),
                "cellpose": version, "torch": torch_version, "device": device,
                "diameter": self.diameter or "auto"}

    def segment(self, image, *, nuclei=None, spacing, do_3D=True, truth=None, kind="nuclei"):
        from .. import _cellpose

        image = np.asarray(image)
        anisotropy = None
        if image.ndim == 3:
            anisotropy = float(spacing[0]) / float(spacing[1])  # as the panel fills it in
        return _cellpose.segment(
            image, nuclei, model=self.model or _cellpose.default_model(), gpu=self.gpu,
            do_3D=do_3D, anisotropy=anisotropy, diameter=self.diameter,
            stitch_threshold=0.0 if do_3D else 0.3,
        )


class BaselineBackend(Backend):
    """Classical segmentation (no deep learning) to exercise the suite."""

    name = "baseline"

    def segment(self, image, *, nuclei=None, spacing, do_3D=True, truth=None, kind="nuclei"):
        from skimage import filters
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        sp = np.asarray(spacing, float)
        if kind == "cells" and nuclei is not None:
            from .. import _secondary

            seeds = self.segment(nuclei, spacing=spacing, kind="nuclei")
            cells, _ = _secondary.seeded_cells(seeds, image, spacing, kind="cytoplasm", smooth=0.5)
            return cells
        img = ndi.gaussian_filter(np.asarray(image, float), sigma=list(1.0 / sp))
        mask = img > filters.threshold_otsu(img)
        mask = ndi.binary_opening(mask, iterations=1)
        mask = ndi.binary_fill_holes(mask)
        dist = ndi.distance_transform_edt(mask, sampling=sp)
        dist = ndi.gaussian_filter(dist, sigma=list(0.7 / sp))
        min_px = max(2, int(round(2.0 / sp.min())))  # peaks at least ~2 µm apart
        peaks = peak_local_max(dist, min_distance=min_px, threshold_abs=1.0, labels=ndi.label(mask)[0],
                               exclude_border=False)
        markers = np.zeros(mask.shape, np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        labels = watershed(-dist, markers, mask=mask)
        sizes = np.bincount(labels.ravel())
        small = np.flatnonzero(sizes < 15)
        labels[np.isin(labels, small[small > 0])] = 0
        return labels.astype(np.uint16)


class TruthBackend(Backend):
    """The truth, renumbered: a perfect segmentation."""

    name = "truth"

    def segment(self, image, *, nuclei=None, spacing, do_3D=True, truth=None, kind="nuclei"):
        if truth is None:
            raise ValueError("the truth back-end needs the truth labels")
        truth = np.asarray(truth)
        ids = np.unique(truth)
        ids = ids[ids != 0]
        rng = np.random.default_rng(0)
        lut = np.zeros(int(truth.max()) + 1, np.uint16)
        lut[ids] = rng.permutation(len(ids)) + 1
        return lut[truth]


def _version(dist):
    from importlib.metadata import version

    try:
        return version(dist)
    except Exception:
        return None


BACKENDS = {"cellpose": CellposeBackend, "baseline": BaselineBackend, "truth": TruthBackend}


def make(name, **kwargs):
    if name not in BACKENDS:
        raise ValueError(f"unknown back-end {name!r}; choose from {', '.join(BACKENDS)}")
    cls = BACKENDS[name]
    return cls(**kwargs) if name == "cellpose" else cls()
