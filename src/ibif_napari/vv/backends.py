"""Segmentation back-ends the V&V suite can test.

* ``cellpose``: the plugin's own segmentation (``ibif_napari._cellpose.segment``, the same
  function the Segmentation section runs in its worker process), with the panel's defaults:
  automatic diameter, flow threshold 0.4, cell probability 0, minimum size 15, and in 3D
  the anisotropy filled in from the voxel size. This is what the segmentation
  requirements are about.
* ``classic`` (also ``baseline``): the Segmentation section's *Threshold + watershed*
  tab (``ibif_napari._classic.segment``) with the tab's defaults: smoothing 1 px, Otsu,
  fill holes, distance-transform watershed, minimum size 15. It needs no model weights,
  so the whole suite can be exercised on any computer; its results say nothing about
  Cellpose (they are the Threshold + watershed tab's results).
* ``truth``: returns the truth labels renumbered (an oracle). Every segmentation metric
  must then be perfect: this checks the suite itself.
"""
from __future__ import annotations

import numpy as np


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

            model = self.model or _cellpose.default_model()
        except Exception as e:  # not installed, or a broken PyTorch / torchvision install
            return False, f"Cellpose cannot be imported ({type(e).__name__}: {e})"
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
            model = self.model or _cellpose.default_model()
        except Exception:
            model = self.model
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
        return {"backend": self.name, "model": model,
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


class ClassicBackend(Backend):
    """The Threshold + watershed tab (no deep learning), with its defaults."""

    name = "classic"

    def describe(self):
        from .._classic import Settings

        return {"backend": self.name, "method": "Threshold + watershed tab",
                "settings": {k: v for k, v in vars(Settings()).items()}}

    def segment(self, image, *, nuclei=None, spacing, do_3D=True, truth=None, kind="nuclei"):
        from .. import _classic

        if kind == "cells" and nuclei is not None:
            # Whole cells: grown from the nuclei, as the Secondary objects section does.
            from .. import _secondary

            seeds = self.segment(nuclei, spacing=spacing, kind="nuclei")
            cells, _ = _secondary.seeded_cells(seeds, image, spacing, kind="cytoplasm", smooth=0.5)
            return cells
        image = np.asarray(image)
        if image.ndim == 3 and not do_3D:  # per plane (the tab's "2D per plane")
            return _classic.segment_planes(image, spacing)[0]
        return _classic.segment(image, spacing)[0]


BaselineBackend = ClassicBackend  # the name used before the tab existed


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


BACKENDS = {"cellpose": CellposeBackend, "classic": ClassicBackend, "baseline": ClassicBackend,
            "truth": TruthBackend}


def make(name, **kwargs):
    if name not in BACKENDS:
        raise ValueError(f"unknown back-end {name!r}; choose from {', '.join(BACKENDS)}")
    cls = BACKENDS[name]
    return cls(**kwargs) if name == "cellpose" else cls()
