"""Cellpose model loading (cached, so it can be preloaded) and 3D segmentation.

Works with Cellpose 3 (cyto3, nuclei, ... models) and Cellpose 4 (cpsam).
"""
from __future__ import annotations

import threading
from importlib.metadata import PackageNotFoundError, version

import numpy as np

# Set by the launcher: "0" = don't preload a model when the panel opens, else a model name.
PRELOAD_ENV = "IBIF_NAPARI_PRELOAD"

_models = {}
_lock = threading.Lock()


def cellpose_major():
    try:
        return int(version("cellpose").split(".")[0])
    except (PackageNotFoundError, ValueError):
        return None


def builtin_models():
    major = cellpose_major()
    if major is None:
        return []
    if major >= 4:
        return ["cpsam"]
    return ["cyto3", "nuclei", "cyto2", "cyto"]


def default_model():
    models = builtin_models()
    return models[0] if models else None


def gpu_available():
    try:
        from cellpose import core

        return bool(core.use_gpu())
    except Exception:
        return False


def get_model(name, gpu=False):
    """Return a cached Cellpose model, downloading its weights on first use.

    `name` is a built-in model name or a path to a custom model file.
    """
    key = (name, bool(gpu))
    with _lock:
        if key not in _models:
            from cellpose import models

            if cellpose_major() >= 4 or name not in builtin_models():
                _models[key] = models.CellposeModel(gpu=gpu, pretrained_model=name)
            else:
                # Cellpose 3's wrapper adds the size model, so diameter=0 estimates it.
                _models[key] = models.Cellpose(gpu=gpu, model_type=name)
        return _models[key]


def segment(
    image,
    nuclei=None,
    *,
    model="cpsam",
    gpu=False,
    do_3D=True,
    anisotropy=None,
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    stitch_threshold=0.0,
    min_size=15,
):
    """Segment a 2D (Y, X) or 3D (Z, Y, X) image; returns integer labels of the same shape.

    `nuclei` is an optional second channel of the same shape. In 3D, `do_3D=True` runs
    Cellpose's true 3D flows; `do_3D=False` segments each plane and links planes whose
    masks overlap by more than `stitch_threshold`.
    """
    image = np.asarray(image)
    is_3d = image.ndim == 3
    if image.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D image, got shape {image.shape}")
    if nuclei is not None:
        nuclei = np.asarray(nuclei)
        if nuclei.shape != image.shape:
            raise ValueError("the nuclear channel must have the same shape as the image")
        x = np.stack([image, nuclei], axis=1 if is_3d else 0)
        channel_axis = 1 if is_3d else 0
    else:
        x = image
        channel_axis = None

    kwargs = dict(
        channel_axis=channel_axis,
        z_axis=0 if is_3d else None,
        do_3D=is_3d and do_3D,
        anisotropy=anisotropy if is_3d else None,
        diameter=diameter or None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        stitch_threshold=stitch_threshold if (is_3d and not do_3D) else 0.0,
        min_size=min_size,
    )
    m = get_model(model, gpu)
    if cellpose_major() < 4:
        # Cellpose 3 picks channels by 1-based index; 0 means grayscale / none.
        kwargs["channels"] = [1, 2] if nuclei is not None else [0, 0]
        if kwargs["diameter"] is None:
            # The Cellpose wrapper (it has a size model) estimates the diameter when given 0.
            kwargs["diameter"] = 0 if hasattr(m, "sz") else 30.0
    if kwargs["do_3D"]:
        kwargs["stitch_threshold"] = 0.0
    masks = m.eval(x, **kwargs)[0]
    masks = np.asarray(masks)
    return masks.astype(np.uint32 if masks.max() > np.iinfo(np.uint16).max else np.uint16)
