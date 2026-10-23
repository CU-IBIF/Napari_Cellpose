"""Cellpose model loading (cached, so it can be preloaded) and 3D segmentation.

Works with Cellpose 3 (cyto3, nuclei, ... models) and Cellpose 4 (cpsam_v2, cpsam, ...).
"""
from __future__ import annotations

import inspect
import os
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.request import urlopen

import numpy as np

# Set by the launcher: "0" = don't preload a model when the panel opens, else a model name.
PRELOAD_ENV = "IBIF_NAPARI_PRELOAD"

# Cellpose 4 weights live here; Cellpose's own downloader has no timeout and reports
# progress only to the terminal, so a stalled download looks like a silent hang.
_CP4_URL = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/"
DOWNLOAD_TIMEOUT_S = 60

_models = {}
_lock = threading.Lock()  # guards loading models into memory
_download_lock = threading.Lock()


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
        try:
            from cellpose import models

            names = list(models.MODEL_NAMES)
        except (ImportError, AttributeError):
            return ["cpsam"]
        # cpdino models need the separate dinov3 package; hide them without it.
        from importlib.util import find_spec

        if find_spec("dinov3") is None:
            names = [n for n in names if not n.startswith("cpdino")]
        return names
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


def is_loaded(name, gpu=False):
    return (name, bool(gpu)) in _models


def ensure_weights(name):
    """Download a Cellpose 4 built-in model's weights if missing, yielding status messages.

    A stalled connection raises an error after DOWNLOAD_TIMEOUT_S instead of hanging.
    Cellpose 3 models and model files on disk are left to Cellpose.
    """
    if (cellpose_major() or 0) < 4 or os.path.exists(name) or name not in builtin_models():
        return
    from cellpose import models

    dst = Path(models.MODEL_DIR) / name
    if dst.exists():
        return
    while not _download_lock.acquire(timeout=0.5):
        yield f"Waiting for the '{name}' download that is already running…"
    try:
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = getattr(models, "_MODEL_URL", _CP4_URL) + name
        yield f"Connecting to {url}…"
        tmp = dst.with_name(dst.name + ".part")
        try:
            with urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response, open(tmp, "wb") as f:
                total = int(response.headers.get("Content-Length") or 0)
                done, last = 0, 0.0
                while chunk := response.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if time.monotonic() - last > 0.5:
                        last = time.monotonic()
                        size = f" of {total / 1e6:.0f} MB" if total else " MB"
                        yield f"Downloading Cellpose model '{name}': {done / 1e6:.0f}{size}"
            os.replace(tmp, dst)
        except BaseException as e:
            tmp.unlink(missing_ok=True)
            if isinstance(e, Exception):
                raise RuntimeError(
                    f"could not download Cellpose model '{name}' from {url} ({e}). "
                    f"Check the internet connection/proxy, or copy the file to {dst}."
                ) from e
            raise
    finally:
        _download_lock.release()


def load_model(name, gpu=False):
    """Download (if needed) and load a model, yielding status messages; returns the model."""
    yield from ensure_weights(name)
    while not _lock.acquire(timeout=0.5):
        yield f"Waiting for Cellpose model '{name}' to finish loading…"
    _lock.release()
    if not is_loaded(name, gpu):
        yield f"Loading Cellpose model '{name}' ({'GPU' if gpu else 'CPU'})…"
    return get_model(name, gpu)


def get_model(name, gpu=False):
    """Return a cached Cellpose model (loading it, and letting Cellpose download it, if needed).

    `name` is a built-in model name or a path to a custom model file.
    """
    key = (name, bool(gpu))
    with _lock:
        if key not in _models:
            from cellpose import models

            if cellpose_major() >= 4 or name not in builtin_models():
                kwargs = {}
                params = inspect.signature(models.CellposeModel).parameters
                if "use_bfloat16" in params:
                    # bfloat16 is fast on GPUs but very slow on most CPUs.
                    kwargs["use_bfloat16"] = bool(gpu)
                _models[key] = models.CellposeModel(gpu=gpu, pretrained_model=name, **kwargs)
            else:
                # Cellpose 3's wrapper adds the size model, so diameter=0 estimates it.
                _models[key] = models.Cellpose(gpu=gpu, model_type=name)
        return _models[key]


def segment(
    image,
    nuclei=None,
    *,
    model=None,
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
    m = get_model(model or default_model(), gpu)
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
