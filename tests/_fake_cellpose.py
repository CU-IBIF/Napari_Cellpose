"""Stand-in for ibif_napari._cellpose, imported by the Cellpose child process in tests."""
import os
import time

import numpy as np


def get_model(name, gpu=False):
    if name == "missing":
        raise FileNotFoundError("no such model")


def segment(image, nuclei=None, **params):
    if params.get("model") == "slow":
        time.sleep(60)
    if params.get("model") == "crash":
        os._exit(3)
    out = np.zeros(np.shape(image), dtype=np.uint16)
    out[..., :2, :2] = 1
    out[..., -2:, -2:] = 2
    return out
