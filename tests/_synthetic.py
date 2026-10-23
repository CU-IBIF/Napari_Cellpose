"""Synthetic fluorescent filaments (blurred lines) with known lengths, for the tracer tests."""
import numpy as np
from scipy import ndimage as ndi

SPACING = (0.4, 0.1, 0.1)  # Z, Y, X in um, like a confocal stack


def draw(shape, segments, spacing=SPACING, width_um=0.3, noise=0.03, seed=0):
    """Image of straight segments [((z, y, x), (z, y, x)), ...] given in um."""
    sp = np.asarray(spacing, float)
    img = np.zeros(shape, np.float32)
    for a, b in segments:
        a, b = np.asarray(a, float) / sp, np.asarray(b, float) / sp
        n = int(np.ceil(np.abs(b - a).max() * 2)) + 1
        for t in np.linspace(0, 1, n):
            idx = np.round(a + t * (b - a)).astype(int)
            if all(0 <= i < s for i, s in zip(idx, shape)):
                img[tuple(idx)] = 1
    sigma = [max(width_um / 2.355 / s, 0.3) for s in sp]
    sigma[0] = max(sigma[0], 0.6 / 2.355 / sp[0])  # the PSF is longer in Z
    img = ndi.gaussian_filter(img, sigma)
    img /= img.max()
    rng = np.random.default_rng(seed)
    return (img + noise * rng.standard_normal(shape)).astype(np.float32)
