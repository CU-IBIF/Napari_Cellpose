"""Synthetic sample data for the manual's screenshots."""
import numpy as np
from scipy import ndimage as ndi

from ibif_napari.vv import synth

SPACING = (1.0, 0.4, 0.4)


def tissue(seed=7):
    """(T, Z, C, Y, X) uint16 with DAPI, Membrane, CD3, CD8; plus channel info."""
    spec = synth.SceneSpec("tissue", (14, 220, 300), SPACING, n_objects=34, cell_scale=1.9,
                           min_gap_um=3.0, nucleus_radius_um=(3.0, 4.2),
                           channels=("nuclei", "cytoplasm", "membrane"),
                           psf_sigma_um=(0.9, 0.35, 0.35), snr=10, seed=seed)
    sc = synth.generate(spec)
    rng = np.random.default_rng(seed)
    cells, nuclei = sc.cells, sc.nuclei
    n = int(cells.max())
    cd3 = np.zeros(n + 1); cd8 = np.zeros(n + 1)
    for i in range(1, n + 1):
        if rng.random() < 0.55:
            cd3[i] = rng.uniform(0.6, 1.0)
            if rng.random() < 0.45:
                cd8[i] = rng.uniform(0.5, 0.9)
        else:
            cd3[i] = rng.uniform(0.0, 0.08)
        if cd8[i] == 0:
            cd8[i] = rng.uniform(0.0, 0.06)
    cyto = (cells > 0) & (nuclei == 0)
    sigma = [s_um / s for s_um, s in zip((0.9, 0.35, 0.35), SPACING)]

    def marker(levels):
        img = np.where(cyto, levels[cells], 0.0)
        img = ndi.gaussian_filter(img, sigma)
        return img

    frames = []
    for t in range(2):
        chans = [sc.acquired["nuclei"].astype(np.float32), sc.acquired["membrane"].astype(np.float32)]
        for lv, peak in ((cd3, 900), (cd8, 800)):
            m = 100 + peak * (1 + 0.1 * t) * marker(lv)
            chans.append(m + rng.normal(0, 40, m.shape))
        if t:
            chans[:2] = [c + rng.normal(0, 30, c.shape) for c in chans[:2]]
        frames.append(np.stack([np.clip(c, 0, 65535) for c in chans], axis=1))  # Z C Y X
    data = np.stack(frames).astype(np.uint16)  # T Z C Y X
    names = ["DAPI", "Membrane", "CD3", "CD8"]
    emissions = [455, 520, 600, 690]
    return data, names, emissions, sc


def neuron(seed=3):
    """A branched neuron-like filament image (Z, Y, X) float32 at (0.5, 0.15, 0.15) µm."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
    from _synthetic import draw
    sp = (0.5, 0.15, 0.15)
    soma = (3.0, 22.0, 24.0)
    segs = []
    rng = np.random.default_rng(seed)

    def branch(start, angle, length, depth):
        end = (float(np.clip(start[0] + rng.uniform(-0.4, 0.4), 2.2, 3.8)),
               start[1] + length * np.sin(angle), start[2] + length * np.cos(angle))
        segs.append((start, end))
        if depth:
            for d in (-0.5, 0.45):
                branch(end, angle + d + rng.uniform(-0.15, 0.15), length * 0.7, depth - 1)

    for a in (0.3, 1.7, 3.0, 4.5):
        start = (soma[0], soma[1] + 2.0 * np.sin(a), soma[2] + 2.0 * np.cos(a))
        branch(start, a, 8.0, 2)
    img = draw((16, 300, 320), segs, spacing=sp, width_um=0.7, noise=0.03, seed=seed)
    zz, yy, xx = np.indices(img.shape)
    blob = (((zz * sp[0] - soma[0]) / 2.0) ** 2 + ((yy * sp[1] - soma[1]) / 2.4) ** 2
            + ((xx * sp[2] - soma[2]) / 2.4) ** 2)
    img = np.maximum(img, np.where(blob < 1, 0.9, 0.9 * np.exp(-(blob - 1) * 3)))
    return (200 + 1000 * img).astype(np.float32), sp
