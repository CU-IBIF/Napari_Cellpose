"""Stand-in for ibif_napari._cellpose used only to make the manual's screenshots
(the Cellpose weights cannot be downloaded in the build environment): a classical
distance-transform watershed of the chosen channel."""
import numpy as np
from scipy import ndimage as ndi


def get_model(name, gpu=False):
    return None


def segment(image, nuclei=None, *, anisotropy=None, min_size=15, **_):
    from skimage import filters, segmentation
    from skimage.feature import peak_local_max
    img = np.asarray(image, np.float32)
    sp = (anisotropy or 1.0, 1.0, 1.0)[-img.ndim:] if img.ndim == 3 else (1.0, 1.0)
    img = ndi.gaussian_filter(img, [1.0 / s for s in sp])
    mask = img > filters.threshold_otsu(img)
    dist = ndi.distance_transform_edt(mask, sampling=sp)
    lab, _ = ndi.label(mask)
    peaks = peak_local_max(dist, min_distance=6, labels=lab, exclude_border=False)
    markers = np.zeros(img.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    out = segmentation.watershed(-dist, markers, mask=mask)
    sizes = np.bincount(out.ravel())
    out[sizes[out] < min_size] = 0
    out, _, _ = segmentation.relabel_sequential(out)
    return out.astype(np.uint16)
