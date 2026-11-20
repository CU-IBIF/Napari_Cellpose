"""Steps planned for later ("coming soon"). They are listed in the library with their
planned settings so the pipeline's shape is visible, but they have no implementation
and cannot be added to a pipeline. Each is a project of its own (see
docs/processing-and-inclusions-plan.md, section 4)."""
from __future__ import annotations

from . import register
from .base import COMING_SOON, Param, StepSpec

ILLUMINATION = register(StepSpec(
    key="illumination",
    title="Illumination correction (BaSiC)",
    category="Intensity",
    status=COMING_SOON,
    description=(
        "Corrects uneven illumination across the field (flat-field and dark-field, as BaSiC "
        "estimates them from the images themselves, Peng et al. 2017) and the loss of "
        "intensity with depth in Z stacks. Feature coming soon."),
    params=(
        Param("flatfield", "Flat-field", "bool", True, "Correct the uneven brightness across the field"),
        Param("darkfield", "Dark-field", "bool", False, "Also correct the camera offset pattern"),
        Param("smoothness", "Smoothness", "float", 1.0, "How smooth the estimated fields are",
              minimum=0.1, maximum=10.0, step=0.1),
        Param("z_decay", "Z intensity decay", "bool", False, "Correct the dimming with depth in a Z stack"),
        Param("estimate_from", "Estimate from", "choice", "this image",
              "Images the correction is estimated from", options=("this image", "every position")),
    ),
))

DECONVOLUTION = register(StepSpec(
    key="deconvolution",
    title="Deconvolution (Richardson–Lucy)",
    category="Restoration",
    status=COMING_SOON,
    description=(
        "Model-based Richardson–Lucy deconvolution with a computed point spread function, on "
        "the GPU when one is available. Feature coming soon."),
    params=(
        Param("na", "Numerical aperture", "float", 1.4, minimum=0.1, maximum=1.7, step=0.05),
        Param("wavelength", "Emission wavelength", "float", 520.0, minimum=300.0, maximum=1000.0,
              step=5.0, unit="nm", decimals=0),
        Param("immersion", "Immersion refractive index", "float", 1.515, minimum=1.0, maximum=1.6,
              step=0.005, decimals=3),
        Param("psf", "PSF model", "choice", "widefield (Gibson–Lanni)",
              options=("widefield (Gibson–Lanni)", "confocal", "two-photon")),
        Param("iterations", "Iterations", "int", 20, minimum=1, maximum=500, step=1, decimals=0),
        Param("regularization", "Regularization (TV)", "float", 0.002, minimum=0.0, maximum=0.1,
              step=0.001, decimals=4),
        Param("gpu", "Use the GPU", "bool", True, "Run on the GPU when one is available"),
    ),
))
