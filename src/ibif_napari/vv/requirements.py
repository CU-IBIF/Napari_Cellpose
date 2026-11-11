"""The requirements the V&V suite checks, with their acceptance criteria.

Each requirement comes from a feature of IBIF napari (the panel or function it covers).
`kind` says what a pass shows:

* verification: the software does exactly what it is specified to do (results are
  compared with an independent calculation of the same thing, usually to round-off);
* validation: the results are good enough for the scientific use (compared with the
  synthetic truth, within a tolerance chosen for typical microscopy work).

Thresholds live in `LIMITS` so they can be reviewed in one place and overridden for a
run (``ibif-napari-vv run --limits my_limits.json``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Acceptance limits (name -> value). Names are referred to by the requirements below.
LIMITS = {
    # Segmentation (Cellpose, plugin defaults)
    "seg2d_f1": 0.90, "seg2d_iou": 0.75, "seg2d_count": 0.10,
    "seg3d_f1": 0.85, "seg3d_iou": 0.65,
    "stitch_f1": 0.80,
    "cells_f1": 0.80, "cells_iou": 0.70,
    "snr_high_f1": 0.90,  # SNR >= 8
    "snr_low_f1": 0.70,  # SNR 4
    "region_f1_drop": 0.05,
    # Segmentation (Threshold + watershed tab, its defaults)
    "classic2d_f1": 0.90, "classic2d_iou": 0.80,
    "classic3d_f1": 0.85, "classic3d_iou": 0.75,
    "classic_snr4_f1": 0.85,
    "touch_f1": 0.85,  # touching nuclei, with the watershed (12 fresh 2D scenes: mean 0.95, lowest 0.85)
    "touch_gain": 0.05,  # F1 gained over connected components alone
    # Secondary objects
    "seeded_iou": 0.80, "seeded_f1": 0.90,
    "membrane_agree": 0.95,  # mean IoU between the cells of different automatic thresholds
    "unseeded_taken": 0.05,  # share of a cell without a seed that other cells may take
    "membrane_spill": 0.10,  # share of grown voxels outside their true cell and its membrane
    "membrane_size_ratio": 1.5,  # largest grown cell / its true cell
    # Measurements
    "exact_rel": 1e-9,
    "analytic_size_rel": 0.10,  # median |voxel size - ellipsoid size| / ellipsoid size
    "centroid_voxels": 0.5,  # centroid error, in voxel diagonals
    "noise_se": 4.0,  # measured mean within this many standard errors of the noise-free mean
    "drift_rel": 0.02,
    "e2e_size_rel": 0.15, "e2e_mean_rel": 0.10,
}


@dataclass
class Requirement:
    id: str
    area: str
    title: str
    statement: str  # "The tool shall ..."
    feature: str  # which part of IBIF napari it covers
    method: str  # how it is checked
    criteria: list  # human-readable acceptance criteria (may use {limit} names)
    kind: str = "validation"
    needs: list = field(default_factory=list)  # e.g. ["segmentation"], ["pylibCZIrw"]
    scenes: list = field(default_factory=list)

    def criteria_text(self, limits=None):
        limits = {**LIMITS, **(limits or {})}
        return [c.format(**limits) for c in self.criteria]


REQUIREMENTS = [
    # ---- the suite itself ------------------------------------------------------------------
    Requirement(
        "VV-H-01", "Harness", "Metrics are correct",
        "The V&V metrics shall score a perfect segmentation as perfect and known defects "
        "exactly as defined.",
        "V&V suite (metrics)", "Test: a renumbered truth, and the truth with one object "
        "removed, one false object added, two objects merged and one split.",
        ["perfect: F1 = AP = mean IoU = 1, no splits or merges",
         "defects: recall, precision, AP and split/merge counts equal their hand-computed values"],
        kind="verification", scenes=["nuclei_2d"]),
    # ---- reading files ---------------------------------------------------------------------
    Requirement(
        "VV-IO-01", "File reading", "Zeiss CZI files open with the right data and metadata",
        "Opening a CZI file shall give one layer per channel with the file's pixel values, "
        "channel names and voxel size (µm), and Z/Y/X axes.",
        "Drag-and-drop reader (.czi)", "Test: write a synthetic 2-channel Z stack as CZI, "
        "open it with the plugin's reader and compare with what was written.",
        ["pixel values identical", "channel names identical", "voxel size identical (to 1e-6 µm)"],
        kind="verification", needs=["czi"], scenes=["cells_3d"]),
    Requirement(
        "VV-IO-02", "File reading", "Leica LIF files open the chosen image correctly",
        "Opening a LIF file with several images shall offer them in a chooser and give the "
        "chosen image's pixel values, channel names, voxel size and time interval.",
        "Drag-and-drop reader (.lif) with image chooser", "Test: write a synthetic LIF "
        "with two images (one a time series), choose each, compare with what was written.",
        ["both images listed", "pixel values identical", "channel names identical",
         "voxel size and time interval identical (to 1e-6)"],
        kind="verification", needs=["lif"], scenes=["timelapse_3d"]),
    # ---- segmentation ---------------------------------------------------------------------
    Requirement(
        "VV-SEG-01", "Segmentation", "2D nuclei are found and outlined",
        "Cellpose segmentation of a 2D nuclear image shall find the nuclei and outline them "
        "accurately.",
        "Segmentation section (2D image)", "Test on the nuclei_2d scene (SNR 12).",
        ["F1 at IoU 0.5 >= {seg2d_f1}", "mean IoU of matched nuclei >= {seg2d_iou}",
         "|count error| <= {seg2d_count:.0%}"],
        needs=["segmentation"], scenes=["nuclei_2d"]),
    Requirement(
        "VV-SEG-02", "Segmentation", "3D nuclei are found in true 3D mode",
        "Cellpose in true 3D mode, with the anisotropy taken from the voxel size, shall find "
        "and outline nuclei in an anisotropic Z stack.",
        "Segmentation section, mode 3D", "Test on the nuclei_3d scene (Z step 1 µm, pixels 0.5 µm).",
        ["F1 at IoU 0.5 >= {seg3d_f1}", "mean IoU of matched nuclei >= {seg3d_iou}"],
        needs=["segmentation"], scenes=["nuclei_3d"]),
    Requirement(
        "VV-SEG-03", "Segmentation", "3D nuclei by 2D planes + stitching",
        "Segmenting each plane and stitching the planes shall give whole 3D nuclei.",
        "Segmentation section, mode 2D per plane + stitch", "Test on the nuclei_3d scene.",
        ["F1 at IoU 0.5 >= {stitch_f1}"],
        needs=["segmentation"], scenes=["nuclei_3d"]),
    Requirement(
        "VV-SEG-04", "Segmentation", "Whole cells with a nuclear channel",
        "Cellpose with a cell (cytoplasm) channel and a nuclear channel shall find and "
        "outline whole cells.",
        "Segmentation section, nuclear channel", "Test on the cells_2d scene.",
        ["F1 at IoU 0.5 >= {cells_f1}", "mean IoU of matched cells >= {cells_iou}"],
        needs=["segmentation"], scenes=["cells_2d"]),
    Requirement(
        "VV-SEG-05", "Segmentation", "Robust to image noise",
        "Segmentation quality shall hold up as the signal-to-noise ratio (SNR) falls to "
        "levels seen in live-cell imaging.",
        "Segmentation section", "Test on the same nuclei at SNR 20, 8, 4 and 2 (SNR 2 is "
        "reported only).",
        ["F1 at IoU 0.5 >= {snr_high_f1} at SNR 20 and 8", "F1 at IoU 0.5 >= {snr_low_f1} at SNR 4"],
        needs=["segmentation"], scenes=["snr_20", "snr_8", "snr_4", "snr_2"]),
    Requirement(
        "VV-SEG-06", "Segmentation", "Segmenting a painted region",
        "Segmenting only a painted region shall read only the region's bounding box, place "
        "the result correctly over the image, keep only objects mostly inside the painted "
        "area, and find them as well as a whole-image run does.",
        "Segmentation section: Region, Draw…, Drop cells mostly outside the painted area",
        "Test: paint a disc over part of nuclei_2d, run the panel's region steps "
        "(region_crop, painted_mask, segment, keep_inside) and compare with the true "
        "nuclei mostly inside the disc, cut to the region's bounding box (what the run sees).",
        ["the image read is the region's bounding box", "no object mostly outside the painted area",
         "F1 at IoU 0.5 within {region_f1_drop} of the whole-image F1 (or better)"],
        needs=["segmentation"], scenes=["nuclei_2d"]),
    Requirement(
        "VV-SEG-07", "Segmentation", "Threshold + watershed does each step as specified",
        "The Threshold + watershed tab shall threshold at the value the chosen method gives, "
        "label connected components exactly, and split objects only within the foreground, "
        "numbering them 1..n.",
        "Segmentation section, Threshold + watershed tab",
        "Test on nuclei_2d and nuclei_3d against an independent calculation with SciPy / "
        "scikit-image: the Otsu threshold of the same normalised, smoothed image; connected "
        "components of the same foreground; a manual threshold with no smoothing or clean-up "
        "against the plain image > value.",
        ["threshold = Otsu of the processed image (relative error <= {exact_rel:g})",
         "Split: none gives exactly the connected components of the foreground",
         "the watershed objects cover exactly the foreground, and each lies within one component",
         "objects numbered 1..n, none smaller than the minimum size",
         "manual threshold: objects = connected components of image > value, exactly"],
        kind="verification", scenes=["nuclei_2d", "nuclei_3d"]),
    Requirement(
        "VV-SEG-08", "Segmentation", "Threshold + watershed finds nuclei in 2D and 3D",
        "With its defaults, the Threshold + watershed tab shall find and outline separated "
        "nuclei in 2D and in anisotropic 3D stacks, also at low signal-to-noise. (It needs no "
        "GPU or model, so it is checked on every run, whichever segmentation back-end is chosen.)",
        "Segmentation section, Threshold + watershed tab (3D, and 2D per plane + stitch)",
        "Test on nuclei_2d (SNR 12), snr_4 and nuclei_3d (Z step 1 µm, pixels 0.5 µm), with the "
        "tab's defaults: smoothing 1 px, Otsu, fill holes, distance watershed, min size 15.",
        ["2D: F1 at IoU 0.5 >= {classic2d_f1}, mean IoU >= {classic2d_iou}",
         "SNR 4: F1 at IoU 0.5 >= {classic_snr4_f1}",
         "3D: F1 at IoU 0.5 >= {classic3d_f1}, mean IoU >= {classic3d_iou}",
         "3D by 2D planes + stitching: F1 at IoU 0.5 >= {classic3d_f1}"],
        scenes=["nuclei_2d", "snr_4", "nuclei_3d"]),
    Requirement(
        "VV-SEG-09", "Segmentation", "Touching nuclei are split by the watershed",
        "Nuclei that touch shall be separated by the watershed rather than merged as one "
        "connected component.",
        "Segmentation section, Threshold + watershed tab: Split touching objects",
        "Test on touching_2d and touching_3d (nuclei packed so that some press into each other), "
        "with the distance watershed (default) and with connected components only.",
        ["with the watershed: F1 at IoU 0.5 >= {touch_f1} in 2D and 3D",
         "the watershed's F1 is at least {touch_gain} higher than connected components alone",
         "the watershed merges fewer nuclei than connected components"],
        scenes=["touching_2d", "touching_3d"]),
    # ---- secondary objects ------------------------------------------------------------------
    Requirement(
        "VV-SEC-01", "Secondary objects", "Expand grows objects by the set distance",
        "Expand shall grow each object by the set distance in µm (anisotropic voxels "
        "included), keep its number, and give contested pixels to the nearer object.",
        "Secondary objects section: Expand", "Test on the truth nuclei of cells_3d, "
        "against an independent distance calculation.",
        ["every expanded pixel is within the distance of its own object",
         "every pixel within the distance of an object is assigned (to the nearest object)",
         "every object keeps its number and all of its pixels"],
        kind="verification", scenes=["cells_3d"]),
    Requirement(
        "VV-SEC-02", "Secondary objects", "Rings and rims have the set gap and width",
        "Ring shall make a band from the gap to gap + width outside each object, and Rim a "
        "band of the width inside it, in µm.",
        "Secondary objects section: Ring, Rim", "Test on the truth nuclei of cells_3d.",
        ["ring pixels lie between gap and gap + width from their own object",
         "rim pixels lie inside their object, closer than the width to its edge",
         "numbers kept"],
        kind="verification", scenes=["cells_3d"]),
    Requirement(
        "VV-SEC-03", "Secondary objects", "Whole cells grown from nuclei match the true cells",
        "Whole cells grown from nuclei in a cell channel shall match the true cells, each "
        "with its nucleus's number; the cytoplasm shall be the cell minus its nucleus.",
        "Secondary objects section: Whole cells from these seeds (Cytoplasm, Membranes), "
        "Also add cytoplasm",
        "Test with the truth nuclei as seeds on cells_2d and cells_3d: the cytoplasm channel "
        "as Cytoplasm and the membrane channel as Membranes, with the panel's defaults (Otsu "
        "threshold x 1.0, no maximum distance). In the membrane channel the inside of each "
        "cell is as dark as the background.",
        ["mean IoU with the true cells >= {seeded_iou}", "F1 at IoU 0.5 >= {seeded_f1}",
         "every cell has its seed's number, and no true cell's membrane is mostly another seed's number",
         "cytoplasm = cell minus nucleus, exactly"],
        scenes=["cells_2d", "cells_3d"]),
    Requirement(
        "VV-SEC-04", "Secondary objects", "Whole cells from a membrane stain stay within their membranes",
        "Whole cells grown from nuclei in a membrane channel shall not depend on which automatic "
        "threshold is chosen, shall stop at their membrane without a maximum distance, and shall "
        "leave a cell that has no nucleus (or an enclosed lumen) empty rather than give it to a "
        "neighbour.",
        "Secondary objects section: Whole cells from these seeds, Channel shows: Membranes",
        "Test with the truth nuclei as seeds on the membrane channel of cells_2d and cells_3d "
        "(the inside of each cell is as dark as the background), with smoothing 0.5 µm and no "
        "maximum distance: (1) Otsu, Li and Triangle; (2) the default Otsu, measuring how much "
        "of each grown cell lies outside its true cell and membrane; (3) each nucleus left out "
        "of the seeds in turn. A sweep of Threshold x from 0.3 to 1.5 is reported as evidence "
        "(not a criterion), to show the range where the method holds.",
        ["Otsu, Li and Triangle give the same cells: mean IoU with Otsu's cells >= {membrane_agree}",
         "with each of them, mean IoU with the true cells >= {seeded_iou}",
         "share of grown voxels outside their true cell and its membrane <= {membrane_spill}",
         "no grown cell larger than {membrane_size_ratio} x its true cell",
         "a cell left without its nucleus: at most {unseeded_taken} of its inside taken by other cells"],
        scenes=["cells_2d", "cells_3d"]),
    # ---- measurements ---------------------------------------------------------------------
    Requirement(
        "VV-MEAS-01", "Measurements", "Object size in µm² / µm³",
        "The measured area or volume shall equal the voxel count times the voxel size, and "
        "agree with the true shape's analytic size.",
        "Measurements section: area / volume, equivalent diameter",
        "Test on the truth nuclei and cells of nuclei_2d and cells_3d.",
        ["size = voxel count x voxel size (relative error <= {exact_rel:g})",
         "median relative difference from the analytic ellipse / ellipsoid size <= {analytic_size_rel:.0%}"],
        kind="verification", scenes=["nuclei_2d", "cells_3d"]),
    Requirement(
        "VV-MEAS-02", "Measurements", "Object centroids in µm",
        "Measured centroids shall be in µm in the image's coordinates (including its offset).",
        "Measurements section: centroid Z/Y/X (µm), click a row to go to the object",
        "Test on timelapse_3d (offset origin) through napari layers, against the analytic "
        "centres and an independent centre-of-mass calculation.",
        ["equal to the independent centre of mass (relative error <= {exact_rel:g})",
         "within {centroid_voxels} voxel diagonal of the true centre"],
        kind="verification", needs=["napari"], scenes=["timelapse_3d"]),
    Requirement(
        "VV-MEAS-03", "Measurements", "Intensity statistics per object",
        "Mean, median, min, max, standard deviation and sum of each channel shall be exact "
        "for each object, and the mean shall estimate the true (noise-free) mean.",
        "Measurements section: <channel> mean / median / min / max / std / sum",
        "Test on cells_3d (3 channels) against independent statistics, and against the "
        "noise-free image.",
        ["all six statistics equal the independent values (relative error <= {exact_rel:g})",
         "mean within {noise_se} standard errors of the noise-free mean for every object"],
        scenes=["cells_3d"]),
    Requirement(
        "VV-MEAS-04", "Measurements", "Time series and joined tables",
        "Measuring a time series shall give one row per object and time point, and the "
        "joined table shall put nucleus, cell and cytoplasm of the same number on one row.",
        "Measurements section: T column, All objects (one row per label)",
        "Test on timelapse_3d (3 time points, intensity rising 25% per step).",
        ["one row per object and time point", "intensity ratio to time 0 within "
         "{drift_rel:.0%} of the true ratio", "joined table: one row per (T, label) with all three objects"],
        needs=["napari"], scenes=["timelapse_3d"]),
    Requirement(
        "VV-MEAS-05", "Measurements", "Measurements on segmented nuclei are accurate",
        "Measurements of segmented nuclei shall be close to those of the true nuclei.",
        "Segmentation section + Measurements section", "Test: segment nuclei_3d, match to the "
        "truth, compare volume and mean intensity of the matched nuclei.",
        ["median relative volume error <= {e2e_size_rel:.0%}",
         "median relative mean-intensity error <= {e2e_mean_rel:.0%}"],
        needs=["segmentation"], scenes=["nuclei_3d"]),
    # ---- saving --------------------------------------------------------------------------
    Requirement(
        "VV-WS-01", "Saving", "A saved workspace reopens exactly",
        "Saving the workspace and opening it again shall restore every image and labels "
        "layer bit for bit, with its voxel size, offset and the measurement tables.",
        "Save workspace section: Save workspace / Open workspace", "Test on cells_3d.",
        ["arrays identical", "scale and translate identical", "measurement tables identical"],
        kind="verification", needs=["napari"], scenes=["cells_3d"]),
]

BY_ID = {r.id: r for r in REQUIREMENTS}


def markdown(limits=None):
    """The requirements as a Markdown document (docs/vv/requirements.md)."""
    lines = [
        "# IBIF napari: V&V requirements",
        "",
        "Generated from `src/ibif_napari/vv/requirements.py` "
        "(`ibif-napari-vv requirements > docs/vv/requirements.md`). Do not edit by hand.",
        "",
        "*Verification* compares the tool with an independent calculation of the same thing; "
        "*validation* compares it with the synthetic truth, within a tolerance.",
        "",
        "| ID | Area | Requirement | Feature | Kind |",
        "|---|---|---|---|---|",
    ]
    for r in REQUIREMENTS:
        lines.append(f"| {r.id} | {r.area} | {r.title} | {r.feature} | {r.kind} |")
    for r in REQUIREMENTS:
        lines += ["", f"## {r.id}: {r.title}", "", f"**Requirement.** {r.statement}", "",
                  f"**Feature.** {r.feature}", "", f"**Method.** {r.method}", "",
                  "**Acceptance criteria.**", ""]
        lines += [f"- {c}" for c in r.criteria_text(limits)]
        extra = []
        if r.scenes:
            extra.append("scenes: " + ", ".join(f"`{s}`" for s in r.scenes))
        if r.needs:
            extra.append("needs: " + ", ".join(r.needs))
        if extra:
            lines += ["", "*" + "; ".join(extra) + "*"]
    return "\n".join(lines) + "\n"
