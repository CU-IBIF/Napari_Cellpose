# IBIF napari: V&V requirements

Generated from `src/ibif_napari/vv/requirements.py` (`ibif-napari-vv requirements > docs/vv/requirements.md`). Do not edit by hand.

*Verification* compares the tool with an independent calculation of the same thing; *validation* compares it with the synthetic truth, within a tolerance.

| ID | Area | Requirement | Feature | Kind |
|---|---|---|---|---|
| VV-H-01 | Harness | Metrics are correct | V&V suite (metrics) | verification |
| VV-IO-01 | File reading | Zeiss CZI files open with the right data and metadata | Drag-and-drop reader (.czi) | verification |
| VV-IO-02 | File reading | Leica LIF files open the chosen image correctly | Drag-and-drop reader (.lif) with image chooser | verification |
| VV-SEG-01 | Segmentation | 2D nuclei are found and outlined | Segmentation section (2D image) | validation |
| VV-SEG-02 | Segmentation | 3D nuclei are found in true 3D mode | Segmentation section, mode 3D | validation |
| VV-SEG-03 | Segmentation | 3D nuclei by 2D planes + stitching | Segmentation section, mode 2D per plane + stitch | validation |
| VV-SEG-04 | Segmentation | Whole cells with a nuclear channel | Segmentation section, nuclear channel | validation |
| VV-SEG-05 | Segmentation | Robust to image noise | Segmentation section | validation |
| VV-SEG-06 | Segmentation | Segmenting a painted region | Segmentation section: Region, Draw…, Drop cells mostly outside the painted area | validation |
| VV-SEG-07 | Segmentation | Threshold + watershed does each step as specified | Segmentation section, Threshold + watershed tab | verification |
| VV-SEG-08 | Segmentation | Threshold + watershed finds nuclei in 2D and 3D | Segmentation section, Threshold + watershed tab (3D, and 2D per plane + stitch) | validation |
| VV-SEG-09 | Segmentation | Touching nuclei are split by the watershed | Segmentation section, Threshold + watershed tab: Split touching objects | validation |
| VV-SEC-01 | Secondary objects | Expand grows objects by the set distance | Secondary objects section: Expand | verification |
| VV-SEC-02 | Secondary objects | Rings and rims have the set gap and width | Secondary objects section: Ring, Rim | verification |
| VV-SEC-03 | Secondary objects | Whole cells grown from nuclei match the true cells | Secondary objects section: Whole cells from these seeds (Cytoplasm, Membranes), Also add cytoplasm | validation |
| VV-SEC-04 | Secondary objects | Whole cells from a membrane stain stay within their membranes | Secondary objects section: Whole cells from these seeds, Channel shows: Membranes | validation |
| VV-MEAS-01 | Measurements | Object size in µm² / µm³ | Measurements section: area / volume, equivalent diameter | verification |
| VV-MEAS-02 | Measurements | Object centroids in µm | Measurements section: centroid Z/Y/X (µm), click a row to go to the object | verification |
| VV-MEAS-03 | Measurements | Intensity statistics per object | Measurements section: <channel> mean / median / min / max / std / sum | validation |
| VV-MEAS-04 | Measurements | Time series and joined tables | Measurements section: T column, All objects (one row per label) | validation |
| VV-MEAS-05 | Measurements | Measurements on segmented nuclei are accurate | Segmentation section + Measurements section | validation |
| VV-WS-01 | Saving | A saved workspace reopens exactly | Save workspace section: Save workspace / Open workspace | verification |

## VV-H-01: Metrics are correct

**Requirement.** The V&V metrics shall score a perfect segmentation as perfect and known defects exactly as defined.

**Feature.** V&V suite (metrics)

**Method.** Test: a renumbered truth, and the truth with one object removed, one false object added, two objects merged and one split.

**Acceptance criteria.**

- perfect: F1 = AP = mean IoU = 1, no splits or merges
- defects: recall, precision, AP and split/merge counts equal their hand-computed values

*scenes: `nuclei_2d`*

## VV-IO-01: Zeiss CZI files open with the right data and metadata

**Requirement.** Opening a CZI file shall give one layer per channel with the file's pixel values, channel names and voxel size (µm), and Z/Y/X axes.

**Feature.** Drag-and-drop reader (.czi)

**Method.** Test: write a synthetic 2-channel Z stack as CZI, open it with the plugin's reader and compare with what was written.

**Acceptance criteria.**

- pixel values identical
- channel names identical
- voxel size identical (to 1e-6 µm)

*scenes: `cells_3d`; needs: czi*

## VV-IO-02: Leica LIF files open the chosen image correctly

**Requirement.** Opening a LIF file with several images shall offer them in a chooser and give the chosen image's pixel values, channel names, voxel size and time interval.

**Feature.** Drag-and-drop reader (.lif) with image chooser

**Method.** Test: write a synthetic LIF with two images (one a time series), choose each, compare with what was written.

**Acceptance criteria.**

- both images listed
- pixel values identical
- channel names identical
- voxel size and time interval identical (to 1e-6)

*scenes: `timelapse_3d`; needs: lif*

## VV-SEG-01: 2D nuclei are found and outlined

**Requirement.** Cellpose segmentation of a 2D nuclear image shall find the nuclei and outline them accurately.

**Feature.** Segmentation section (2D image)

**Method.** Test on the nuclei_2d scene (SNR 12).

**Acceptance criteria.**

- F1 at IoU 0.5 >= 0.9
- mean IoU of matched nuclei >= 0.75
- |count error| <= 10%

*scenes: `nuclei_2d`; needs: segmentation*

## VV-SEG-02: 3D nuclei are found in true 3D mode

**Requirement.** Cellpose in true 3D mode, with the anisotropy taken from the voxel size, shall find and outline nuclei in an anisotropic Z stack.

**Feature.** Segmentation section, mode 3D

**Method.** Test on the nuclei_3d scene (Z step 1 µm, pixels 0.5 µm).

**Acceptance criteria.**

- F1 at IoU 0.5 >= 0.85
- mean IoU of matched nuclei >= 0.65

*scenes: `nuclei_3d`; needs: segmentation*

## VV-SEG-03: 3D nuclei by 2D planes + stitching

**Requirement.** Segmenting each plane and stitching the planes shall give whole 3D nuclei.

**Feature.** Segmentation section, mode 2D per plane + stitch

**Method.** Test on the nuclei_3d scene.

**Acceptance criteria.**

- F1 at IoU 0.5 >= 0.8

*scenes: `nuclei_3d`; needs: segmentation*

## VV-SEG-04: Whole cells with a nuclear channel

**Requirement.** Cellpose with a cell (cytoplasm) channel and a nuclear channel shall find and outline whole cells.

**Feature.** Segmentation section, nuclear channel

**Method.** Test on the cells_2d scene.

**Acceptance criteria.**

- F1 at IoU 0.5 >= 0.8
- mean IoU of matched cells >= 0.7

*scenes: `cells_2d`; needs: segmentation*

## VV-SEG-05: Robust to image noise

**Requirement.** Segmentation quality shall hold up as the signal-to-noise ratio (SNR) falls to levels seen in live-cell imaging.

**Feature.** Segmentation section

**Method.** Test on the same nuclei at SNR 20, 8, 4 and 2 (SNR 2 is reported only).

**Acceptance criteria.**

- F1 at IoU 0.5 >= 0.9 at SNR 20 and 8
- F1 at IoU 0.5 >= 0.7 at SNR 4

*scenes: `snr_20`, `snr_8`, `snr_4`, `snr_2`; needs: segmentation*

## VV-SEG-06: Segmenting a painted region

**Requirement.** Segmenting only a painted region shall read only the region's bounding box, place the result correctly over the image, keep only objects mostly inside the painted area, and find them as well as a whole-image run does.

**Feature.** Segmentation section: Region, Draw…, Drop cells mostly outside the painted area

**Method.** Test: paint a disc over part of nuclei_2d, run the panel's region steps (region_crop, painted_mask, segment, keep_inside) and compare with the true nuclei mostly inside the disc, cut to the region's bounding box (what the run sees).

**Acceptance criteria.**

- the image read is the region's bounding box
- no object mostly outside the painted area
- F1 at IoU 0.5 within 0.05 of the whole-image F1 (or better)

*scenes: `nuclei_2d`; needs: segmentation*

## VV-SEG-07: Threshold + watershed does each step as specified

**Requirement.** The Threshold + watershed tab shall threshold at the value the chosen method gives, label connected components exactly, and split objects only within the foreground, numbering them 1..n.

**Feature.** Segmentation section, Threshold + watershed tab

**Method.** Test on nuclei_2d and nuclei_3d against an independent calculation with SciPy / scikit-image: the Otsu threshold of the same normalised, smoothed image; connected components of the same foreground; a manual threshold with no smoothing or clean-up against the plain image > value.

**Acceptance criteria.**

- threshold = Otsu of the processed image (relative error <= 1e-09)
- Split: none gives exactly the connected components of the foreground
- the watershed objects cover exactly the foreground, and each lies within one component
- objects numbered 1..n, none smaller than the minimum size
- manual threshold: objects = connected components of image > value, exactly

*scenes: `nuclei_2d`, `nuclei_3d`*

## VV-SEG-08: Threshold + watershed finds nuclei in 2D and 3D

**Requirement.** With its defaults, the Threshold + watershed tab shall find and outline separated nuclei in 2D and in anisotropic 3D stacks, also at low signal-to-noise. (It needs no GPU or model, so it is checked on every run, whichever segmentation back-end is chosen.)

**Feature.** Segmentation section, Threshold + watershed tab (3D, and 2D per plane + stitch)

**Method.** Test on nuclei_2d (SNR 12), snr_4 and nuclei_3d (Z step 1 µm, pixels 0.5 µm), with the tab's defaults: smoothing 1 px, Otsu, fill holes, distance watershed, min size 15.

**Acceptance criteria.**

- 2D: F1 at IoU 0.5 >= 0.9, mean IoU >= 0.8
- SNR 4: F1 at IoU 0.5 >= 0.85
- 3D: F1 at IoU 0.5 >= 0.85, mean IoU >= 0.75
- 3D by 2D planes + stitching: F1 at IoU 0.5 >= 0.85

*scenes: `nuclei_2d`, `snr_4`, `nuclei_3d`*

## VV-SEG-09: Touching nuclei are split by the watershed

**Requirement.** Nuclei that touch shall be separated by the watershed rather than merged as one connected component.

**Feature.** Segmentation section, Threshold + watershed tab: Split touching objects

**Method.** Test on touching_2d and touching_3d (nuclei packed so that some press into each other), with the distance watershed (default) and with connected components only.

**Acceptance criteria.**

- with the watershed: F1 at IoU 0.5 >= 0.85 in 2D and 3D
- the watershed's F1 is at least 0.05 higher than connected components alone
- the watershed merges fewer nuclei than connected components

*scenes: `touching_2d`, `touching_3d`*

## VV-SEC-01: Expand grows objects by the set distance

**Requirement.** Expand shall grow each object by the set distance in µm (anisotropic voxels included), keep its number, and give contested pixels to the nearer object.

**Feature.** Secondary objects section: Expand

**Method.** Test on the truth nuclei of cells_3d, against an independent distance calculation.

**Acceptance criteria.**

- every expanded pixel is within the distance of its own object
- every pixel within the distance of an object is assigned (to the nearest object)
- every object keeps its number and all of its pixels

*scenes: `cells_3d`*

## VV-SEC-02: Rings and rims have the set gap and width

**Requirement.** Ring shall make a band from the gap to gap + width outside each object, and Rim a band of the width inside it, in µm.

**Feature.** Secondary objects section: Ring, Rim

**Method.** Test on the truth nuclei of cells_3d.

**Acceptance criteria.**

- ring pixels lie between gap and gap + width from their own object
- rim pixels lie inside their object, closer than the width to its edge
- numbers kept

*scenes: `cells_3d`*

## VV-SEC-03: Whole cells grown from nuclei match the true cells

**Requirement.** Whole cells grown from nuclei in a cell channel shall match the true cells, each with its nucleus's number; the cytoplasm shall be the cell minus its nucleus.

**Feature.** Secondary objects section: Whole cells from these seeds (Cytoplasm, Membranes), Also add cytoplasm

**Method.** Test with the truth nuclei as seeds on cells_2d and cells_3d: the cytoplasm channel as Cytoplasm and the membrane channel as Membranes, with the panel's defaults (Otsu threshold x 1.0, no maximum distance). In the membrane channel the inside of each cell is as dark as the background.

**Acceptance criteria.**

- mean IoU with the true cells >= 0.8
- F1 at IoU 0.5 >= 0.9
- every cell has its seed's number, and no true cell's membrane is mostly another seed's number
- cytoplasm = cell minus nucleus, exactly

*scenes: `cells_2d`, `cells_3d`*

## VV-SEC-04: Whole cells from a membrane stain stay within their membranes

**Requirement.** Whole cells grown from nuclei in a membrane channel shall not depend on which automatic threshold is chosen, shall stop at their membrane without a maximum distance, and shall leave a cell that has no nucleus (or an enclosed lumen) empty rather than give it to a neighbour.

**Feature.** Secondary objects section: Whole cells from these seeds, Channel shows: Membranes

**Method.** Test with the truth nuclei as seeds on the membrane channel of cells_2d and cells_3d (the inside of each cell is as dark as the background), with smoothing 0.5 µm and no maximum distance: (1) Otsu, Li and Triangle; (2) the default Otsu, measuring how much of each grown cell lies outside its true cell and membrane; (3) each nucleus left out of the seeds in turn. A sweep of Threshold x from 0.3 to 1.5 is reported as evidence (not a criterion), to show the range where the method holds.

**Acceptance criteria.**

- Otsu, Li and Triangle give the same cells: mean IoU with Otsu's cells >= 0.95
- with each of them, mean IoU with the true cells >= 0.8
- share of grown voxels outside their true cell and its membrane <= 0.1
- no grown cell larger than 1.5 x its true cell
- a cell left without its nucleus: at most 0.05 of its inside taken by other cells

*scenes: `cells_2d`, `cells_3d`*

## VV-MEAS-01: Object size in µm² / µm³

**Requirement.** The measured area or volume shall equal the voxel count times the voxel size, and agree with the true shape's analytic size.

**Feature.** Measurements section: area / volume, equivalent diameter

**Method.** Test on the truth nuclei and cells of nuclei_2d and cells_3d.

**Acceptance criteria.**

- size = voxel count x voxel size (relative error <= 1e-09)
- median relative difference from the analytic ellipse / ellipsoid size <= 10%

*scenes: `nuclei_2d`, `cells_3d`*

## VV-MEAS-02: Object centroids in µm

**Requirement.** Measured centroids shall be in µm in the image's coordinates (including its offset).

**Feature.** Measurements section: centroid Z/Y/X (µm), click a row to go to the object

**Method.** Test on timelapse_3d (offset origin) through napari layers, against the analytic centres and an independent centre-of-mass calculation.

**Acceptance criteria.**

- equal to the independent centre of mass (relative error <= 1e-09)
- within 0.5 voxel diagonal of the true centre

*scenes: `timelapse_3d`; needs: napari*

## VV-MEAS-03: Intensity statistics per object

**Requirement.** Mean, median, min, max, standard deviation and sum of each channel shall be exact for each object, and the mean shall estimate the true (noise-free) mean.

**Feature.** Measurements section: <channel> mean / median / min / max / std / sum

**Method.** Test on cells_3d (3 channels) against independent statistics, and against the noise-free image.

**Acceptance criteria.**

- all six statistics equal the independent values (relative error <= 1e-09)
- mean within 4.0 standard errors of the noise-free mean for every object

*scenes: `cells_3d`*

## VV-MEAS-04: Time series and joined tables

**Requirement.** Measuring a time series shall give one row per object and time point, and the joined table shall put nucleus, cell and cytoplasm of the same number on one row.

**Feature.** Measurements section: T column, All objects (one row per label)

**Method.** Test on timelapse_3d (3 time points, intensity rising 25% per step).

**Acceptance criteria.**

- one row per object and time point
- intensity ratio to time 0 within 2% of the true ratio
- joined table: one row per (T, label) with all three objects

*scenes: `timelapse_3d`; needs: napari*

## VV-MEAS-05: Measurements on segmented nuclei are accurate

**Requirement.** Measurements of segmented nuclei shall be close to those of the true nuclei.

**Feature.** Segmentation section + Measurements section

**Method.** Test: segment nuclei_3d, match to the truth, compare volume and mean intensity of the matched nuclei.

**Acceptance criteria.**

- median relative volume error <= 15%
- median relative mean-intensity error <= 10%

*scenes: `nuclei_3d`; needs: segmentation*

## VV-WS-01: A saved workspace reopens exactly

**Requirement.** Saving the workspace and opening it again shall restore every image and labels layer bit for bit, with its voxel size, offset and the measurement tables.

**Feature.** Save workspace section: Save workspace / Open workspace

**Method.** Test on cells_3d.

**Acceptance criteria.**

- arrays identical
- scale and translate identical
- measurement tables identical

*scenes: `cells_3d`; needs: napari*
