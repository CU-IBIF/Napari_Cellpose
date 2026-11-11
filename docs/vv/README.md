# Verification & validation (V&V) of IBIF napari

This suite shows, with evidence, that IBIF napari does what it is meant to do. It generates
synthetic microscopy images whose true content is known exactly, runs the plugin's own code
on them (the same functions the panels call), compares the results with the truth using
standard metrics, and writes a report that explains why the tool can be trusted, or where it
cannot.

- **Requirements:** [requirements.md](requirements.md). There are 22, each tied to a feature
  of the tool, with numeric acceptance criteria.
- **Code:** `src/ibif_napari/vv/`, with the command `ibif-napari-vv`.
- **Tests of the suite itself:** `tests/test_vv.py`.
- **Automatic runs:** `.github/workflows/vv.yml` runs the suite with real Cellpose on pull
  requests that touch the checked code, and on demand.

## Quick start

```bash
pip install -e ".[vv]"                       # in the IBIF napari environment
ibif-napari-vv all vv                        # generate + run (Cellpose) + report
# open vv/results/report.html
```

Or step by step:

```bash
ibif-napari-vv generate vv/truth             # the synthetic test data (truth)
ibif-napari-vv run vv/truth vv/results       # run the checks (Cellpose, CPU)
ibif-napari-vv run vv/truth vv/results --gpu # ... on the GPU
ibif-napari-vv report vv/results             # report.html + report.md
```

Useful options for `run` and `all`:

| Option | What it does |
|---|---|
| `--quick` (`generate`, `all`) | Smaller scenes. The full suite with Cellpose takes about 35 min on a 4-core CPU (GitHub's runner), much less on a GPU; the quick one is a few times faster. |
| `--backend cellpose` | The default. Uses the plugin's Cellpose segmentation with the panel's defaults. |
| `--backend classic` | The Segmentation section's *Threshold + watershed* tab, with its defaults (`baseline` is the old name). It needs no model weights, so the whole suite runs anywhere in a few minutes; the segmentation requirements then score that tab, and do **not** verify Cellpose. VV-SEG-07 to 09 test the tab on every run, whatever the back-end. |
| `--backend truth` | An oracle that returns the truth. Every segmentation metric must then be perfect: a self-check of the suite. |
| `--model NAME`, `--diameter PX`, `--gpu` | Test another Cellpose model, a fixed diameter, or the GPU. |
| `--only VV-SEG-01 VV-MEAS-03` | Run only some requirements. |
| `--limits limits.json` | Override acceptance limits, e.g. `{"seg3d_f1": 0.9}`. The names are in `LIMITS` in `requirements.py`. |
| `--strict` | Exit with status 1 unless every requirement passed. A requirement that was NOT RUN counts as a failure. |

The first Cellpose run downloads the model weights (about 1 GB for `cpsam`). If they cannot
be downloaded, the segmentation requirements are reported as **NOT RUN** with the reason; the
other requirements still run.

## What is in a results folder

```
results/
  report.html        the report (self-contained: open or share this one file)
  report.md          the same in Markdown (summary table and evidence)
  results.json       every number: criteria, metrics, timings, environment, limits
  tables/*.csv       per-object tables behind each requirement (matching, sizes, intensities…)
  figures/*.png      the figures in the report
  predictions/*.tif  each segmentation that was scored (open it next to the truth in napari)
  files/             the synthetic CZI and LIF files used for the file-reading checks
  workspace/         the workspace saved and reopened by VV-WS-01
```

## The test data (truth)

`ibif-napari-vv generate` writes one folder per scene. Each scene comes from fixed seeds,
so the data are identical on every computer and the truth needs no storage: it can always
be regenerated.

```
truth/<scene>/
  scene.json            all parameters (shape, voxel size, PSF, SNR, seed, …)
  nuclei.tif …          the "acquired" channels (uint16, as from a camera)
  ideal_nuclei.tif …    the same without noise (float)
  nuclei_truth.tif      true nuclei (labels)
  cells_truth.tif       true cells (labels; cell i contains nucleus i)
  truth.csv             per object: shape, analytic size, voxel counts, centroid, the
                        expected intensity statistics of every channel (noisy and noise-free)
```

How a scene is made (`synth.py`):

1. **Objects.** Nuclei are ellipsoids (ellipses in 2D) with random size, elongation and
   orientation, placed without overlap, and measured in µm. Each nucleus sits in a cell body
   about 2× larger. Neighbouring cells share a border, as in a Voronoi tessellation.
2. **Truth labels** are drawn on the pixel grid (anisotropic voxels included).
3. **Phantom channels:** *nuclei* (each nucleus a constant brightness), *cytoplasm* (each
   cell body a constant brightness, dark nucleus) and *membrane* (bright cell borders).
4. **Optics and camera:** a Gaussian PSF in µm (wider in Z), a background offset, and
   Gaussian noise set by the signal-to-noise ratio SNR = mean nuclear signal / noise σ.
   The result is rounded to uint16.
5. **Expected values** in `truth.csv` are computed with plain NumPy (`np.bincount`, sorting),
   independently of the plugin's code (which uses `scipy.ndimage`).

| Scene | Size | Voxel (µm) | Purpose |
|---|---|---|---|
| `nuclei_2d` | 256×256 | 0.5 | 2D nuclei, SNR 12 |
| `nuclei_3d` | 20×112×112 | 1.0×0.5×0.5 | 3D nuclei, anisotropic voxels |
| `cells_2d` | 256×256 | 0.5 | nuclei + cytoplasm + membrane channels |
| `cells_3d` | 20×112×112 | 1.0×0.5×0.5 | 3D cells: secondary objects, measurements, files, workspace |
| `timelapse_3d` | 3 T × 8×48×48 | 1.0×0.5×0.5 | intensities +25% per time point, offset origin |
| `touching_2d` | 256×256 | 0.5 | 50 nuclei packed so that some press into each other (12 contacts) |
| `touching_3d` | 20×112×112 | 1.0×0.5×0.5 | 16 nuclei, 3 contacts, anisotropic voxels |
| `snr_20`, `snr_8`, `snr_4`, `snr_2` | 256×256 | 0.5 | the same nuclei at falling SNR |

## How results are scored

Segmentation uses the standard instance-segmentation metrics (`metrics.py`). Objects are
paired one-to-one by intersection over union (IoU) with the Hungarian algorithm, and a pair
counts as a true positive (TP) at threshold *t* when its IoU ≥ *t*.

- precision = TP / (TP + FP), recall = TP / (TP + FN), **F1** = 2TP / (2TP + FP + FN)
- **AP** = TP / (TP + FP + FN), as used by Cellpose and the Data Science Bowl; **mAP** is its
  mean over IoU 0.50:0.05:0.95
- **mean IoU** and Dice of the matched pairs; **PQ** = SQ × RQ (panoptic quality)
- **splits / merges**: an object covered (≥ 50%) by several objects of the other side
- **count error** = (found − true) / true

Measurements are checked in two ways:

- **Verification:** the tool's numbers equal an independent calculation to round-off
  (relative error ≤ 1e-9).
- **Validation:** they are close to the truth within a stated tolerance. An example: the
  measured mean is within 4 standard errors (σ/√n) of the noise-free mean.

A requirement **passes** when all its criteria pass. **NOT RUN** means a prerequisite was
missing (Cellpose weights, pylibCZIrw…), with the reason. **ERROR** means the check crashed;
the traceback is in the report.

## What the suite found (and was fixed)

The first runs found two real problems. Both are fixed in the same change as the suite
(a third, found later, is listed after them):

1. **Leica LIF time series had no time interval** (VV-IO-02). bioio-lif only reports the
   interval for plate scans, so every ordinary LIF time series opened without seconds on the
   time slider. The reader now takes the interval from the file's T dimension (`_lif.py`).
   A unit test was added in `tests/test_lif.py`.
2. **"Whole cells from these seeds" with a cytoplasm channel could let one cell flood its
   neighbour** (VV-SEC-03), when neighbouring cells were about equally bright and the stain
   left the nucleus dark. Of 9 cells in `cells_3d`, one was swallowed (F1 0.89).
   `_secondary.seeded_cells` now gives the seeds the brightness of the cytoplasm next to
   them and uses a compact watershed on the scaled gradient, like CellProfiler's propagation.
   On 12 freshly seeded scenes that the suite does not use (to avoid tuning to the test
   data), F1 rose from 0.91 to 1.00 and mean IoU from 0.93 to 0.95.
3. **"Whole cells from these seeds" with a membrane channel gave a cell's membrane to its
   neighbour** (VV-SEC-03, found while making the user guide's screenshots; the suite then
   only tested the cytoplasm channel). Inside a cell a membrane stain is as dark as the
   background, so the Otsu foreground kept only the membranes. The watershed was limited to
   that foreground: a seed could not cross the dark inside to reach its own membrane, but
   touching membranes connected it to its neighbours', which it took. On a 34-cell 3D
   tissue scene, 18 cells had their membrane mostly labelled as another cell. Now each dark
   region next to a seed counts as the inside of that cell, and dark regions that touch no
   seed are flooded as the outside. Cells and outside meet on the ridge of the membrane,
   and each cell then keeps the membrane voxel just outside that ridge. VV-SEC-03 now also
   runs the membrane channel with the panel's defaults. On 12 freshly seeded
   `cells_2d`/`cells_3d` scenes, membrane F1 rose from 0.20 to 1.00 and mean IoU from 0.28
   to 0.94. The cytoplasm results did not change. VV-SEC-04 then added checks specific to
   membrane stains: Otsu, Li and Triangle must give the same cells, cells must stop at their
   membrane without a maximum distance, and a cell whose nucleus is missing must stay empty.
   It also reports a sweep of *Threshold ×* as evidence. That sweep shows a limit that is
   not fixed: from about ×1.2 (3D) or ×1.5 (2D) gaps open in the dimmer membranes and one
   cell can take the whole background (61× and 105× its true size). Keep *Threshold ×* near 1
   for membrane stains, or set a maximum distance.

## The Threshold + watershed tab

VV-SEG-07 to VV-SEG-09 cover the classical segmentation tab (`_classic.py`). It runs on the
CPU in well under a second per scene, so these run on every V&V run and in the unit tests,
whatever the segmentation back-end:

- **VV-SEG-07 (verification):** the threshold equals an independent Otsu of the same
  processed image; *Split: none* is exactly SciPy's connected-components labelling; the
  watershed only divides the foreground; a manual threshold with nothing else is exactly
  `image > value`.
- **VV-SEG-08 (validation):** separated nuclei in 2D, at SNR 4, in 3D and in 3D by planes +
  stitching. On the suite's scenes F1 = 1.00 (0.97 by planes) with mean IoU 0.91-0.96.
- **VV-SEG-09 (validation):** touching nuclei. Connected components merge them (F1 0.71 in
  2D, 0.90 in 3D); the distance watershed splits them (F1 0.97 and 1.00).

To avoid tuning to the test data, the defaults were also run on 12 freshly seeded copies of
each scene that the suite does not use. Separated nuclei (2D, 3D, SNR 4): F1 1.00 on every
one. Touching nuclei: watershed F1 mean 0.95 (lowest 0.85) in 2D and 0.98 (lowest 0.90) in 3D,
against 0.78 (lowest 0.63) and 0.75 (lowest 0.54) for connected components. Hence the 0.85
limit: nuclei pressed hard together, with no neck between them, can still merge. The images
are synthetic, with even staining and a dark background; on real tissue, check a few images
by eye, and prefer Cellpose for crowded or irregular cells.

## Adding a requirement

1. Add a `Requirement` to `REQUIREMENTS` in `requirements.py`: a new ID, the feature it
   covers, the method, and criteria that use names in `LIMITS`.
2. Write its check in `checks.py` with `@check("VV-…")`. Call the plugin's own functions,
   record each criterion with `res.check(name, measured, op, limit)`, and add evidence
   sentences, tables and figures.
3. Add a scene to `standard_scenes()` in `synth.py` if it needs new data.
4. Regenerate the document with
   `ibif-napari-vv requirements > docs/vv/requirements.md`. `tests/test_vv.py` checks that
   it is up to date and that every requirement has a check.

## Scope and limitations

- The images are synthetic: uniform staining, ellipsoidal shapes, a Gaussian PSF and
  Gaussian noise. Real samples (texture, irregular and clustered cells, uneven
  illumination) are harder. Treat the segmentation scores as an upper bound, and confirm
  them on a few annotated real images from your own instruments.
- ND2 files cannot be written by open tools, so the ND2 reader is covered by its unit tests,
  not by this suite.
- Interactive steps (painting, the 3D brush, dialogs, remote servers and clusters) are
  covered by the unit tests in `tests/`. This suite calls the same functions the panels
  call.
- Segmentation results depend on the Cellpose version, model and device. The report records
  them; re-run the suite after changing any of them.
