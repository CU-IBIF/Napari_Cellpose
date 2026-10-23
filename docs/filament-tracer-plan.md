# Plan: a filament tracer for IBIF napari

Status: **draft for review**. No code has been written yet. The questions in the last
section decide the scope.

## Goal

Trace filament-like structures in ND2/CZI/LIF images, in 2D or 3D, the way Imaris'
Filament Tracer does. Examples are neurons and dendrites, axons, vessels, actin or
microtubule bundles and mitochondrial networks. The tracer should then report
measurements such as total length, branch count, branch lengths, branch-point
positions, tortuosity and approximate diameter. Results should appear as napari layers
and export to CSV (and SWC for neurons).

## Option A: use existing napari plugins

These install alongside our package with **Plugins > Install/Uninstall Plugins**. That
already works today, because IBIF napari is ordinary napari.

| Plugin | What it does | Fit |
|---|---|---|
| [napari-tracing](https://pypi.org/project/napari-tracing/) (mapmanager) | Semi-automatic: click a start and end point and it finds the brightest path between them (A* / bidirectional search, [brightest-path-lib](https://pmc.ncbi.nlm.nih.gov/articles/PMC10370081)). Works in 2D and 3D. Saves and loads SWC. | Good for a few neurons or vessels traced by hand. Not automatic, and gives no branch statistics. |
| [napari-filaments](https://github.com/hanjinliu/napari-filaments) (hanjinliu) | Manual tracing with Shapes paths, fits splines, measures length at sub-pixel precision. | 2D only. Good for single filaments such as microtubules or actin. |
| [napari-filament-annotator](https://pypi.org/project/napari-filament-annotator) (amedyukhina) | 3D manual annotation from two projections, refined with active contours. | Slow by hand. Suits small numbers of 3D filaments. |
| [skan](https://github.com/jni/skan) napari plugin | Skeletonizes a Labels layer and gives skeleton/branch statistics. | Analysis only. Needs a binary mask as input, and its napari GUI is still basic ([v0.11 notes](https://github.com/jni/skan/releases/tag/v0.11.0)). |

**Pros:** almost no work for us, and it is available today.
**Cons:**
- None of them is an automatic Imaris-style tracer.
- Each has its own UI and its own idea of axes and units. They don't know our
  `ibif_axes` metadata, the T/P sliders, Region, Z planes or the facility server.
- Their maintenance status varies. Some have had few releases, so check compatibility
  with napari ≥ 0.5 before recommending them.

**Recommendation for A:** list them in the README as "also works", with a short how-to.
Don't make them dependencies. Test napari-tracing and skan against the napari version
we ship before listing them.

## Option B: a built-in "Filament Tracer" panel (recommended)

This would be a third dock panel, built from well-maintained Python libraries and
reusing what the Cellpose panel already has.

### Libraries

| Step | Library | Notes |
|---|---|---|
| Ridge / tube enhancement | `skimage.filters.sato`, `meijering`, `frangi` | Already installed with napari. Multi-scale and 3D, with `sigmas` set in µm and converted using the voxel size. |
| Threshold and cleanup | `skimage.filters` (Otsu/Li/manual), `skimage.morphology.remove_small_objects` | Already installed. |
| Skeletonize | `skimage.morphology.skeletonize` (2D and 3D) | Already installed. |
| Graph and statistics | [`skan`](https://github.com/jni/skan): `Skeleton`, `summarize` | New dependency (pulls in numba and pandas). Uses the voxel `spacing`, so lengths come out in µm. |
| Diameter | `scipy.ndimage.distance_transform_edt(mask, sampling=voxel)` sampled along the skeleton | Already installed. |
| Point-to-point tracing (phase 3) | `brightest-path-lib` | New, small and pure Python/numpy. The same engine as napari-tracing. |
| Optional ML enhancement (later) | A Cellpose or other custom model trained on filaments, run through the existing process/server | Only if classic filters aren't good enough on real data. |

Install `skan` and `brightest-path-lib` with an extra, `pip install "ibif-napari[filaments]"`.
The panel would then show a clear "install …" message when they are missing, the same
way the readers handle `bioio-czi`. Alternatively make them core dependencies if
everyone will use the tracer (see questions).

### Pipeline (one volume)

```
image (ZYX crop) ─▶ ridge filter (σ range in µm) ─▶ threshold ─▶ remove small objects
   ─▶ skeletonize ─▶ skan graph ─▶ prune spurs < L µm ─▶ measurements + layers
                                     └─▶ diameter from distance transform of the mask
```

This goes in a new module `src/ibif_napari/_filaments.py`. It would be pure functions on
numpy arrays, with no Qt, so it can be tested and later run on the facility server:

- `enhance(volume, voxel, sigmas_um, method) -> float array`
- `segment(enhanced, threshold, min_size) -> bool mask`
- `trace(mask, voxel, prune_um) -> (skeleton, branch_table, graph)`
- `measure(...) -> pandas.DataFrame`, one row per branch: id, filament id, length µm,
  euclidean length, tortuosity, mean intensity, mean diameter µm, branch type
  (end–end, end–junction, junction–junction).

### Panel (`_filament_widget.py`), modelled on `_cellpose_widget.py`

- **Image** combo. Voxel size comes from the layer scale and axes from `layer_axes`.
- **Filament diameter range (µm)**, the min and max for the filter sigmas, and the
  **Filter** method (Sato / Meijering / Frangi).
- **Threshold**: auto (Otsu) or a manual slider. A **Preview** button shows the enhanced
  image and the mask on the current plane, so you can tune before tracing.
- **Min object size**, **Prune spurs shorter than (µm)**.
- **Region** and **Z planes**: the same widgets and the same `_region.py` helpers
  (`region_crop`, `ZExtendedRegion`), so only the painted block is processed.
- **Run on**: current time point/position or every time point/position, the same as
  Cellpose.
- **Trace** / **Cancel**, progress bar and status line.

### Outputs in napari

- **Labels layer** `"<image> filaments"`: the skeleton, coloured by filament (connected
  component), with the image's scale/translate so it overlays correctly.
- **Shapes layer** (paths) or **Points**: branch points and end points, with a
  `features` table so clicking one shows its statistics. Optionally a Vectors/Shapes
  path layer per branch for 3D rendering.
- **Table**: a small table widget in the panel (per filament and per branch) with
  **Export CSV**. **Export SWC** for neuron users.
- The layer `metadata` keeps the parameters used, so a result can be reproduced.

### Where it runs

The first version runs in a `thread_worker` in napari. These filters are numpy/scipy,
so a thread is fine and Cancel takes effect between steps. Large volumes are a concern:
a 3D multi-scale Sato filter on 2k×2k×100 is slow and uses a lot of memory. If that is
a problem in practice, move it to the existing child process (`_process.py`, with a
second backend) to get instant Cancel. After that it could go to the facility server
(`_server.py` gets a `"trace"` op). The pure-function design keeps that move cheap.
GPU acceleration (cuCIM / CuPy versions of the same filters) is a possible later step.

### Phases

1. **Core and automatic tracing.** Write `_filaments.py`, the panel, the Labels and
   Points outputs and CSV export, using Region, Z planes and the current-volume scope.
   Add tests on synthetic tubes (straight line, Y junction, helix, noisy background)
   that check length within ~5 % and the correct branch and junction counts. Add a
   README section, `napari.yaml` widget registration and docking in `launcher.py`
   (tabbed). Bump the version.
2. **Time and position series.** Support "every time point/position". Add per-frame
   summary columns (total length, number of branches) for growth or retraction
   studies.
3. **Semi-automatic editing.** Add brightest-path tracing between two clicked points,
   either to add missed segments or to trace by hand. Allow deleting a branch by
   clicking it, then recomputing the statistics.
4. **Optional.** Soma detection (reuse Cellpose labels as seed points for the trees),
   per-tree Sholl analysis, spine detection, and running on the facility server or GPU.

Rough effort: phase 1 takes about 3–5 days including tests, phase 2 about 1 day,
phase 3 about 2–3 days. Phase 4 depends on which items are chosen.

### Risks

- **Classic filters on noisy or dense data.** Skeletons of thick or touching
  structures produce spurious spurs and loops. Pruning helps, and so do the preview
  and the phase 3 editing. Test early on real facility images.
- **Anisotropic Z.** Filters and skeletonization assume roughly isotropic voxels. The
  options are to scale sigmas per axis (supported by `sato`/`meijering`) or to
  resample Z to the XY pixel size before filtering and map the results back.
- **Memory on big volumes.** Region and Z planes limit the problem. Process by chunks
  later if needed.
- **New dependency weight.** `skan` pulls in numba. Keep it optional if that's a
  concern.

## Recommendation

Do **both, cheaply**. Document napari-tracing and skan as compatible plugins (Option A)
now. Build the native panel (Option B) in phases, starting with automatic tracing
using scikit-image and skan. That gives users an Imaris-like workflow that fits the
file readers, dimension sliders, Region and facility server, using libraries that are
already installed or are well maintained.

## Questions for review

1. **Which structures and samples matter most** (neurons/dendrites, vessels,
   cytoskeleton, mitochondria)? Could you share 2–3 example files? This decides the
   default filter, thresholds and whether we need spines or somata.
2. **Mostly 2D or 3D, and are time-lapses common?** That affects phase 2 priority.
3. **Which measurements are required?** For example length, branches, diameter, Sholl,
   tortuosity, or parity with specific Imaris statistics.
4. **Automatic first, or click-to-trace first?** If users mainly trace a few neurons
   by hand, phase 3 (brightest path) could come first.
5. **Optional extra or core dependency** for `skan` / `brightest-path-lib`?
6. **Should tracing also run on the facility GPU server** from the start?
