# Plan: Processing section, and inclusions in Secondary objects

Status: **in progress**: M1 done (0.33.0), M2 done (0.33.1), M3 done (0.34.0; VV-SEC-07 limit to confirm on the first CI run), M4 done (0.35.0), M5 done (0.35.1). Target: 0.33.0 – 0.37.0.

## Decisions (from review, 2026-11-20)

1. **Inclusion output**: two layers, as it works today. One numbers each inclusion
   separately and stores its parent; the other numbers every inclusion by its cell.
2. **Watershed split**: **on by default** (*Watershed (distance)*). Inclusion counts and
   layers change from 0.32 for touching inclusions; the release notes say so, and *None*
   gives the old behaviour.
3. **Cellpose for spots**: a built-in model is enough. Validation uses the default
   built-in model; custom models stay selectable but are not validated in this round.
4. **Stubs**: **not addable** to the pipeline yet. They show in the library with a
   "coming soon" badge and a description only.
5. **Registration scope**: both **T** and **Z** in the first version.
   **Channel-to-channel (chromatic) alignment is later** (out of scope, §4).
6. **IMART**: re-implementing its approach (rigid, then non-rigid) in Python is
   accepted. Matching IMART's own output is not required.
7. **Server and cluster**: pipelines should run there **in this round if possible**.
   This is milestone M7 (§3).
8. **Real data**: the requester chooses the datasets, tests and signs off (M8).

This plan covers two requests:

1. Extend **Secondary objects**. Split touching "pieces" with a watershed in 2D and 3D,
   rename "pieces" to **inclusions**, and let *Structures inside each object → Find:
   Spots* run either **Cellpose** or **Intensity + watershed**.
2. Add a **Processing** section between *Configuration* and *Segmentation*. It holds an
   ordered list of processing steps chosen from a library. Selecting a step shows its
   settings in a pane below the list. The first three steps are illumination correction
   (BaSiC), stack registration (IMART-style) and model-based Richardson–Lucy
   deconvolution. Only **registration** is built in this plan. The other two are stubs
   that say "Feature coming soon".

---

## 1. User needs → requirements

User needs (UN) come from the request. Each requirement (RQ) is written so that it can be
tested, and each one points to the V&V check that tests it (§5). Requirement IDs follow
the existing suite (`src/ibif_napari/vv/requirements.py`).

### 1A. Secondary objects: inclusions

| UN | User need | Requirements |
|---|---|---|
| UN-1 | Touching structures inside an object (vesicles, puncta, nucleoli) should be counted and measured one by one, not merged into one blob. | **RQ-INC-01** *Structures inside each object* shall offer *Split touching inclusions: None / Watershed (distance) / Watershed (intensity)*, for Spots and Regions, on 2D images and on 3D stacks with anisotropic voxels. The default shall be *Watershed (distance)*. **RQ-INC-02** The watershed shall use the voxel spacing (µm) and a *minimum inclusion diameter* to merge seeds closer than that, as in the Threshold + watershed tab. **RQ-INC-03** The watershed shall never move an inclusion out of its parent object, and an inclusion shall never span two parents. |
| UN-2 | The word "pieces" should be "inclusions" everywhere. | **RQ-INC-04** Every place a user can see it shall say "inclusion(s)": the panel labels and tooltips, the measurement column (`inclusions`), cytometry feature names and short labels (`incl`), the README and the manual. **RQ-INC-05** Workspaces and CSVs saved by ≤ 0.32 (metadata `ibif_pieces`, column `pieces`) shall still open, with the old names mapped to the new ones. |
| UN-3 | For spots, choose between Cellpose and a classical intensity method. | **RQ-INC-06** When *Find* is *Spots*, a *Method* selector shall offer **Intensity + watershed** (the current background subtraction and threshold, then the RQ-INC-01 split; the default) and **Cellpose**. **RQ-INC-07** *Cellpose* shall run a chosen model (the default built-in model unless the user picks another) with a spot diameter on the chosen channel. It shall run on whatever backend the Segmentation tab is set to (this computer, server or cluster), and each spot shall go to the parent that holds most of its voxels. Spots mostly outside every parent (more than 50 % outside by default, adjustable) shall be dropped. **RQ-INC-08** If no Cellpose backend is available, the Cellpose option shall be disabled and its tooltip shall say why. |
| UN-1/3 | Each inclusion should stay linked to its cell. | **RQ-INC-09** The output shall be two layers. *`<name> inclusions`* numbers every inclusion on its own (1…N) and stores its parent's label. *`<name>`* keeps today's convention, with every inclusion carrying its parent's number. The measurements shall give the individual layer a `parent` column and the parent-numbered layer an `inclusions` count column, and the count shall match the number of individual inclusions with that parent. |

### 1B. Processing section

| UN | User need | Requirements |
|---|---|---|
| UN-4 | A place to correct the image before segmenting it. | **RQ-PROC-01** The panel shall show a *Processing* button (new icon) between *Configuration* and *Segmentation*. Its tooltip shall follow the existing style. |
| UN-5 | Build an ordered pipeline from a library of routines. | **RQ-PROC-02** The section shall show a **Library** list of available steps, a **Pipeline** list (add, remove, move up/down, drag to reorder, enable/disable a step), and a **step settings** pane under it that shows the fields of the selected pipeline step. **RQ-PROC-03** Every step shall be described by a schema (name, category, status, parameters with type, default, range, unit and tooltip) so the settings pane is built automatically and new routines need no UI code. **RQ-PROC-04** The pipeline shall run in order on a chosen image (all of its channels, and every T / position, or only the current one). It shall write new layers named `<name> [processed]` and leave the original untouched. Each new layer shall keep its scale, axes and a record of the steps that made it (`metadata["ibif_processing"]`). **RQ-PROC-05** Runs shall be cancellable, show progress, and keep napari responsive (worker thread). **RQ-PROC-06** The pipeline (steps, order, settings) shall be saved and restored with the workspace, and can be exported and imported as JSON so it can be reused and sent to a server or cluster (RQ-PROC-08). |
| UN-6 | Show the planned routines now, build them later. | **RQ-PROC-07** *Illumination correction (BaSiC)* and *Deconvolution (Richardson–Lucy, model PSF, GPU when available)* shall be listed in the library with a "coming soon" badge. Selecting one in the library shall show a short description, its planned parameters (disabled), and "Feature coming soon". A stub **cannot be added** to the pipeline: *Add* and drag-and-drop are disabled for it, and a pipeline JSON that names a stub is refused on import with a message naming it. |
| UN-8 | Run the processing where the GPU and the memory are. | **RQ-PROC-08** *Run on* shall offer the same choices as the Segmentation tab: this computer, a facility server, or a cluster session. The server shall run a pipeline sent as JSON, accepting only step keys and parameters in its own registry (no arbitrary code), and return the processed image and the transforms table. **RQ-PROC-09** A remote run shall give the same result as a local run of the same pipeline on the same data (within float32 round-off). **RQ-PROC-10** A server too old to run pipelines, or an upload over the server's size limit, shall be detected before sending, and the panel shall say which, and suggest running per position or time range. |
| UN-7 | Correct motion in time-lapse and Z stacks, for example intravital imaging (IMART). | **RQ-REG-01** *Stack registration* shall correct motion along **T** (frame to frame / to a reference) and along **Z** (plane to plane); both are in the first version, chosen by the user per step (adding the step twice does both). **RQ-REG-02** Transform models: *Translation* (sub-pixel), *Rigid* (translation + rotation), *Affine*, and *Non-rigid* (a dense displacement field applied after the rigid step, as IMART does with rigid followed by non-rigid registration). **RQ-REG-03** Reference: first frame, mean of the first *n* frames, a chosen frame, or *previous frame (cumulative)*. **RQ-REG-04** The transform shall be estimated on one reference channel (optionally smoothed or cropped to an ROI) and the same transform applied to every channel. **RQ-REG-05** Edges uncovered by the motion shall be filled with zero, NaN or the nearest value, or the result cropped to the region covered by every frame. **RQ-REG-06** The transforms shall be saved in a table (one row per frame or plane: shifts in px and µm, rotation, quality score) in the layer metadata, and exported as CSV. They shall be re-applicable to another layer, including Labels with nearest-neighbour interpolation. **RQ-REG-07** For 3D time-lapse (T Z Y X), *T* registration shall estimate one 3D transform per time point (translation and rigid at least). |

### Non-functional requirements

- **RQ-NF-01** Registration uses only NumPy, SciPy and scikit-image (already dependencies):
  `phase_cross_correlation`, `optical_flow_tvl1` / `optical_flow_ilk` and `ndimage` warps.
  No MATLAB and no new heavy packages. GPU (CuPy / torch) is optional and comes later.
- **RQ-NF-02** Performance: 2D translation on 512 × 512 × 200 frames takes under 30 s on a
  4-core laptop CPU. Non-rigid on the same data takes under 5 min. Memory stays within
  about 3 × the size of one channel (the image is processed frame by frame from dask).
- **RQ-NF-03** Everything new works headless (pure functions in `_processing/`, Qt only in
  the widget), so it can be tested in CI and run on the server or cluster.
- **RQ-NF-04** A new client keeps working with an old server for segmentation; only
  Processing is unavailable there (RQ-PROC-10).

---

## 2. Design

### 2A. Inclusions (`_secondary.py`, `_secondary_widget.py`, `_measure.py`, cytometry)

- `structures(..., split="none"|"distance"|"intensity", min_diameter=...)` returns the
  parent-numbered labels as today, **plus** the individual inclusion labels and a
  `{inclusion: parent}` map. The split reuses the seed-merging watershed from
  `_classic.py` (move it into a shared helper so both tabs behave the same), run inside
  each parent's mask so RQ-INC-03 holds by construction.
- `structures_cellpose(parents, image, spacing, model, diameter, ...)` calls the existing
  `_cellpose.segment` (local) or the remote and cluster path the Segmentation tab uses,
  then assigns parents by majority overlap. Cellpose's built-in models are trained on
  cells and nuclei, so small spots may need a custom model. This is a known risk that the
  V&V check measures (§5).
- Rename: `count_pieces` becomes `count_inclusions`, with the old name kept as an alias
  for one release. The metadata key `ibif_pieces` becomes `ibif_inclusions`, and both are
  read. Measurements write `inclusions`. Workspace and CSV loaders map `pieces` to
  `inclusions`. Cytometry: update the size-feature matcher in `_cytometry.py` (around
  line 213) and the `pcs` → `incl` abbreviation in `_cyto_labels.py`.

### 2B. Processing (`_processing/` package, `_processing_widget.py`)

```
_processing/
  __init__.py       registry: STEPS = {key: StepSpec}; register(step)
  base.py           StepSpec(key, title, category, status, params, run), Param(...)
  pipeline.py       Pipeline (ordered, JSON round-trip), run(layer/array, spacing, axes, progress, cancel)
  registration.py   estimate(...) -> Transforms table; apply(...); pure functions
  stubs.py          BaSiC, Richardson–Lucy: specs with status="coming_soon", planned params
_processing_widget.py  Library list | Pipeline list (+/−/↑/↓/enable) | settings form | Run on | Run / Cancel / progress
```

- Stubs: the *Add* button and drag-and-drop check `status`; `Pipeline.from_json` raises
  on a stub key, so a hand-edited or old JSON cannot bring one in.

- `SECTIONS` in `_panel.py` gets
  `Section("processing", "Processing", ..., "._processing_widget:ProcessingWidget")`
  between configuration and segmentation, plus `icons/processing.png`.
- The settings form is built from `Param`s (float/int/choice/bool/channel/layer), in the
  same style as the existing `_spin` helpers.
- Workspace: `_workspace.register_extension("processing", save, load)`.
- Output layers are added next to their source and become the default input of
  Segmentation (it already lists image layers).

### Running on the server or cluster (M7)

The cluster runs `ibif-napari-server` as a Slurm job behind an SSH tunnel, so supporting
the server gives the cluster too.

- **Server** (`_server.py`): a new job op `"process"`. The upload is an `.npz` with
  `image` (up to 5D: T, C, Z, Y, X; the segmentation upload is limited to 2D/3D today),
  `spacing`, `axes` and `pipeline` (JSON). `_check_pipeline` validates every step key and
  parameter against the registry, as `_check_params` does for Cellpose. The result
  `.npz` holds `image` and `transforms`. The job goes through the same queue, progress
  and cancel as segmentation. `/status` reports `"processing": <registry version>` so
  the client can tell whether a server supports it.
- **Client** (`_remote.py`): `RemoteProcess.process(image, spacing, axes, pipeline)`,
  a generator like `call`, with the same progress messages and cancel.
- **Large data**: the image is sent one position (and, if it is over the limit, one
  channel set) at a time. Registration along T needs the whole time series of a
  position, so a series over the server's `max_mb` is refused up front (RQ-PROC-10)
  rather than split.
- **Cluster batch** (`_batch.py`): out of scope for this round unless M7 goes quickly;
  the interactive cluster session covers it.

### Registration algorithm (IMART-style, two stages)

1. **Pre-filter** the reference channel: Gaussian blur and an optional percentile clip
   (reduces the effect of noise and of bright flashing cells).
2. **Rigid / translation**: `phase_cross_correlation` with upsampling (sub-pixel) against
   the reference. Rotation comes from log-polar phase correlation. Affine comes from an
   iterative intensity-based refinement. In cumulative mode, shifts are chained frame to
   frame and the drift is optionally smoothed.
3. **Non-rigid** (optional): TV-L1 optical flow between each rigidly aligned frame and
   the reference. The flow is regularised by a smoothing sigma (µm) and a maximum
   displacement cap, then applied with `ndimage.map_coordinates`.
4. **Quality**: normalised cross-correlation with the reference before and after, per
   frame, stored in the transforms table and shown as a sparkline in the status.

IMART itself is MATLAB software (Dunn, Lorenz, Salama & Delp, *IntraVital* 2014). As
agreed in review, this plan **re-implements its approach** (rigid, then non-rigid)
rather than calling IMART, and does not aim to match IMART's output exactly.

---

## 3. Milestones and deliverables

| # | Milestone | Deliverables | Exit criteria |
|---|---|---|---|
| **M0** | Plan review | This document, with the decisions answered | **Done** (Decisions, top of this file) |
| **M1** | Pieces → inclusions (0.33.0) | Rename in UI, `_measure`, cytometry, docs and manual; back-compat loaders; unit tests | RQ-INC-04/05 pass; a 0.32 workspace fixture opens with an `inclusions` column |
| **M2** | Inclusion watershed 2D/3D, on by default (0.33.x) | Shared watershed helper; `split` option (default distance); individual + parent-numbered layers; `parent` column; tests; V&V VV-SEC-05/06; release note on the changed default | RQ-INC-01/02/03/09 pass; CI and V&V green |
| **M3** | Spots: Cellpose or Intensity + watershed (0.34.0) | Method selector; `structures_cellpose` (local, server, cluster); disabled state; V&V VV-SEC-07 with the default built-in model | RQ-INC-06/07/08 pass. Validation limits met, or the gap reported with the measured F1 |
| **M4** | Processing framework + stubs (0.35.0) | Panel button and icon; `_processing` registry, schema-driven form, pipeline list; BaSiC and R–L stubs (library only, not addable); workspace and JSON persistence; Qt tests | RQ-PROC-01…07 pass (the pipeline runs end to end with a pass-through test step) |
| **M5** | Registration v1: translation and rigid, T and Z (0.35.x) | `registration.py` estimate/apply; T and Z modes; multi-channel; references; edge handling; transforms table and CSV; re-apply to labels; V&V VV-PROC-01…04, 07 | RQ-REG-01, 03…07 pass (translation and rigid); NF-02 met for translation |
| **M6** | Registration v2: affine and non-rigid (0.36.0) | Affine refinement; optical-flow stage; quality scores; V&V VV-PROC-05/06 | RQ-REG-02 complete; NF-02 met for non-rigid |
| **M7** | Processing on the server and cluster (0.37.0) | Server `"process"` op with `_check_pipeline`; `/status` capability; `RemoteProcess.process`; *Run on* in the panel; size and version checks; tests with the in-process server and fake Slurm; V&V VV-PROC-08 | RQ-PROC-08/09/10 and NF-04 pass. If it slips, 0.36.0 ships with local runs only and M7 follows on its own |
| **M8** | Real-data testing and release | Requester's datasets; README, manual chapter with screenshots, `docs/vv/requirements.md` regenerated | Acceptance criteria in §5.4 met; requester signs off |

M1–M3 (Secondary objects) and M4 (framework) are independent and can be built in
parallel. M5 needs M4; M7 needs M5 (it can start in parallel with M6).

---

## 4. Stubs: what "coming soon" includes now

Stubs appear in the library only; they cannot be added to the pipeline until built.

| Step | Shown now | Planned parameters (disabled) | Later mini-project |
|---|---|---|---|
| Illumination correction (BaSiC) | Description, badge, planned fields | Flat-field / dark-field on/off, smoothness, Z-illumination (per-plane intensity decay) correction, estimate from current or all positions | BaSiCPy integration, Z decay model |
| Deconvolution (R–L, model PSF) | Description, badge, planned fields, GPU detected yes/no | NA, emission λ, refractive index, iterations, regularisation (TV), PSF model (Gibson–Lanni / widefield / confocal), GPU on/off | GPU R–L (CuPy / torch), PSF generation, tiling |

Also deferred (not in the library yet): **channel-to-channel (chromatic) alignment**,
and processing in cluster batch jobs (`_batch.py`).

---

## 5. Verification & validation strategy

This plan uses the repo's existing V&V approach. **Verification** compares the tool with
an independent calculation of the same thing, usually to round-off. **Validation**
compares it with synthetic truth, within a tolerance chosen for microscopy work. Every
RQ maps to at least one test.

### 5.1 Levels

1. **Unit tests (pytest, every push, `ci.yml`)**: pure functions on small synthetic
   arrays, under a few seconds each.
2. **Widget tests (pytest-qt, offscreen)**: panel order, enabling and disabling, stubs,
   settings pane switching, list reorder, cancel, workspace round-trip.
3. **V&V suite (`ibif-napari-vv`, `vv.yml`)**: new requirements with acceptance limits in
   `LIMITS`, synthetic truth generated in `vv/synth.py`, and figures in the report. Add
   `_secondary.py`, `_processing/**` and `_measure.py` to the workflow paths (some are
   there already).
4. **Real-data acceptance (M8)**: the requester's datasets, tested and signed off by
   the requester.

### 5.2 New V&V requirements (draft limits; final values set from the first runs, as before)

| ID | Kind | Check | Draft acceptance |
|---|---|---|---|
| VV-SEC-05 | validation | Touching synthetic spots (2D and anisotropic 3D) inside synthetic cells, with the watershed split | Inclusion F1 ≥ 0.85; count per parent exact in ≥ 90 % of cells; F1 gain over no split ≥ 0.05 |
| VV-SEC-06 | verification | Every individual inclusion lies inside exactly one parent; `parent` column = parent label; `inclusions` count = number of individual inclusions with that parent | Exact (0 violations) |
| VV-SEC-07 | validation | Spots by Cellpose vs truth (real Cellpose, CPU) | Inclusion F1 ≥ 0.80 at SNR ≥ 8 (reported against Intensity + watershed on the same scenes) |
| VV-PROC-01 | verification | Pipeline runs steps in order; output = composition of the steps; original layer unchanged; metadata records the steps | Exact |
| VV-PROC-02 | validation | Known sub-pixel translations (±20 px, T and Z), with noise at SNR 4 and 8 | RMS residual ≤ 0.1 px (SNR 8), ≤ 0.25 px (SNR 4) |
| VV-PROC-03 | validation | Known rotations (±10°) and translations | Rotation error ≤ 0.2°; RMS residual ≤ 0.3 px |
| VV-PROC-04 | verification | Multi-channel: transform from channel A applied to B equals applying the known transform; a stack with no motion → identity (‖shift‖ ≤ 0.02 px); labels stay integer with no new IDs | Exact / to limit |
| VV-PROC-05 | validation | Synthetic smooth non-rigid warp (breathing-like) on top of rigid drift | Median end-point error ≤ 0.5 px; NCC to the reference improves by ≥ 0.1 |
| VV-PROC-06 | validation | End to end: segment the cells in a drifting synthetic time-lapse, with and without registration | Per-cell centroid track jitter falls by ≥ 80 %; segmentation F1 does not drop |
| VV-PROC-07 | verification | Intensity preserved: mean intensity of the overlapping region before and after (translation, linear interpolation) | Relative difference ≤ 1 % |
| VV-PROC-08 | verification | The same pipeline run locally and on the server (in-process server; also through the cluster tunnel with fake Slurm) | Same image within float32 round-off; same transforms table |

### 5.3 Unit and widget tests (examples)

- `test_secondary.py`: two touching discs or spheres in one parent split into 2 with the
  default settings and stay 1 with *None*; inclusions never cross parents; anisotropic spacing
  changes the seed merging as expected; `parent` map; old `pieces` metadata is read.
- `test_secondary_widget.py`: *Method* appears only for Spots; Cellpose is disabled with
  no backend; the split and min-diameter fields appear and hide correctly.
- `test_processing.py`: registry and schema; JSON round-trip; pipeline order; a stub
  cannot be added (button disabled, drag refused, JSON import refused); cancel mid-run
  leaves no partial layer.
- `test_server.py` / `test_cluster.py`: the `"process"` op; unknown step or parameter
  refused; cancel; an old server (no capability) disables Processing remotely; an
  upload over `max_mb` is refused before sending.
- `test_registration.py`: shift recovery (property-based, random shifts); Z vs T axis;
  edge fill modes; CSV export; re-apply to labels.
- `test_panel.py`: button order is Configuration, Processing, Segmentation, …
- `test_workspace.py`: the pipeline and the processed-layer metadata survive save/load.

### 5.4 Real-data acceptance (M8)

- The requester chooses the datasets, runs the tests and signs off. Suggested coverage:
  an intravital time-lapse with breathing and heartbeat motion, a Z stack with
  plane-to-plane drift, and a fixed sample with dense vesicles or puncta.
- Registration metrics: mean NCC between consecutive frames before and after; residual
  motion of 5–10 hand-marked landmarks (≤ 1 px median); a reviewer's visual check of a
  kymograph or orthogonal view.
- Inclusions: the requester counts inclusions in 20 cells by hand; tool count within
  ±10 % on average, with no systematic bias.
- If the server or cluster is available, one dataset is also run there and compared
  with the local result.
- Sign-off is recorded in `docs/vv/` with the dataset IDs and the V&V report version.

### 5.5 Traceability

`vv/requirements.py` gains the IDs above with `feature` strings that point at the panel
fields. `docs/vv/requirements.md` is regenerated. Each PR description lists the RQ and
VV IDs it closes.

---

## 6. Decisions

All eight review questions are answered; see **Decisions** at the top of this file.

## References

- Dunn KW, Lorenz KS, Salama P, Delp EJ. *IMART software for correction of motion
  artifacts in images collected in intravital microscopy.* IntraVital 2014; 3:e28210.
  [PMC4469201](https://pmc.ncbi.nlm.nih.gov/articles/PMC4469201)
- Peng T et al. *A BaSiC tool for background and shading correction of optical
  microscopy images.* Nat Commun 2017 (BaSiCPy for the later mini-project).
- Guizar-Sicairos M et al. *Efficient subpixel image registration algorithms.* Opt Lett
  2008 (`skimage.registration.phase_cross_correlation`).
