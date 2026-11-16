# Trajectories across datasets: atlas and cohort analysis (plan)

**Status: planned, not built.** The Trajectory tab (Cytometry workbench, phase 6) works
on **one dataset**: the objects in use. This document describes the next step. A pipeline
is developed and checked on one dataset in IBIF napari, then applied **unchanged** to a set
of datasets. Their results are analysed together as an atlas, or as a cohort
meta-analysis comparing conditions (reference vs. disease, AKI vs. CKD, treated vs.
untreated).

The model is the multi-sample design of Asghari et al., *Sci Adv* 11:eadv8918 (2025;
[PMC12356270](https://pmc.ncbi.nlm.nih.gov/articles/PMC12356270/)):

- The CODEX discovery set was 9 specimens (5 reference, 4 disease). Each tissue was
  processed on its own, z-scaled, combined, clustered, and the PT cells subset.
- Trajectories were run on reference only, disease only, and both integrated (figs. 5,
  S12).
- PhenoCycler validation covered 49 samples, with trajectories from the
  THY1+PROM1+VCAM1+ cells (fig. 6).
- Neighbourhoods were typed ("spatial by cell", 55 µm) and related to disease by odds
  ratios (fig. 8).

The engine-side design is in vtea-python
[`docs/TRAJECTORIES.md`](https://github.com/winfrees/vtea-python/blob/main/docs/TRAJECTORIES.md).
This document covers what IBIF napari contributes and how a user would work.

## 1. What a dataset contributes today

Everything that turns an image into pseudotime is already saved with an IBIF napari
workspace:

| step | where it lives now |
| --- | --- |
| segmentation and measurements | Segmentation / Measurements settings, the object table |
| cell types | phenotype rules (`Cytometry.tree`, saved as rules, not object lists) or a classification model (model library, by hash) |
| neighbourhoods | Neighbourhoods settings (radius / k, number of types, classes) |
| trajectory | Trajectory settings (`analyses[<name>]`: population, classes, features, asinh, space, components, start, ends, distance) and its lineage structure and curves |

Phenotype rules and model cards are already meant to "type the next acquisition"
unchanged. The other settings are only stored per workspace.

## 2. Freezing a pipeline

New: **Export pipeline…** in the Cytometry section. It writes one versioned bundle:

- the measurement settings and channel names (with aliases, so `THY1` matches
  `Alexa647 THY1` in another panel);
- the scaling (asinh cofactor, z-score per dataset or to a reference);
- the phenotype tree, or the classification model's hash and card;
- the neighbourhood definition;
- the trajectory settings, including the start class chosen on the development dataset;
- the IBIF napari and vtea-core versions.

The bundle is hashed. Every result made with it records the hash, so a cohort table can
never mix pipelines silently.

A **Check on this dataset** button applies a bundle to the open dataset and reports what
does not fit: missing channels, classes with no objects, a start class that is empty.

## 3. Running a cohort

A cohort is a manifest (CSV or YAML) with one row per dataset: path, sample, donor,
condition, tissue / region, acquisition batch. Running it is headless. It uses the same
`_batch` machinery as batch segmentation, and runs locally, on the facility server or on
a SLURM cluster (`_cluster`). Each dataset gives an object table (Parquet) with its types,
neighbourhood types and the bundle hash.

## 4. One reference space, one set of lineages

Trajectories fitted on each dataset separately cannot be compared: "t2" in one sample
need not be "t2" in another. So:

1. **Types** come from the frozen rules or model, never from clustering each dataset
   anew.
2. **Normalisation** is per dataset: z-score within the dataset, or quantile-matching to
   reference datasets. Optional batch correction is reported with its effect: Harmony on
   the PCA, or a reference-mapping model.
3. **Reference space**: PCA (and optionally UMAP) fitted on reference datasets, or on a
   balanced subsample of all of them. Every dataset is *projected* into it.
4. **Lineages** (tree, curves, start) are fitted once in that space. Every object of every
   dataset is then projected onto the fixed curves to get its pseudotime. A pseudotime
   of 0.7 on t2 then means the same thing in every sample.

## 5. Statistics at the level of samples

The unit of replication is the sample or donor, not the cell: 100,000 cells from 4 donors
is n = 4. Per sample and lineage, the analysis summarises:

- lineage occupancy (the fraction of the population on each lineage, and beyond a
  pseudotime);
- the pseudotime distribution (median, quantiles, density in bins);
- feature trends (pseudotime spectra) and neighbourhood-type composition along
  pseudotime.

The comparisons:

- **conditions**: mixed models with donor as a random effect;
- **does a marker rise later or less in disease?** A condition × pseudotime interaction;
- **lineage occupancy and neighbourhood use**: odds ratios with 95% CI, as in the paper's
  fig. 8;
- **progression**: differential progression and differential fate tests (condiments:
  Roux de Bézieux et al., *Nat Commun* 15:833, 2024);
- multiple testing: Benjamini–Hochberg over features and lineages.

## 6. Robustness, by default

Every cohort report includes:

- lineages per condition and integrated (as fig. S12 did);
- alternative start classes;
- a bootstrap over samples for the stability of the tree and of each lineage;
- leave-one-dataset-out.

A lineage that appears in only one of these is flagged.

## 7. In IBIF napari

- **Cytometry section**: **Export pipeline…**, **Check on this dataset**, and
  **Project onto a reference trajectory**. The last places the open dataset's objects on
  a cohort's lineages, so one image can be read against the atlas: where are this
  biopsy's PT cells along t2?
- **A Cohort window**: the manifest, run status per dataset, and the summary plots
  (occupancy and pseudotime by condition, spectra per condition, neighbourhood
  composition along pseudotime per condition). It also exports the per-sample tables for
  R or Python.
- The Trajectory tab stays as it is for one dataset. The cohort analysis is a separate,
  headless workflow, so it can run on a cluster without a display.

## 8. Order of work

1. Pipeline bundle: export, hash, check on a dataset. Builds on workspace saving and
   model cards.
2. vtea-core: a public "project onto curves" function, the reference-space fit and
   projection (PCA, and UMAP `transform`), and per-sample summaries.
3. Headless cohort runner: manifest, `_batch` / `_cluster` integration, Parquet outputs
   with provenance.
4. Sample-level statistics and the robustness report.
5. The Cohort window and "project onto a reference trajectory" in the GUI.
