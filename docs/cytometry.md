# Cytometry

The Cytometry section turns measured objects into cell types. You gate objects on a
plot or on the image, gate within gates, and build a **hierarchy of phenotypes** (for
example *Immune > T cell > CD8 T cell*). You can then show any level of it on the
image, and export it.

The analysis engine is [vtea-core](https://github.com/winfrees/vtea-python), VTEA's
headless Python library. IBIF napari is the interface on top of it. New analysis
methods (spatial features, neighbourhoods, reference annotation) are added to
vtea-core, so VTEA and IBIF napari share them.

## Using it

1. **Measure the objects first.** In the Measurements section, tick the objects
   (e.g. cells) and the channels, and click **Measure**.
2. **Choose the objects.** Open the **Cytometry** section and pick the table (one row
   per object and time point / position) and the labels layer the objects are in.
   Then click **Use these objects**.
3. **Open the Cytometry workbench.** It is a separate window, so the plot can be large
   next to the image. It stays above napari and closes with it.
   - **Show** chooses which objects are plotted: all of them, a gate, an image gate or
     a phenotype. A gate you draw now is drawn **within** them. Within a gate, the new
     gate is a subgate; within a phenotype, it only ever contains that phenotype's
     objects.
   - **X / Y** can be any measurement, on a **linear**, **log10** or **asinh** scale.
     asinh(x/150) behaves like a log for bright objects but keeps zero and negative
     values, which suits background-subtracted intensities.
   - **Colour** the points by the shown population, by phenotype, or by any
     measurement.
   - **Rectangle / Polygon / Lasso gate**: draw on the plot, then name the gate. For a
     polygon, click its corners and click the first corner again to close it.
   - **Gate on image…**: draw rectangles or polygons on the image (a Shapes layer is
     added), then click **Make image gate**. An object is in an image gate when its
     centre is inside one of the regions. Regions are drawn in Y/X, so a 3D object
     counts when its centre's Y/X is inside, whatever its Z.
   - Click a point to go to that object in the viewer. Click an object in the viewer
     to ring it on the plot.
4. **Make phenotypes.** Select a gate and click **Make phenotype…**, choosing the
   phenotype it belongs **within**. Or click **Add…** to write a rule yourself, for
   example:
   - `` `gate: CD3+` AND NOT `gate: CD4+` ``
   - `` `CD8 mean` > 200 ``
   - `` `image gate: tumour` ``

   Names with spaces go in backticks. **Insert a column…** writes them for you. A
   phenotype with an empty rule is "the rest of its parent": everything in the parent
   that an earlier sibling did not claim (e.g. *Other T*).
5. **Use the result.**
   - **Show phenotypes on image** colours every object by its phenotype, either the
     deepest (most specific) one or the one at a chosen level of the hierarchy.
   - **Export table…** writes the objects with every gate column, their phenotype, its
     path, and the phenotype at each level (CSV or Excel).

The tree shows, for each phenotype, its objects and its % of the parent. If an object
matches more than one sibling, the first sibling (top of the list) gets it, and the
others report the overlap. This usually means a gate was drawn too wide.

Everything (gates, image gates, phenotypes and which objects they apply to) is saved
with the workspace and restored when it is opened. Phenotypes are saved as rules, not
as lists of objects, so the same tree can type the next acquisition.

## Roadmap

The section is being built in stages. **Phase 1** (gating, subgating, image gates and
the phenotype hierarchy) is done. The later phases use the same object table and the
same phenotype tree. The larger tools get their own tabs in the workbench rather than
crowding the section.

| Phase | What | Engine |
|---|---|---|
| 1 ✅ | Plot and image gates, subgates, gates within phenotypes, phenotype hierarchy, show on image, export, workspace | vtea-core `gates`, `classes.hierarchy` (new) |
| 2 | **Dimensionality reduction and clustering** to find object types: feature choice and scaling (z-score, asinh), PCA / UMAP / t-SNE, k-means / Gaussian mixture / Leiden. A cluster × marker heat map to name clusters. Label clusters as phenotypes. Re-cluster within a phenotype, so you can work iteratively from coarse to fine. | vtea-core `reduction`, `clustering` |
| 3 | **Spatial features per object**, once some identities are known. Distance to the nearest object of each phenotype, by centroid or by **surface** (exact distance between masks, in µm, 3D included). Number and fraction of each phenotype within a radius. Contacts (touching masks). These features can be gated and clustered like any measurement, to split types further (e.g. CD8 T cells touching tumour cells). | vtea-core (new `spatial` module) |
| 4 | **Neighbourhood analysis.** Cellular neighbourhoods: composition of each cell's *k* nearest neighbours or of the cells within a radius, clustered into neighbourhood types (the CODEX / Schürch–Nolan approach used by Hickey et al., Cell Systems 2024). Multi-scale composition (k = 10…200) as in RIBCA's tissue regions. Cell-type × neighbourhood enrichment, and pairwise neighbourhood enrichment with permutation z-scores. Neighbourhood maps on the image. Neighbourhood features reflected back onto cells for gating. | vtea-core `neighborhoods` (extended) |
| 5 | **Reference-based annotation**, in a separate integrated window. STELLAR (Brbić et al., Nature Methods 2022): a graph neural network over a cell graph. It transfers labels from an annotated dataset and finds new cell types. Optional install (PyTorch Geometric), run locally or on the facility GPU server / cluster. RIBCA (Sun et al., Cell Systems 2025): pretrained per-panel models for 2D multiplexed images, with confidence. | vtea-core plug-ins |

### Notes on the methods

- **Cellular neighbourhoods (Phase 4).** Hickey et al., *Cell Systems* 15, 322–338
  (2024), doi:10.1016/j.cels.2024.03.004, use the CODEX multicellular neighbourhood
  analysis on both their images and their simulations. The standard form of that
  analysis is a window of the 10 nearest cells, its cell-type composition, and k-means
  into ~10 neighbourhoods. This build could not read the paper's exact parameters (the
  journal was not reachable), so they are to be checked against its STAR Methods
  before Phase 4 is finalised. Distances are in µm, and 3D works without change.
- **RIBCA** (Sun et al., *Cell Systems* 2025; bioRxiv 10.1101/2024.09.12.612510; code
  `sun-huangqingbo/multiplexed-image-annotator`):
  - ViT classifiers on 40×40 patches, one per marker panel (immune base / extended /
    full, structural, nerve).
  - Missing markers are imputed by a masked autoencoder.
  - Its spatial analysis is kNN cell-type co-occurrence and multi-scale neighbourhood
    composition (k = 10…200), reduced with PCA and clustered (k-means) into tissue
    regions.
  - It is 2D only, the marker names must match exactly, and the weights come from
    Google Drive.
  - The licence needs checking: GitHub says GPL-3.0, the package metadata says BSD.
- **STELLAR** (Brbić et al., *Nature Methods* 19, 1411–1418, 2022; code
  `snap-stanford/stellar`, MIT):
  - Node features are marker intensities. Edges join cells closer than a distance
    threshold (default 50, in the data's units).
  - The loss is pair consistency + margin cross-entropy − 0.3 × entropy. Novel types
    are seeded from Louvain clusters with high prediction entropy.
  - Its code is 2D, but the graph only needs centroids, so 3D works by giving it Z.
  - Its pinned dependencies are old (PyTorch 1.9, PyG 2.0), so it needs a port to
    current versions.
