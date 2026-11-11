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
2. **Choose the objects.** Open the **Cytometry** section. The objects you measured are
   already chosen: the table (one row per object and time point / position; the joined
   *All objects* table when several layers were measured) and the labels layer they are
   in. To analyse another table, pick it and click **Use these objects**. Measuring the
   objects in use again updates them and keeps the gates, phenotypes and clusterings.
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
   - As soon as a gate is drawn, its objects are coloured on their segmented pixels in
     the image (the *<objects> selection* labels layer); selecting a gate in the list
     shows that one. Ctrl- or Shift-click selects several gates: each is shown in its
     colour, objects in more than one in white. Delete removes every selected gate.
   - **Points** (plot toolbar or right-click): point size, style and opacity.
   - The workbench opens as large as fits on the screen: a 4:3 plot, and the gate and
     phenotype lists at about half its width (drag the divider to change it). In a joined table (*All objects*), centres and positions come from
     the measurements of the chosen objects layer (e.g. `cells: centroid Y (µm)`).
   - Click a point to go to that object in the viewer. Click an object in the viewer
     to ring it on the plot.
4. **Or find the types by clustering** (the workbench's **Clustering** tab):
   - **Objects**: all of them, a gate, or a phenotype. Clustering a phenotype splits
     it further, so you can work from coarse to fine: cluster all cells, name the
     immune cluster, then cluster within *Immune*.
   - **Features**: which measurements to cluster on. **Intensities** ticks the mean
     intensity of every channel.
   - **asinh(x/150) first** compresses bright intensities. **Scaling** puts the
     features on one scale: z-score, robust (median/IQR), min-max, or none.
   - **Map**: UMAP, t-SNE or PCA, for looking at the objects in two dimensions.
   - **Clustering**:
     - **Leiden** and **Louvain** decide the number of clusters themselves (higher
       **Resolution** gives more, smaller clusters).
     - **k-means** and **Gaussian mixture** make **k** clusters.
     - **k-means, best k** tries 2…k and keeps the best by BIC.
   - The **Seed** makes a run repeatable. **Reproducible map (slower)** keeps a UMAP
     map exactly the same for a seed, but UMAP then runs on one core; unticked (the
     default) it uses every core, several times faster, and the layout varies a
     little. Clusters do not depend on it.
   - Speed: the neighbour search is made once (every core) and shared by Leiden /
     Louvain and UMAP; Leiden runs on igraph's C implementation (vtea-core
     `implementation="igraph"`, about 14x faster than leidenalg); t-SNE uses openTSNE;
     UMAP's numba code is compiled in the background when the workbench opens. The
     run's time is shown and saved with its settings.
   - **Run** (in the background). Each run becomes columns of the object table:
     `<name> cluster` (1 = the largest cluster; 0 = not clustered) and the map's
     `<name> UMAP 1/2`.
   - The results show a **heat map** of the clusters (z-score of each feature's
     mean) and a table with each cluster's size and markers (e.g.
     *CD8 mean+ CD3 mean+ / CD4 mean−*). Long names are shortened to fit (shared file
     name dropped, layer and statistic abbreviated: *Cell Alexa 647 mean*, *Nuc vol*);
     hover a label or a marker cell for the full name.
   - Type a **phenotype name** next to clusters and click **Make phenotypes**. Clusters
     given the same name become one phenotype, which is how an over-split population
     is merged.
   - **Plot the map** shows the map on the Gating tab, coloured by cluster, where you
     can gate on it. **Show clusters on image** colours the objects by cluster.
5. **Make phenotypes.** Select a gate and click **Make phenotype…**, choosing the
   phenotype it belongs **within**. Or click **Add…** to write a rule yourself, for
   example:
   - `` `gate: CD3+` AND NOT `gate: CD4+` ``
   - `` `CD8 mean` > 200 ``
   - `` `image gate: tumour` ``
   - `` `Clustering 1 cluster` in [3, 7] ``
   - `` `Clustering 1 cluster` == 2 & `cells: sample Alexa647 mean` < 100 ``

   Names with spaces or symbols go in backticks or quotes. A gate can be given by its
   name (`'CD3+'`), a measurement by its heat-map label (`'Cell Alexa647 mean'`) or by
   words only one column has (`'CD4 mean'`); an ambiguous name lists the candidates.
   The rule is stored with the exact names (`Cytometry.resolve_rule`). **Insert a
   column…** lists gates, image gates, clusters and measurements; the dialog's
   **Check** shows the number of objects as you type, or the problem. Phenotypes at
   one level are exclusive (the first that matches wins). A phenotype with an empty
   rule is "the rest of its parent": everything in the parent that an earlier sibling
   did not claim (e.g. *Other T*).
6. **Use the result.**
   - **Show phenotypes on image** (workbench: **Show on image**) colours every object by
     its phenotype, either the deepest (most specific) one or the one at a chosen level
     of the hierarchy.
   - **Export…** (workbench) writes the objects with every gate column, their phenotype, its
     path, and the phenotype at each level (CSV or Excel).

The tree shows, for each phenotype, its objects and its % of the parent. If an object
matches more than one sibling, the first sibling (top of the list) gets it, and the
others report the overlap. This usually means a gate was drawn too wide.

Everything (gates, image gates, clustering results, phenotypes and which objects they
apply to) is saved
with the workspace and restored when it is opened. Phenotypes are saved as rules, not
as lists of objects, so the same tree can type the next acquisition.

## Roadmap

The section is being built in stages. **Phase 1** (gating, subgating, image gates and
the phenotype hierarchy) and **Phase 2** (maps and clustering) are done. The later phases use the same object table and the
same phenotype tree. The larger tools get their own tabs in the workbench rather than
crowding the section.

| Phase | What | Engine |
|---|---|---|
| 1 ✅ | Plot and image gates, subgates, gates within phenotypes, phenotype hierarchy, show on image, export, workspace | vtea-core `gates`, `classes.hierarchy` (new) |
| 2 ✅ | **Dimensionality reduction and clustering** to find object types: feature choice and scaling (z-score, asinh), PCA / UMAP / t-SNE, k-means / Gaussian mixture / Leiden. A cluster × marker heat map to name clusters. Label clusters as phenotypes. Re-cluster within a phenotype, so you can work iteratively from coarse to fine. | vtea-core `reduction`, `clustering`, `clustering.profiles` (new) |
| 3 | **Spatial features per object**, once some identities are known. Distance to the nearest object of each phenotype, by centroid or by **surface** (exact distance between masks, in µm, 3D included). Number and fraction of each phenotype within a radius. Contacts (touching masks). These features can be gated and clustered like any measurement, to split types further (e.g. CD8 T cells touching tumour cells). | vtea-core (new `spatial` module) |
| 4 | **Neighbourhood analysis.** Cellular neighbourhoods: composition of each cell's *k* nearest neighbours or of the cells within a radius, clustered into neighbourhood types (the CODEX / Schürch–Nolan approach). Cell-type × neighbourhood enrichment, and pairwise neighbourhood enrichment with permutation z-scores. Neighbourhood maps on the image. Neighbourhood features reflected back onto cells for gating. | vtea-core `neighborhoods` (extended) |
| 5 | **Reference-based annotation with STELLAR**, in a separate integrated window. STELLAR ([snap-stanford/stellar](https://github.com/snap-stanford/stellar); Brbić et al., Nature Methods 2022) is a graph neural network over a cell graph. It transfers labels from an annotated dataset and finds new cell types. Optional install (PyTorch Geometric), run locally or on the facility GPU server / cluster. | vtea-core plug-in |

### Notes on the methods

- **STELLAR** (Brbić et al., *Nature Methods* 19, 1411–1418, 2022; code
  [snap-stanford/stellar](https://github.com/snap-stanford/stellar), MIT):
  - Node features are marker intensities. Edges join cells closer than a distance
    threshold (default 50, in the data's units).
  - The loss is pair consistency + margin cross-entropy − 0.3 × entropy. Novel types
    are seeded from Louvain clusters with high prediction entropy.
  - Its code is 2D, but the graph only needs centroids, so 3D works by giving it Z.
  - Its pinned dependencies are old (PyTorch 1.9, PyG 2.0), so it needs a port to
    current versions.
