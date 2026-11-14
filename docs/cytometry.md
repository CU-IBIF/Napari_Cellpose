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
   - **Gate on image…**: a labels layer (*Cytometry image gate*) is added over the
     objects and selected for painting, with the same brush as the segmentation
     region: set the brush size and the eraser in napari's layer controls; in the 3D
     view the brush paints balls on the image. Paint over the objects to gate and
     click **Make image gate**, then name it. An object is in the gate when its centre
     is on a painted pixel. What is painted in 2D counts on every Z plane; a region
     painted with the 3D ball brush keeps its depth. Delete the layer to cancel.
   - Objects in an image gate are **ringed** on the plot in the gate's colour,
     whatever the plot shows, so you can see where a region's objects fall. Select a
     gate and click **Colour…** to change its colour (its outline, its objects on the
     image, and its rings). **Points > Ring objects in image gates** turns the rings
     off.
   - As soon as a gate is drawn, its objects are coloured on their segmented pixels in
     the image (the *<objects> selection* labels layer); selecting a gate in the list
     shows that one. Ctrl- or Shift-click selects several gates: each is shown in its
     colour, objects in more than one in white. Delete removes every selected gate.
   - **Points** (plot toolbar or right-click): point size, style and opacity. Under
     **Size**, **Custom size…** sets a diameter in pt, and **Size by a measurement…**
     sizes each point by a measurement (e.g. volume), from a smallest to a largest
     diameter over the 1st–99th percentile of its values.
   - The rectangle, polygon and lasso gates follow the mouse smoothly however many
     points there are: while a gate is drawn only its outline is redrawn (matplotlib
     blitting), not the points.
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
5. **Describe objects by their surroundings** (the workbench's **Spatial** tab), once
   some objects have identities:
   - **Classes**: the phenotypes (deepest, or at a level of the hierarchy) or the
     clusters of a clustering. **Towards** picks the classes to measure towards.
   - **Distance to the nearest (centres)**: to the nearest *other* object of each
     class, centroid to centroid.
   - **Distance to the nearest (mask to mask)**: the exact distance between the
     masks, voxel to voxel, in µm and in 3D (anisotropic voxels included); touching
     objects are one voxel apart. Slower than centres on large images.
   - **Objects within the radius**: how many objects of each class are within the
     radius, and what fraction of all the neighbours they are.
   - **Contacts**: how many objects of each class touch the mask.
   - Each result becomes columns of the object table (`Spatial 1 distance to Tumour
     (µm)`, `Spatial 1 n CD8 T within 30`, ...). Plot them, gate on them, write rules
     with them (`` `Spatial 1 contacts Tumour` > 0 AND `gate: CD8+` ``) or cluster on
     them, like any measurement. Distances never cross between time points or
     positions. Engine: vtea-core `spatial`.
6. **Find neighbourhood types** (the **Neighbourhoods** tab):
   - Each object's neighbourhood is itself and the objects within a **radius**, or
     its **k** nearest. Its composition is the fraction of each class in it.
   - **Types**: k-means on the compositions sorts neighbourhoods into types (the
     CODEX / Schürch–Nolan cellular neighbourhoods). Each object gets its
     neighbourhood's type (`Neighbourhoods 1 neighbourhood`, 1 = the most common)
     and its composition (`Neighbourhoods 1 fraction T`), as table columns.
   - **Show**: the composition of each type; which classes are enriched in each
     type (log2 observed / expected); and which classes are neighbours of which
     more than chance: the neighbour graph is kept and the classes are permuted
     among the objects (**Permutations**), giving z-scores.
   - **Colour the plot by type**, **Show types on image**. Engine: vtea-core
     `neighborhoods` and `spatial.enrichment`.
7. **Classify objects with a trained model** (the **Classification** tab):
   - **Train a model…**: labels from the phenotypes (deepest, or a level), features
     from the measurements (intensities by default), hidden layers, and a
     **Validation** split: by object, or **by time point / position** so that each
     image is wholly on one side (the honest test for new images). **Train** runs in
     the background and can be stopped; the balanced accuracy and confusion matrix on
     the held-out objects are shown. **Save to library** writes the model with its
     card (classes, colours, features, training data, validation, notes). **Start
     from** fine-tunes a saved model on new labels (same classes and features).
   - **Model**: a model from the library (`~/.ibif-napari/models`, or the folder in
     `IBIF_NAPARI_MODELS`), or **Browse…** for a folder. Its card is shown, and its
     features are matched to the table's columns by name; missing ones are listed.
   - **Classify** gives each object `<name> class`, `<name> confidence` and
     `<name> p <class>`: plot, gate (e.g. confidence ≥ 0.8), **Colour the plot by
     class**, **Show classes on image**, and **Make phenotypes** (rename or merge
     classes, within a parent, only above a confidence). The classes can also be what
     the Spatial and Neighbourhoods tabs class objects by.
   - The first model family is a generic ANN (a multilayer perceptron over the
     measurements, CPU only). Models are folders of JSON and plain arrays, never
     pickles, so a shared library is safe to open. Engine: vtea-core
     `classification`. NephNet3D and STELLAR follow (see
     [classification-plan.md](classification-plan.md)).
8. **Make phenotypes.** Selecting phenotypes in the list shows their objects on the
   image, as selecting gates does: Ctrl- or Shift-click several, each is shown in its
   colour and objects in more than one in white. Select a gate and click **Make phenotype…**, choosing the
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
9. **Use the result.**
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
the phenotype hierarchy), **Phase 2** (maps and clustering), **Phase 3** (spatial
features) and **Phase 4** (neighbourhoods) are done. The later phases use the same object table and the
same phenotype tree. The larger tools get their own tabs in the workbench rather than
crowding the section.

| Phase | What | Engine |
|---|---|---|
| 1 ✅ | Plot and image gates, subgates, gates within phenotypes, phenotype hierarchy, show on image, export, workspace | vtea-core `gates`, `classes.hierarchy` (new) |
| 2 ✅ | **Dimensionality reduction and clustering** to find object types: feature choice and scaling (z-score, asinh), PCA / UMAP / t-SNE, k-means / Gaussian mixture / Leiden. A cluster × marker heat map to name clusters. Label clusters as phenotypes. Re-cluster within a phenotype, so you can work iteratively from coarse to fine. | vtea-core `reduction`, `clustering`, `clustering.profiles` (new) |
| 3 ✅ | **Spatial features per object**, once some identities are known. Distance to the nearest object of each phenotype, by centroid or by **surface** (exact distance between masks, in µm, 3D included). Number and fraction of each phenotype within a radius. Contacts (touching masks). These features can be gated and clustered like any measurement, to split types further (e.g. CD8 T cells touching tumour cells). | vtea-core `spatial` (new) |
| 4 ✅ | **Neighbourhood analysis.** Cellular neighbourhoods: composition of each cell's *k* nearest neighbours or of the cells within a radius, clustered into neighbourhood types (the CODEX / Schürch–Nolan approach). Cell-type × neighbourhood enrichment, and pairwise neighbourhood enrichment with permutation z-scores. Neighbourhood maps on the image. Neighbourhood features reflected back onto cells for gating. | vtea-core `neighborhoods` (extended) |
| 5 (in progress: 5.0, 5.1 done) | **Supervised object classification**: a Classification tab (apply a model; class and confidence per object; make phenotypes) and a Training workbench (labels from phenotypes, gates or clicked objects; train or fine-tune; validate; model library). First models: a generic ANN on measurements, **NephNet3D** on 3D nuclear crops (Woloshuk et al. 2021), and **STELLAR** reference-based annotation (Brbić et al. 2022). Training locally, on the facility GPU server or a cluster. See [classification-plan.md](classification-plan.md). | vtea-core `classification` (extended) |
| Next | **Spatial hierarchy**: neighbourhoods as objects, built again on themselves at any depth; one workbench with a level selector and the same tabs at every level; expanded neighbourhood features; level-aware classification. See [spatial-hierarchy-plan.md](spatial-hierarchy-plan.md). | vtea-core `hierarchy` (new) |

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

### Neighbourhood analysis with VTEA in the kidney

The Neighbourhoods tab follows the cell-centric neighbourhood analysis our group has
published with VTEA (the "spatial by cell" method; composition as class sums and
fractions; neighbourhoods clustered into niches):

- Winfree S, McNutt AT, Khochare S, et al. Integrated cytometry with machine learning
  applied to high-content imaging of human kidney tissue for in situ cell
  classification and neighborhood analysis. *Laboratory Investigation* 103(6), 100104
  (2023). doi:10.1016/j.labinv.2023.100104. VTEA tissue cytometry with NephNet3D
  classification of kidney cell types from a nuclear stain, and neighbourhood analysis
  of the classified cells in situ.
- Lake BB, Menon R, Winfree S, et al. An atlas of healthy and injured cell states and
  niches in the human kidney. *Nature* 619, 585–594 (2023).
  doi:10.1038/s41586-023-05769-3. Injury neighbourhoods located by spatial
  transcriptomics, and large-scale 3D imaging analysis of about 1.2 million
  neighbourhoods linking them to active immune responses.
- Asghari M, Sabo AR, Barwinska D, et al. Integration of spatial protein imaging and
  transcriptomics in the human kidney tracks the regenerative potential of proximal
  tubules. *Science Advances* 11(33), eadv8918 (2025). doi:10.1126/sciadv.adv8918.
  Cell-centric neighbourhoods built with VTEA's "spatial by cell" method (55 µm
  diameter, matched to the Visium spot), described by class sums and fractions,
  clustered with Louvain into 19 niches, compared between reference and disease
  tissue by odds ratio.
