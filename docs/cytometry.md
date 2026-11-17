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
   - **Level** (at the top) chooses which objects every tab works on: the segmented
     objects, or a spatial level built on them (neighbourhood windows or regions, at
     any depth; *Spatial level 1* is built on the objects, *Spatial level 2* on level 1,
     and so on). Each level has its own gates, phenotypes, clusterings, spatial
     features and classifications, its own plot set-up, and is saved with the
     workspace. The path beside it shows what the level is built on. With only the
     segmented objects, the selector is greyed out.
   - **Map** shows the **hierarchy map**, docked on the left once there is a spatial level:
     every level as a tree (● segmentation, ◆ windows, ▣ regions) with its number of
     objects, its type mix and how it was built. Click a level to work on it. The
     **Selection** column counts, at every other level, the objects related to the ones
     shown (members below, containers above); double-click a count to go there with them.
     Right-click to rename, export, delete, or make an *out of date* level again: a level
     turns red when its parent's objects, classes or types changed after it was built.
   - **Making a level.** On the **Neighbourhoods** tab, run a neighbourhood analysis,
     then **Make a level from this**: its windows (one around each object, overlapping)
     become the objects of a new level, with their composition (`fraction <class>`),
     size and the neighbourhood type of the object each is around (`neighbourhood
     type`). **Make regions…** instead merges contiguous objects of the same type into
     regions (functional units, no overlap): objects whose masks touch (above the
     segmentation: windows whose centre objects touch, regions whose members touch),
     or whose centres are within a distance; for windows, only when their members sit
     close to each other's (*Member closeness*, 0–1), so side-by-side structures stay
     apart. At the new level, *Neighbourhood types (from the level below)* are classes
     to build on again. **Windows around** / **Counting** restrict which objects get a
     window and which are counted in one (default: all).
   - **On the image**, a level is shown through the segmented objects: each object
     takes the value of its own window, or of its region. Surface distances and
     contacts (Spatial tab) need masks, so they are measured on the segmented objects.
   - **Follow selection to** takes the objects shown (Show) to another level: the
     members they are made of below, or the objects containing them above. It appears
     there as a *selection*: show it, gate within it, make it a phenotype, or delete it
     in the gate list.
   - **Columns from levels…** brings measurements of another level onto these objects:
     from the windows or regions containing each (their mean, the most common type, or
     its own window's), or summarised over each one's members.
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
   - **Train a model…** opens the **Training workbench**, a window of its own:
     - **Training set**: classes (with colours) and labelled objects. Add labels from
       phenotypes, clusters or a classification (each object with its class), from a
       gate (as the active class), by **clicking** objects (with *Label by clicking*
       on, each object picked in the viewer or clicked on the Gating plot is labelled
       as the active class), or **Import…** a CSV (`label`, `T` / `P`, `class`). The
       training set is saved with the workspace.
     - **Examples**: a thumbnail of each labelled object (the chosen image around
       it, its outline in its class's colour), all or by class, and after training
       the held-out objects the model got wrong. Click one to go to it; **Remove
       label** or **Move to active class**.
     - **Model**: the generic ANN for now (NephNet3D and STELLAR are listed, to
       come), its features, hidden layers, the **Validation** split (by object, or
       **by time point / position** so that each image is wholly on one side), and
       **Start from** a new model or a saved one to fine-tune (new classes are
       added; its features are used). **Train** runs in the background with live
       loss and balanced-accuracy curves; **Stop** keeps the best weights so far.
       **Results**: balanced accuracy, the confusion matrix, per-class precision,
       recall and F1. **Save to library** writes the model and its card, and the
       Classification tab offers it straight away.
   - **Model**: a model from the library (`~/.ibif-napari/models`, or the folder in
     `IBIF_NAPARI_MODELS`), or **Browse…** for a folder. Its card is shown, and its
     features are matched to the table's columns by name; missing ones are listed.
   - **Classify** gives each object `<name> class`, `<name> confidence` and
     `<name> p <class>`: plot, gate (e.g. confidence ≥ 0.8), **Colour the plot by
     class**, **Show classes on image**, and **Make phenotypes** (rename or merge
     classes, within a parent, only above a confidence). The classes can also be what
     the Spatial and Neighbourhoods tabs class objects by.
   - **A model applies to the kind of object it was trained on.** Its card records
     what it reads (measurements, image crops around masks, or the cells' neighbour
     graph) and the level it was trained at. The **Model** list shows every model but
     greys out those that do not fit the level being analysed, with the reason in its
     tooltip and on the card: a model trained on cells is not offered for
     neighbourhood windows, one trained on windows not for cells, and an image-crop
     model such as NephNet3D (nuclei) or a cell-graph model such as STELLAR only on
     the segmented objects. Train a model *at* a spatial level (with that level's
     phenotypes as labels) to classify its windows or regions. A model used at another
     depth than it was trained at (spatial level 2 with a level-1 model) is allowed,
     with a caution to check its features mean the same there. **Start from** offers
     only models for the current level. Cellular classifications reach the levels
     above as classes the neighbourhoods are measured over.
   - The first model family is a generic ANN (a multilayer perceptron over the
     measurements, CPU only). Models are folders of JSON and plain arrays, never
     pickles, so a shared library is safe to open. Engine: vtea-core
     `classification`. NephNet3D and STELLAR follow (see
     [classification-plan.md](classification-plan.md)).
8. **Follow lineages through the types** (the **Trajectory** tab), once objects have
   types:
   - **Objects**: all, a gate, or a phenotype, e.g. only the proximal tubule cells.
     **Classes**: phenotypes, clusters, a classification's classes, or a neighbourhood
     analysis's **types**. A trajectory of neighbourhood types follows how one niche
     turns into another; the neighbourhoods' composition is ticked as its features.
   - **Space**: PCA of the scaled features (**Components**), or a clustering's map
     (UMAP / t-SNE). **Start**: the class at pseudotime 0, chosen from biology (the
     healthy type, or a candidate transitional state). **Ends** (optional): classes
     that must end lineages.
   - **Distance**:
     - Slingshot's covariance-scaled distance (the default). On simulated
       trajectories it found the topology at least as often as Euclidean, and found
       trifurcations where Euclidean never did.
     - Or between class means (Euclidean).
     - With either, a class should cover each branch point. If the first classes of
       two branches are joined to each other rather than to the trunk, re-cluster or
       merge classes there.
   - **Knots** (tradeSeq, 6 by default): more follow short-lived changes; too few can
     bend a curve's ends.
   - **Engine**:
     - vtea-core's ports of slingshot and tradeSeq (the default);
     - or the R packages themselves (Bioconductor slingshot + tradeSeq through
       Rscript), offered when R with both is installed (`VTEA_RSCRIPT` points to an
       Rscript off the PATH).
   - **Also follow** (optional): a second classing whose fractions are followed along
     pseudotime. For example, neighbourhood types along a cell-type trajectory, or cell
     types along a neighbourhood trajectory.
   - **Run** (background). The classes are joined into a minimum spanning tree rooted
     at the start, with one lineage per leaf (*t1*, *t2*, …). Each lineage is fitted
     with a principal curve; lineages that share their first classes are shrunk onto
     a common trunk (Slingshot; Street et al. 2018). Each object gets a pseudotime per
     lineage and a weight. Then a Gaussian GAM per feature along the lineages, and its
     Wald tests (tradeSeq; Van den Berge et al. 2020):
     - association: does it change along a lineage?
     - start vs end;
     - different ends between lineages;
     - different patterns between lineages.

     Each test is run globally and per lineage or pair, with BH q-values.
   - Columns: `<name> pseudotime` (weighted mean over the object's lineages),
     `<name> lineage` (main lineage, 1 = t1), `<name> pseudotime t<j>` (empty off the
     lineage), and `<name> PC 1/2` when the space was PCA.
   - **Show**:
     - lineages on the map (by class, with the tree and the curves);
     - pseudotime on the map;
     - a feature along pseudotime per lineage: the GAM's fitted curve and 95% band over
       the binned means ± SEM of the z-scored feature (the "pseudotime spectrum" of
       Asghari et al. 2025, figs. 5 and 6);
     - every feature along one lineage (heat map);
     - which features change: −log10 q of each tradeSeq test (rows: test and lineage /
       pair; columns: features);
     - classes, and the followed classes, along pseudotime (stacked fractions).
   - **Colour the plot by pseudotime**, **Show pseudotime on image**. Engine: vtea-core
     `trajectory` (`slingshot`, `fit_gam` and the tests, `feature_trends`,
     `composition_along`, `r_engine`). See vtea-python `docs/TRAJECTORIES.md` for what
     is ported and the validation: simulations with a known truth, and parity with R by
     a CI workflow that has not run yet.
9. **Make phenotypes.** Selecting phenotypes in the list shows their objects on the
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
10. **Use the result.**
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
features), **Phase 4** (neighbourhoods) and **Phase 6** (trajectories, one dataset) are
done; Phase 5 is in progress. The later phases use the same object table and the
same phenotype tree. The larger tools get their own tabs in the workbench rather than
crowding the section.

| Phase | What | Engine |
|---|---|---|
| 1 ✅ | Plot and image gates, subgates, gates within phenotypes, phenotype hierarchy, show on image, export, workspace | vtea-core `gates`, `classes.hierarchy` (new) |
| 2 ✅ | **Dimensionality reduction and clustering** to find object types: feature choice and scaling (z-score, asinh), PCA / UMAP / t-SNE, k-means / Gaussian mixture / Leiden. A cluster × marker heat map to name clusters. Label clusters as phenotypes. Re-cluster within a phenotype, so you can work iteratively from coarse to fine. | vtea-core `reduction`, `clustering`, `clustering.profiles` (new) |
| 3 ✅ | **Spatial features per object**, once some identities are known. Distance to the nearest object of each phenotype, by centroid or by **surface** (exact distance between masks, in µm, 3D included). Number and fraction of each phenotype within a radius. Contacts (touching masks). These features can be gated and clustered like any measurement, to split types further (e.g. CD8 T cells touching tumour cells). | vtea-core `spatial` (new) |
| 4 ✅ | **Neighbourhood analysis.** Cellular neighbourhoods: composition of each cell's *k* nearest neighbours or of the cells within a radius, clustered into neighbourhood types (the CODEX / Schürch–Nolan approach). Cell-type × neighbourhood enrichment, and pairwise neighbourhood enrichment with permutation z-scores. Neighbourhood maps on the image. Neighbourhood features reflected back onto cells for gating. | vtea-core `neighborhoods` (extended) |
| 5 (in progress: 5.0–5.2 done) | **Supervised object classification**: a Classification tab (apply a model; class and confidence per object; make phenotypes) and a Training workbench (labels from phenotypes, gates or clicked objects; train or fine-tune; validate; model library). First models: a generic ANN on measurements, **NephNet3D** on 3D nuclear crops (Woloshuk et al. 2021), and **STELLAR** reference-based annotation (Brbić et al. 2022). Training locally, on the facility GPU server or a cluster. See [classification-plan.md](classification-plan.md). | vtea-core `classification` (extended) |
| 6 ✅ | **Trajectories** of cell types and neighbourhood types in one dataset: lineages from a chosen start class and pseudotime (Slingshot), a GAM per feature along the lineages and its tests (tradeSeq: association, start vs end, different ends, different patterns), class and neighbourhood-type fractions along pseudotime. Ports in vtea-core, validated on simulations; or the R packages when installed. After Asghari et al., *Sci Adv* 2025, figs. 5–6. | vtea-core `trajectory` (new) |
| Next | **Spatial hierarchy**: neighbourhoods as objects, built again on themselves at any depth; one workbench with a level selector and the same tabs at every level; expanded neighbourhood features; level-aware classification. See [spatial-hierarchy-plan.md](spatial-hierarchy-plan.md). | vtea-core `hierarchy` (new) |
| 7 (planned) | **Atlas / cohort analysis**: a pipeline built and checked here (phenotype rules or a classification model, neighbourhood definition, trajectory settings) frozen and applied unchanged to many datasets; a shared reference space; lineages fitted once and every dataset projected onto them; per-sample summaries and sample-level statistics across conditions. See [trajectory-atlas-plan.md](trajectory-atlas-plan.md). | vtea-core (headless cohort workflow) |

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
