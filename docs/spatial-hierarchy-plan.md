# Spatial hierarchy: neighbourhoods as objects, at any depth (plan)

Status: proposal. Builds on phase 4 (neighbourhoods) and phase 5 (classification) of
[cytometry.md](cytometry.md); the engine work goes in vtea-core, the interface in the
Cytometry workbench, as for phases 1–5.

## 0. Where neighbourhoods stand today (review)

**Overlap is allowed, and is built correctly.** A cell-centric neighbourhood
(`radius` or `nearest`) is drawn around *every* object, so an object is a member of its
own neighbourhood and of every neighbour's. vtea-core holds membership as a
many-to-many table (`NeighborhoodSet.membership()`), never as one parent per object. On
500 random cells (radius 30, k = 10) every cell is a member of 3–26 neighbourhoods,
median 11–15.

What was wrong, and is fixed on this branch:

1. **vtea-core, k nearest with tied positions.** The tree query asks for k + 1 points
   on the assumption that the seed comes back first. Objects at the same centroid tie at
   distance 0, so the seed could be missing from the result and was then added back,
   which gave k + 1 *others* (k + 2 members). The seed is now dropped wherever it is
   and the k nearest others kept. Test: `test_nearest_keeps_exactly_k_others_when_positions_tie`.
2. **Workbench, overlap was thrown away.** The Neighbourhoods tab built overlapping
   neighbourhoods but kept only each cell's *own* neighbourhood (its type and
   composition). How many neighbourhoods a cell belongs to, and what kind they are, was
   lost. The tab now also writes `<name> memberships` (how many neighbourhoods contain
   the cell; a feature, so it can be gated and clustered) and `<name> member
   neighbourhood` (the most common type among them, via vtea-core
   `reflect_neighborhoods(relation="member")`), and says so in the result text.

Still seed-only, by design, and worth knowing: the "Cell types in each type" view
counts each cell once, under its own neighbourhood's type. A membership-weighted
version (each (member, neighbourhood) pair counted) is cheap to add and is listed under
H2.

## 1. The model: levels of objects

```
level 0  objects          segmented cells / nuclei                  (labels layer)
level 1  neighbourhoods   built on level-0 objects, classed by       (members: level-0 ids)
                          level-0 phenotypes / clusters / classes
level 2  neighbourhoods   built on level-1 objects, classed by       (members: level-1 ids)
                          level-1 types / phenotypes / clusters
...
```

Every level is the same kind of thing: **an object table** (one row per object, with
`centroid-*` and features), **a membership** to the level below (sparse,
many-to-many, overlap allowed), and **provenance** (method, radius / k, which class
column of the level below it was built on, randomisation, seed). Level 0 is the only
level with masks and image crops; every higher level has positions, members and
features.

The recursion is natural: the *types* found at level n are the *classes* used to
measure composition at level n + 1, exactly as phenotypes are at level 1 today.

### Two kinds of higher-level object

Centred neighbourhoods give one neighbourhood per object, so level n has as many
objects as level n − 1. That is right for "what surrounds this cell" but it never
produces countable structures (a glomerulus, a follicle, a tubule segment). So the
plan supports two builders per level:

| builder | overlap | count | answers | precedent |
|---|---|---|---|---|
| **Windows** (centred: radius, k nearest; or grid) | yes | = objects below (or grid points) | what surrounds each object | CODEX cellular neighbourhoods (Schürch et al. 2020) |
| **Regions** (contiguous objects of the same type merged: connected components of the neighbour graph restricted to one type, with a minimum size) | no (a partition) | few | which structures exist, their size, shape and makeup | UTAG domains (Kim et al. 2022); CellCharter spatial clusters (Varrone et al. 2024); tissue motifs (Bhate et al. 2022) |

The two are not interchangeable, and neither is the default for the other:

- **Windows describe state and change.** Because they overlap, an object of level
  n − 1 sits in many windows, so windows grade continuously across tissue. They are
  the tool for alteration: immune cells infiltrating tubulointerstitium show up as a
  gradient of window composition, not as a boundary. Windows on windows (level ≥ 2)
  stay windows, each level reaching further: a level-2 window describes what
  surrounds the surroundings, which is how an infiltrate's front and its
  hinterland are told apart.
- **Regions are functional tissue units** (a glomerulus, a tubule segment, a
  follicle): countable, non-overlapping, with a size and a shape of their own. They
  are built when the question is about units rather than gradients.

A typical chain: cells → windows (types: "tubular", "glomerular", "immune-rich") →
windows on those windows (where infiltration is advancing); and, alongside,
cells → regions (tubules, glomeruli) → windows over regions (which units sit next
to which: Bhate's tissue motifs).

**How each is drawn in 3D:**

- Regions follow **touching masks** by default (the contact graph between level-0
  masks; radius or k nearest as alternatives), so a tubule is traced along its
  epithelium and not bridged across the lumen or the interstitium to its neighbour.
  At level ≥ 1 windows are merged by the **seed-distance, member-weighted rule**
  below, which reduces to "seeds touch" when its distance is 0.
- Windows stay **radius or k nearest**, centred on every object of level n − 1, so
  objects at level n − 1 are included in many windows.

### Merging windows into regions: a k-d tree problem

Two windows A and B of the same type (seeds a and b) are joined in two stages.

1. **Candidates by seed distance.** A and B are candidates when their seeds are within
   a distance **d**: `cKDTree(seed positions).query_pairs(d)`, plus seeds whose masks
   touch. d = 0 is exactly "seeds touch"; a larger d bridges small gaps (a missed
   segmentation, a thin lumen) without needing contact. In 3D the distance is in µm
   with the voxel size, as for every neighbourhood. Sharing members never makes a
   candidate on its own: overlapping windows share members almost everywhere, so that
   rule would merge whole tissues.
2. **Weight by member proximity.** Each candidate pair gets a score from how close
   the members of each window are to the members of the other:

   ```
   w(A, B) = mean over m in A ∪ B of  exp(−δ(m)² / 2σ²)
   δ(m)    = distance from m to the nearest member of the other window
             (0 for a member both windows share)
   ```

   δ comes from k-d tree queries on the members' level-0 positions (one tree for the
   whole level; each pair queries only its few dozen members, so the cost is linear in
   the number of candidate pairs). The pair is joined when w ≥ **τ**. A window whose
   members sit right against the other's scores near 1; two windows whose seeds are
   close but whose members extend in different directions (two tubules running side
   by side, a tubule beside a glomerulus) score low and stay apart. A check on two
   parallel rows of cells (10 µm spacing, rows 16 µm apart, windows of radius 12 µm,
   σ = 10 µm): neighbouring windows along a row score 0.87, windows facing each other
   across the rows 0.28, so τ = 0.5 joins each row and keeps the two apart.

Joined pairs form a graph; its connected components (union-find), at least
`min_size` windows, are the regions. Every window of a region keeps its own
membership, so the region's cells are the union of its windows' members.

Defaults, all recorded in the region level's provenance: d = 0 (touching, the
conservative start), σ = the median nearest-neighbour distance between level-0
objects (one cell spacing), τ = 0.5. The Neighbourhoods tab shows a preview count
of regions as d and τ move, before the level is made.

Regions of regions (level ≥ 2 regions) use the same rule with the region's centre in
place of the seed. At level 0 (regions of cells) the rule is not needed: cells are
joined by touching masks, or by the same seed-distance stage with d > 0.

**Building on a subset.** A level is built on all objects of the level below by
default. Optionally it can be built on a population (a gate, a phenotype, a cluster,
a type), e.g. windows only over tubular neighbourhoods. Two separate choices, both
recorded in the provenance: which objects **seed** windows, and which objects can be
**members**. Seeding on a subset while counting every object as a member is usually
what is wanted (the windows of tubular neighbourhoods, with whatever infiltrates
them); restricting members too gives the subset's own arrangement.

## 2. vtea-core (engine)

New module `vtea_core.hierarchy` (neighbourhoods stay as they are and become one
builder):

```python
@dataclass
class ObjectLevel:
    name: str                 # "Objects", "Neighbourhoods 1", ...
    depth: int                # 0 = segmentation
    table: pd.DataFrame       # id, centroid-*, features
    id_column: str
    parent: str | None        # the level its members are rows of
    membership: scipy.sparse.csr_matrix | None   # (n_this, n_parent), binary
    seeds: np.ndarray | None  # parent row each object is centred on (windows), else None
    provenance: dict          # builder, params, class column used, vtea-core version

class Hierarchy:
    levels: dict[str, ObjectLevel]
    def build(self, parent, *, builder="windows"|"regions", classes, seeds=None, members=None,
              **params) -> ObjectLevel    # seeds / members: optional row masks of the parent
    def expand(self, level, to=0) -> csr_matrix        # members at any depth (product of memberships, binarised)
    def reflect(self, level, onto, columns, how="own"|"member")   # down: objects carry what they are part of
    def aggregate(self, level, columns, how="mean"|...)            # up: a level summarises its members
    def save(self, folder) / load(folder)
```

- Membership as sparse matrices makes "which cells are in this level-2 object" one
  product (`M2 @ M1`), and reflection and aggregation matrix products, so it scales to
  millions of cells. `NeighborhoodSet` gets `to_sparse()` / `from_sparse()`.
- Regions builder: `regions(parent, types, graph="contact"|"radius"|"knn", min_size)`,
  contact by default: touching masks at level 0 (from the labels image, 3D included);
  above it, the seed-distance, member-weighted rule of §1 (`d`, `sigma`, `tau`
  parameters; candidates from `cKDTree.query_pairs`, member distances from k-d tree
  queries, components by union-find). Members are the parent objects of each component; centre =
  mean of members; shape (volume, length along the principal axis, elongation) is
  measured from the members' masks.
- Windows builder: today's `build_neighborhoods`, plus `seeds` and `members` subsets.
- Persistence: one folder per level (table as parquet or CSV, membership as `.npz`,
  `level.json` with provenance and format version).
- Protocol steps (`workflow` registry) for build / reflect / aggregate, so a hierarchy
  is reproducible from a protocol, as phase 4 already is.

## 3. The workbench: one window, a level selector

```
┌ Cytometry workbench ─────────────────────────────────────────────────────┐
│ Level: [ Objects (cells) ▾ ]   cells › Neighbourhoods 1 › Regions 1      │
│        ├ Objects (cells)                         12,431 objects          │
│        ├ Spatial level 1: Neighbourhoods 1 (on cells)    12,431          │
│        └ Spatial level 2: Regions 1 (on Neighbourhoods 1)    86          │
├──────────────────────────────────────────────────────────────────────────┤
│ Gating │ Clustering │ Spatial │ Neighbourhoods │ Classification          │
```

- **Same core tabs at every level.** Gating, Clustering, Spatial, Neighbourhoods and
  Classification work on "the current level's object table". This is mostly a state
  refactor, not new tabs: today's `Cytometry` (table, gates, phenotype tree, analyses,
  derived columns, view) becomes `LevelState`, and `Cytometry` holds the levels and the
  current one. `state_of(viewer)` keeps returning something with today's interface (the
  current level), so the tabs need few changes.
- **Making a level.** The Neighbourhoods tab gets "Make a level from this" (windows) and
  "Make regions from these types" (regions), each with an optional "Seeds" and
  "Members" population (default: all objects). The new level appears in the selector;
  its table starts with the neighbourhood features (composition, size, density, and
  the H2 families) and the type as a column, so it can be gated, clustered and
  phenotyped immediately.
- **Seeing a level on the image.** Level 0 paints masks as now. A higher level paints
  through its membership: windows paint each seed cell (own) or each cell by majority
  over the windows containing it (member); regions paint the union of their members'
  masks as a labels layer, so a region is outlined on the image and can be clicked.
- **Moving between levels with a selection.** "Show members" (a gate or phenotype at
  level n → population at level n − 1 or 0) and "Show containers" (cells selected at
  level 0 → the level-n objects containing them). Populations in the Show box gain
  "members of …" / "part of …" entries.
- **Columns crossing levels.** Reflect (down) and aggregate (up) are offered per level,
  so "CD8 T cells in tumour-interface regions" is a level-0 gate on a reflected level-2
  column.
- **Workspace.** Version 2 saves every level (`cytometry/levels/<name>/`); a version 1
  workspace loads as level 0 alone.

## 4. The hierarchy map

Depth is unlimited, and levels branch: one level can be the parent of several (cells →
windows and cells → regions; windows → windows at two radii). So the hierarchy is a
tree, and a drop-down alone stops being a good overview after a few levels. The
**hierarchy map** is a panel docked on the left of the workbench (collapsible, and
openable as its own window) that shows the whole tree and is the main way to move
around it. The level drop-down stays, as a compact shortcut that mirrors it.

```
┌ Hierarchy ──────────────────────────────┐
│ ● Objects (cells)            12,431     │  level 0 · segmentation "cells"
│ ├─◆ Neighbourhoods 1         12,431     │  windows · r 50 µm · by phenotype
│ │  ├─◆ Neighbourhoods 2      12,431     │  windows · r 50 µm · by NH1 type
│ │  │  └─◆ Neighbourhoods 3   12,431     │  windows · k 15 · by NH2 type
│ │  └─▣ Regions 2                 41     │  regions · contact · NH1 type 3
│ └─▣ Regions 1                    86     │  regions · contact · tubular
│    └─◆ Neighbourhoods 4          86     │  windows · r 200 µm · by region type
└─────────────────────────────────────────┘
  ◆ windows (overlapping)   ▣ regions (functional units)   ● segmentation
```

**Each node** is one level: its name, kind (segmentation, windows, regions), object
count, and a one-line provenance (builder, radius / k / contact, the class column of
the parent it was built on, seed and member populations if a subset was used). A small
bar under the name shows the level's type mix (its current classes, coloured as on the
plot), so the map is also a summary of what was found at each depth. A tooltip gives
the full provenance and when it was built.

**Navigating:**

- Click a node: make it the current level. All tabs switch to it; the plot keeps its
  axes where the columns exist.
- Right-click a node: build a child level (windows, regions) from it, rename, show on
  the image, export its table, or delete it with its descendants (after a confirmation
  that lists them).
- A node whose parent was rebuilt or changed is marked **stale** (its build is no
  longer reproducible from the current parent) with "Rebuild" offered; nothing is
  rebuilt silently.

**Following objects through the hierarchy.** With objects selected at any level (a
gate, a phenotype, clicked objects on the image), the map shows, at every other node,
how many objects are related to them: below the selection, their **members** (expanded
through every level in between); above it, their **containers** (every object that
contains any of them). Clicking that count makes the node current with that
population shown, so one can go from a region to its windows to its cells and back
without writing a gate at each level. For windows, members and containers count
overlap honestly: a cell in seven windows of the selection counts once as a member,
and the map says how many windows it is in on average.

**Where it lives in the code.** The map is a view of `Hierarchy` (vtea-core) and of the
per-level states (workbench); it holds no state of its own beyond which nodes are
expanded, which is saved with the workspace view. Member and container counts use the
sparse membership products of §2, so they stay interactive on millions of cells.

## 5. Neighbourhood features (research)

Today a neighbourhood is described by its class composition (fractions and counts),
size and mean distance; types are k-means on raw fractions. That is the CODEX recipe
and a good default, but it ignores what the members express, how they are arranged,
and scale. Proposed feature families, each a function in
`vtea_core.neighborhoods.features` returning named columns, and each a check box in
the Neighbourhoods tab:

| family | features | why / precedent |
|---|---|---|
| Composition (exists) | fraction and count per class | Schürch et al. 2020 |
| Weighted composition | Gaussian distance-weighted fractions (σ set in µm) | less sensitive to the hard radius edge; BANKSY's neighbour terms are Gaussian-weighted (Singhal et al. 2024) |
| Multi-scale composition | fractions at 2–3 radii or graph hops, concatenated | CellCharter aggregates features over several neighbour layers (Varrone et al. 2024) |
| Diversity | Shannon entropy, Simpson index, richness, dominant class and its share | separates mixed interfaces from pure compartments |
| Member markers | mean / median / SD of member intensities and sizes, seed excluded or not | UTAG message passing (Kim et al. 2022), BANKSY neighbour mean; needs no phenotypes |
| Arrangement | density (members per µm³), nearest-neighbour distance among members, gyration tensor eigenvalues → elongation / planarity / orientation, convex hull volume, seed offset from the members' centre | shape of the local tissue; elongation picks out tubules and vessels, seed offset picks out edges |
| Internal contacts | fraction of member–member neighbour pairs that are each class pair; class assortativity | whether types mix or sit side by side within the window |
| Asymmetry / interface | composition difference between opposite half-windows (largest over directions); distance to the nearest neighbourhood of another type | gradients and borders; BANKSY's azimuthal term; CN interfaces (Schürch et al. 2020) |
| Overlap | number of containers (memberships), at every level | how embedded an object is |
| Level-crossing (level ≥ 2) | composition over level n − 1 types *and* over expanded level-0 classes; aggregates of level n − 1 features | a level-2 object is described by both its parts and its cells |

Clustering choices for types: k-means (today), Gaussian mixture with a stability / BIC
choice of k (CellCharter's approach), Leiden on the feature graph; and for
mixed-membership neighbourhoods a topic model (Spatial-LDA, Chen et al. 2020). For
composition, a centred log-ratio or square-root transform before clustering keeps rare
classes from being swamped.

**The research step (H2) is a benchmark, not a guess.** Synthetic tissues with known
structure (two halves, layered tubule-like bands, scattered aggregates, 2D and
anisotropic 3D), plus one public dataset with published neighbourhoods (the CODEX
colorectal cancer data of Schürch et al. 2020), scored by how well each feature set
and clustering recovers the truth (adjusted Rand index), its stability across seeds
and radii, and its run time. The defaults in the tab come from that table.

## 6. Classification is level-specific

A model applies to the kind of object it was trained on. The model card
(`ModelCard.inputs`) gains:

- `level_kind`: `"segmentation"` (level 0), `"windows"`, `"regions"`, or `"any"`.
- `input_kind`: `"crops"` (image crops around masks: **segmentation only**, so
  NephNet3D, which classifies nuclei from 3D DAPI crops, never appears as applicable
  at a spatial level), `"features"` (the generic ANN: any level, but bound to the
  feature names and the builder provenance it was trained with), `"graph"` (STELLAR:
  cells and a neighbour graph; level 0).

The Classification tab lists every model but disables the ones that do not fit the
current level, with the reason ("NephNet3D classifies nuclei from 3D image crops; it
applies to segmented objects, not neighbourhoods"). Cellular classifications still
inform higher levels the intended way: as classes the composition is measured over.
Training at higher levels (e.g. an ANN separating "glomerulus" from "tubulointerstitium"
regions from labelled regions) uses the same Training workbench with labels from that
level's phenotypes.

## 7. Milestones

| | what | where |
|---|---|---|
| H0 ✅ | Overlap review: k-nearest tie fix; memberships and member type kept in the workbench | vtea-core, workbench |
| H1 | `vtea_core.hierarchy`: `ObjectLevel`, sparse membership, expand / reflect / aggregate, regions builder, persistence, protocol steps, tests | vtea-core |
| H2 | Feature families and the benchmark; membership-weighted cell type × type enrichment; defaults chosen from results | vtea-core |
| H3 | `LevelState` refactor and the level selector, with only level 0 (no behaviour change; all current tests pass); workspace v2 | workbench |
| H4 | Make a level (windows, regions); every tab at level ≥ 1; painting levels on the image; members / containers populations; reflect and aggregate | workbench |
| H4b | Hierarchy map: the tree of levels with counts, provenance and type mix; click to switch level; build / delete / rebuild from a node; stale marking; members and containers of a selection at every node | workbench |
| H5 | Level-aware model cards and Classification tab | vtea-core, workbench |
| H6 | Docs, V&V cases for levels (synthetic tissue with known structure at two levels) | both |

H1 and H3 are independent and can proceed in parallel; H4 needs both. A read-only
map (nodes, counts, click to switch) comes with H3 as soon as there is more than one
level; the rest of the map is H4b.

## 8. Decisions

1. **Level ≥ 2 stays windows** when built from windows. Regions are a separate
   builder for functional tissue units, not a default for deeper levels; windows
   describe alteration and change (e.g. immune infiltration), which needs overlap.
2. **Subsets are an option, not a requirement**: a level is built on all objects by
   default, with optional seed and member populations.
3. **Regions follow touching masks** (in 3D), so a tubule is traced along its
   epithelium. **Windows keep overlap**: an object at level n − 1 can be in any number
   of windows at level n.
4. **Merging windows into regions is a k-d tree problem**: candidates are windows of
   the same type whose seeds are within a distance d (d = 0: seeds touch); a candidate
   pair is joined when its members are close to each other's (Gaussian-weighted
   member-to-member distance, threshold τ). Sharing members alone never merges.
5. **Any depth is allowed**, and levels may branch. A **hierarchy map** gives the
   overview and the navigation (§4); the level drop-down is a shortcut that mirrors it.

## 9. Open questions

None at present.

## References

- Schürch CM et al. Coordinated cellular neighborhoods orchestrate antitumoral immunity
  at the colorectal cancer invasive front. *Cell* 182, 1341–1359 (2020).
- Bhate SS, Barlow GL, Schürch CM, Nolan GP. Tissue schematics map the specialization of
  immune tissue motifs and their appropriation by tumors. *Cell Systems* 13, 109–130
  (2022).
- Kim J et al. Unsupervised discovery of tissue architecture in multiplexed imaging
  (UTAG). *Nature Methods* 19, 1653–1661 (2022).
- Varrone M et al. CellCharter reveals spatial cell niches associated with tissue
  remodeling and cell plasticity. *Nature Genetics* 56, 74–84 (2024).
- Singhal V et al. BANKSY unifies cell typing and tissue domain segmentation for
  scalable spatial omics data analysis. *Nature Genetics* 56, 431–441 (2024).
- Chen Z et al. Modeling multiplexed images with Spatial-LDA reveals novel tissue
  microenvironments. *Journal of Computational Biology* 27, 1204–1218 (2020).
- Woloshuk A et al. In situ classification of cell types in human kidney tissue using
  3D nuclear staining. *Cytometry A* 99, 707–721 (2021).
