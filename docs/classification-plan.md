# Cytometry phase 5: supervised object classification (plan)

Phases 1–4 find cell types by gating, clustering and spatial context. Phase 5 adds the
fourth route: **learn the types from labelled examples and predict them for every
object**. It becomes a **Classification** tab in the Cytometry workbench (apply a model)
and a separate **Training workbench** window (build a training set, train or fine-tune a
model, check it, save it). Three model families come first:

| Model | Learns from | Sees | Typical use |
| --- | --- | --- | --- |
| **Generic ANN** (multilayer perceptron) | labelled objects | each object's row of measurements (intensities, size, shape; optionally spatial and neighbourhood features) | multiplex marker panels; fast, CPU only |
| **NephNet3D** | labelled nuclei | a 3D image crop around each object (e.g. DAPI) | cell types from nuclear morphology alone, no type markers (Woloshuk et al. 2021) |
| **STELLAR** | an annotated reference dataset | measurements **and** the cell's neighbourhood graph | transfer annotations from one dataset to another, and find new types (Brbić et al. 2022) |

The tab is built so that other per-object classifiers (gradient boosting, other CNNs,
BioImage Model Zoo models) can be added as further model families without changing
the interface.

## 1. What the user does

### Classification tab (Cytometry workbench)

1. **Model**: pick a saved model (from the model library, a file, or the facility's
   shared models). Its card shows what it expects (classes, channels, crop size, voxel
   size, features) and how well it validated.
2. **Map inputs**: match the model's inputs to this dataset: features to table
   columns (matched by name, as phenotype rules are), image channels to layers, voxel
   size to the image's. What does not match is named before anything runs.
3. **Objects**: all, a gate or a phenotype (classify within *Immune*, say).
4. **Run** (in the background; locally, on the facility GPU server, or on a cluster).
5. The result is columns of the object table, like a clustering:
   `<name> class` (the predicted class), `<name> confidence` (top probability) and
   `<name> p <class>` per class. Plot, colour, gate (e.g. *confidence > 0.8*), and
   **Make phenotypes** from the predicted classes. **Show on image** colours objects by
   class; low-confidence objects can be shown hatched or grey.

### Training workbench (separate window)

A second window, because training is its own task with its own views (examples,
curves, confusion matrix), and it must not squeeze the Cytometry workbench.

1. **Labels**: the training labels come from:
   - phenotypes or gates (each becomes a class);
   - **clicking objects** on the image or the plot and assigning a class (a label
     palette, as in the image-gate painting tool);
   - an imported table (`label`, `class`) or a previous session's labels.
   An **example gallery** shows crops of labelled objects per class, to spot mistakes.
2. **Model family** and its settings (section 3). **Start from**: new weights, or an
   existing model to **fine-tune** (refine) on these labels.
3. **Split**: training / validation by object, or, better, **by image or by region**
   (held-out images), so that neighbouring cells of the same image do not leak between
   training and validation. Class balance: weighted loss or balanced sampling.
4. **Train** (locally or remotely). Live loss and validation curves; Stop keeps the
   best weights so far.
5. **Check**: confusion matrix, per-class precision / recall, balanced accuracy, and
   the misclassified objects in the gallery (click one to go to it in the viewer).
6. **Save** the model with its card (section 4) to the model library; it then appears
   in the Classification tab.
7. **Active learning** (later milestone): the workbench proposes the objects the model
   is least sure of; label them and retrain.

## 2. Architecture

The engine goes in vtea-core, next to the code it already has:
`vtea_core.classification` holds `extract_crops`, `class_map`, a small generic 3D CNN
(`Cell3DClassifier`) and the VAEs, ported from the Java `vtea.deeplearning`. IBIF
napari is the interface, as for phases 1–4.

```
vtea_core.classification
├── crops.py          (exists)  per-object image crops, measurement-table order
├── cnn.py            (exists)  small generic 3D CNN
├── vae.py            (exists)  VAEs
├── models.py         (new)     ObjectClassifier protocol, model folders, model cards
├── training.py       (new)     splits (by object / image / region), class weights,
│                               training loop, early stopping, metrics, confusion matrix
├── ann.py            (new)     generic ANN (MLP) over measurement features
├── nephnet.py        (new)     NephNet3D, ported from the Java implementation
└── stellar.py        (new)     STELLAR on current PyTorch / PyTorch Geometric

ibif_napari
├── _cyto_classify.py (new)     Classification tab (apply a model)
├── _train_workbench.py (new)   Training workbench window
└── _server.py / _cluster.py    (extended) training and prediction jobs remotely
```

Every model family implements one protocol, so the tab and the workbench never need to
know which one they are driving:

```python
class ObjectClassifier(Protocol):
    family: str                       # "ann", "nephnet3d", "stellar", ...
    classes: list[str]
    def inputs(self) -> InputSpec     # features, channels, crop size, voxel size, graph
    def fit(self, data: TrainingData, *, start_from=None, on_epoch=None, stop=None) -> History
    def predict_proba(self, data: ObjectData) -> np.ndarray   # objects x classes
    def save(self, folder) -> None    # weights + model card
```

`ObjectData` is built from what the workbench already has: the object table (features),
the labels layer and image layers (crops), and the Phase 4 neighbour graph
(`vtea_core.spatial.neighbor_graph`) for STELLAR.

**Dependencies stay optional.** torch is already the `deeplearning` extra of vtea-core;
PyTorch Geometric would be a new `stellar` extra. Without them the families stay in
the menu and say what to install, as the clustering methods do. The generic ANN needs
no new dependency if it is built on scikit-learn's MLP; a torch MLP is used when torch
is present, for GPU and fine-tuning.

## 3. The three model families

### 3.1 Generic ANN (milestone 5.1, first)

- Input: chosen measurement columns (default: every channel's mean intensity and the
  size measurements), optionally spatial and neighbourhood features. Transform
  (asinh, as for clustering) and scaling are fitted on the training set and **saved
  with the model**, so prediction scales new data the same way.
- Network: a multilayer perceptron, 1–3 hidden layers (default 64-32), ReLU, dropout,
  softmax output; weighted cross-entropy for imbalanced classes; Adam; early stopping
  on validation loss.
- Fine-tuning: continue training from saved weights; new classes add output units.
- Why first: it exercises the whole route (labels → split → train → validate → save
  → predict → phenotypes) on CPU in seconds, so the tab, the workbench and the model
  card are proven before any GPU model is wired in.

### 3.2 NephNet3D (milestone 5.3)

NephNet3D (Woloshuk et al., *Cytometry A* 2021) classifies kidney cell types in situ
from 3D nuclear (DAPI) image volumes alone, trained on nuclei labelled with VTEA; it
reported 80.26% balanced accuracy and outperformed classical machine learning and
shape-based classifiers.

- **Port, not reinvent.** The reference is the Java implementation in VTEA
  (`vtea.deeplearning`, NephNet3D) and the paper. vtea-core's existing `Cell3DClassifier`
  is explicitly *not* NephNet3D's layer layout, so `nephnet.py` will reproduce the
  published network: its layers, crop size, channels, normalisation and training
  recipe. **These are to be taken from the Java source and the paper; they are not
  restated here** (the paper could not be read while writing this plan).
- Input: `extract_crops` (exists) gives fixed-size crops centred on each object, in
  table order, z-scored per crop, optionally masked to the object. Two additions:
  **resampling to the model's voxel size** (a model trained at one Z step must see the
  same physical size at another), and **augmentation** (flips, 90° rotations in XY,
  small intensity changes) for training.
- Weights: Java checkpoints (JavaCPP / PyTorch bindings) cannot be loaded directly,
  as with the VAEs. Options, in order of preference: (1) export the original weights
  from the Java / original training run to a PyTorch state dict; (2) retrain on the
  original labelled kidney nuclei if the dataset is available; (3) train from scratch
  on new labels.
- **Parity check**: on the original data and split, the port must reach the published
  balanced accuracy within a stated tolerance before it is offered as "NephNet3D"
  rather than "3D CNN".
- **Fine-tuning** for new tissue or a new microscope: start from the kidney weights,
  freeze the convolutional layers, retrain the classifier head (and optionally
  unfreeze the last block) on a few hundred labelled nuclei. This is the main way
  NephNet3D will be used outside its training tissue.
- Compute: GPU for training (facility server or cluster); prediction is feasible on a
  CPU for thousands of nuclei, GPU for whole-slide volumes.

### 3.3 STELLAR (milestone 5.4)

STELLAR (Brbić et al., *Nature Methods* 2022) is a graph neural network: it learns cell
embeddings from the cells' features **and** their spatial neighbourhood, transfers
labels from an annotated reference dataset to an unannotated one, and proposes novel
cell types in the target.

- Inputs, as the reference implementation (snap-stanford/stellar) takes them: node
  features (marker intensities) for both datasets, labels for the reference, and a
  graph (edges) for each, built from cell positions with a distance threshold. In IBIF
  napari: features from the object table, labels from phenotypes, edges from
  `vtea_core.spatial.neighbor_graph` (radius or k nearest, in µm, 2D or 3D: the graph
  only needs centroids, so 3D works).
- **It is transductive**: training uses the reference and the target together, so
  "applying" STELLAR to a new dataset is a training run. In the interface it therefore
  lives in the Training workbench ("Annotate this dataset from a reference"), with the
  reference chosen from saved datasets or workspaces; the Classification tab shows its
  result like any other.
- **Port to current libraries.** The reference code was verified under Python 3.8,
  PyTorch 1.9.1 and PyG 2.0, and pins old scanpy / scikit-learn / pandas. The port
  re-implements the model and loss on current PyTorch and PyTorch Geometric inside
  `vtea_core.classification.stellar`, validated against the reference implementation
  on its public CODEX data (Dryad) before use.
- **License**: confirm the repository's license before copying any code; if it is
  not compatible with vtea-core's GPL-2.0, implement from the paper.
- Memory: graph construction on the published datasets needs tens of GB of RAM (the
  README cites about 32 GB for HuBMAP); large datasets go to the cluster.
- Novel types: STELLAR's novel-class proposals come back as extra classes
  (`novel 1`, `novel 2`, ...) for the user to inspect and name.

## 4. Models: files, cards and library

A model is a folder, as vtea-core's VAE checkpoints already are:

```
<model>/
  model.pt          weights (PyTorch state dict), or model.joblib for scikit-learn
  config.json       family, architecture settings, classes
  card.json         the model card (below)
```

The **model card** is what makes a model reusable and reviewable:

- classes and their colours; the family and version of vtea-core that trained it;
- inputs: features (names, transform, scaling parameters), or channels, crop size,
  voxel size and normalisation, or graph construction;
- training data: datasets / workspaces, number of objects per class, how labels were
  made (phenotypes, gates, clicked), the split;
- validation: balanced accuracy, per-class precision / recall, confusion matrix;
- what it was fine-tuned from, if anything; who trained it and when; free notes.

The **model library** is a folder (per user, plus an optional shared facility folder
on the server) listed in the Classification tab. Models are saved with the workspace
by reference (path and hash); predictions are saved as columns, so a workspace reopens
without the model.

## 5. Running remotely, and compute quotas

Training GPU models goes through the routes IBIF napari already has for Cellpose:

- **Facility GPU server** (`ibif-napari-server`): add `train` and `classify` job types
  next to segmentation. Training jobs are long, so they need a progress endpoint,
  cancel, and a per-user limit on concurrent and total training time so that one
  user's training does not block everyone's segmentation queue.
- **Slurm cluster** (NSF ACCESS etc.): a training run is a batch job with its own time
  limit; the Training workbench submits it, follows it, and fetches the model folder.
- Compute and storage **quotas** to respect: GPU hours on the allocation (training
  NephNet3D or STELLAR can take hours; the workbench shows an estimate before
  submitting), the cluster's storage quota for uploaded crops / graphs and model
  folders (cleaned up after fetching, as segmentation batches are), and the GPU's
  memory (batch size chosen to fit, with a fallback on out-of-memory).
- Data: crops and tables leave the computer only for the facility server or the user's
  own cluster account; national clusters are generally not for HIPAA-regulated data
  (as for Cellpose, `docs/cluster.md`).

## 6. Milestones

| # | Milestone | Done when |
| --- | --- | --- |
| 5.0 | **Common ground** in vtea-core: `ObjectClassifier` protocol, model folders and cards, `training.py` (splits by object / image / region, class weights, metrics, confusion matrix). | Unit tests: a toy classifier round-trips through save / load; image-level split never shares an image between train and validation. |
| 5.1 | **Generic ANN** in vtea-core, and the **Classification tab** in IBIF napari (apply a model; class, confidence and per-class probability columns; make phenotypes; show on image). | On the cell-types test table, an ANN trained on half the objects predicts the other half with balanced accuracy > 0.95; the tab runs it and makes phenotypes. |
| 5.2 | **Training workbench** window: labels from phenotypes, gates and clicked objects; example gallery; split and class balance; train with live curves; confusion matrix; save to the library; fine-tune from a saved model. | A model trained in the workbench appears in the Classification tab; fine-tuning on a new class adds it. Window fits 1366×768. |
| 5.3 | **NephNet3D** port: architecture and preprocessing from the Java implementation and the paper; crop resampling to the model's voxel size; augmentation; weight export or retraining; fine-tuning (frozen convolutions). | Parity: on the original kidney data and split, balanced accuracy within the agreed tolerance of the published 80.26%. Fine-tuning on a small new set improves on training from scratch. |
| 5.4 | **STELLAR** port on current PyTorch / PyG, graph from `neighbor_graph`, reference-to-target annotation and novel types in the Training workbench. | On the public CODEX data, agreement with the reference implementation's annotations; runs in 3D on a synthetic volume. License confirmed. |
| 5.5 | **Remote training and prediction**: `train` / `classify` jobs on the facility server (with per-user limits) and on Slurm clusters; progress, cancel, fetch. | A NephNet3D fine-tune runs on the facility server from the workbench and returns a model folder. |
| 5.6 | **Active learning** and review: least-confident objects proposed for labelling; confidence thresholds as gates; model comparison on the same validation set. | Labelling the proposed objects improves balanced accuracy faster than labelling random ones, on a test set. |

5.0 and 5.1 can start straight away; 5.3 needs the Java source / original weights and
data; 5.4 needs the license check.

## 7. Open questions

- **NephNet3D assets**: are the original weights (Java / training run) and the labelled
  kidney nuclei available to export or retrain from, and what tolerance on the
  published balanced accuracy counts as parity?
- **Shared models**: should the facility server host a shared model library (e.g. a
  reviewed kidney NephNet3D), and who approves models into it?
- **Quotas**: which limits apply on the facility server (training hours per user,
  concurrent jobs) and on the cluster allocations used for training.
- **STELLAR reference data**: which annotated datasets will serve as references
  (the public CODEX sets for validation; in-house datasets in practice)?

## References

- Woloshuk A, Khochare S, Almulhim AF, et al. In situ classification of cell types in
  human kidney tissue using 3D nuclear staining. *Cytometry A* 99(7):707–721 (2021).
  [doi:10.1002/cyto.a.24274](https://doi.org/10.1002/cyto.a.24274)
- Brbić M, Cao K, Hickey JW, et al. Annotation of spatially resolved single-cell data
  with STELLAR. *Nature Methods* 19:1411–1418 (2022).
  [doi:10.1038/s41592-022-01651-8](https://doi.org/10.1038/s41592-022-01651-8);
  code [snap-stanford/stellar](https://github.com/snap-stanford/stellar).
