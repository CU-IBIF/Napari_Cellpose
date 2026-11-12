"""The Training workbench: build a training set, train a classifier, check it, save it.

A window of its own (opened from the Classification tab), because training is a task
with its own views. Milestone 5.2 of `docs/classification-plan.md`.

**Training set** (left): classes with colours, and labelled objects. Labels come from
- phenotypes (or clusters, or a classification): each object labelled with its class;
- a gate, image gate or phenotype: all its objects labelled as the active class;
- **clicking objects**: with *Label by clicking* on, each object selected in the
  viewer (the objects layer's picker) or clicked on the Gating plot is labelled as
  the active class;
- a table (CSV: `label`, `T` / `P` where present, `class`).
The set is saved with the workspace.

**Examples** (middle): a thumbnail of each labelled object (an image layer around
the object, its outline in its class's colour), by class; after training, the
held-out objects the model got wrong. Click one to go to it; remove or relabel
selected ones.

**Model** (right): the family (generic ANN now; NephNet3D and STELLAR to come), its
features and settings, the validation split (by object, or by time point /
position), and what to start from (a new model, or fine-tuning a saved one - new
classes are added). **Train** runs in the background with live loss and validation
curves; **Stop** keeps the best weights so far. **Results**: balanced accuracy,
per-class precision / recall / F1, the confusion matrix. **Save to library** writes the
model and its card; the Classification tab then offers it.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QObject, QSize, Qt, Signal
from qtpy.QtGui import QIcon, QImage, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import _cytometry as cy
from ._cyto_classify import SPLITS, _combo, _html, _metrics_html, _safe, object_data

FAMILIES = {"Generic ANN (measurements)": "ann", "NephNet3D (3D nuclear crops) — coming": None,
            "STELLAR (reference annotation) — coming": None}
THUMB = 56  # thumbnail size (pixels)
MAX_THUMBS = 240  # examples shown at once


# ---- training ----------------------------------------------------------------------------


def train_model(state, labels, features, *, hidden=(64, 32), asinh=True, split_by="object",
                fraction=0.2, seed=0, start_from=None, on_epoch=None, stop=None):
    """Train a generic ANN on the labelled rows: returns {"model", "history", "metrics",
    "validation_rows", "predicted"} (the held-out rows and what the model said for
    them). `start_from`: a model folder to fine-tune. Safe in a background thread once
    the inputs are prepared (`prepare_training`)."""
    job = prepare_training(state, labels, features, hidden=hidden, asinh=asinh, split_by=split_by,
                           fraction=fraction, seed=seed, start_from=start_from)
    return job(on_epoch, stop)


def prepare_training(state, labels, features, *, hidden=(64, 32), asinh=True, split_by="object",
                     fraction=0.2, seed=0, start_from=None):
    """Everything that reads the viewer's state, done now (in the GUI thread); returns
    job(on_epoch, stop) for the training itself."""
    from vtea_core.classification import ANNClassifier, evaluate, load_model, predictions, split
    from vtea_core.classification.training import labelled

    labels = np.asarray(labels, dtype=object)
    rows = np.flatnonzero(labelled(labels))
    if len(set(labels[rows])) < 2:
        raise ValueError("label objects of at least two classes first")
    if not features:
        raise ValueError("tick at least one feature")
    groups = None
    if split_by == "group":
        keys = [k for k in ("T", "P") if k in state.table.columns]
        if not keys:
            raise ValueError("these objects have no time points or positions to split by")
        groups = state.table[keys].astype(str).agg("/".join, axis=1).to_numpy()[rows]
    train, val = split(labels[rows], groups, validation=fraction, random_state=seed)
    start = load_model(start_from) if start_from else None
    if start is not None:
        features = list(start.inputs()["features"])
    data = object_data(state, features, rows)
    y = labels[rows].astype(str)
    model = ANNClassifier(features, hidden=hidden, transform="asinh" if asinh else "none", random_state=seed)

    def job(on_epoch=None, stop=None):
        history = model.fit(data.subset(train), y[train], start_from=start,
                            validation=(data.subset(val), y[val]) if len(val) else None,
                            on_epoch=on_epoch, stop=stop)
        predicted = (predictions(model, data.subset(val))["class"].to_numpy(dtype=object) if len(val)
                     else np.zeros(0, dtype=object))
        metrics = evaluate(y[val], predicted, model.classes)
        model.card.validation = {**metrics, "split": split_by, "held_out": fraction}
        if start is not None:
            model.card.fine_tuned_from = start.card.notes or start.card.created
            model.card.training["new_classes"] = model.new_classes(start)
        return {"model": model, "history": history, "metrics": metrics, "validation_rows": rows[val],
                "predicted": predicted, "truth": y[val]}

    return job


class _Progress(QObject):
    epoch = Signal(int, object)  # emitted from the training thread, delivered in the GUI thread


# ---- thumbnails ---------------------------------------------------------------------------


def object_thumbnail(state, image, row, size=THUMB, color=None):
    """An RGB (size, size, 3) uint8 picture of table row `row`'s object in image layer
    `image`: the plane through its centre (a max projection of ±2 planes in 3D),
    scaled by the layer's contrast, with the object's outline in `color`."""
    table = state.table
    cols = state.centroid_columns()
    rec = table.iloc[row]
    out = np.zeros((size, size, 3), np.uint8)
    if image is None or "Y" not in cols:
        return out
    data = image.data
    n = min(cy._spatial_ndim(image), 3)
    lead = data.ndim - n
    keys = [k for k in table.columns if cy._is_key(k) and k != "label"][:lead]
    pos = tuple(int(rec[k]) for k in keys) + (0,) * (lead - len(keys))
    world = {a: float(rec[c]) for a, c in cols.items()}
    scale, translate = np.asarray(image.scale[-n:], float), np.asarray(image.translate[-n:], float)
    axes = "ZYX"[-n:]
    centre = [int(round((world.get(a, translate[i]) - translate[i]) / scale[i])) for i, a in enumerate(axes)]
    half = size // 2
    sl = []
    for i, a in enumerate(axes):
        extent = data.shape[lead + i]
        if a == "Z":
            sl.append(slice(max(centre[i] - 2, 0), min(centre[i] + 3, extent)))
        else:
            sl.append(slice(max(centre[i] - half, 0), min(centre[i] - half + size, extent)))
    try:
        crop = np.asarray(data[pos + tuple(sl)], dtype=float)
    except (IndexError, ValueError):
        return out
    if n == 3:
        crop = crop.max(axis=0)
    lo, hi = (image.contrast_limits if image.contrast_limits else (crop.min(), crop.max()))
    gray = np.clip((crop - lo) / ((hi - lo) or 1), 0, 1) * 255
    y0 = max(half - centre[-2], 0)
    x0 = max(half - centre[-1], 0)
    h, w = gray.shape
    for c in range(3):
        out[y0:y0 + h, x0:x0 + w, c] = gray.astype(np.uint8)
    labels = state.labels_layer()
    if color and labels is not None and labels.data.shape[-n:] == data.shape[-n:]:
        try:
            lab = np.asarray(labels.data[pos + tuple(sl)])
        except (IndexError, ValueError):
            lab = None
        if lab is not None:
            mask = lab == int(rec["label"])
            if n == 3:
                mask = mask.any(axis=0)
            from scipy import ndimage as ndi

            edge = mask & ~ndi.binary_erosion(mask)
            rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
            region = out[y0:y0 + h, x0:x0 + w]
            region[edge] = rgb
    return out


def _pixmap(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


# ---- the window ---------------------------------------------------------------------------


class TrainingWorkbench(QWidget):
    def __init__(self, bench, parent=None):
        super().__init__(parent or bench, Qt.Window)
        self.setWindowTitle("Training workbench")
        self.bench = bench
        self.state = bench.state
        self.viewer = bench.viewer
        self.result = None
        self._worker = None
        self._stop = False
        self._navigating = False
        self._progress = _Progress()
        self._progress.epoch.connect(self._epoch)
        split = QSplitter(Qt.Horizontal)
        QVBoxLayout(self).addWidget(split)
        split.addWidget(self._training_set_pane())
        split.addWidget(self._examples_pane())
        split.addWidget(self._model_pane())
        split.setChildrenCollapsible(False)
        split.setSizes([320, 420, 420])
        self.split = split
        self.state.listen_training(self.refresh_training)
        self.state.listen(self.refresh)
        bench.object_selected.append(self._object_selected)
        self.refresh()
        self.fit_to_screen()

    # ---- panes ------------------------------------------------------------------------------

    def _training_set_pane(self):
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("<b>Training set</b>"))
        self.class_table = QTableWidget(0, 2)
        self.class_table.setHorizontalHeaderLabels(["Class", "Objects"])
        self.class_table.horizontalHeader().setStretchLastSection(True)
        self.class_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.class_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.class_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.class_table.itemSelectionChanged.connect(self._class_selected)
        layout.addWidget(self.class_table, 1)
        row = QHBoxLayout()
        for text, slot in (("Add…", self.add_class), ("Colour…", self.recolor_class), ("Remove", self.remove_class)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, f=slot: f())
            row.addWidget(b)
        layout.addLayout(row)
        form = QFormLayout()
        self.active = _combo(12)
        self.active.setToolTip("The class objects are labelled as (clicking, a gate, Move to)")
        form.addRow("Active class", self.active)
        self.click = QCheckBox("Label by clicking")
        self.click.setToolTip("Each object picked in the viewer (the objects layer's picker) or clicked on the "
                              "Gating plot is labelled as the active class")
        self.click.toggled.connect(self._click_toggled)
        form.addRow("", self.click)
        self.source = _combo(14)
        self.source.setToolTip("Label objects from phenotypes, clusters or a classification (each with its "
                               "class), or from a gate (as the active class)")
        add = QPushButton("Add labels")
        add.clicked.connect(lambda _=False: self.add_from_source())
        srow = QHBoxLayout()
        srow.addWidget(self.source, 1)
        srow.addWidget(add)
        form.addRow("From", srow)
        layout.addLayout(form)
        row = QHBoxLayout()
        imp = QPushButton("Import…")
        imp.setToolTip("Labels from a CSV table with 'label' (and T / P) and 'class' columns")
        imp.clicked.connect(lambda _=False: self.import_labels())
        clear = QPushButton("Clear labels")
        clear.clicked.connect(lambda _=False: self.clear_labels())
        row.addWidget(imp)
        row.addWidget(clear)
        layout.addLayout(row)
        self.set_summary = QLabel()
        self.set_summary.setWordWrap(True)
        layout.addWidget(self.set_summary)
        return pane

    def _examples_pane(self):
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("<b>Examples</b>"))
        form = QFormLayout()
        self.image = _combo(14)
        self.image.setToolTip("The image the thumbnails are cut from")
        self.image.currentIndexChanged.connect(lambda _: self.fill_gallery())
        form.addRow("Image", self.image)
        self.show_what = _combo(14)
        self.show_what.currentIndexChanged.connect(lambda _: self.fill_gallery())
        form.addRow("Show", self.show_what)
        layout.addLayout(form)
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListView.IconMode)
        self.gallery.setIconSize(QSize(THUMB, THUMB))
        self.gallery.setResizeMode(QListView.Adjust)
        self.gallery.setMovement(QListView.Static)
        self.gallery.setSpacing(2)
        self.gallery.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.gallery.itemClicked.connect(self._example_clicked)
        layout.addWidget(self.gallery, 1)
        row = QHBoxLayout()
        remove = QPushButton("Remove label")
        remove.clicked.connect(lambda _=False: self.relabel_selected(None))
        move = QPushButton("Move to active class")
        move.clicked.connect(lambda _=False: self.relabel_selected(self.active.currentText() or None))
        row.addWidget(remove)
        row.addWidget(move)
        layout.addLayout(row)
        self.gallery_note = QLabel()
        self.gallery_note.setWordWrap(True)
        layout.addWidget(self.gallery_note)
        return pane

    def _model_pane(self):
        self.model_tabs = QTabWidget()
        settings = QWidget()
        form = QFormLayout(settings)
        self.family = QComboBox()
        for text, family in FAMILIES.items():
            self.family.addItem(text, family)
            if family is None:
                self.family.model().item(self.family.count() - 1).setEnabled(False)
        form.addRow("Model", self.family)
        self.features = QListWidget()
        self.features.setMinimumHeight(90)
        form.addRow("Features", self.features)
        quick = QHBoxLayout()
        for text, how in (("Intensities", "intensity"), ("All", "all"), ("None", "none")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, h=how: self.check_features(h))
            quick.addWidget(b)
        form.addRow("", quick)
        self.asinh = QCheckBox("asinh(x/150) first (for intensities)")
        self.asinh.setChecked(True)
        form.addRow("", self.asinh)
        self.hidden = QLineEdit("64, 32")
        self.hidden.setToolTip("Neurons in each hidden layer, e.g. 64, 32")
        form.addRow("Hidden layers", self.hidden)
        self.split_by = QComboBox()
        for text, how in SPLITS.items():
            self.split_by.addItem(text, how)
        form.addRow("Validation", self.split_by)
        self.fraction = QDoubleSpinBox()
        self.fraction.setRange(0.05, 0.5)
        self.fraction.setSingleStep(0.05)
        self.fraction.setValue(0.2)
        form.addRow("Held out", self.fraction)
        self.start_from = _combo(16)
        self.start_from.setToolTip("Train from scratch, or fine-tune a saved model (its features; new classes "
                                   "are added)")
        form.addRow("Start from", self.start_from)
        self.seed = QSpinBox()
        self.seed.setRange(0, 1_000_000)
        form.addRow("Seed", self.seed)
        row = QHBoxLayout()
        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(lambda _=False: self.train())
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(lambda _=False: setattr(self, "_stop", True))
        row.addWidget(self.train_btn)
        row.addWidget(self.stop_btn)
        form.addRow(row)
        self.model_tabs.addTab(settings, "Settings")

        results = QWidget()
        rl = QVBoxLayout(results)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.curves_fig = Figure(figsize=(4, 2.2), layout="constrained")
        self.curves = FigureCanvasQTAgg(self.curves_fig)
        self.curves.setMinimumHeight(150)
        rl.addWidget(self.curves, 1)
        self.metrics = QLabel("Train a model to see how it does on the held-out objects.")
        self.metrics.setWordWrap(True)
        self.metrics.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.metrics.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.metrics)
        rl.addWidget(scroll, 1)
        form = QFormLayout()
        self.model_name = QLineEdit()
        form.addRow("Model name", self.model_name)
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("what it is for, the tissue, the stain…")
        form.addRow("Notes", self.notes)
        rl.addLayout(form)
        self.save_btn = QPushButton("Save to library")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(lambda _=False: self.save())
        rl.addWidget(self.save_btn)
        self.model_tabs.addTab(results, "Results")
        self.status = QLabel()
        self.status.setWordWrap(True)
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.model_tabs, 1)
        layout.addWidget(self.status)
        return pane

    def fit_to_screen(self, available=None):
        from ._cyto_workbench import _screen_size

        if available is None:
            available = _screen_size(self.parentWidget())
        width = int(min(available[0] * 0.92, 1280))
        height = int(min(available[1] * 0.88, 760))
        self.resize(max(width, min(self.minimumSizeHint().width(), available[0])), height)
        return self.width(), self.height()

    # ---- refreshing ---------------------------------------------------------------------------

    def refresh(self):
        """The choices that follow the analysis (sources, features, images, models)."""
        state = self.state
        keep = self.source.currentData()
        self.source.clear()
        if state.table is not None:
            for label, key in state.class_choices():
                self.source.addItem(label, _key("classes", key))
            for kind, key, label in state.populations():
                if kind != "all":
                    self.source.addItem(f"{label.strip()} → active class", _key(kind, key))
        index = self.source.findData(keep)
        self.source.setCurrentIndex(max(index, 0) if self.source.count() else -1)
        checked = set(self.checked_features())
        first = self.features.count() == 0
        self.features.clear()
        for f in state.measured_features() if state.table is not None else []:
            item = QListWidgetItem(f)
            item.setToolTip(f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            on = f in checked or (first and f.endswith(" mean"))
            item.setCheckState(Qt.Checked if on else Qt.Unchecked)
            self.features.addItem(item)
        from napari.layers import Image

        keep = self.image.currentText()
        self.image.blockSignals(True)
        self.image.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.image.addItem(layer.name)
        if keep and self.image.findText(keep) >= 0:
            self.image.setCurrentText(keep)
        self.image.blockSignals(False)
        from vtea_core.classification import list_models, read_card

        keep = self.start_from.currentData()
        self.start_from.clear()
        self.start_from.addItem("(new model)", None)
        for folder in list_models(cy.model_library()):
            try:
                if read_card(folder).family == "ann":
                    self.start_from.addItem(f"fine-tune {folder.name}", str(folder))
            except ValueError:
                continue
        self.start_from.setCurrentIndex(max(self.start_from.findData(keep), 0))
        groups = [k for k in ("T", "P") if state.table is not None and k in state.table.columns
                  and state.table[k].nunique() > 1]
        self.split_by.model().item(1).setEnabled(bool(groups))
        if not groups and self.split_by.currentData() == "group":
            self.split_by.setCurrentIndex(0)
        if not self.model_name.text():
            self.model_name.setText(f"{state.layer_name or 'objects'} ANN")
        self.refresh_training()

    def refresh_training(self):
        """The class list, the active class, the summary and the gallery."""
        state = self.state
        counts = state.training_counts()
        selected = self.selected_class()
        self.class_table.blockSignals(True)
        self.class_table.setRowCount(len(counts))
        for r, (c, n) in enumerate(counts.items()):
            item = QTableWidgetItem(c)
            item.setForeground(_qcolor(state.training_classes.get(c, "#888888")))
            self.class_table.setItem(r, 0, item)
            self.class_table.setItem(r, 1, QTableWidgetItem(f"{n:,}"))
            if c == selected:
                self.class_table.selectRow(r)
        self.class_table.blockSignals(False)
        keep = self.active.currentText()
        self.active.blockSignals(True)
        self.active.clear()
        self.active.addItems(list(state.training_classes))
        if keep in state.training_classes:
            self.active.setCurrentText(keep)
        self.active.blockSignals(False)
        keep = self.show_what.currentData()
        self.show_what.blockSignals(True)
        self.show_what.clear()
        self.show_what.addItem("All labelled", "all")
        for c in state.training_classes:
            self.show_what.addItem(c, _key("class", c))
        if self.result is not None:
            self.show_what.addItem("Wrong on held-out objects", "wrong")
        index = self.show_what.findData(keep)
        self.show_what.setCurrentIndex(max(index, 0))
        self.show_what.blockSignals(False)
        total = sum(counts.values())
        self.set_summary.setText(f"{total:,} labelled objects in {sum(1 for n in counts.values() if n)} classes."
                                 if total else "No labelled objects yet: add labels from phenotypes or a "
                                               "gate, or label by clicking.")
        self.train_btn.setEnabled(self._worker is None and sum(1 for n in counts.values() if n) >= 2)
        self.fill_gallery()

    def selected_class(self):
        items = self.class_table.selectedItems()
        return self.class_table.item(items[0].row(), 0).text() if items else None

    def _class_selected(self):
        c = self.selected_class()
        if c:
            self.active.setCurrentText(c)

    # ---- the training set --------------------------------------------------------------------

    def add_class(self, name=None):
        if name is None:
            name, ok = QInputDialog.getText(self, "New class", "Name of the class:")
            if not ok:
                return None
        try:
            name = self.state.add_training_class(name)
        except ValueError as e:
            self.status.setText(str(e))
            return None
        self.active.setCurrentText(name)
        return name

    def recolor_class(self, color=None):
        c = self.selected_class() or self.active.currentText()
        if not c:
            return None
        if color is None:
            picked = QColorDialog.getColor(_qcolor(self.state.training_classes[c]), self, f"Colour of {c}")
            if not picked.isValid():
                return None
            color = picked.name()
        self.state.add_training_class(c, color)
        return color

    def remove_class(self):
        c = self.selected_class() or self.active.currentText()
        if c:
            self.state.remove_training_class(c)

    def add_from_source(self):
        data = self.source.currentData()
        if not data:
            return 0
        kind, key = _unkey(data)
        try:
            if kind == "classes":
                n = self.state.label_from_classes(key)
            else:
                cls = self.active.currentText()
                if not cls:
                    self.status.setText("Add a class (or pick the active class) first.")
                    return 0
                n = self.state.label_population(kind, key, cls)
        except (ValueError, KeyError) as e:
            self.status.setText(str(e))
            return 0
        self.status.setText(f"{n:,} objects labelled.")
        return n

    def import_labels(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Labels", "", "CSV (*.csv)")
            if not path:
                return None
        try:
            found, missing = self.state.import_training_labels(path)
        except (ValueError, OSError) as e:
            self.status.setText(str(e))
            return None
        self.status.setText(f"{found:,} objects labelled" + (f"; {missing:,} rows matched no object" if missing
                                                              else "") + ".")
        return found

    def clear_labels(self):
        self.state.training = {}
        self.state.training_changed()

    def _click_toggled(self, on):
        if on and not self.active.currentText():
            self.status.setText("Add a class first: clicked objects are labelled as the active class.")
            self.click.setChecked(False)
            return
        self.status.setText(f"Pick objects in the viewer (the objects layer's colour picker) or click points on "
                            f"the Gating plot: each is labelled {self.active.currentText()}." if on else "")

    def _object_selected(self, row):
        if self.click.isChecked() and not self._navigating and self.active.currentText():
            if self.state.label_objects([row], self.active.currentText()):
                self.status.setText(f"Object {int(self.state.table.iloc[row]['label'])} labelled "
                                    f"{self.active.currentText()}.")

    # ---- the gallery ------------------------------------------------------------------------

    def gallery_rows(self):
        """The rows the gallery shows, and the class to outline each in."""
        state = self.state
        what = self.show_what.currentData() or "all"
        if what == "wrong" and self.result is not None:
            r = self.result
            wrong = r["truth"] != r["predicted"]
            return [(int(row), t) for row, t in zip(r["validation_rows"][wrong], r["truth"][wrong])]
        rows = sorted(state.training.items())
        if what.startswith("class\t"):
            rows = [(r, c) for r, c in rows if c == _unkey(what)[1]]
        return rows

    def fill_gallery(self):
        self.gallery.clear()
        if self.state.table is None:
            return
        layer = self.viewer.layers[self.image.currentText()] if self.image.currentText() in self.viewer.layers \
            else None
        rows = self.gallery_rows()
        shown = rows[:MAX_THUMBS]
        for row, cls in shown:
            color = self.state.training_classes.get(cls, "#ffffff")
            item = QListWidgetItem()
            item.setIcon(QIcon(_pixmap(object_thumbnail(self.state, layer, row, THUMB, color))))
            label = int(self.state.table.iloc[row]["label"])
            tip = f"{cls}: object {label}"
            if self.show_what.currentData() == "wrong" and self.result is not None:
                k = list(self.result["validation_rows"]).index(row)
                tip += f", predicted {self.result['predicted'][k]}"
            item.setToolTip(tip)
            item.setData(Qt.UserRole, row)
            self.gallery.addItem(item)
        more = len(rows) - len(shown)
        self.gallery_note.setText(f"{len(rows):,} objects" + (f" (the first {len(shown)} shown)" if more else "")
                                  + (". Click one to go to it." if rows else ""))

    def _example_clicked(self, item):
        self._navigating = True  # going to an object is not labelling it
        try:
            self.bench.go_to_row(int(item.data(Qt.UserRole)))
        finally:
            self._navigating = False

    def selected_examples(self):
        return [int(i.data(Qt.UserRole)) for i in self.gallery.selectedItems()]

    def relabel_selected(self, cls):
        rows = self.selected_examples()
        if rows:
            self.state.label_objects(rows, cls)

    def check_features(self, how, names=()):
        for i in range(self.features.count()):
            item = self.features.item(i)
            on = {"all": True, "none": False, "intensity": item.text().endswith(" mean"),
                  "names": item.text() in names}[how]
            item.setCheckState(Qt.Checked if on else Qt.Unchecked)

    def checked_features(self):
        return [self.features.item(i).text() for i in range(self.features.count())
                if self.features.item(i).checkState() == Qt.Checked]

    # ---- training ------------------------------------------------------------------------

    def train(self, wait=False):
        """Train (in the background unless `wait`). Returns the result when waiting."""
        try:
            hidden = tuple(int(v) for v in self.hidden.text().replace(";", ",").split(",") if v.strip())
            if not hidden or min(hidden) < 1:
                raise ValueError("Hidden layers: whole numbers separated by commas, e.g. 64, 32.")
            job = prepare_training(self.state, self.state.training_labels(), self.checked_features(),
                                   hidden=hidden, asinh=self.asinh.isChecked(),
                                   split_by=self.split_by.currentData(), fraction=self.fraction.value(),
                                   seed=self.seed.value(), start_from=self.start_from.currentData())
        except ValueError as e:
            self.status.setText(str(e))
            return None
        self._stop = False
        self._history = {"epoch": [], "train_loss": [], "val_loss": [], "val_balanced_accuracy": []}
        self.model_tabs.setCurrentIndex(1)
        self.save_btn.setEnabled(False)

        def progress(epoch, info):
            self._progress.epoch.emit(epoch, dict(info))

        if wait:
            result = job(progress, lambda: self._stop)
            self._done(result)
            return result
        from napari.qt.threading import thread_worker

        @thread_worker(ignore_errors=True)
        def work():
            return job(progress, lambda: self._stop)

        worker = work()
        worker.returned.connect(self._done)
        worker.errored.connect(lambda e: self.status.setText(f"Training failed: {e}"))
        worker.finished.connect(self._finished)
        self._worker = worker
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("Training…")
        worker.start()
        return worker

    def _epoch(self, epoch, info):
        h = self._history
        h["epoch"].append(epoch)
        for key in ("train_loss", "val_loss", "val_balanced_accuracy"):
            h[key].append(info.get(key, np.nan))
        if epoch % 5 == 0 or epoch == 1:
            self.draw_curves()
            self.status.setText(f"Epoch {epoch}" + (f": held-out balanced accuracy "
                                                    f"{info['val_balanced_accuracy']:.3f}"
                                                    if "val_balanced_accuracy" in info else ""))

    def draw_curves(self):
        h = getattr(self, "_history", None)
        fig = self.curves_fig
        fig.clf()
        if not h or not h["epoch"]:
            self.curves.draw_idle()
            return
        ax = fig.add_subplot(111)
        ax.plot(h["epoch"], h["train_loss"], label="training loss", color="#1f77b4", lw=1)
        if np.isfinite(h["val_loss"]).any():
            ax.plot(h["epoch"], h["val_loss"], label="held-out loss", color="#ff7f0e", lw=1)
        ax.set_xlabel("epoch", fontsize=7)
        ax.set_ylabel("loss", fontsize=7)
        ax.tick_params(labelsize=6)
        if np.isfinite(h["val_balanced_accuracy"]).any():
            ax2 = ax.twinx()
            ax2.plot(h["epoch"], h["val_balanced_accuracy"], color="#2ca02c", lw=1, label="balanced accuracy")
            ax2.set_ylim(0, 1.02)
            ax2.set_ylabel("balanced accuracy", fontsize=7, color="#2ca02c")
            ax2.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper right", frameon=False)
        self.curves.draw_idle()

    def _done(self, result):
        self.result = result
        self.draw_curves()
        state, model = self.state, result["model"]
        model.card.colors = {c: state.training_classes[c] for c in model.classes if c in state.training_classes}
        model.card.training.update(source=state.source, objects_layer=state.layer_name,
                                   labels="training set")
        self.metrics.setText(_metrics_html(result["metrics"], result["history"])
                             + _per_class_html(result["metrics"]))
        new = model.card.training.get("new_classes")
        self.status.setText("Trained" + (f"; new classes: {', '.join(new)}" if new else "") + ".")
        self.save_btn.setEnabled(True)
        self.refresh_training()  # offers "Wrong on held-out objects"

    def _finished(self):
        self._worker = None
        self.stop_btn.setEnabled(False)
        self.train_btn.setEnabled(True)

    def save(self, name=None):
        """Save the trained model to the library (and offer it on the Classification tab)."""
        if self.result is None:
            return None
        model = self.result["model"]
        name = (name or self.model_name.text()).strip() or "model"
        library = cy.model_library()
        folder = library / _safe(name)
        k = 2
        while folder.exists():
            folder = library / f"{_safe(name)} {k}"
            k += 1
        model.card.notes = self.notes.text().strip()
        saved = str(model.save(folder))
        self.status.setText(f"Saved as {folder.name}; choose it on the Classification tab.")
        self.save_btn.setEnabled(False)
        self.bench.classification.use_model(saved)
        self.refresh()
        return saved

    def closeEvent(self, event):
        self.click.setChecked(False)
        super().closeEvent(event)


def _key(kind, key):
    """A choice as a string (Qt keeps strings as item data, not tuples)."""
    return f"{kind}\t{key}"


def _unkey(value):
    kind, _, key = value.partition("\t")
    return kind, key


def _per_class_html(metrics):
    rows = "".join(
        f"<tr><td>{_html(c)}</td><td align='right'>{_fmt(m['precision'])}</td>"
        f"<td align='right'>{_fmt(m['recall'])}</td><td align='right'>{_fmt(m['f1'])}</td>"
        f"<td align='right'>{m['support']}</td></tr>"
        for c, m in metrics["per_class"].items())
    return ("<br><table cellspacing='4'><tr><th align='left'>class</th><th>precision</th><th>recall</th>"
            f"<th>F1</th><th>objects</th></tr>{rows}</table>")


def _fmt(v):
    return "" if v is None else f"{v:.2f}"


def _qcolor(hex_color):
    from qtpy.QtGui import QColor

    return QColor(hex_color)


def open_training(bench):
    """The workbench's Training workbench window, created on first use and raised."""
    window = getattr(bench, "_training_window", None)
    if window is None:
        window = TrainingWorkbench(bench)
        bench._training_window = window
    window.show()
    window.raise_()
    window.activateWindow()
    return window


__all__ = ["TrainingWorkbench", "open_training", "object_thumbnail", "train_model", "prepare_training"]
