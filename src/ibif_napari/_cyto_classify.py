"""The Classification tab of the Cytometry workbench (phase 5): supervised object
classification.

**Apply a model**: pick a model from the model library (or a folder), check that the
table has what it needs (its features are matched to the table's columns by name, as
phenotype rules are), choose the objects, and Run. Each object gets "<name> class",
"<name> confidence" and "<name> p <class>" columns: plot and gate them (e.g. a
confidence of at least 0.8), colour the plot or the image by class, and **Make
phenotypes** from the predicted classes.

**Train a model…** opens the Training workbench (`_cyto_training`): build a training
set, train or fine-tune a model, check it, and save it with its card to the library.

Models are vtea-core `classification` model folders: plain arrays and JSON, never a
pickle, so models shared through a library folder are safe to open. The first family
is the generic ANN (a multilayer perceptron over measurements, CPU only).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import _cytometry as cy

SPLITS = {"By object (stratified by class)": "object", "By time point / position": "group"}


def _combo(min_chars=18):
    box = QComboBox()
    box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(min_chars)
    return box


def model_input_columns(state, features):
    """{model feature: table column} for the features the table has (exact name, else
    matched as phenotype rules match names), and [features it does not have]."""
    found, missing = {}, []
    for feature in features:
        column, _ = state.match_column(feature) if state.table is not None else (None, None)
        if column is None:
            missing.append(feature)
        else:
            found[feature] = column
    return found, missing


def object_data(state, features, rows):
    """vtea-core ObjectData for table rows `rows`, with the model's feature names as
    columns (taken from the matched table columns)."""
    import pandas as pd
    from vtea_core.classification import ObjectData

    found, missing = model_input_columns(state, features)
    if missing:
        raise ValueError("the table has no column for: " + ", ".join(missing))
    table = pd.DataFrame({f: state.column(found[f])[rows] for f in features})
    return ObjectData(table)


class ClassificationTab(QWidget):
    kind = "classification"

    def __init__(self, bench):
        super().__init__()
        self.bench = bench
        self.state = bench.state
        self._model_path = None
        split = QSplitter(Qt.Horizontal)
        QVBoxLayout(self).addWidget(split)

        # ---- settings ----
        left = QWidget()
        form = QFormLayout(left)
        self.model = _combo(20)
        self.model.setToolTip("A saved classification model (the model library, or Browse…)")
        self.model.currentIndexChanged.connect(self._model_chosen)
        form.addRow("Model", self.model)
        row = QHBoxLayout()
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.setToolTip("Use a model folder from elsewhere (e.g. a shared facility folder)")
        self.browse_btn.clicked.connect(lambda _=False: self.browse())
        self.train_btn = QPushButton("Train a model…")
        self.train_btn.setToolTip("Open the Training workbench: label examples, train a model, check it, "
                                  "save it to the library")
        self.train_btn.clicked.connect(lambda _=False: self.train_dialog())
        row.addWidget(self.browse_btn)
        row.addWidget(self.train_btn)
        form.addRow("", row)
        self.card = QLabel()
        self.card.setWordWrap(True)
        self.card.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow(self.card)
        self.population = _combo(16)
        self.population.setToolTip("Classify these objects: all, a gate, or a phenotype")
        form.addRow("Objects", self.population)
        self.name = QLineEdit()
        form.addRow("Name", self.name)
        self.run_btn = QPushButton("Classify")
        self.run_btn.clicked.connect(lambda _=False: self.run())
        form.addRow(self.run_btn)
        self.status = QLabel()
        self.status.setWordWrap(True)
        form.addRow(self.status)
        split.addWidget(left)

        # ---- results ----
        right = QWidget()
        rl = QVBoxLayout(right)
        row = QHBoxLayout()
        row.addWidget(QLabel("Results"))
        self.analysis = QComboBox()
        self.analysis.currentTextChanged.connect(self.show_analysis)
        row.addWidget(self.analysis, 1)
        rl.addLayout(row)
        self.info = QLabel()
        self.info.setWordWrap(True)
        rl.addWidget(self.info)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Class", "Objects", "Mean confidence", "Phenotype"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setToolTip("Type a phenotype name per class (the same name merges classes; empty: skip)")
        rl.addWidget(self.table, 1)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Within"))
        self.parent_box = _combo(12)
        prow.addWidget(self.parent_box, 1)
        prow.addWidget(QLabel("Confidence ≥"))
        self.min_conf = QDoubleSpinBox()
        self.min_conf.setRange(0.0, 1.0)
        self.min_conf.setSingleStep(0.05)
        self.min_conf.setToolTip("Only objects predicted at least this sure join the phenotypes (0: all)")
        prow.addWidget(self.min_conf)
        self.make_btn = QPushButton("Make phenotypes")
        self.make_btn.clicked.connect(lambda _=False: self.make_phenotypes())
        prow.addWidget(self.make_btn)
        rl.addLayout(prow)
        brow = QHBoxLayout()
        self.plot_btn = QPushButton("Colour the plot by class")
        self.plot_btn.clicked.connect(lambda _=False: self.colour_plot())
        self.image_btn = QPushButton("Show classes on image")
        self.image_btn.clicked.connect(lambda _=False: self.show_on_image())
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(lambda _=False: self.delete())
        for b in (self.plot_btn, self.image_btn, self.delete_btn):
            brow.addWidget(b)
        rl.addLayout(brow)
        split.addWidget(right)
        split.setSizes([360, 760])

    # ---- the model library ------------------------------------------------------------------

    def models(self):
        """[(name, folder)] of the models in the library."""
        from vtea_core.classification import list_models

        return [(p.name, p) for p in list_models(cy.model_library())]

    def refresh(self):
        state = self.state
        from ._cyto_workbench import _fill, _pop_key

        keep = self.model.currentData()
        self.model.blockSignals(True)
        self.model.clear()
        for name, folder in self.models():
            self.model.addItem(name, str(folder))
        if self._model_path and self.model.findData(self._model_path) < 0:
            self.model.addItem(Path(self._model_path).name, self._model_path)
        index = self.model.findData(keep or self._model_path)
        self.model.setCurrentIndex(max(index, 0) if self.model.count() else -1)
        self.model.blockSignals(False)
        _fill(self.population, [(label.strip(), _pop_key(kind, key)) for kind, key, label in state.populations()]
              if state.table is not None else [], self.population.currentData())
        keep_parent = self.parent_box.currentData()
        self.parent_box.clear()
        self.parent_box.addItem("(top level)", None)
        for node in state.tree:
            self.parent_box.addItem("  " * (state.tree.depth(node.name) - 1) + node.name, node.name)
        self.parent_box.setCurrentIndex(max(self.parent_box.findData(keep_parent), 0))
        current = self.analysis.currentText()
        mine = state.analyses_of(self.kind)
        self.analysis.blockSignals(True)
        self.analysis.clear()
        self.analysis.addItems(mine)
        if current in mine:
            self.analysis.setCurrentText(current)
        elif mine:
            self.analysis.setCurrentIndex(len(mine) - 1)
        self.analysis.blockSignals(False)
        if not self.name.text() or self.name.text() in state.analyses:
            self.name.setText(state.unique_analysis_name("Classes"))
        self.train_btn.setEnabled(state.table is not None)
        self._model_chosen()
        self.show_analysis(self.analysis.currentText())

    def browse(self, folder=None):
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Model folder", str(cy.model_library()))
            if not folder:
                return
        self.use_model(folder)

    def use_model(self, folder):
        """Choose the model in `folder` (also used by tests and scripts)."""
        from vtea_core.classification import read_card

        try:
            read_card(folder)
        except ValueError as e:
            self.status.setText(str(e))
            return False
        self._model_path = str(folder)
        self.refresh()
        self.model.setCurrentIndex(self.model.findData(str(folder)))
        return True

    def _model_chosen(self, *_):
        from vtea_core.classification import read_card

        folder = self.model.currentData()
        ready = False
        if not folder:
            self.card.setText("No model yet: <b>Train a model…</b> from your phenotypes, or <b>Browse…</b> "
                              "for a model folder.")
        else:
            try:
                card = read_card(folder)
            except ValueError as e:
                self.card.setText(f"<span style='color:#d62728'>{e}</span>")
            else:
                text = card.summary()
                if card.notes:
                    text += f"<br><i>{_html(card.notes)}</i>"
                if self.state.table is not None:
                    found, missing = model_input_columns(self.state, card.inputs.get("features", []))
                    if missing:
                        text += ("<br><span style='color:#d62728'>This table has no column for: "
                                 + _html(", ".join(missing)) + "</span>")
                    else:
                        ready = True
                        renamed = {f: c for f, c in found.items() if f != c}
                        if renamed:
                            text += "<br>matched: " + _html(", ".join(f"{f} → {c}" for f, c in renamed.items()))
                self.card.setText(text)
        self.run_btn.setEnabled(ready)

    # ---- running ------------------------------------------------------------------------------

    def run(self):
        """Classify the chosen objects with the chosen model. Returns the stored result."""
        from vtea_core.classification import load_model, model_hash, predictions

        from ._cyto_workbench import _pop_parse

        state = self.state
        folder = self.model.currentData()
        if state.table is None or not folder:
            return None
        name = self.name.text().strip() or state.unique_analysis_name("Classes")
        started = time.perf_counter()
        try:
            state._check_analysis_name(name)
            model = load_model(folder)
            population = _pop_parse(self.population.currentData())
            rows = np.flatnonzero(state.mask(*population))
            data = object_data(state, model.inputs().get("features", []), rows)
            result = predictions(model, data)
        except (ValueError, KeyError) as e:
            self.status.setText(str(e))
            return None
        settings = {"model": str(folder), "model_name": Path(folder).name, "model_hash": model_hash(folder),
                    "family": model.family, "population": list(population),
                    "seconds": round(time.perf_counter() - started, 2)}
        info = state.add_classification(name, settings, rows, result, model.classes, model.card.colors)
        done = int(result["class"].notna().sum())
        self.status.setText(f"{name}: {done:,} of {len(rows):,} objects classified"
                            + (f" ({len(rows) - done:,} miss a feature)" if done < len(rows) else "") + ".")
        self.analysis.setCurrentText(name)
        return info

    # ---- results ------------------------------------------------------------------------------

    def show_analysis(self, name):
        state = self.state
        has = bool(name) and name in state.analyses
        for b in (self.make_btn, self.plot_btn, self.image_btn, self.delete_btn):
            b.setEnabled(has)
        self.table.setRowCount(0)
        if not has:
            self.info.setText("Classify objects with a trained model; the predicted classes become columns "
                              "of the table, and can be made phenotypes.")
            return
        info = state.analyses[name]
        pop = info.get("population") or ["all", None]
        where = "all objects" if pop[0] == "all" else f"{pop[0]} {pop[1]}"
        self.info.setText(f"<b>{name}</b>: {info['n_objects']:,} objects ({where}) by model "
                          f"<i>{_html(info.get('model_name', ''))}</i> ({info.get('family', '')}).")
        counts = state.classification_counts(name)
        self.table.setRowCount(len(counts))
        for r, (c, n, conf) in enumerate(counts):
            item = QTableWidgetItem(c)
            item.setForeground(_qcolor(info["colors"][c]))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 0, item)
            for col, text in ((1, f"{n:,}"), (2, "" if conf != conf else f"{conf:.2f}")):
                cell = QTableWidgetItem(text)
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, col, cell)
            self.table.setItem(r, 3, QTableWidgetItem("" if c in state.tree else c))
        self.table.resizeColumnsToContents()

    def phenotype_names(self):
        return {self.table.item(r, 0).text(): (self.table.item(r, 3).text() if self.table.item(r, 3) else "")
                for r in range(self.table.rowCount())}

    def make_phenotypes(self):
        name = self.analysis.currentText()
        if name not in self.state.analyses:
            return []
        try:
            made = self.state.phenotypes_from_classification(
                name, self.phenotype_names(), self.parent_box.currentData(), self.min_conf.value() or None)
        except (ValueError, KeyError) as e:
            self.status.setText(str(e))
            return []
        self.status.setText("Made phenotypes: " + ", ".join(n.name for n in made) + ".")
        return made

    def colour_plot(self):
        name = self.analysis.currentText()
        if name in self.state.analyses:
            self.bench.color_by.setCurrentText(self.state.analyses[name]["columns"]["class"])
            self.bench.tabs.setCurrentIndex(0)

    def show_on_image(self):
        name = self.analysis.currentText()
        if name not in self.state.analyses:
            return None
        try:
            layer = self.state.show_classification(name)
        except ValueError as e:
            self.status.setText(str(e))
            return None
        self.status.setText(f"Shown in layer {layer.name}.")
        return layer

    def delete(self):
        name = self.analysis.currentText()
        if name in self.state.analyses:
            self.state.remove_analysis(name)
            broken = self.state.broken_phenotypes()
            self.status.setText(f"Deleted {name}." + (f" Phenotypes that used it need a new rule: "
                                                      f"{', '.join(broken)}." if broken else ""))

    def train_dialog(self):
        """Open the Training workbench (a window of its own)."""
        from ._cyto_training import open_training

        return open_training(self.bench)


def _metrics_html(metrics, history):
    bacc = metrics.get("balanced_accuracy")
    stopped = (f"{history['epochs']} epochs, weights of epoch {history['best_epoch']} kept"
               if history["best_epoch"] != history["epochs"] else f"{history['epochs']} epochs")
    lines = [f"Balanced accuracy on held-out objects: <b>{bacc:.3f}</b> (accuracy {metrics['accuracy']:.3f}, "
             f"{metrics['n']:,} objects; {stopped})"
             if bacc is not None else "No held-out objects to check on."]
    classes = metrics["classes"]
    head = "".join(f"<th>{_html(c)}</th>" for c in classes)
    body = "".join(f"<tr><th align='left'>{_html(c)}</th>" + "".join(f"<td align='right'>{v}</td>" for v in row)
                   + f"<td align='right'>{_pct(metrics['per_class'][c]['recall'])}</td></tr>"
                   for c, row in zip(classes, metrics["confusion"]))
    lines.append(f"<table cellspacing='4'><tr><th>true \\ predicted</th>{head}<th>recall</th></tr>{body}</table>")
    return "<br>".join(lines)


def _pct(value):
    return "" if value is None else f"{100 * value:.0f}%"


def _safe(name):
    return "".join(c if c.isalnum() or c in " -_.+" else "_" for c in name).strip() or "model"


def _html(text):
    import html

    return html.escape(str(text))


def _qcolor(hex_color):
    from qtpy.QtGui import QColor

    return QColor(hex_color)


__all__ = ["ClassificationTab", "model_input_columns", "object_data"]
