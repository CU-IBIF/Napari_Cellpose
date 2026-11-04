"""Workspace panel: save / open the whole session, and save layers as files."""
from __future__ import annotations

from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import _workspace


class WorkspaceWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None
        layout = QVBoxLayout(self)
        intro = QLabel(
            "A workspace keeps everything to come back to later or to share: all layers with "
            "their display settings, the view and the measurement tables. Microscope files "
            "are referenced, not copied, unless you include copies below."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.save_btn = QPushButton("Save workspace…")
        self.save_btn.clicked.connect(lambda: self.save())
        layout.addWidget(self.save_btn)
        self.include_images = QCheckBox("Include copies of the images (to share without the original files)")
        self.include_images.setToolTip("Can be as large as the original files")
        layout.addWidget(self.include_images)
        self.open_btn = QPushButton("Open workspace…")
        self.open_btn.clicked.connect(lambda: self.open())
        layout.addWidget(self.open_btn)
        self.layers_btn = QPushButton("Save selected layers…")
        self.layers_btn.setToolTip("Images and labels as TIFF, shapes and points as CSV")
        self.layers_btn.clicked.connect(lambda: self.save_selected())
        layout.addWidget(self.layers_btn)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

    def _busy(self, busy):
        for b in (self.save_btn, self.open_btn, self.layers_btn):
            b.setEnabled(not busy)
        self.progress.setVisible(busy)

    # ---- save -----------------------------------------------------------------------------

    def save(self, path=None):
        if self._worker is not None:
            return
        if not len(self.viewer.layers):
            self.status.setText("Nothing to save: no layers are open.")
            return
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save workspace", "workspace.ibif",
                                                  "IBIF workspace (*.ibif)")
            if not path:
                return
        include = self.include_images.isChecked()
        viewer = self.viewer

        @thread_worker
        def work():
            done = []
            folder, notes = _workspace.save_workspace(
                viewer, path, include, progress=lambda n, total: done.append((n, total)))
            yield (1, 1)
            return folder, notes

        def saved(result):
            folder, notes = result
            text = f"Saved workspace {folder}"
            self.status.setText(text + (". Not saved: " + "; ".join(notes) if notes else "."))

        self._run(work(), saved, "Saving the workspace…")

    def save_selected(self, folder=None):
        layers = list(self.viewer.layers.selection)
        if not layers:
            self.status.setText("Select the layers to save in the layer list first.")
            return
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Folder for the selected layers")
            if not folder:
                return

        @thread_worker
        def work():
            yield (0, 1)
            return _workspace.save_layers(layers, folder)

        self._run(work(), lambda paths: self.status.setText(f"Saved {len(paths)} files in {folder}."),
                  f"Saving {len(layers)} layers…")

    def _run(self, worker, on_done, message):
        worker.returned.connect(on_done)
        worker.errored.connect(lambda e: self.status.setText(f"Failed: {e}"))
        worker.finished.connect(self._finished)
        self._worker = worker
        self._busy(True)
        self.progress.setRange(0, 0)
        self.status.setText(message)
        worker.start()

    def _finished(self):
        self._worker = None
        self._busy(False)

    # ---- open ------------------------------------------------------------------------------

    def open(self, path=None, replace=None):
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Open workspace (the .ibif folder)")
            if not path:
                return None
        if replace is None:
            replace = True
            if len(self.viewer.layers):
                answer = QMessageBox.question(
                    self, "Open workspace", "Close the layers that are open now?\n"
                    "(No: add the workspace's layers to them)",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes,
                )
                if answer == QMessageBox.Cancel:
                    return None
                replace = answer == QMessageBox.Yes
        try:
            notes = _workspace.load_workspace(self.viewer, path, replace=replace)
        except (OSError, ValueError) as e:
            self.status.setText(f"Could not open {path}: {e}")
            return None
        text = f"Opened {path}."
        if notes:
            text += " Problems: " + "; ".join(notes)
        self.status.setText(text)
        return notes
