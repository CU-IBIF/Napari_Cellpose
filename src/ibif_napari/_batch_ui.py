"""The Batch jobs dialog: follow, fetch, cancel, resubmit and remove cluster batch jobs."""
from __future__ import annotations

import threading

from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from . import _batch
from ._process import Cancelled

COLUMNS = ["Submitted", "Cluster", "Labels layer", "Positions", "Job", "Status"]


class _Signals(QObject):
    status = Signal(str, object)  # record id, status dict (or an error string)
    progress = Signal(int, int)
    message = Signal(str)
    fetched = Signal(object, object, object)  # record, labels, missing
    done = Signal()


class BatchJobsDialog(QDialog):
    """`on_labels(record, labels, path)` is called (on the GUI thread) after a fetch."""

    def __init__(self, parent=None, prompts=None, on_labels=None, records_path=None, clusters=None):
        super().__init__(parent)
        from ._cluster import load_profiles

        self.setWindowTitle("Batch jobs")
        self.resize(860, 420)
        self.prompts = prompts
        self.on_labels = on_labels
        self.records_path = records_path
        self.profiles = {p.name: p for p in (clusters if clusters is not None else load_profiles())}
        self._clients = {}
        self._busy = False
        self._refill = False  # reload the table when the background work ends
        self.statuses = {}

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Batch jobs segment a whole series on a cluster; napari does not need to stay open. "
            "Fetch results loads the labels as a layer and saves them as a TIFF next to the image."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.buttons = {}
        for text, slot in (("Refresh", self.refresh), ("Fetch results", self.fetch),
                           ("Cancel job", self.cancel_job), ("Resubmit", self.resubmit), ("Remove", self.remove)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
            self.buttons[text] = b
        row.addStretch(1)
        layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.accept)
        layout.addWidget(close)

        self.signals = _Signals(self)
        self.signals.status.connect(self._set_status)
        self.signals.progress.connect(self._set_progress)
        self.signals.message.connect(self.message.setText)
        self.signals.fetched.connect(self._fetched)
        self.signals.done.connect(self._done)
        self._fill()
        if self.records:
            self.refresh()

    # -- table ----------------------------------------------------------------------------

    def _fill(self):
        self.records = list(reversed(_batch.load_records(self.records_path)))  # newest first
        self.table.setRowCount(len(self.records))
        for r, rec in enumerate(self.records):
            values = [rec.submitted.replace("T", " "), rec.cluster, rec.name, str(rec.count), rec.job_id,
                      self._status_text(rec)]
            for c, v in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(v))
        if self.records and self.table.currentRow() < 0:
            self.table.selectRow(0)
        if not self.records:
            self.message.setText("No batch jobs yet. Submit one from the Cellpose panel.")

    def _status_text(self, rec):
        st = self.statuses.get(rec.id)
        if st is None:
            return "…"
        if isinstance(st, str):
            return st
        text = f"{st['state']}: {st['done']}/{st['total']}"
        if rec.fetched:
            text += " (fetched)"
        return f"{text}. {st['message']}"

    def _set_status(self, rec_id, st):
        self.statuses[rec_id] = st
        for r, rec in enumerate(self.records):
            if rec.id == rec_id:
                self.table.setItem(r, 5, QTableWidgetItem(self._status_text(rec)))

    def selected(self):
        r = self.table.currentRow()
        return self.records[r] if 0 <= r < len(self.records) else None

    # -- background work ------------------------------------------------------------------

    def _client(self, rec):
        profile = self.profiles.get(rec.cluster)
        if profile is None:
            raise RuntimeError(f"no cluster profile named {rec.cluster!r} (see Clusters…)")
        if rec.cluster not in self._clients:
            self._clients[rec.cluster] = _batch.BatchClient(profile, self.prompts)
        return self._clients[rec.cluster]

    def _background(self, what, fn):
        if self._busy:
            return False
        self._busy = True
        for b in self.buttons.values():
            b.setEnabled(False)
        self.message.setText(what)

        def run():
            try:
                fn()
            except Cancelled:
                self.signals.message.emit("Cancelled.")
            except Exception as e:
                self.signals.message.emit(f"Error: {e}")
            finally:
                self.signals.done.emit()

        threading.Thread(target=run, daemon=True).start()
        return True

    def _done(self):
        if self._refill:
            self._refill = False
            self._fill()
        self._busy = False
        self.progress.setVisible(False)
        for b in self.buttons.values():
            b.setEnabled(True)

    def _set_progress(self, done, total):
        self.progress.setVisible(True)
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    # -- actions ---------------------------------------------------------------------------

    def refresh(self):
        records = list(self.records)

        def work():
            for rec in records:
                try:
                    self.signals.status.emit(rec.id, self._client(rec).status(rec))
                except Cancelled:
                    raise
                except Exception as e:
                    self.signals.status.emit(rec.id, f"unknown ({e})")
            self.signals.message.emit("Status updated.")

        self._background("Checking the cluster…", work)

    def fetch(self):
        rec = self.selected()
        if rec is None:
            return

        def work():
            gen = self._client(rec).fetch(rec)
            try:
                while True:
                    self.signals.progress.emit(*next(gen))
            except StopIteration as stop:
                labels, missing = stop.value
            self.signals.fetched.emit(rec, labels, missing)

        self._background(f"Downloading the labels of batch {rec.id}…", work)

    def _fetched(self, rec, labels, missing):
        try:
            path = _batch.save_tiff(labels, rec)
        except OSError as e:
            path = None
            self.message.setText(f"Could not save the TIFF: {e}")
        if path is not None:
            rec.fetched = str(path)
            _batch.update_record(rec, self.records_path)
            note = f" {len(missing)} of {rec.count} positions had no result yet (left empty)." if missing else ""
            self.message.setText(f"Saved {path}.{note}")
        if self.on_labels is not None:
            self.on_labels(rec, labels, path)
        self._fill()

    def cancel_job(self):
        rec = self.selected()
        if rec is None:
            return

        def work():
            self._client(rec).cancel(rec)
            self.signals.message.emit("Job cancelled.")

        self._background(f"Cancelling job {rec.job_id}…", work)

    def resubmit(self):
        rec = self.selected()
        if rec is None:
            return

        def work():
            self._client(rec).resubmit(rec, self.records_path)
            self.signals.message.emit(f"Resubmitted as job {rec.job_id}; finished positions are skipped.")

        self._refill = self._background(f"Resubmitting batch {rec.id}…", work)

    def remove(self):
        rec = self.selected()
        if rec is None:
            return
        if QMessageBox.question(
            self, "Remove batch",
            f"Delete batch {rec.id} on {rec.cluster} (its uploaded images and results) and end its job?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self._remove(rec)

    def _remove(self, rec):
        def work():
            self._client(rec).remove(rec, self.records_path)
            self.signals.message.emit(f"Removed batch {rec.id}.")

        self._refill = self._background(f"Removing batch {rec.id}…", work)

    def done(self, result):
        for client in self._clients.values():
            client.close()
        super().done(result)
