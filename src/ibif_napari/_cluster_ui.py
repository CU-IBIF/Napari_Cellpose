"""Qt side of cluster connections: login prompts as dialogs, and the Clusters dialog."""
from __future__ import annotations

import dataclasses
import threading

from qtpy.QtCore import QObject, Qt, QThread, Signal
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._cluster import ClusterError, ClusterProfile, check_login, install, load_profiles, save_profiles
from ._process import Cancelled


class QtPrompts(QObject):
    """SSH prompts (two-factor codes, key passphrases, new host keys) shown as dialogs.

    Safe to call from a worker thread: the dialog opens on the GUI thread and the worker
    waits for the answer.
    """

    _request = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self._request.connect(self._on_request, Qt.QueuedConnection)

    def _call(self, fn, *args):
        if QThread.currentThread() == self.thread():
            return fn(*args)
        box = {"done": threading.Event()}
        self._request.emit((box, fn, args))
        box["done"].wait()
        return box["result"]

    def _on_request(self, payload):
        box, fn, args = payload
        try:
            box["result"] = fn(*args)
        finally:
            box["done"].set()

    def ask(self, title, instructions, prompts):
        return self._call(self._ask, title, instructions, prompts)

    def confirm(self, title, text):
        return self._call(self._confirm, title, text)

    def _ask(self, title, instructions, prompts):
        dlg = QDialog(self._parent)
        dlg.setWindowTitle(title or "Login")
        layout = QVBoxLayout(dlg)
        if instructions:
            label = QLabel(instructions)
            label.setWordWrap(True)
            layout.addWidget(label)
        form = QFormLayout()
        edits = []
        for text, echo in prompts:
            edit = QLineEdit()
            if not echo:
                edit.setEchoMode(QLineEdit.Password)
            form.addRow(text.strip(), edit)
            edits.append(edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.Accepted:
            return None
        return [e.text() for e in edits]

    def _confirm(self, title, text):
        answer = QMessageBox.question(self._parent, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes


# ---- profiles dialog -------------------------------------------------------------------

_LABELS = {
    "name": "Name",
    "host": "Login host",
    "user": "Username",
    "port": "SSH port",
    "key_file": "SSH key file",
    "setup": "Setup lines",
    "workdir": "Work folder",
    "activate": "Activate environment",
    "partition": "Partition",
    "account": "Account (allocation)",
    "gpus": "GPUs (--gpus=)",
    "time": "Time limit",
    "cpus": "CPUs",
    "mem": "Memory",
    "extra_sbatch": "Extra #SBATCH lines",
    "idle_minutes": "Stop after idle (min)",
    "server_args": "Extra server options",
    "batch_time": "Batch job time limit",
}
_HINTS = {
    "host": "e.g. login.example-cluster.org",
    "key_file": "empty: SSH agent and ~/.ssh/id_* keys",
    "setup": "run first, e.g. module load python/3.11",
    "activate": "empty: the environment that Install creates",
    "account": "your ACCESS allocation account on this system",
    "extra_sbatch": "#SBATCH --qos=... (one per line)",
    "server_args": "e.g. --models-dir /shared/cellpose-models",
    "batch_time": "for whole series, e.g. 12:00:00",
}


class ClustersDialog(QDialog):
    def __init__(self, parent=None, path=None, prompts=None):
        super().__init__(parent)
        self.setWindowTitle("GPU clusters")
        self.resize(820, 620)
        self.path = path
        self.prompts = prompts or QtPrompts(self)
        self.profiles = load_profiles(path)
        self._busy = False

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Slurm clusters (for example an NSF ACCESS system) where IBIF napari starts a GPU job "
            "on Connect. Fill in your login and allocation, Install once, then Test login."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        row = QHBoxLayout()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show)
        row.addWidget(self.list, 1)

        right = QVBoxLayout()
        form = QFormLayout()
        self.fields = {}
        for f in dataclasses.fields(ClusterProfile):
            if f.name in ("setup", "extra_sbatch"):
                w = QPlainTextEdit()
                w.setFixedHeight(54)
                w.setPlaceholderText(_HINTS.get(f.name, ""))
            elif f.name in ("port", "cpus", "idle_minutes"):
                w = QSpinBox()
                w.setRange(0, 100000)
            else:
                w = QLineEdit()
                w.setPlaceholderText(_HINTS.get(f.name, ""))
            if f.name == "key_file":
                box = QWidget()
                h = QHBoxLayout(box)
                h.setContentsMargins(0, 0, 0, 0)
                h.addWidget(w, 1)
                browse = QPushButton("…")
                browse.clicked.connect(self._browse_key)
                h.addWidget(browse)
                form.addRow(_LABELS[f.name], box)
            else:
                form.addRow(_LABELS[f.name], w)
            self.fields[f.name] = w
        right.addLayout(form)
        buttons = QHBoxLayout()
        self.buttons = []
        for text, slot in (("Save", self._save_current), ("New", self._new), ("Remove", self._remove),
                           ("Test login", self.test), ("Install / update on cluster", self.install),
                           ("Job script", self._show_script)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
            self.buttons.append(b)
        right.addLayout(buttons)
        row.addLayout(right, 3)
        layout.addLayout(row, 3)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 2)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.accept)
        layout.addWidget(close)
        self._append = _Appender(self)
        self._fill_list()

    # -- profile editing -----------------------------------------------------------------

    def _fill_list(self, select=0):
        self.list.blockSignals(True)
        self.list.clear()
        self.list.addItems([p.name for p in self.profiles])
        self.list.blockSignals(False)
        if self.profiles:
            self.list.setCurrentRow(min(select, len(self.profiles) - 1))
        else:
            self._new()

    def _show(self, row):
        if 0 <= row < len(self.profiles):
            self._set_fields(self.profiles[row])

    def _set_fields(self, profile):
        for name, w in self.fields.items():
            value = getattr(profile, name)
            if isinstance(w, QSpinBox):
                w.setValue(int(value))
            elif isinstance(w, QPlainTextEdit):
                w.setPlainText(str(value))
            else:
                w.setText(str(value))

    def _new(self):
        self.list.setCurrentRow(-1)
        self._set_fields(ClusterProfile(name="", host="", user=""))

    def profile(self):
        values = {}
        for name, w in self.fields.items():
            if isinstance(w, QSpinBox):
                values[name] = w.value()
            elif isinstance(w, QPlainTextEdit):
                values[name] = w.toPlainText().strip()
            else:
                values[name] = w.text().strip()
        values["name"] = values["name"] or values["host"]
        return ClusterProfile(**values)

    def _save_current(self):
        p = self.profile()
        if not p.host or not p.user:
            self.log.appendPlainText("Enter at least the login host and username.")
            return None
        row = self.list.currentRow()
        names = [x.name for x in self.profiles]
        if not 0 <= row < len(self.profiles):
            row = names.index(p.name) if p.name in names else len(self.profiles)
        if row < len(self.profiles):
            self.profiles[row] = p
        else:
            self.profiles.append(p)
        save_profiles(self.profiles, self.path)
        self._fill_list(row)
        self.log.appendPlainText(f"Saved {p.name}.")
        return p

    def _remove(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.profiles):
            del self.profiles[row]
            save_profiles(self.profiles, self.path)
            self._fill_list(row)

    def _browse_key(self):
        path, _ = QFileDialog.getOpenFileName(self, "SSH private key")
        if path:
            self.fields["key_file"].setText(path)

    def _show_script(self):
        self.log.setPlainText(self.profile().job_script("<session>"))

    # -- remote actions (in a background thread, prompts as dialogs) ---------------------

    def _background(self, what, fn):
        if self._busy:
            return
        p = self._save_current()
        if p is None:
            return
        self._busy = True
        for b in self.buttons:
            b.setEnabled(False)
        self.log.appendPlainText(f"--- {what} on {p.host} ---")

        def run():
            try:
                self._append.line.emit(fn(p))
            except Cancelled:
                self._append.line.emit("Cancelled.")
            except (ClusterError, OSError) as e:
                self._append.line.emit(f"Error: {e}")
            except Exception as e:  # e.g. paramiko errors
                self._append.line.emit(f"Error: {type(e).__name__}: {e}")
            finally:
                self._append.done.emit()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def _finished(self):
        self._busy = False
        for b in self.buttons:
            b.setEnabled(True)

    def test(self):
        def run(p):
            info = check_login(p, self.prompts)
            version = info["server_version"] or "not installed yet (use Install)"
            return f"Logged in. IBIF napari on the cluster: {version}\nPartitions (name, GPUs):\n{info['partitions']}"

        self._background("Test login", run)

    def install(self):
        def run(p):
            ok = install(p, self.prompts, stream=self._append.line.emit)
            return "Installed." if ok else "Install failed: see the log above."

        self._background("Install", run)


class _Appender(QObject):
    line = Signal(str)
    done = Signal()

    def __init__(self, dialog):
        super().__init__(dialog)
        self.line.connect(dialog.log.appendPlainText)
        self.done.connect(dialog._finished)
