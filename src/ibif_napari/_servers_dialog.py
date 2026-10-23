"""Dialog to add, edit, test and remove Cellpose servers (facility GPU workstations)."""
from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

from ._remote import RemoteError, RemoteProcess, load_servers, save_servers


class ServersDialog(QDialog):
    def __init__(self, parent=None, path=None):
        super().__init__(parent)
        self.setWindowTitle("Cellpose servers")
        self.resize(560, 420)
        self.path = path
        self.servers = load_servers(path)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Facility GPU workstations running ibif-napari-server. Ask the facility for the "
            "address and your access token."
        ))
        row = QHBoxLayout()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show)
        row.addWidget(self.list, 1)

        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("Facility GPU")
        self.url = QLineEdit()
        self.url.setPlaceholderText("http://gpu-workstation.example.edu:8765")
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.Password)
        self.cafile = QLineEdit()
        self.cafile.setPlaceholderText("only for a self-signed HTTPS certificate")
        browse = QPushButton("…")
        browse.clicked.connect(self._browse_ca)
        ca_row = QHBoxLayout()
        ca_row.addWidget(self.cafile, 1)
        ca_row.addWidget(browse)
        form.addRow("Name", self.name)
        form.addRow("Address", self.url)
        form.addRow("Access token", self.token)
        form.addRow("Certificate", ca_row)
        right = QVBoxLayout()
        right.addLayout(form)
        buttons = QHBoxLayout()
        for text, slot in (("Save", self._save_current), ("New", self._new), ("Remove", self._remove),
                           ("Test connection", self.test)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        right.addLayout(buttons)
        self.result = QLabel()
        self.result.setWordWrap(True)
        right.addWidget(self.result)
        right.addStretch(1)
        row.addLayout(right, 2)
        layout.addLayout(row)

        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.accept)
        layout.addWidget(close)
        self._fill_list()

    def _fill_list(self, select=0):
        self.list.blockSignals(True)
        self.list.clear()
        self.list.addItems([s["name"] for s in self.servers])
        self.list.blockSignals(False)
        if self.servers:
            self.list.setCurrentRow(min(select, len(self.servers) - 1))
        else:
            self._new()

    def _show(self, row):
        if 0 <= row < len(self.servers):
            s = self.servers[row]
            self.name.setText(s.get("name", ""))
            self.url.setText(s.get("url", ""))
            self.token.setText(s.get("token", ""))
            self.cafile.setText(s.get("cafile", ""))
            self.result.clear()

    def _new(self):
        self.list.clearSelection()
        self.list.setCurrentRow(-1)
        for w in (self.name, self.url, self.token, self.cafile):
            w.clear()
        self.result.clear()

    def _entry(self):
        url = self.url.text().strip()
        if url and "://" not in url:
            url = "http://" + url
        entry = {
            "name": self.name.text().strip() or url,
            "url": url,
            "token": self.token.text().strip(),
        }
        if self.cafile.text().strip():
            entry["cafile"] = self.cafile.text().strip()
        return entry

    def _save_current(self):
        entry = self._entry()
        if not entry["url"] or not entry["token"]:
            self.result.setText("Enter the address and your access token.")
            return
        row = self.list.currentRow()
        names = [s["name"] for s in self.servers]
        if 0 <= row < len(self.servers):
            self.servers[row] = entry
        elif entry["name"] in names:
            row = names.index(entry["name"])
            self.servers[row] = entry
        else:
            self.servers.append(entry)
            row = len(self.servers) - 1
        save_servers(self.servers, self.path)
        self._fill_list(row)
        self.result.setText(f"Saved {entry['name']}.")

    def _remove(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.servers):
            del self.servers[row]
            save_servers(self.servers, self.path)
            self._fill_list(row)

    def _browse_ca(self):
        path, _ = QFileDialog.getOpenFileName(self, "Server certificate", filter="Certificates (*.pem *.crt);;All files (*)")
        if path:
            self.cafile.setText(path)

    def test(self):
        entry = self._entry()
        try:
            info = RemoteProcess(entry["url"], entry["token"], name=entry["name"],
                                 cafile=entry.get("cafile"), timeout=10).status()
        except RemoteError as e:
            self.result.setText(f"Not connected: {e}")
            return False
        self.result.setText(
            f"Connected as {info.get('user')}. {info.get('device')}, Cellpose {info.get('cellpose')}, "
            f"server {info.get('version')}. {info.get('queued', 0)} job(s) waiting."
        )
        return True
