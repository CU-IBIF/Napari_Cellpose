"""Dialog to pick which images of a multi-image file (e.g. a Leica LIF project) to open."""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


class ImageChooser(QDialog):
    def __init__(self, title, scenes, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Open images from {title}")
        self.resize(520, 480)
        self._scenes = scenes
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{len(scenes)} images in this file. Tick the ones to open:"))

        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by name…")
        self.filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter)

        self.list = QListWidget()
        for i, s in enumerate(scenes):
            item = QListWidgetItem(f"{s.name}    ({s.summary})")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i == 0 else Qt.Unchecked)
            item.setData(Qt.UserRole, i)
            self.list.addItem(item)
        self.list.itemDoubleClicked.connect(self._open_only)
        self.list.itemChanged.connect(self._update)
        layout.addWidget(self.list, 1)

        row = QHBoxLayout()
        for text, state in (("Select all shown", Qt.Checked), ("Select none", Qt.Unchecked)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, st=state: self._set_all(st))
            row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)

        self.stack = QCheckBox("Stack images of the same size as positions (one slider)")
        layout.addWidget(self.stack)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update()

    def _items(self):
        return [self.list.item(i) for i in range(self.list.count())]

    def _apply_filter(self, text):
        for item in self._items():
            item.setHidden(text.lower() not in item.text().lower())

    def _set_all(self, state):
        for item in self._items():
            if not item.isHidden():
                item.setCheckState(state)

    def _open_only(self, item):
        for other in self._items():
            other.setCheckState(Qt.Checked if other is item else Qt.Unchecked)
        self.accept()

    def selected(self):
        return [item.data(Qt.UserRole) for item in self._items() if item.checkState() == Qt.Checked]

    def _update(self, *_):
        chosen = [self._scenes[i] for i in self.selected()]
        same = len(chosen) > 1 and len({(s.data.shape, s.axes) for s in chosen}) == 1
        self.stack.setEnabled(same)
        if not same:
            self.stack.setChecked(False)
        self.buttons.button(QDialogButtonBox.Open).setEnabled(bool(chosen))


def choose_images(title, scenes):
    """Ask which images to open: returns (indices, stack_as_positions), or None if
    cancelled. Without a running Qt application (scripts), every image is opened."""
    app = QApplication.instance()
    if app is None:
        return list(range(len(scenes))), False
    dialog = ImageChooser(title, scenes, parent=app.activeWindow())
    if dialog.exec() != QDialog.Accepted:
        return None
    return dialog.selected(), dialog.stack.isChecked()
