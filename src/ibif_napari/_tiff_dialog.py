"""Dialog: how the planes of a TIFF are arranged (dimension order and sizes)."""
from __future__ import annotations


from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLayout,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import _tiff

NAMES = {"X": "X (width)", "Y": "Y (height)", "Z": "Z (slices)", "C": "C (channels)",
         "T": "T (time points)", "P": "P (positions / series)", "S": "S (RGB samples)"}


class DimensionOrderDialog(QDialog):
    """Asks the order of the dimensions, fastest first (e.g. XYCZT), and their sizes; the
    sizes must account for every value in the file."""

    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.info = info
        self.setWindowTitle(f"Dimensions of {info.name}" if info.name else "Dimensions")
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SetMinimumSize)  # grows with the rows of the order
        about = QLabel(f"<b>{info.summary()}</b>.<br>Give the order of the dimensions from the "
                       "<b>fastest</b> changing to the slowest, like ImageJ / Bio-Formats: "
                       "<i>XYCZT</i> means the planes go channel 1, channel 2, … of the first "
                       "slice, then the next slice. Dashes are fine (XY-Z-C-T); X and Y need not "
                       "come first (CXYZT: channels interleaved pixel by pixel). "
                       "Letters: X, Y, Z, C, T, P (positions), S (RGB).")
        about.setWordWrap(True)
        about.setMinimumWidth(520)
        about.setMinimumHeight(about.heightForWidth(520))
        layout.addWidget(about)
        form = QFormLayout()
        self.order = QComboBox()
        self.order.setEditable(True)
        y, x = info.plane
        start = info.layout.order
        choices = [start] + [p for p in _tiff.PRESETS if p != start]
        self.order.addItems(choices)
        self.order.setCurrentText(start)
        self.order.setToolTip("Fastest changing dimension first")
        form.addRow("Dimension order", self.order)
        layout.addLayout(form)
        self.grid_box = QWidget()
        self.grid = QGridLayout(self.grid_box)
        self.grid.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.grid_box)
        self.check = QLabel()
        self.check.setWordWrap(True)
        self.check.setMinimumHeight(2 * self.check.fontMetrics().lineSpacing())
        layout.addWidget(self.check)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        # Sizes remembered across order changes; X / Y from the planes.
        self._sizes = {"X": x, "Y": y, **info.layout.sizes}
        self._spins = {}
        self._edited = set()  # sizes typed by the user (the others are worked out)
        self.order.currentTextChanged.connect(self._order_changed)
        self._order_changed(start)

    def _order_text(self):
        return self.order.currentText()

    def _order_changed(self, text):
        for spin in self._spins.values():
            self._sizes[spin.objectName()] = spin.value()
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self._spins = {}
        try:
            order = _tiff.clean_order(text)
        except ValueError as e:
            self.check.setText(f"<font color='#d62728'>{e}</font>")
            self.buttons.button(QDialogButtonBox.Open).setEnabled(False)
            return
        # Unknown sizes (a dimension new to this order) are worked out from the rest.
        sizes = {a: self._sizes.get(a, 0) for a in order}
        if sum(1 for a in order if not sizes[a]) == 1:
            sizes = _tiff.fill_sizes(order, sizes, self.info.total)
        for row, a in enumerate(order):
            self.grid.addWidget(QLabel(NAMES[a]), row, 0)
            spin = QSpinBox()
            spin.setObjectName(a)
            spin.setRange(1, 10 ** 9)
            spin.setValue(max(1, sizes.get(a) or 1))
            spin.valueChanged.connect(lambda _=0, a=a: self._size_changed(a))
            self.grid.addWidget(spin, row, 1)
            self._spins[a] = spin
        self._validate()


    def _size_changed(self, axis):
        self._edited.add(axis)
        self._complete()
        self._validate()

    def _complete(self):
        """When the sizes do not add up, work out one the user has not typed (the slowest
        such dimension that fits) from the others."""
        layout = self.chosen()
        if layout is None or layout.total == self.info.total:
            return
        for a in reversed(layout.order):
            if a in self._edited or a in "XY":
                continue
            rest = layout.total // layout.sizes[a]
            if rest and self.info.total % rest == 0:
                spin = self._spins[a]
                spin.blockSignals(True)
                spin.setValue(self.info.total // rest)
                spin.blockSignals(False)
                return

    def _validate(self, *_):
        layout = self.chosen()
        ok = layout is not None and layout.total == self.info.total
        if layout is None:
            return
        product = " × ".join(f"{a} {layout.sizes[a]}" for a in reversed(layout.order))
        if ok:
            self.check.setText(f"<font color='#2ca02c'>{product} = {layout.total:,} values ✓</font>")
        else:
            planes = self.info.total / max(1, layout.sizes.get("X", 1) * layout.sizes.get("Y", 1))
            self.check.setText(f"<font color='#d62728'>{product} = {layout.total:,} values, but the "
                               f"file has {self.info.total:,} ({planes:g} planes of "
                               f"{self.info.plane[1]} × {self.info.plane[0]}).</font>")
        self.buttons.button(QDialogButtonBox.Open).setEnabled(ok)

    def set_order(self, text, **sizes):
        """Choose an order and sizes (also used by tests and scripts)."""
        self._sizes.update(sizes)
        self.order.setCurrentText(text)
        self._order_changed(text)
        for a, n in sizes.items():
            if a in self._spins:
                self._spins[a].setValue(n)
                self._size_changed(a)

    def chosen(self):
        """The chosen Layout (None while the order is invalid)."""
        try:
            order = _tiff.clean_order(self._order_text())
        except ValueError:
            return None
        if set(order) != set(self._spins):
            return None
        return _tiff.Layout(order, {a: self._spins[a].value() for a in order})


__all__ = ["DimensionOrderDialog"]
