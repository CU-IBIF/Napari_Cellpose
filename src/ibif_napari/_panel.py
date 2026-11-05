"""The IBIF napari panel: a row of icon buttons that switches between the sections.

Configuration · Segmentation · Secondary objects · Measurements · Cytometry · Filament
tracing · Save workspace. Each button's name and description are in its tooltip; the
current section's name is shown above it. Cytometry is a placeholder for now (disabled).
The panel opens on Configuration, and goes back to it when a workspace is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

ICONS = Path(__file__).with_name("icons")
ICON_SIZE = 36
# The selected section's button is highlighted; works on napari's dark and light themes.
BUTTON_STYLE = """
QToolButton { border: 1px solid transparent; border-radius: 6px; padding: 3px; }
QToolButton:hover:enabled { background-color: rgba(128, 128, 128, 0.25); }
QToolButton:checked { background-color: rgba(70, 130, 230, 0.35); border-color: rgba(70, 130, 230, 0.9); }
"""


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    tip: str
    widget: str | None  # "module:Class", or None for a section that is not built yet


SECTIONS = (
    Section("configuration", "Configuration",
            "Time, position and Z sliders, channels, 2D/3D view, projections and the 3D brush",
            "._dims_widget:DimensionsWidget"),
    Section("segmentation", "Segmentation",
            "Cellpose segmentation in 2D or 3D, on this computer, a facility server or a cluster",
            "._cellpose_widget:CellposeWidget"),
    Section("secondary_objects", "Secondary objects",
            "Expand, rings, rims and whole cells grown from the segmented nuclei",
            "._secondary_widget:SecondaryWidget"),
    Section("measurements", "Measurements",
            "Size, position and intensity statistics of every object, as tables",
            "._measure_widget:MeasureWidget"),
    Section("cytometry", "Cytometry",
            "Coming soon: cell/object typing (supervised and unsupervised), neighbourhood analyses "
            "and other cytometry of the measured objects",
            None),
    Section("filament_tracing", "Filament tracing",
            "Trace neurons, microtubules and other filaments in 2D or 3D",
            "._filament_widget:FilamentWidget"),
    Section("save_workspace", "Save workspace",
            "Save or open the whole workspace (layers, view, measurements), or save single layers",
            "._workspace_widget:WorkspaceWidget"),
)
DEFAULT = "configuration"


def _make(spec, viewer):
    module, cls = spec.split(":")
    return getattr(import_module(module, __package__), cls)(viewer)


class IbifPanel(QWidget):
    """All IBIF napari sections in one dock, chosen with icon buttons."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.pages = {}  # key -> section widget (None when not built yet)
        self.buttons = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        bar = QHBoxLayout()
        bar.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.stack = QStackedWidget()
        self._index = {}
        for section in SECTIONS:
            button = QToolButton()
            button.setIcon(QIcon(str(ICONS / f"{section.key}.png")))
            button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
            button.setAutoRaise(True)
            button.setCheckable(True)
            button.setStyleSheet(BUTTON_STYLE)
            button.setAccessibleName(section.title)
            button.setObjectName(f"ibif_{section.key}")
            button.setToolTip(f"<b>{section.title}</b><br>{section.tip}")
            button.clicked.connect(lambda _=False, key=section.key: self.show_section(key))
            self._group.addButton(button)
            self.buttons[section.key] = button
            bar.addWidget(button)
            page = _make(section.widget, napari_viewer) if section.widget else None
            if page is None:
                button.setEnabled(False)  # Cytometry: not built yet
            else:
                self._index[section.key] = self.stack.addWidget(_scrolled(page))
            self.pages[section.key] = page
        bar.addStretch(1)
        layout.addLayout(bar)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        self.title = QLabel()
        font = self.title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.title.setFont(font)
        layout.addWidget(self.title)
        layout.addWidget(self.stack, 1)

        self.show_section(DEFAULT)
        from . import _workspace

        _workspace.on_loaded(napari_viewer, lambda: self.show_section(DEFAULT))

    def show_section(self, key):
        """Switch to a section by key (see SECTIONS); returns its widget."""
        if key not in self._index:
            raise ValueError(f"{key!r} is not an available section")
        self.stack.setCurrentIndex(self._index[key])
        self.buttons[key].setChecked(True)
        self.title.setText(next(s.title for s in SECTIONS if s.key == key))
        return self.pages[key]

    def current_section(self):
        index = self.stack.currentIndex()
        return next(k for k, i in self._index.items() if i == index)


def _scrolled(widget):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area
