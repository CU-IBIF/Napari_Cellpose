"""The hierarchy map of the Cytometry workbench (spatial hierarchy, H4b).

A panel docked on the left of the workbench that shows every level of the spatial
hierarchy as a tree - the segmented objects, the windows and regions built on them,
and what is built on those, to any depth - and is the main way to move around it:

- each node: the level's name, kind (● segmentation, ◆ windows, ▣ regions), object
  count, how it was built, and a bar of its type mix (its phenotypes, or else the
  neighbourhood types of the level below, coloured as on the plot);
- click a node to work on that level (every tab follows);
- right-click for: show on the image, make the level again from its parent (when it is
  out of date), rename, export its table, delete it with everything built on it;
- a level whose parent changed since it was built is marked *out of date* (red), with
  the reason in its tooltip; nothing is rebuilt silently;
- the **Selection** column follows the objects shown on the plot (Show) through the
  hierarchy: at every other level, how many objects are related to them - their
  members below, the objects containing them above. Double-click a count to go there
  with those objects shown (Follow selection).
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QFileDialog,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import _cytometry as cy

MARKS = {"objects": "●", "windows": "◆", "regions": "▣"}
BAR = QSize(64, 10)
COL_LEVEL, COL_OBJECTS, COL_SELECTION = 0, 1, 2


def type_mix(state):
    """[(fraction, colour)] of a level's objects by type: its phenotypes (deepest), or
    else the neighbourhood types of the level below; [] when it has neither."""
    if state.table is None or not len(state.table):
        return []
    n = len(state.table)
    if len(state.tree):
        assign = state.assignment()["phenotype"].to_numpy()
        out = []
        for node in state.tree:
            share = float((assign == node.name).sum()) / n
            if share > 0:
                out.append((share, node.color or "#bdbdbd"))
        return out
    if cy.LEVEL_TYPE in state.table.columns:
        values = state.table[cy.LEVEL_TYPE].to_numpy(np.int64)
        ids, counts = np.unique(values[values > 0], return_counts=True)
        return [(c / n, cy.PALETTE[(int(i) - 1) % len(cy.PALETTE)]) for i, c in zip(ids, counts)]
    return []


def mix_icon(parts):
    """A small horizontal bar of `parts` [(fraction, colour)]."""
    pixmap = QPixmap(BAR)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setPen(Qt.NoPen)
    x = 0.0
    for share, colour in parts:
        width = share * BAR.width()
        painter.setBrush(QColor(colour))
        painter.drawRect(int(round(x)), 0, max(int(round(x + width)) - int(round(x)), 1), BAR.height())
        x += width
    painter.end()
    return QIcon(pixmap)


def built_text(info):
    """How a level was built, in a few words."""
    p = info.get("provenance") or {}
    if info["kind"] == "objects":
        return "segmentation"
    if info["kind"] == "windows":
        how = f"k {p['k']}" if p.get("method") == "nearest" else f"r {p.get('radius', 0):g}"
        classes = p.get("classes")
        return f"windows · {how}" + (f" · by {classes}" if classes else "")
    how = "touching" if p.get("touching", True) else ""
    if p.get("d"):
        how = (how + " or " if how else "") + f"within {p['d']:g}"
    return "regions · " + (how or "joined") + (f" · closeness {p['tau']:g}" if p.get("tau") is not None else "")


class HierarchyMap(QWidget):
    """The tree of levels; `bench` is the Workbench it navigates."""

    def __init__(self, bench):
        super().__init__()
        self.bench = bench
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("<b>Hierarchy</b>")
        title.setToolTip(__doc__.split("\n\n")[1])
        layout.addWidget(title)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Level", "Objects", "Selection"])
        self.tree.setIconSize(BAR)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(14)
        self.tree.setTextElideMode(Qt.ElideMiddle)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(COL_LEVEL, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_OBJECTS, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_SELECTION, QHeaderView.ResizeToContents)
        self.setMinimumWidth(280)
        self.tree.itemClicked.connect(self._clicked)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        layout.addWidget(self.tree, 1)
        self.legend = QLabel("● segmentation  ◆ windows  ▣ regions")
        self.legend.setStyleSheet("color: #808080")
        layout.addWidget(self.legend)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.items = {}

    @property
    def levels(self):
        return self.bench.state

    # ---- filling ----------------------------------------------------------------------------

    def refresh(self):
        levels = self.levels
        self.tree.blockSignals(True)
        self.tree.clear()
        self.items = {}
        related = self.selection_counts()
        for _depth, name in levels.level_tree():
            info = levels.level_info(name)
            state = levels.level(name)
            item = QTreeWidgetItem([f"{MARKS.get(info['kind'], '')} {name}", f"{info['objects']:,}",
                                    related.get(name, "")])
            item.setData(COL_LEVEL, Qt.UserRole, name)
            item.setIcon(COL_OBJECTS, mix_icon(type_mix(state)))
            stale = levels.staleness(name)
            tip = f"{name}: {built_text(info)}" + (f"\nOut of date: {stale}" if stale else "")
            item.setToolTip(COL_LEVEL, tip)
            item.setToolTip(COL_OBJECTS, f"{info['objects']:,} objects; the bar is their type mix")
            if stale:
                item.setForeground(COL_LEVEL, QColor("#d62728"))
                item.setText(COL_LEVEL, item.text(COL_LEVEL) + "  (out of date)")
            if name in related and name != levels.current_level:
                item.setToolTip(COL_SELECTION, "Related to the objects shown: double-click to go there")
            sub = QTreeWidgetItem([built_text(info), "", ""])
            sub.setToolTip(COL_LEVEL, built_text(info))
            sub.setFlags(Qt.ItemIsEnabled)
            sub.setForeground(COL_LEVEL, QColor("#808080"))
            sub.setData(COL_LEVEL, Qt.UserRole, name)
            item.addChild(sub)
            parent = self.items.get(info["parent"])
            (parent.addChild(item) if parent is not None else self.tree.addTopLevelItem(item))
            self.items[name] = item
            if name == levels.current_level:
                font = item.font(COL_LEVEL)
                font.setBold(True)
                item.setFont(COL_LEVEL, font)
                self.tree.setCurrentItem(item)
        self.tree.expandAll()
        self.tree.blockSignals(False)

    def selection_counts(self):
        """{level: "n"} - how many objects of each other level are related to the ones
        shown (Show) at the current level; the current level gives its own count."""
        from . import _cyto_levels

        levels = self.levels
        here = levels.current_level
        if levels.table is None or len(levels.level_names()) < 2:
            return {}
        try:
            kind, key = self.bench.current_population()
            mask = np.asarray(levels.mask(kind, key), bool)
        except (KeyError, ValueError, TypeError):
            return {}
        out = {here: f"{int(mask.sum()):,} shown"}
        for name in levels.level_names():
            if name == here or levels.level_info(name)["objects"] == 0:
                continue
            try:
                out[name] = f"{int(_cyto_levels.related(levels, here, mask, name).sum()):,}"
            except (KeyError, ValueError):
                continue
        return out

    # ---- acting -----------------------------------------------------------------------------

    def _clicked(self, item, _column):
        name = item.data(COL_LEVEL, Qt.UserRole)
        if name and name != self.levels.current_level:
            self.bench.set_level(name)

    def _double_clicked(self, item, column):
        name = item.data(COL_LEVEL, Qt.UserRole)
        if column == COL_SELECTION and name and name != self.levels.current_level and item.text(COL_SELECTION):
            self.follow(name)

    def follow(self, name):
        """Go to level `name` with the objects related to those shown."""
        return self.bench.follow_selection(name)

    def _menu(self, point):
        item = self.tree.itemAt(point)
        if item is None:
            return
        name = item.data(COL_LEVEL, Qt.UserRole)
        menu = QMenu(self)
        menu.addAction("Work on this level", lambda: self.bench.set_level(name))
        if name != self.levels.current_level:
            menu.addAction("Follow the shown objects here", lambda: self.follow(name))
        if name != cy.ROOT_LEVEL:
            if self.levels.staleness(name):
                menu.addAction("Make again from its parent", lambda: self.rebuild(name))
            menu.addAction("Rename…", lambda: self.rename(name))
        menu.addAction("Export its table…", lambda: self.export(name))
        if name != cy.ROOT_LEVEL:
            menu.addSeparator()
            menu.addAction("Delete…", lambda: self.delete(name))
        menu.exec_(self.tree.viewport().mapToGlobal(point))

    def rebuild(self, name, confirm=True):
        levels = self.levels
        gone = [n for n in levels.level_names() if name in levels._ancestors(n)]
        if confirm and gone:
            answer = QMessageBox.question(
                self, "Make the level again",
                f"Making {name} again drops what was built on it: {', '.join(gone)}. Continue?")
            if answer != QMessageBox.Yes:
                return None
        try:
            state = levels.rebuild_level(name)
        except ValueError as e:
            self.status.setText(str(e))
            return None
        self.status.setText(f"{name} made again: {len(state.table):,} objects.")
        return state

    def rename(self, name, new=None):
        if new is None:
            new, ok = QInputDialog.getText(self, "Rename level", "New name:", text=name)
            if not ok:
                return False
        try:
            self.levels.rename_level(name, new)
        except ValueError as e:
            self.status.setText(str(e))
            return False
        return True

    def export(self, name, path=None):
        """Write a level's table (with its analysis columns) as CSV."""
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, f"Export {name}", f"{name}.csv", "CSV (*.csv)")
            if not path:
                return None
        frame = self.levels.level(name).frame()
        frame.to_csv(path, index=False)
        self.status.setText(f"{name}: {len(frame):,} rows written.")
        return path

    def delete(self, name, confirm=True):
        levels = self.levels
        gone = [name] + [n for n in levels.level_names() if name in levels._ancestors(n)]
        if confirm:
            answer = QMessageBox.question(self, "Delete level", f"Delete {', '.join(gone)}?")
            if answer != QMessageBox.Yes:
                return []
        removed = levels.remove_level(name)
        self.status.setText(f"Deleted {', '.join(removed)}.")
        return removed
