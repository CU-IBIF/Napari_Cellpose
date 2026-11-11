"""Segmentation section: two tabs, Cellpose (deep learning) and Threshold + watershed
(classical, CPU only)."""
from __future__ import annotations

from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

CELLPOSE_TAB, CLASSIC_TAB = "Cellpose", "Threshold + watershed"


class SegmentationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        from ._cellpose_widget import CellposeWidget
        from ._classic_widget import ClassicWidget

        self.viewer = napari_viewer
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.cellpose = CellposeWidget(napari_viewer)
        self.classic = ClassicWidget(napari_viewer)
        self.tabs.addTab(self.cellpose, CELLPOSE_TAB)
        self.tabs.addTab(self.classic, CLASSIC_TAB)
        self.tabs.setTabToolTip(0, "Deep-learning segmentation, here or on a GPU server or cluster")
        self.tabs.setTabToolTip(1, "Classical threshold and watershed on this computer's CPU: no GPU, "
                                   "model or server needed")
        layout.addWidget(self.tabs)
        # Without Cellpose here and no server to run it on, start on the tab that works.
        if not self.cellpose._local_available and not self.cellpose.run_btn.isEnabled():
            self.tabs.setCurrentWidget(self.classic)

    def show_tab(self, name):
        """Switch to a tab by name (CELLPOSE_TAB or CLASSIC_TAB); returns its widget."""
        widget = {CELLPOSE_TAB: self.cellpose, CLASSIC_TAB: self.classic}[name]
        self.tabs.setCurrentWidget(widget)
        return widget


__all__ = ["SegmentationWidget", "CELLPOSE_TAB", "CLASSIC_TAB"]
