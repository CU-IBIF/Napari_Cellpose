"""The user guide (manual/README.md, installed with the package), shown in a window from
napari's Help menu: Help > IBIF napari user guide."""
from __future__ import annotations

import re
from pathlib import Path

from qtpy.QtCore import Qt, QTimer, QUrl
from qtpy.QtGui import QColor, QDesktopServices, QImage, QPalette, QTextCursor
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

GUIDE = Path(__file__).with_name("manual") / "README.md"
ONLINE = "https://github.com/CU-IBIF/Napari_Cellpose/blob/main/src/ibif_napari/manual/README.md"
ACTION_ID = "ibif_napari.user_guide"
TITLE = "IBIF napari user guide"

_window = None
_registered = False


def guide_text():
    """The guide's Markdown, without the Mermaid diagram (Qt cannot draw it; the table
    after it says the same)."""
    text = GUIDE.read_text(encoding="utf-8")
    return re.sub(r"```mermaid\n.*?```\n", "", text, flags=re.S)


def slug(heading):
    """GitHub's anchor for a heading, e.g. '7. Segmentation with Cellpose' ->
    '7-segmentation-with-cellpose' (what the guide's contents links use)."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


class GuideWindow(QWidget):
    """The guide in a window of its own: links, contents, images scaled to fit, Find."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(TITLE)
        self.resize(980, 860)
        layout = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.find_box = QLineEdit()
        self.find_box.setPlaceholderText("Find in the guide… (Enter for the next match)")
        self.find_box.returnPressed.connect(self.find_next)
        bar.addWidget(self.find_box, 1)
        top = QPushButton("Contents")
        top.setToolTip("Back to the start of the guide")
        top.clicked.connect(lambda: self.browser.verticalScrollBar().setValue(0))
        bar.addWidget(top)
        open_file = QPushButton("Open the .md file")
        open_file.setToolTip(str(GUIDE))
        open_file.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(GUIDE))))
        bar.addWidget(open_file)
        online = QPushButton("Online")
        online.setToolTip("The guide on GitHub (latest version)")
        online.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(ONLINE)))
        bar.addWidget(online)
        layout.addLayout(bar)

        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self.follow)
        palette = self.browser.palette()  # links readable on napari's dark and light themes
        dark = palette.color(QPalette.Base).lightness() < 128
        palette.setColor(QPalette.Link, QColor("#7fb2ff" if dark else "#1f5fbf"))
        self.browser.setPalette(palette)
        layout.addWidget(self.browser, 1)

        self._sizes = {}  # image name -> natural QSize
        self._fit = QTimer(self)
        self._fit.setSingleShot(True)
        self._fit.setInterval(150)
        self._fit.timeout.connect(self.fit_images)
        self.load()

    # ---- content --------------------------------------------------------------------------

    def load(self):
        doc = self.browser.document()
        doc.setBaseUrl(QUrl.fromLocalFile(str(GUIDE.parent) + "/"))
        self.browser.setSearchPaths([str(GUIDE.parent)])
        self.browser.setMarkdown(guide_text())
        self.fit_images()

    def headings(self):
        """{anchor: text block} of every heading."""
        out = {}
        block = self.browser.document().begin()
        while block.isValid():
            if block.blockFormat().headingLevel() > 0:
                out.setdefault(slug(block.text()), block)
            block = block.next()
        return out

    def fit_images(self):
        """Scale the screenshots to the window's width (never above their own size)."""
        width = max(self.browser.viewport().width() - 40, 200)
        doc = self.browser.document()
        block = doc.begin()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                frag = it.fragment()
                fmt = frag.charFormat()
                if frag.isValid() and fmt.isImageFormat():
                    img = fmt.toImageFormat()
                    size = self._natural_size(img.name())
                    if size is not None and size.width() > 0:
                        w = min(size.width(), width)
                        img.setWidth(w)
                        img.setHeight(size.height() * w / size.width())
                        cursor = QTextCursor(doc)
                        cursor.setPosition(frag.position())
                        cursor.setPosition(frag.position() + frag.length(), QTextCursor.KeepAnchor)
                        cursor.setCharFormat(img)
                it += 1
            block = block.next()

    def _natural_size(self, name):
        if name not in self._sizes:
            path = GUIDE.parent / name
            image = QImage(str(path))
            self._sizes[name] = None if image.isNull() else image.size()
        return self._sizes[name]

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit.start()

    # ---- navigation -----------------------------------------------------------------------

    def follow(self, url):
        """Contents links scroll to their heading; web links open in the browser."""
        if url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)
        elif url.hasFragment() and (not url.path() or url.path().endswith("README.md")):
            self.go_to(url.fragment())
        elif url.isLocalFile() or url.isRelative():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(GUIDE.parent / url.path())))

    def go_to(self, anchor):
        block = self.headings().get(anchor)
        if block is None:
            return False
        top = self.browser.document().documentLayout().blockBoundingRect(block).top()
        self.browser.setTextCursor(QTextCursor(block))
        self.browser.verticalScrollBar().setValue(int(top))
        return True

    def find_next(self):
        text = self.find_box.text()
        if not text:
            return False
        if self.browser.find(text):
            return True
        self.browser.moveCursor(QTextCursor.Start)  # wrap around
        return self.browser.find(text)


def open_guide(parent=None):
    """Show the guide window (one per session), above `parent` (napari's window)."""
    global _window
    if parent is None:
        try:
            import napari

            viewer = napari.current_viewer()
            parent = viewer.window._qt_window if viewer is not None else None
        except Exception:  # noqa: BLE001  (no viewer: a window of its own)
            parent = None
    if _window is None:
        _window = GuideWindow(parent)
    _window.show()
    _window.raise_()
    _window.activateWindow()
    return _window


def install_help_menu(viewer=None):
    """Add 'IBIF napari user guide' to napari's Help menu (once per session).

    Plugins cannot contribute to the Help menu through their manifest, so the action is
    registered with napari's app model; if that private API changes, the action is added to
    this viewer's Help menu directly instead.
    """
    global _registered
    if _registered:
        return True
    try:
        from app_model.types import Action
        from napari._app_model import get_app_model
        from napari._app_model.constants import MenuId

        app = get_app_model()
        if ACTION_ID not in app.commands:
            app.register_action(Action(
                id=ACTION_ID, title=TITLE, callback=open_guide,
                menus=[{"id": MenuId.MENUBAR_HELP, "group": "0_ibif_napari"}],
            ))
        _registered = True
        return True
    except Exception:  # noqa: BLE001
        menu = getattr(getattr(viewer, "window", None), "help_menu", None)
        if menu is None:
            return False
        if not any(a.text() == TITLE for a in menu.actions()):
            menu.addAction(TITLE, open_guide)
        return True


__all__ = ["GUIDE", "GuideWindow", "install_help_menu", "open_guide"]
