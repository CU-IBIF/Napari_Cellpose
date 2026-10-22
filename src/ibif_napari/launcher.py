"""`ibif-napari`: napari with the Dimensions and Cellpose panels docked and a model preloaded.

This is ordinary napari: other plugins install and run as usual (Plugins menu), and
`napari file.nd2` also works because the ND2 reader is a regular napari plugin.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from ._cellpose import PRELOAD_ENV

PLUGIN = "ibif-napari"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ibif-napari", description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="ND2 (or any napari-readable) files to open")
    parser.add_argument("--model", help="Cellpose model to preload (default: the newest built-in)")
    parser.add_argument("--no-preload", action="store_true", help="don't load Cellpose at startup")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="download the Cellpose model weights and exit (run once after installing)",
    )
    args = parser.parse_args(argv)

    if args.download_models:
        return download_models(args.model)

    # Show Cellpose's own messages (model loading, 3D progress) in the terminal.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))
    logging.getLogger("cellpose").addHandler(handler)
    logging.getLogger("cellpose").setLevel(logging.INFO)

    # Read by the Cellpose panel when it is created: "0" = don't preload, else a model name.
    os.environ[PRELOAD_ENV] = "0" if args.no_preload else (args.model or "")

    import napari

    viewer = napari.Viewer(title="IBIF napari")
    viewer.window.add_plugin_dock_widget(PLUGIN, "Dimensions")
    viewer.window.add_plugin_dock_widget(PLUGIN, "Cellpose 3D", tabify=True)
    for path in args.files:
        viewer.open(path, plugin=PLUGIN if path.lower().endswith(".nd2") else None)
    napari.run()
    return 0


def download_models(model=None):
    from . import _cellpose

    names = [model] if model else _cellpose.builtin_models()[:1]
    if not names:
        print("Cellpose is not installed.", file=sys.stderr)
        return 1
    for name in names:
        for message in _cellpose.load_model(name, gpu=False):
            print(message, flush=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
