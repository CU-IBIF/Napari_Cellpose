"""Verification and validation (V&V) of IBIF napari with synthetic data.

    ibif-napari-vv generate truth/                 # synthetic images with known truth
    ibif-napari-vv run truth/ results/             # run the checks (Cellpose by default)
    ibif-napari-vv report results/                 # results/report.html and report.md
    ibif-napari-vv all vv/                         # all three in one go

See docs/vv/README.md.
"""
