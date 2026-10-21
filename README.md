# IBIF napari: ND2 + 3D Cellpose

[napari](https://napari.org) with three things built in:

- **Nikon ND2 files** open directly (drag and drop, File > Open, or `ibif-napari file.nd2`).
  They load lazily, so large time lapses open quickly. Each channel becomes its own layer
  with its Nikon name, a colour based on its emission wavelength, and the real voxel size
  in µm.
- **Dimensions panel**: labelled sliders for time, stage position and Z. Time is shown in
  s/min and Z in µm. It also has channel on/off switches, a 2D/3D view toggle, a max
  projection over Z, and "Extract current time/position", which copies the current
  volume out as new layers.
- **Cellpose 3D panel**: segments the selected channel, optionally with a nuclear channel.
  You can run true 3D (anisotropy is filled in from the voxel size) or 2D per plane with
  stitching. It runs on the current time point/position or on all of them, and writes a
  labels layer. The Cellpose model is loaded in the background at startup, so the first
  run doesn't wait for it.

Everything else is ordinary napari. Plugins install and run as usual from
**Plugins > Install/Uninstall Plugins**, and this package is itself a normal napari
plugin. You can open the panels from **Plugins > IBIF napari** in any napari, and
`napari file.nd2` uses the ND2 reader.

## Install

Use a fresh environment (Python 3.10–3.12 recommended):

```bash
conda create -n ibif-napari python=3.11 -y
conda activate ibif-napari
# NVIDIA GPU (optional, much faster for 3D): install the CUDA build of PyTorch first,
# see https://pytorch.org/get-started/locally/ ; Apple Silicon uses its GPU automatically.
pip install "git+https://github.com/CU-IBIF/Napari_Cellpose"
ibif-napari --download-models     # fetch the Cellpose weights once
```

Alternatively, `conda env create -f environment.yml` does the first and third steps.

## Use

```bash
ibif-napari                      # napari with both panels docked
ibif-napari sample.nd2           # ...and open a file
ibif-napari --model nuclei       # preload a different model (Cellpose 3)
ibif-napari --no-preload         # start without loading Cellpose
```

Typical 3D workflow:

1. Open the ND2 file. Use the Dimensions panel to go to the time point/position you want
   and switch on the channels you need.
2. In **Cellpose 3D**, pick the image (e.g. the membrane/cytoplasm channel) and, if you
   have one, the nuclear channel. Keep the mode on *3D*.
3. Leave diameter on *auto* or enter the typical cell diameter in pixels, then
   **Run Cellpose**. Use *Every time point / position* to segment the whole series.

Which models you get depends on the Cellpose version installed. Cellpose 4 has `cpsam`
only. Cellpose 3 has `cyto3`, `nuclei`, `cyto2` and `cyto`. To use your own trained
model, type its path in the Model box or click **…**.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests don't need real ND2 files or Cellpose weights. They feed synthetic arrays
through the reader and replace the model with a stub.
