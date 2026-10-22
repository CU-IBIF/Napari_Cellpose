# IBIF napari: ND2/CZI/LIF + 3D Cellpose

[napari](https://napari.org) with three things built in:

- **Nikon ND2, Zeiss CZI and Leica LIF files** open directly (drag and drop,
  File > Open, or `ibif-napari file.nd2`). They load lazily, so large time lapses open quickly. Each
  channel becomes its own layer with its name from the file, a colour based on its
  emission wavelength, and the real voxel size in µm. For CZI files, scenes of the same
  size become the "position" slider (scenes of different sizes open as separate layers),
  and tiled (mosaic) acquisitions are stitched. A LIF project with several images opens
  a chooser listing each image and its size. Tick the ones to open, or double-click one.
  Images of the same size can optionally be stacked onto the position slider.
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
`napari file.nd2` / `.czi` / `.lif` use our readers. If another installed plugin can
also read CZI or LIF (e.g. napari-aicsimageio), napari asks which to use and can remember the
choice.

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

## Update

```bash
conda activate ibif-napari
pip install --upgrade "git+https://github.com/CU-IBIF/Napari_Cellpose"
```

This also installs any new dependencies, such as file-format readers. If a file type
reports a missing package (e.g. `No module named 'bioio_czi'`), install the package the
message names, e.g. `pip install bioio-czi bioio-lif`.

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

**Segmenting only part of the image.** Big volumes are slow, especially on CPU. To
segment just the cells you care about:

1. Click **Draw…** next to *Region*. This adds a "region" labels layer over the image in
   paint mode.
2. Paint roughly over the cells (any label colour). The painting applies to every Z
   plane. Alternatively pick an existing Labels layer, or a Shapes layer with a
   rectangle, as the region.
3. Optionally narrow **Z planes** (e.g. 10 to 25).
4. **Run Cellpose.** Only the bounding box of the region, within those Z planes, is read
   and segmented. With *Drop cells mostly outside the painted area* ticked, objects
   that are mostly outside your painted area are removed, so an irregular painted
   region is fine. The result is a labels layer of just that block, positioned over the
   image.

Which models you get depends on the Cellpose version installed. Cellpose 4 has `cpsam_v2`
(the default) and `cpsam`, plus `cpdino` models in 4.2+. Cellpose 3 has `cyto3`, `nuclei`, `cyto2` and `cyto`. To use your own trained
model, type its path in the Model box or click **…**.

## Troubleshooting Cellpose

- **First run downloads the model.** Cellpose 4's weights are about 1.2 GB, downloaded
  from huggingface.co into `~/.cellpose/models` (on Windows
  `C:\Users\<you>\.cellpose\models`). The panel shows the download progress. If the
  connection stalls for 60 s you get an error instead of a hang. If your network blocks
  huggingface.co, download the file (e.g. `cpsam_v2`) on another machine and copy it
  into that folder. You can also run `ibif-napari --download-models` once in a terminal.
- **CPU is slow for 3D.** Expect minutes per volume; use *Region* to segment a
  smaller block. CPU use only starts once the status line says "Segmenting…". On CPU
  the model runs in 32-bit floats, which is much faster than Cellpose's 16-bit default
  on most CPUs. An NVIDIA GPU (CUDA build of PyTorch) is roughly 10–50× faster.
- **Cancel is immediate.** Cellpose runs in a separate background process that keeps the
  model loaded. Cancel stops that process at once, which frees the CPU/GPU, and the
  model is reloaded in the background for the next run. If Cellpose crashes or runs
  out of memory, napari keeps running and the panel shows the error.
- **More detail:** start with `ibif-napari` from a terminal. Cellpose's own log
  (model loading, 3D progress) is printed there.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Bump `version` in `pyproject.toml` and `src/ibif_napari/__init__.py` with each change,
so `pip install --upgrade` sees a new version and installs new dependencies.

The tests don't need real ND2 files or Cellpose weights. They feed synthetic arrays
through the reader and replace the model with a stub.
