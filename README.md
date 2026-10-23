# IBIF napari: ND2/CZI/LIF + 3D Cellpose + filament tracing

[napari](https://napari.org) with four things built in, plus an optional
[facility GPU server](docs/facility-server.md) for Cellpose:

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
- **Filament Tracer panel**: traces neurons, microtubules and other filaments in 2D or 3D,
  automatically and/or by clicking along them. It reports Imaris-style statistics (length,
  segments, branch and terminal points, diameter, volume, straightness, branch depth,
  Sholl) and exports them as CSV, with the tracing as SWC. See
  [Filament tracing](#filament-tracing).

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
ibif-napari                      # napari with the panels docked
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

1. Optionally set **Z planes** (e.g. 10 to 25). You can also change it later.
2. Click **Draw…** next to *Region*. This adds a "region" labels layer over the image in
   paint mode.
3. Paint roughly over the cells (any label colour) on any plane. For a Z stack, the
   shape is copied to every plane of the Z planes range and shown only there, so you
   see exactly the block that will be segmented.
   - Erasing or undoing on one plane applies to all of them.
   - Changing *Z planes* moves the shape to the new planes.
   - Painting on a plane outside the range widens the range to include it.

   Alternatively pick an existing Labels layer, or a Shapes layer with a rectangle, as
   the region.
4. **Run Cellpose.** Only the bounding box of the region, within those Z planes, is read
   and segmented. With *Drop cells mostly outside the painted area* ticked, objects
   that are mostly outside your painted area are removed, so an irregular painted
   region is fine. The result is a labels layer of just that block, positioned over the
   image.

Which models you get depends on the Cellpose version installed. Cellpose 4 has `cpsam_v2`
(the default) and `cpsam`; its `cpdino` models appear once the separate `dinov3` package
is installed. Cellpose 3 has `cyto3`, `nuclei`, `cyto2` and `cyto`. To use your own trained
model, type its path in the Model box or click **…**. On a facility server, custom models
are placed in the server's `--models-dir` folder and appear in the list by file name.

## Filament tracing

The **Filament Tracer** panel works on the current time point/position, like Cellpose, and
can be limited to a **Region** and **Z planes** in the same way.

Two other filament plugins are installed too and appear under **Plugins**:
**napari-tracing** (click-to-trace with SWC files) and, on Python 3.11 or newer,
**napari-filaments** (2D spline tracing of single filaments). skan's **Skeleton Widget**
is also there. The panel below is the one that knows about voxel sizes, regions and the
dimension sliders.

**Automatic tracing**

1. Choose the image (the channel with the neurons or microtubules).
2. Set **Diameter (µm)** to the range of filament thicknesses to find. Typical ranges
   are 0.3 to 1.5 for dendrites and 0.2 to 0.6 for microtubules.
3. Click **Trace automatically**. Filaments are found with a multi-scale "tubeness"
   filter, which uses the real voxel size so anisotropic Z is handled. The result is
   thresholded, skeletonized and cleaned up, then measured.

   Results:
   - a **filaments** labels layer, coloured by filament;
   - a **filament points** layer: root (yellow), branch points (magenta) and terminal
     points (cyan);
   - a table of filaments in the panel.
4. The status line shows the threshold used. Raise **Threshold** to keep only bright,
   clear filaments, or lower it to find faint ones, then trace again. The slow filtering
   step is reused, so this is quick.

   Other settings:
   - **Prune spurs** trims short side branches caused by noise.
   - **Min length** drops short fragments.

**Click to trace** works on its own or to fix up an automatic trace:

1. Click **Click to trace**. A "trace clicks" points layer is added in add mode.
2. In the 2D view, click along a filament on the plane where it is in focus. Each click
   traces the brightest path from the previous click, also across Z planes.
   - A click near an already traced filament joins it, so you can add missing branches.
   - **New filament** starts the next click somewhere new.
   - **Undo last** removes the last traced piece.
3. The statistics update after each click.

To remove wrong pieces, select the filaments layer and use napari's eraser, then click
**Measure**. Note that **Trace automatically** replaces the filaments layer, including
click traces. Do the automatic trace first, then fix it up.

**Statistics.** **Export…** saves four CSV files plus an SWC file:

- `<name>_filaments.csv`, one row per filament: total length, number of segments, branch
  points and terminal points, length-weighted mean diameter, volume and surface area,
  mean intensity, maximum branch depth, straightness, and the position of the root
  (beginning point).
- `<name>_segments.csv`, one row per segment (between branch/terminal points): type,
  length, straight-line distance, straightness, mean diameter, volume, area, mean
  intensity, branch depth, distance from the root along the filament, elevation angle
  from the XY plane, and start/end positions.
- `<name>_points.csv`: root, branch and terminal points.
- `<name>_sholl.csv`: Sholl intersections every **Sholl step** µm from the root.
- `<name>.swc`: the tracing for neuron-morphology tools such as NeuroMorpho, SNT or
  neuTube.

How the measurements are made:

- Lengths and positions are in µm in the image's coordinates.
- Diameters come from the distance to the edge of the segmented filament. They are
  approximate, especially near the resolution limit.
- The root is the thickest node of each filament, which is usually the soma for a
  neuron.
- Branch depth counts the branch points between the root and a segment.

**Speed.** Filtering a 30 × 512 × 512 stack takes about 30 s on a laptop CPU. Use
**Region** for larger stacks.

## Facility GPU server

A facility GPU workstation can run Cellpose for everyone. Install the same package on it
and start `ibif-napari-server`. Users then choose it under **Run on** in the Cellpose panel
(add it with **Servers…** and the access token the facility gives them). Only the
selected region is sent; Cancel stops the run on the server at once. Users do not need
Cellpose or a GPU on their own computers to use it.

Setup, users, HTTPS, firewall and running it at start-up:
[docs/facility-server.md](docs/facility-server.md).

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
