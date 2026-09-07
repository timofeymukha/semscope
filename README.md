# semview

**Spectral-element-aware visualization of 2D CG SEM data** (Nek5000 / Neko field
files), built on [pySEMTools](https://github.com/ExtremeFLOW/pySEMTools) for
MPI-parallel IO.

Generic viewers (ParaView, VisIt) treat a spectral-element solution as a cloud
of GLL points connected by linear cells.  semview instead understands the data
the way the solver does:

* **elements vs. GLL nodes** — the mesh is a set of curved quadrilateral
  elements, each carrying an `n × n` tensor-product Lagrange basis on
  Gauss–Lobatto–Legendre nodes;
* **the field is a polynomial on every element**, so it can be evaluated at
  *any* point, not only at the GLL nodes.  When you zoom in, semview
  oversamples the basis instead of showing linear facets;
* **spectrally exact derivatives** (vorticity, divergence, gradients) and
  **exact point/line probes** by inverting the polynomial geometry map;
* **resolution diagnostics** from the decay of the Legendre spectrum.

Two ways to use it:

| | |
|---|---|
| **Scripting API** (pyvista-like) | `semview.Plotter` builds matplotlib figures layer by layer |
| **GUI** | `semview case0.nek5000` opens a WebGL viewer in the browser; the Lagrange basis is evaluated on the GPU at every pixel |

## Installation

```bash
pip install -e .            # needs numpy, mpi4py and extremeflow-pysemtools
pip install -e ".[plot]"    # + matplotlib for the scripting backend
```

## Quick start — scripting

```python
import semview

ds = semview.open("mixlay0.nek5000")          # .nek5000 metafile, a case0.f00012 file, or a glob
print(ds)                                     # Dataset('mixlay', steps=1, nelv=1600, order=7)
data = ds[-1]                                 # SEMData2D for the last step (mesh shared between steps)

pl = semview.Plotter(figsize=(9, 6))
pl.add_field(data, "vorticity", cmap="RdBu_r", clim=(-3, 3))   # derived on the fly, spectrally exact
pl.add_contours(data, "p", levels=15, colors="w", linewidths=0.5)
pl.add_mesh(data, color="k", linewidth=0.3)                     # true curved element edges
pl.set_view((8, 12), (5, 9))                                    # zoom: oversampling adapts to the pixel size
pl.save("vorticity.png")
```

`add_field(..., method=...)` selects how the pseudo-color surface is built:

* `"spectral"` (default) — every element is oversampled onto an `m × m` grid with
  the Lagrange basis; `m` follows the element's size **in pixels**
  (`resolution="auto"`), so zooming never reveals linear facets;
* `"pixel"` — evaluate the expansion exactly at every screen pixel (point
  location + Newton inversion of the geometry map);
* `"nodal"` — GLL nodes with linear shading, i.e. what a generic VTK pipeline
  shows; handy for side-by-side comparisons.

Other layers: `add_nodes` (GLL nodes), `add_vectors`, `add_streamlines`,
`add_element_ids`, `add_points`.

### Working with the data directly

```python
data.fields            # {'u': (nelv, n, n), 'v': ..., 'p': ...}  indexed [element, j, i]
data.x, data.y         # GLL node coordinates, same shape
data["vorticity"]      # derived quantities: speed, vorticity, divergence, jacobian, "du/dx", ...
data.gradient("p")     # (dp/dx, dp/dy) at the GLL nodes
X, Y, U = data.resample("u", 16)             # oversample every element on a 16 x 16 grid
u = data.sample("u", x, y)                   # exact values at arbitrary points (NaN outside)
dist, vals = data.sample_line(["u", "v"], (0, 7), (20, 7), 500)
xg, yg, grid = data.to_grid("u", nx=800)     # Cartesian resampling (e.g. for FFTs / exports)
loc = data.locate(x, y)                      # element index and (r, s) of points
data.spectral_decay("u")                     # per-element under-resolution indicator
data.write("out0.f00000")                    # back to a Nek5000 file through pysemtools
```

### MPI

Everything reads through pysemtools' parallel readers, so large files can be
opened with

```bash
mpirun -n 8 python my_script.py
```

Each rank holds its share of the elements (`data.is_distributed`); call
`data.gather()` to assemble the full field on rank 0 before plotting
(`semview.load(path)` does this for you and returns `None` on other ranks).

## Quick start — GUI

```bash
semview case0.nek5000            # opens http://127.0.0.1:8765 in your browser
semview                          # start empty and use the Open… dialog
mpirun -n 4 semview big0.nek5000 # parallel reads; rank 0 serves the browser
semview --info case0.nek5000     # print what is in the file
semview --png out.png --field vorticity --cmap RdBu_r --mesh case0.nek5000
```

What the GUI does differently from a generic viewer:

* the **field is evaluated on the GPU at every pixel** from the element's nodal
  values (fragment shader), and the **geometry** is tessellated adaptively
  from the same polynomial basis — infinite zoom, no linear artefacts, no
  resampling round-trips to the server;
* toggle **spectral ↔ nodal (linear)** rendering to see what the linear
  interpolation of a generic pipeline hides;
* **element edges**, **GLL nodes** and anti-aliased **iso-lines** are drawn
  analytically in the shader at any zoom level;
* **hover probe**: the element id, the reference coordinates `(r, s)` (found by
  Newton inversion of the geometry map) and the exact field value; click to
  pin, `Shift`-drag (or the *Line probe* button) for **line probes**: several
  lines can coexist (a legend in the popup shows, hides, selects or deletes
  each), end points can be dragged afterwards (the body drags the whole line),
  and the angle snaps to 10° steps with the *snap angle* option or while
  holding `Ctrl`. The chart shows only the parts inside the domain, marks the
  element boundaries crossed by the selected line, follows field and time-step
  changes, and can be zoomed (wheel: distance, `Shift`+wheel: value), panned,
  reset (double-click), moved by its header and resized from its corner; the
  tick grid and the element-boundary markers can each be toggled; while the
  chart is zoomed, a translucent halo on the plot marks the part of the line it
  shows;
* derived quantities (vorticity, divergence, speed) and the **spectral-decay
  resolution indicator** per element;
* a **right-click menu** on the plot: copy coordinates or the probe readout,
  pin the probe, start a line probe, zoom to the element under the cursor,
  export line samples as CSV, and **copy a Python snippet** that reproduces the
  current view with the scripting API (field, colormap, range, layers, view
  limits, probes);
* time series from the `.nek5000` metafile with playback and prefetching;
* colormaps come from matplotlib, so figures and GUI look the same.

**Undo/redo** (`Ctrl+Z`, `Ctrl+Shift+Z` or `Ctrl+Y`, also in the right-click
menu) covers line probes (create, move, delete, show/hide, clear), the pinned
probe and view changes; zoom/pan gestures collapse into single steps.

Keyboard: `Space` play/pause · `←/→` step · `m` edges · `c` iso-lines · `n` nodes ·
`1/2` spectral/nodal · `l` line probe · `Del` remove selected line · `r` reset view ·
`s` screenshot · `o` open · `Esc` clear probes.

The GUI is a thin client: the server streams each field once as `float32`
(a 100k-element, order-7 field is 25 MB) and everything else — zoom, LOD,
iso-lines, edges, probes — happens in the browser.  The Python side keeps a
memory-bounded cache of recently visited steps (`Dataset(cache_bytes=...)`).

## How it works

`semview/spectral.py` holds the numerics: GLL nodes and weights, barycentric
Lagrange interpolation, differentiation matrices and tensor-product
application to `(nelv, n, n)` arrays.  `semview/locate.py` finds the element
containing a point with a background grid over element bounding boxes followed
by a vectorised Newton iteration on the polynomial map `(r, s) → (x, y)`.
`semview/dataset.py` wraps the pysemtools `Mesh`/`FieldRegistry` objects into
`SEMData2D` (one step) and `Dataset` (a time series that reuses the mesh of the
first file).  The GUI (`semview/gui`) is a standard-library HTTP server that
streams `float32` nodal arrays to a WebGL2 front-end; the arrays are uploaded
into textures unchanged and the shaders do the rest.

## Development

```bash
pytest                    # unit tests (uses the pySEMTools example data when present)
python examples/scripting_demo.py
```
