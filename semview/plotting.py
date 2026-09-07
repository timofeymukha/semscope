"""Matplotlib front-end: a small pyvista-like ``Plotter`` for 2D SEM data.

The plotter records *layers* and renders them when the figure is shown or
saved, so the view can be changed after layers were added.  Pseudo-color
layers use the spectral basis to oversample each element adaptively: the
number of sub-cells per element is chosen from the element's size **in
pixels**, so zooming in never shows the piecewise-linear artefacts of
GLL-node-only rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import spectral as sp
from .dataset import SEMData2D

__all__ = ["Plotter", "quick_plot"]


def _mpl():
    import matplotlib

    return matplotlib


@dataclass
class _Layer:
    kind: str
    data: SEMData2D
    opts: dict[str, Any] = field(default_factory=dict)


class Plotter:
    """Compose a figure of SEM data layer by layer.

    Parameters
    ----------
    figsize, dpi : passed to ``matplotlib.figure.Figure``
    ax : an existing axes to draw into (optional)
    px_per_cell : target size in pixels of one oversampled sub-cell when
        ``resolution="auto"``; smaller values mean finer oversampling.
    """

    def __init__(self, figsize=(8, 5), dpi=150, ax=None, px_per_cell: float = 3.0, max_resolution: int = 64, background="white"):
        import matplotlib.pyplot as plt

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            self.ax = ax
            self.fig = ax.figure
        self.fig.patch.set_facecolor(background)
        self.ax.set_aspect("equal")
        self.px_per_cell = px_per_cell
        self.max_resolution = max_resolution
        self._layers: list[_Layer] = []
        self._view: tuple | None = None
        self._colorbars: list = []
        self._rendered = False
        self._artists: list = []

    # ------------------------------------------------------------ layers
    def add_field(
        self,
        data: SEMData2D,
        name: str,
        cmap="viridis",
        clim=None,
        resolution="auto",
        method: str = "spectral",
        alpha: float = 1.0,
        log: bool = False,
        colorbar: bool = True,
        label: str | None = None,
        zorder: float = 1.0,
    ) -> "Plotter":
        """Pseudo-color plot of a stored or derived field.

        method : ``"spectral"`` (adaptive oversampling of the polynomial basis,
            Gouraud shaded), ``"nodal"`` (GLL nodes only with linear shading —
            what a generic VTK/ParaView pipeline shows) or ``"pixel"``
            (evaluate the expansion exactly at every screen pixel).
        resolution : ``"auto"`` or the number of sub-cells per element side.
        """
        self._layers.append(_Layer("field", data, dict(name=name, cmap=cmap, clim=clim, resolution=resolution, method=method, alpha=alpha, log=log, colorbar=colorbar, label=label, zorder=zorder)))
        return self

    def add_contours(self, data: SEMData2D, name: str, levels=10, colors="k", linewidths=0.6, cmap=None, resolution="auto", labels=False, alpha=1.0, zorder=3.0, filled=False, clim=None, **kw) -> "Plotter":
        """Iso-lines of a field, computed on the spectrally oversampled elements."""
        self._layers.append(_Layer("contours", data, dict(name=name, levels=levels, colors=colors, linewidths=linewidths, cmap=cmap, resolution=resolution, labels=labels, alpha=alpha, zorder=zorder, filled=filled, clim=clim, kw=kw)))
        return self

    def add_mesh(self, data: SEMData2D, color="black", linewidth=0.4, alpha=0.8, resolution: int = 8, zorder=4.0) -> "Plotter":
        """Element edges, drawn as the actual (possibly curved) polynomial edges."""
        self._layers.append(_Layer("mesh", data, dict(color=color, linewidth=linewidth, alpha=alpha, resolution=resolution, zorder=zorder)))
        return self

    def add_nodes(self, data: SEMData2D, color="black", size=2.0, alpha=0.9, zorder=5.0, marker=".") -> "Plotter":
        """The GLL nodes themselves."""
        self._layers.append(_Layer("nodes", data, dict(color=color, size=size, alpha=alpha, zorder=zorder, marker=marker)))
        return self

    def add_vectors(self, data: SEMData2D, u: str = "u", v: str = "v", spacing: int = 30, scale=None, color="black", width=0.003, zorder=6.0, **kw) -> "Plotter":
        """Velocity arrows sampled (spectrally) on a uniform grid of ``spacing`` points across the view."""
        self._layers.append(_Layer("vectors", data, dict(u=u, v=v, spacing=spacing, scale=scale, color=color, width=width, zorder=zorder, kw=kw)))
        return self

    def add_streamlines(self, data: SEMData2D, u: str = "u", v: str = "v", density=1.0, color="white", linewidth=0.7, resolution: int = 300, zorder=6.0, **kw) -> "Plotter":
        """Streamlines of ``(u, v)`` from a spectrally exact resampling onto a Cartesian grid."""
        self._layers.append(_Layer("streamlines", data, dict(u=u, v=v, density=density, color=color, linewidth=linewidth, resolution=resolution, zorder=zorder, kw=kw)))
        return self

    def add_element_ids(self, data: SEMData2D, fontsize=6, color="black", zorder=7.0, use_global: bool = True) -> "Plotter":
        """Label each element with its (global) id."""
        self._layers.append(_Layer("ids", data, dict(fontsize=fontsize, color=color, zorder=zorder, use_global=use_global)))
        return self

    def add_points(self, x, y, color="red", size=20, marker="o", zorder=8.0, **kw) -> "Plotter":
        """Mark arbitrary points (e.g. probe locations)."""
        self._layers.append(_Layer("points", None, dict(x=np.asarray(x), y=np.asarray(y), color=color, size=size, marker=marker, zorder=zorder, kw=kw)))
        return self

    # ------------------------------------------------------------ view
    def set_view(self, xlim=None, ylim=None) -> "Plotter":
        self._view = (tuple(xlim) if xlim is not None else None, tuple(ylim) if ylim is not None else None)
        return self

    def zoom(self, center, width, height=None) -> "Plotter":
        """Center the view on ``center`` with the given data-space width (and height)."""
        cx, cy = center
        if height is None:
            bb = self.ax.get_window_extent()
            height = width * bb.height / max(bb.width, 1)
        return self.set_view((cx - width / 2, cx + width / 2), (cy - height / 2, cy + height / 2))

    def set_title(self, title: str) -> "Plotter":
        self.ax.set_title(title)
        return self

    def set_labels(self, xlabel="x", ylabel="y") -> "Plotter":
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        return self

    # ------------------------------------------------------------ rendering helpers
    def _limits(self):
        if self._view is not None and self._view[0] is not None and self._view[1] is not None:
            return self._view
        xs, ys = [], []
        for L in self._layers:
            if L.data is not None:
                b = L.data.bounds
                xs += [b[0], b[1]]
                ys += [b[2], b[3]]
        if not xs:
            return (0, 1), (0, 1)
        xl = (min(xs), max(xs)) if (self._view is None or self._view[0] is None) else self._view[0]
        yl = (min(ys), max(ys)) if (self._view is None or self._view[1] is None) else self._view[1]
        return xl, yl

    def _pixels_per_unit(self, xlim, ylim):
        bb = self.ax.get_window_extent()
        return bb.width / max(xlim[1] - xlim[0], 1e-300), bb.height / max(ylim[1] - ylim[0], 1e-300)

    def _visible(self, data: SEMData2D, xlim, ylim):
        b = data.element_bounds()
        return (b[:, 1] >= xlim[0]) & (b[:, 0] <= xlim[1]) & (b[:, 3] >= ylim[0]) & (b[:, 2] <= ylim[1])

    def _resolution_per_element(self, data: SEMData2D, xlim, ylim, resolution, px_per_cell=None):
        """Sub-cells per element side from the element's on-screen size."""
        if resolution != "auto":
            return np.full(data.nelv, int(resolution))
        sx, sy = self._pixels_per_unit(xlim, ylim)
        b = data.element_bounds()
        px = np.maximum((b[:, 1] - b[:, 0]) * sx, (b[:, 3] - b[:, 2]) * sy)
        m = np.ceil(px / (px_per_cell or self.px_per_cell)).astype(int)
        return np.clip(m, 2, self.max_resolution)

    def _triangulate(self, data: SEMData2D, names, xlim, ylim, resolution, px_per_cell=None, oversample_min=None):
        """Oversample visible elements and return a Triangulation plus values.

        Elements are grouped by their oversampling level so that each group is a
        single tensor-product resampling.
        """
        from matplotlib.tri import Triangulation

        vis = np.nonzero(self._visible(data, xlim, ylim))[0]
        if vis.size == 0:
            return None, None
        m_all = self._resolution_per_element(data, xlim, ylim, resolution, px_per_cell)
        if oversample_min is not None:
            m_all = np.maximum(m_all, oversample_min)
        single = isinstance(names, str)
        names = [names] if single else list(names)
        X, Y, F, T = [], [], {nm: [] for nm in names}, []
        offset = 0
        for m in np.unique(m_all[vis]):
            els = vis[m_all[vis] == m]
            mm = int(m) + 1  # points per side
            xx = sp.resample_elements(data.x[els], mm)
            yy = sp.resample_elements(data.y[els], mm)
            for nm in names:
                F[nm].append(sp.resample_elements(data[nm][els], mm).reshape(-1))
            X.append(xx.reshape(-1))
            Y.append(yy.reshape(-1))
            # triangles of an mm x mm grid, replicated for each element
            j, i = np.meshgrid(np.arange(mm - 1), np.arange(mm - 1), indexing="ij")
            a = (j * mm + i).ravel()
            b = a + 1
            c = a + mm
            d = c + 1
            tri = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)], 0)
            tri = tri[None, :, :] + (offset + np.arange(els.size) * mm * mm)[:, None, None]
            T.append(tri.reshape(-1, 3))
            offset += els.size * mm * mm
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        T = np.concatenate(T)
        tri = Triangulation(X, Y, T)
        vals = {nm: np.concatenate(F[nm]) for nm in names}
        return tri, (vals[names[0]] if single else vals)

    def _norm(self, values, clim, log):
        import matplotlib.colors as mcolors

        finite = values[np.isfinite(values)]
        if clim is None:
            if finite.size == 0:
                clim = (0.0, 1.0)
            else:
                clim = (float(finite.min()), float(finite.max()))
                if clim[0] == clim[1]:
                    clim = (clim[0] - 0.5, clim[1] + 0.5)
        if log:
            return mcolors.LogNorm(vmin=max(clim[0], 1e-300), vmax=clim[1])
        return mcolors.Normalize(vmin=clim[0], vmax=clim[1])

    # ------------------------------------------------------------ rendering
    def render(self) -> "Plotter":
        """Build all matplotlib artists for the current view (called by ``show``/``save``)."""
        for a in self._artists:
            try:
                a.remove()
            except Exception:  # noqa: BLE001 - artist may already be gone
                pass
        self._artists = []
        for cb in self._colorbars:
            try:
                cb.remove()
            except Exception:  # noqa: BLE001
                pass
        self._colorbars = []
        xlim, ylim = self._limits()
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        # make sure the axes has its final pixel size before choosing resolutions
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.get_renderer()
        except Exception:  # noqa: BLE001
            pass
        for L in sorted(self._layers, key=lambda L: L.opts.get("zorder", 0)):
            getattr(self, f"_render_{L.kind}")(L, xlim, ylim)
        self._rendered = True
        return self

    def _render_field(self, L: _Layer, xlim, ylim):
        o = L.opts
        data = L.data
        method = o["method"]
        norm = self._norm(np.asarray(data[o["name"]]), o["clim"], o["log"])
        if method == "pixel":
            sx, sy = self._pixels_per_unit(xlim, ylim)
            nx = int(max(2, round((xlim[1] - xlim[0]) * sx)))
            ny = int(max(2, round((ylim[1] - ylim[0]) * sy)))
            _, _, vals = data.to_grid(o["name"], xlim, ylim, nx=nx, ny=ny)
            art = self.ax.imshow(vals, extent=(*xlim, *ylim), origin="lower", cmap=o["cmap"], norm=norm, alpha=o["alpha"], interpolation="nearest", zorder=o["zorder"], aspect="auto")
        else:
            if method == "nodal":
                tri, vals = self._triangulate(data, o["name"], xlim, ylim, data.n - 1)
                # "nodal" means: use the GLL nodes as they are (linear shading between them)
            elif method == "spectral":
                tri, vals = self._triangulate(data, o["name"], xlim, ylim, o["resolution"])
            else:
                raise ValueError(f"unknown method {method!r}")
            if tri is None:
                return
            art = self.ax.tripcolor(tri, vals, shading="gouraud", cmap=o["cmap"], norm=norm, alpha=o["alpha"], zorder=o["zorder"], rasterized=True)
        self._artists.append(art)
        if o["colorbar"]:
            cb = self.fig.colorbar(art, ax=self.ax, pad=0.02, fraction=0.046)
            cb.set_label(o["label"] if o["label"] is not None else o["name"])
            self._colorbars.append(cb)

    def _render_contours(self, L: _Layer, xlim, ylim):
        o = L.opts
        tri, vals = self._triangulate(L.data, o["name"], xlim, ylim, o["resolution"], px_per_cell=self.px_per_cell / 2, oversample_min=4)
        if tri is None:
            return
        kw = dict(levels=o["levels"], alpha=o["alpha"], zorder=o["zorder"], **o["kw"])
        if o["cmap"] is not None:
            kw["cmap"] = o["cmap"]
            kw["norm"] = self._norm(vals, o["clim"], False)
        else:
            kw["colors"] = o["colors"]
        if o["filled"]:
            cs = self.ax.tricontourf(tri, vals, **kw)
        else:
            cs = self.ax.tricontour(tri, vals, linewidths=o["linewidths"], **kw)
            if o["labels"]:
                self.ax.clabel(cs, fontsize=6, inline=True)
        self._artists.append(cs)

    def _render_mesh(self, L: _Layer, xlim, ylim):
        from matplotlib.collections import LineCollection

        o = L.opts
        data = L.data
        vis = np.nonzero(self._visible(data, xlim, ylim))[0]
        if vis.size == 0:
            return
        m = int(o["resolution"]) + 1
        J = sp.interpolation_matrix(sp.gll_nodes(data.n), sp.uniform_nodes(m))  # (m, n)
        x = data.x[vis]
        y = data.y[vis]
        segs = []
        for xe, ye in ((x[:, 0, :], y[:, 0, :]), (x[:, -1, :], y[:, -1, :]), (x[:, :, 0], y[:, :, 0]), (x[:, :, -1], y[:, :, -1])):
            px = xe @ J.T  # (nvis, m)
            py = ye @ J.T
            segs.append(np.stack([px, py], axis=-1))
        segs = np.concatenate(segs, axis=0)
        lc = LineCollection(segs, colors=o["color"], linewidths=o["linewidth"], alpha=o["alpha"], zorder=o["zorder"], capstyle="round")
        self.ax.add_collection(lc)
        self._artists.append(lc)

    def _render_nodes(self, L: _Layer, xlim, ylim):
        o = L.opts
        data = L.data
        vis = self._visible(data, xlim, ylim)
        art = self.ax.scatter(data.x[vis].ravel(), data.y[vis].ravel(), s=o["size"], c=o["color"], alpha=o["alpha"], zorder=o["zorder"], marker=o["marker"], linewidths=0)
        self._artists.append(art)

    def _render_vectors(self, L: _Layer, xlim, ylim):
        o = L.opts
        n = int(o["spacing"])
        xg = np.linspace(xlim[0], xlim[1], n + 2)[1:-1]
        ny = max(2, int(round(n * (ylim[1] - ylim[0]) / max(xlim[1] - xlim[0], 1e-300))))
        yg = np.linspace(ylim[0], ylim[1], ny + 2)[1:-1]
        XG, YG = np.meshgrid(xg, yg)
        vals = L.data.sample([o["u"], o["v"]], XG, YG)
        U, V = vals[o["u"]], vals[o["v"]]
        ok = np.isfinite(U) & np.isfinite(V)
        art = self.ax.quiver(XG[ok], YG[ok], U[ok], V[ok], scale=o["scale"], color=o["color"], width=o["width"], zorder=o["zorder"], **o["kw"])
        self._artists.append(art)

    def _render_streamlines(self, L: _Layer, xlim, ylim):
        o = L.opts
        xg, yg, vals = L.data.to_grid([o["u"], o["v"]], xlim, ylim, nx=int(o["resolution"]))
        U = np.nan_to_num(vals[o["u"]])
        V = np.nan_to_num(vals[o["v"]])
        art = self.ax.streamplot(xg, yg, U, V, density=o["density"], color=o["color"], linewidth=o["linewidth"], zorder=o["zorder"], **o["kw"])
        self._artists.append(art.lines)
        self._artists.append(art.arrows)

    def _render_ids(self, L: _Layer, xlim, ylim):
        o = L.opts
        data = L.data
        vis = np.nonzero(self._visible(data, xlim, ylim))[0]
        cx = data.x[vis].mean(axis=(1, 2))
        cy = data.y[vis].mean(axis=(1, 2))
        ids = data.elmap[vis] if (o["use_global"] and data.elmap is not None) else vis + data.offset_el
        for e, x, y in zip(ids, cx, cy):
            self._artists.append(self.ax.text(x, y, str(int(e)), fontsize=o["fontsize"], color=o["color"], ha="center", va="center", zorder=o["zorder"]))

    def _render_points(self, L: _Layer, xlim, ylim):
        o = L.opts
        art = self.ax.scatter(o["x"], o["y"], s=o["size"], c=o["color"], marker=o["marker"], zorder=o["zorder"], **o["kw"])
        self._artists.append(art)

    # ------------------------------------------------------------ output
    def show(self, block=True):
        import matplotlib.pyplot as plt

        self.render()
        plt.show(block=block)
        return self

    def save(self, path: str, **kwargs) -> str:
        """Render and save the figure (any matplotlib format: png, pdf, svg, ...)."""
        self.render()
        kwargs.setdefault("bbox_inches", "tight")
        kwargs.setdefault("facecolor", self.fig.get_facecolor())
        self.fig.savefig(path, **kwargs)
        return path

    def close(self):
        import matplotlib.pyplot as plt

        plt.close(self.fig)


def quick_plot(data: SEMData2D, name: str, mesh: bool = False, **kw) -> Plotter:
    """One-liner: ``quick_plot(data, "u", cmap="magma").show()``."""
    pl = Plotter()
    pl.add_field(data, name, **kw)
    if mesh:
        pl.add_mesh(data)
    pl.set_title(f"{data.name} — {name}  (t = {data.time:g})")
    return pl
