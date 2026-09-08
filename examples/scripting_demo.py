"""Scripting API tour: a few figures from the pySEMTools mixing-layer example.

Run with:  python examples/scripting_demo.py [path/to/mixlay0.f00001]
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import semscope  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/timofey/code/pySEMTools/examples/data/mixlay0.f00001"
out = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out, exist_ok=True)

data = semscope.load(path)
print(data)

# 1) full view: vorticity from the field calculator (spectrally exact derivatives) with the element mesh
data.define("vorticity", "dx(v) - dy(u)")
pl = semscope.Plotter(figsize=(9, 6.3), dpi=130)
pl.add_field(data, "vorticity", cmap="RdBu_r", clim=(-3, 3))
pl.add_mesh(data, color="k", linewidth=0.25, alpha=0.5)
pl.set_title(f"{data.name}: vorticity, t = {data.time:.2f}")
pl.set_labels()
pl.save(os.path.join(out, "demo_vorticity.png"))

# 2) zoom on a vortex: nodal (ParaView-like) vs spectral rendering
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), dpi=130)
for ax, method in zip(axs, ["nodal", "spectral"]):
    p = semscope.Plotter(ax=ax)
    p.add_field(data, "t", cmap="magma", method=method, colorbar=False)
    p.add_mesh(data, color="w", linewidth=0.6)
    p.add_nodes(data, color="cyan", size=4)
    p.add_contours(data, "t", levels=12, colors="w", linewidths=0.4)
    p.set_view((8.6, 9.9), (6.9, 8.0))
    p.render()
    ax.set_title(f"{method} rendering")
fig.savefig(os.path.join(out, "demo_nodal_vs_spectral.png"), bbox_inches="tight")

# 3) exact line probe through the shear layer with element boundaries
dist, vals = data.sample_line(["u", "v"], (0.0, 7.0), (20.0, 7.0), 2000)
loc = data.locate(*(dist / 20.0 * 20.0, 7.0 * (dist * 0 + 1)))
fig, ax = plt.subplots(figsize=(9, 3), dpi=130)
ax.plot(dist, vals["u"], label="u", lw=1.2)
ax.plot(dist, vals["v"], label="v", lw=1.2)
edges = dist[1:][loc.elem[1:] != loc.elem[:-1]]
for e in edges:
    ax.axvline(e, color="0.85", lw=0.5, zorder=0)
ax.set_xlabel("x")
ax.set_title("velocity along y = 7 (grey lines: element boundaries)")
ax.legend()
fig.savefig(os.path.join(out, "demo_line.png"), bbox_inches="tight")

print("figures written to", out)
