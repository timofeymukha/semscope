"""Build an analytic curved-element field, look at it, and write it to a Nek5000 file.

Run with:  python examples/synthetic_demo.py
Then open the GUI with:  semscope annulus0.f00000
"""

import numpy as np

import semscope
from semscope import synthetic

data = synthetic.taylor_green(synthetic.annulus(nr=4, ntheta=32, n=8, r_in=0.5, r_out=1.5), k=2.5)
synthetic.add_analytic_field(data, "s0", lambda x, y: np.tanh(4 * (np.hypot(x, y) - 1.0)) * np.cos(6 * np.arctan2(y, x)))

print(data)
print("max |div u| (spectral accuracy: small, decays exponentially with order):", np.abs(data["dx(u) + dy(v)"]).max())
data.define("vort", "dx(v) - dy(u)")
print("vorticity error vs analytic:", np.abs(data["vort"] - 2 * 2.5 * np.sin(2.5 * data.x) * np.sin(2.5 * data.y)).max())

pl = semscope.Plotter(figsize=(7, 7))
pl.add_field(data, "s0", cmap="twilight")
pl.add_mesh(data, color="w", linewidth=0.5)
pl.add_streamlines(data, density=1.2, color="k", linewidth=0.6)
pl.save("annulus.png")
data.write("annulus0.f00000")
print("wrote annulus.png and annulus0.f00000")
