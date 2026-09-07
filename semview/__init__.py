"""semview — spectral-element-aware visualization of 2D CG SEM data.

Two ways to use it:

* **Scripting** (pyvista-like)::

      import semview
      data = semview.load("case0.nek5000", step=-1)
      pl = semview.Plotter()
      pl.add_field(data, "vorticity", cmap="RdBu_r", clim=(-5, 5))
      pl.add_mesh(data)
      pl.save("vorticity.png")

* **GUI**: ``semview case0.nek5000`` (or ``semview.serve(...)``) opens a
  WebGL viewer in the browser that evaluates the spectral basis on the GPU.
"""

import os as _os
import warnings as _warnings

# pysemtools reads these switches at import time: keep its per-read INFO logging
# quiet, and hide a numpy-2.5 deprecation notice raised inside its reader.
_os.environ.setdefault("PYSEMTOOLS_HIDE_LOG", "true")
_warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array", category=DeprecationWarning)

from .dataset import SEMData2D, Dataset, open, load, DERIVED_QUANTITIES  # noqa: F401,E402
from .locate import ElementLocator  # noqa: F401,E402
from .boundary import Boundary  # noqa: F401,E402
from . import spectral  # noqa: F401,E402

__version__ = "0.1.0"


def __getattr__(name):
    # Lazy imports keep `import semview` light (no matplotlib / web server).
    if name == "Plotter":
        from .plotting import Plotter

        return Plotter
    if name == "serve":
        from .gui.server import serve

        return serve
    raise AttributeError(name)


__all__ = ["SEMData2D", "Dataset", "open", "load", "Plotter", "serve", "spectral", "ElementLocator", "Boundary", "DERIVED_QUANTITIES"]
