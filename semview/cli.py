"""Command line entry point: ``semview [options] [FILE]``."""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="semview",
        description="Spectral-element-aware viewer for 2D Nek5000/Neko field files. "
        "Opens a browser GUI; run under mpirun to read large files in parallel.",
    )
    p.add_argument("file", nargs="?", help=".nek5000 metafile, a case0.f00000 field file, a glob, or a saved *.semview.json session")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.add_argument("--info", action="store_true", help="print dataset information and exit")
    p.add_argument("--png", metavar="OUT", help="render FIELD to a PNG with the scripting backend and exit")
    p.add_argument("--field", default=None, help="field to render with --png (default: first field)")
    p.add_argument("--step", type=int, default=-1, help="time step for --png / --info")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--mesh", action="store_true", help="overlay element edges with --png")
    args = p.parse_args(argv)

    if args.info or args.png:
        if not args.file:
            p.error("a FILE is required")
        import semview

        if args.png and args.file.endswith(".semview.json"):
            from semview.session import plotter_from_session

            pl, data = plotter_from_session(args.file)
            if pl is None:  # non-root MPI rank
                return 0
            pl.save(args.png)
            print(f"wrote {args.png} from session {args.file}")
            return 0
        if args.file.endswith(".semview.json"):
            from semview.session import load_session

            args.file = load_session(args.file)["dataset"]["path"]
        ds = semview.open(args.file)
        data = ds[args.step].gather(ds.comm)
        if data is None:  # non-root MPI rank
            return 0
        if args.info:
            print(ds)
            print(f"  files      : {len(ds)} step(s), first {ds.files[0]}")
            print(f"  elements   : {data.glb_nelv}   order N = {data.order} ({data.n}x{data.n} GLL nodes)")
            print(f"  bounds     : x in [{data.bounds[0]:g}, {data.bounds[1]:g}], y in [{data.bounds[2]:g}, {data.bounds[3]:g}]")
            print(f"  time(s)    : {ds.times}")
            print(f"  fields     : {data.field_names}")
            print(f"  derived    : {[k for k in data.available if k not in data.field_names]}")
            for nm in data.field_names:
                a = data[nm]
                print(f"    {nm:10s} min {a.min(): .6g}  max {a.max(): .6g}")
            return 0
        from semview.plotting import Plotter

        name = args.field or data.field_names[0]
        pl = Plotter()
        pl.add_field(data, name, cmap=args.cmap)
        if args.mesh:
            pl.add_mesh(data)
        pl.set_title(f"{ds.name}: {name}  (t = {data.time:g})")
        pl.save(args.png)
        print(f"wrote {args.png}")
        return 0

    from semview.gui.server import serve

    serve(args.file, port=args.port, host=args.host, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
