"""Re-render notebook figures at slide resolution.

Executes a notebook's code cells headlessly (Agg) and saves every figure a cell
leaves open to results/figures/talk/<notebook>/c<cell>_<n>.png. Text is enlarged
uniformly by --font-scale: each figure is shrunk by that factor in inches and
saved at a matching higher dpi, so pixel size is unchanged and every label,
default or explicit, grows by the same ratio. The notebooks' own savefig calls
are suppressed, so the figures they normally write are left untouched.
Inference/plotting only; the binding-site notebook reads its cached variant calls.

    python scripts/render_talk_figures.py lineage_split binding_site_visuals
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "talk"


def run(name, k, dpi):
    nb = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    own_savefig = Figure.savefig
    Figure.savefig = lambda self, *a, **kw: None          # notebook's own writes are skipped
    plt.show = lambda *a, **kw: None
    os.chdir(ROOT / "notebooks")
    ns = {"__name__": "__main__", "display": lambda *a, **kw: None}
    saved, failed = [], []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = "\n".join(l for l in cell.source.splitlines() if not l.lstrip().startswith(("%", "!")))
        try:
            exec(compile(src, f"{name}[{i}]", "exec"), ns)
        except Exception:
            failed.append(i)
            print(f"[{name}] cell {i} failed:\n{traceback.format_exc(limit=3)}", file=sys.stderr)
        for n, num in enumerate(plt.get_fignums()):
            p = out / f"c{i}_{n}.png"
            fig = plt.figure(num)
            w, h = fig.get_size_inches()
            fig.set_size_inches(w / k, h / k)
            own_savefig(fig, p, dpi=dpi * k, bbox_inches="tight")
            saved.append(p.name)
        plt.close("all")
    Figure.savefig = own_savefig
    print(f"[{name}] saved {len(saved)}: {', '.join(saved)}; failed cells: {failed or 'none'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+")
    ap.add_argument("--font-scale", type=float, default=1.2)
    ap.add_argument("--dpi", type=int, default=250)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    for nb_name in a.notebooks:
        run(nb_name, a.font_scale, a.dpi)
