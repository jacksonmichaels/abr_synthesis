"""Pull the executed figures out of notebooks/claims.ipynb as standalone PNGs.

    python results/figures/claims/_export.py

**These are exports, not sources.** Regenerate by rebuilding and re-executing
the notebook first, then re-running this:

    python scripts/build_claims.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/claims.ipynb
    python results/figures/claims/_export.py

Files are named `<n>_<the section heading above them>.png`, so the ordering here
is the notebook's ordering and a figure can be traced back to the prose that
qualifies it.
"""
import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = next(p for p in HERE.parents if (p / "bigtb_ref.py").exists())
NB = PROJECT / "notebooks" / "claims.ipynb"

nb = json.loads(NB.read_text())
for f in HERE.glob("*.png"):
    f.unlink()

label, n, rows = "00-setup", 0, []
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if cell["cell_type"] == "markdown":
        m = re.search(r"^#{2,3}\s+(.+)$", src, re.M)
        if m:
            heading = m.group(1).strip()
            label = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")[:44]
        continue
    for out in cell.get("outputs", []):
        png = (out.get("data") or {}).get("image/png")
        if png:
            n += 1
            name = f"{n:02d}_{label}.png"
            (HERE / name).write_bytes(base64.b64decode(png))
            rows.append((name, heading))

print(f"{n} figures -> {HERE}")
for name, heading in rows:
    print(f"  {name:52s} {heading}")
