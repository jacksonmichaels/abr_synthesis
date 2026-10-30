"""Per-cell table of the v3 grid: done / running / pending / none, per drug.
    python results/experiments/locusfusion_v3/status_table.py
"""
import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from babysit import ARMS, STATE, done, drugs_of, key

q = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i %T"],
                   capture_output=True, text=True).stdout.split("\n")
st = dict(l.split() for l in q if l.strip())
cells = json.loads(STATE.read_text())["cells"] if STATE.exists() else {}
print("| arm | cell | done | running | pending | none |\n|---|---|---|---|---|---|")
tot = [0, 0, 0, 0]
for arm, prefix, exps, _ in ARMS:
    for exp in exps:
        n = [0, 0, 0, 0]
        for d in drugs_of(arm):
            c = (arm, exp, d)
            s = st.get(cells.get(key(c), {}).get("job"))
            i = 0 if done(c) else 1 if s == "RUNNING" else 2 if s else 3
            n[i] += 1
        tot = [a + b for a, b in zip(tot, n)]
        print(f"| {arm} | {prefix}{exp} | " + " | ".join(map(str, n)) + " |")
print("| **total** | | " + " | ".join(f"**{x}**" for x in tot) + " |")
