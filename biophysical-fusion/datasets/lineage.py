"""Per-isolate M. tuberculosis lineage calls, keyed by New_ID.

Calls are made once by scripts/call_lineages.py (TB-Profiler barcode, Napier et al.
2020) and stored read-only in data_snapshot/lineage/ (see the README there), so
nothing downstream re-genotypes. $ABR_LINEAGE_CSV overrides the path.

    from datasets.lineage import load_lineage
    lin = load_lineage()                 # DataFrame indexed by New_ID
    groups = lin.loc[ids, "level2"]      # e.g. lineage4.3, for GroupKFold
"""
import os
from pathlib import Path

import pandas as pd

LINEAGE_CSV = os.environ.get("ABR_LINEAGE_CSV") or str(
    Path(__file__).resolve().parents[2] / "data_snapshot" / "lineage" / "lineage_calls.csv")


def load_lineage(path=None):
    """Lineage table indexed by New_ID: lineage, level1-3, status, support, ..."""
    df = pd.read_csv(path or LINEAGE_CSV, dtype={"New_ID": str})
    return df.set_index("New_ID")
