"""Moved to canonical/locusfusion.py -- edit the model there. This module
re-exports it so locusfusion_v4.train, its tests and older scripts keep working."""
from canonical.locusfusion import (LF_DEFAULTS, LocusFusionNet, LocusFusionV4Net,  # noqa: F401
                                   V4_DEFAULTS, proximity_table)
