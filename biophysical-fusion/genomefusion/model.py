"""Moved to canonical/genomefusion.py -- edit the model there. This module
re-exports it so genomefusion.train, its tests and older scripts keep working.

BUCKETS is read from canonical.genomefusion at call time; to change it at
runtime, set canonical.genomefusion.BUCKETS, not this module's copy."""
from canonical.genomefusion import (BUCKETS, GF_DEFAULTS, GenomeFusionNet,  # noqa: F401
                                    _sinusoid, proximity_table)
