"""
Training engines.

  core.py        masked weighted BCE + EarlyStopper (shared primitives)
  multimodal.py  single-drug CV + held-out-test engine  -> run_modal_cv
  multidrug.py   multi-drug (MD-CNN style) engine       -> run_multidrug_cv
  curves.py      per-epoch history plotting             -> save_curves

Both engines follow BIG-TB's SD-CNN protocol; see each module's docstring for
where they deviate and why.
"""
from .core import EarlyStopper, masked_weighted_bce  # noqa: F401
from .curves import save_curves  # noqa: F401
from .multidrug import run_multidrug_cv  # noqa: F401
from .multimodal import run_modal_cv  # noqa: F401
