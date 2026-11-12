"""
Training engines for the stacked CNN.

  core.py        masked weighted BCE + EarlyStopper (shared primitives)
  multimodal.py  single-drug CV + held-out-test engine  -> run_modal_cv
  multidrug.py   multi-drug engine                      -> run_multidrug_cv
  checkpoint.py  weights + rebuild config
  curves.py      per-epoch history plotting             -> save_curves

Both engines follow BIG-TB's SD-CNN protocol; see each module's docstring for
where they deviate and why.
"""
