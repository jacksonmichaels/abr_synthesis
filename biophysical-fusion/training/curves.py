"""Training-curve capture + plotting — "was the epoch cap enough?".

The training engines (training.multimodal._train / training.multidrug._train) record
a per-epoch ``history`` dict on every CV fold:

    {"train_loss": [...], "val_auc" | "val_loss": [...]}   # one entry per epoch

which lands in the run's result JSON under ``cv_folds[i]["history"]``.
``save_curves`` renders those into one PNG next to the JSON: train loss on the
left, the early-stopping metric on the right, one line per fold, with each
fold's best epoch marked. If the curves are still moving at the right edge — or
the best epochs cluster near the cap — the epoch budget is too small.
"""


def save_curves(folds, path, title=""):
    """Write a two-panel curve plot for `folds` (the result dict's cv_folds).

    No-op when no fold carries a history (e.g. early stopping disabled) or when
    matplotlib isn't installed — plotting must never sink a training run.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[curves] matplotlib not available; skipping curve plot", flush=True)
        return None

    hists = [(f.get("fold", i), f.get("history") or {}, f.get("best_epoch"))
             for i, f in enumerate(folds)]
    hists = [h for h in hists if h[1].get("train_loss")]
    if not hists:
        return None
    # the val series key is whatever the run monitored ('val_auc' / 'val_loss')
    val_key = next((k for _, h, _ in hists for k in h
                    if k.startswith("val_") and h[k]), None)

    n_panels = 2 if val_key else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4), squeeze=False)
    axes = axes[0]
    for fold, h, best_epoch in hists:
        ep = range(1, len(h["train_loss"]) + 1)
        line, = axes[0].plot(ep, h["train_loss"], lw=1.2, label=f"fold {fold}")
        if val_key and h.get(val_key):
            axes[1].plot(range(1, len(h[val_key]) + 1), h[val_key],
                         lw=1.2, color=line.get_color(), label=f"fold {fold}")
            if best_epoch:
                axes[1].axvline(best_epoch, color=line.get_color(),
                                ls=":", lw=0.9, alpha=0.7)
    axes[0].set(xlabel="epoch", ylabel="train loss", title="training loss")
    if val_key:
        axes[1].set(xlabel="epoch", ylabel=val_key,
                    title=f"{val_key} (dotted = best epoch)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    best = [b for _, _, b in hists if b]
    subtitle = f"  |  best epochs: {best}" if best else ""
    fig.suptitle(f"{title}{subtitle}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":  # re-plot a finished run: python training/curves.py <run_dir|json...>
    import json
    import sys
    from pathlib import Path

    targets = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        targets += sorted(p.glob("*.json")) if p.is_dir() else [p]
    for jf in targets:
        r = json.loads(jf.read_text())
        folds = list(r.get("cv_folds") or [])
        if r.get("test_history"):
            folds.append({"fold": "full-train", "history": r["test_history"]})
        out = save_curves(folds, jf.with_name(f"{jf.stem}_curves.png"),
                          title=f"{jf.stem}  (epoch cap {r.get('epochs')})")
        print(f"{jf.name}: {'wrote ' + out.name if out else 'no history recorded'}")
