# Former worktree `.claude/worktrees/lf-token-offset-fix` (branch `worktree-lf-token-offset-fix`)

Removed from git's worktree list 2026-10-30. The branch is kept (never merged
into `locus-fusion`; it holds `datasets/tokens.py`, `models/variant_tokens.py`
and other files not on the main line).

- `tree/` -- the worktree as it was, minus its `.git` pointer. Kept because its
  352 git-ignored run outputs (locusfusion_v2 result JSONs etc.) exist nowhere else.
- `uncommitted.patch` -- its uncommitted edits (`git apply` on the branch).
- `untracked/` + `untracked.txt` -- its untracked files. The two slurm manifests
  were also copied into biophysical-fusion/slurm_logs/manifests/.

Worktrees `lasso-wholeseq` and `token-census` were removed outright: no
uncommitted work, and their ignored outputs were identical copies of files
elsewhere. Their branches `worktree-lasso-wholeseq` / `worktree-token-census` remain.
