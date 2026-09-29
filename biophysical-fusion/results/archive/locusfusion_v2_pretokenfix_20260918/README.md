# locusfusion_v2, before the reference channels. Archived 2026-09-24.

The grid submitted 2026-09-18 and cancelled on 2026-09-24, run on the FIRST v2
tokenizer: no reference information on the token, an uncalled base indistinguishable
from a column matching H37Rv, and v1's positional wavelength band.

**Why it was stopped.** Complete at per-drug loci, it scored 0.8926 macro
(all_modalities) against `locusfusion_fixed`'s 0.9089 — behind on all 11 drugs and
all five modality sets. Against `newmodels_full`, the SAME architecture on the
PRE-fix v1 tokenizer, it was level (+0.0006, and within 0.002 in every cell). So
this tokenizer reproduced the old v1, and the 0.017 that `locusfusion_fixed` gained
came from the half of that fix v2 did not have: the reference symbol on the token,
occupancy read as alt != ref, and a positional band fitted to the data.

Kept because it is the control for the retokenized run: v2's own design changes
(one merged axis, 16 tokens per locus, reference-frame translation, [NULL]) are
what differ between this and `newmodels_full`, and they cost nothing measurable.

Complete cells: sd_* (5 modality sets x 11 drugs). Partial: sd19_* (7-11 drugs).
Not run: md_*. Excluded from results_frame -- it is under results/archive/.
