"""
Backward-compatibility shim. The amino-acid biochemistry (property tables,
genetic code, translation, featurizers) moved into ``datasets/biochem.py`` when
data handling was consolidated under the ``datasets/`` package. Import from
``datasets.biochem`` in new code; this re-export keeps older imports working.
"""
from datasets.biochem import (  # noqa: F401
    AA_PROPERTY,
    AMINO_ACIDS,
    CODON_TABLE,
    N_PROPERTIES,
    PROPERTY_NAMES,
    biophysical_matrix,
    one_hot_aa,
    translate_codon,
    translate_seq,
    upsample_to_nt,
)
