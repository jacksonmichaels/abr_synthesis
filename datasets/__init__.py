"""
Per-isolate input data.

  loader.py      single-drug loader (load_dataset) + locus-set policy
                 (drug_loci, LOCUS_SETS, loci_on_disk)
  multidrug.py   multi-drug loader (load_multidrug_dataset), label matrix
  dna / protein / biophysical / regulatory .py   the four modalities, one
                 FeatureBlock per locus/region each
  sequences.py   FASTA / phenotype reading, nucleotide one-hot
  biochem.py     genetic code, translation, amino-acid property tables
  cds.py         where the CDS sits inside an aligned locus; genome placement
  who_catalogue.py, regulatory.py   WHO 2023 Tables 21/22, per-drug promoters
  lineage.py     TB-Profiler lineage calls
"""
