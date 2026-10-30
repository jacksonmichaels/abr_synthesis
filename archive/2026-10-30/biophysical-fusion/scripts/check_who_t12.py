"""Gate for results/experiments/who_t12: every gene FASTA in the WHO tier 1+2
genotype folder is usable by locusfusion_v2, and each drug resolves its loci.

    python scripts/check_who_t12.py      # exit 1 on any failure
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import cds  # noqa: E402
from datasets import ALL_DRUGS  # noqa: E402
from datasets.loader import drug_loci  # noqa: E402

G = "/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/gene_msa/who_t12_genotypes"
fails = []
cds_nt = cds._cds_by_gene()
for f in sorted(Path(G).glob("*.fasta")):
    g = f.stem
    if not f.exists():
        fails.append(f"{g}: missing (dangling link)"); continue
    n = sum(1 for line in open(f) if line.startswith(">"))
    ref = cds._reference_row(f.as_posix())
    span = cds.genome_span(G, g)
    win = cds.cds_columns(G, g) or cds.cds_columns(G, g, feature="rRNA")
    ok = n == 17943 and ref is not None and span is not None and win is not None
    codons = ""
    if g in cds_nt and win:
        codons = f"{len(cds_nt[g]) // 3} codons"
    print(f"{'PASS' if ok else 'FAIL'}  {g:8} records {n}  ref {'yes' if ref else 'NO'}  "
          f"genome {span}  window {win}  {codons}")
    if not ok:
        fails.append(g)
for d in ALL_DRUGS:
    loci = drug_loci(d, locus_set="who_t12")
    miss = [g for g in loci if not (Path(G) / f"{g}.fasta").exists()]
    print(f"{d:13} {len(loci):2d} loci  missing {miss}")
    if miss:
        fails.append(f"{d} missing {miss}")
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
