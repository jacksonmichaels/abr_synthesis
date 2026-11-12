#!/bin/bash
# Re-download the GenomeFusion public reference data. Login node only
# (compute nodes may lack internet). Small files; safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
PY=python   # abr_env active

mkdir -p rlc mycobrowser essentiality
RAW=https://raw.githubusercontent.com/farhat-lab/mtb-illumina-wgs-evaluation/main
curl -fsSL -o rlc/RLC_Regions.H37Rv.bed \
  "$RAW/References/Mtb_H37Rv_MaskingSchemes/RLC_Regions.H37Rv.bed"
curl -fsSL -o rlc/RLC_Regions.Plus.LowPmapK50E4.H37Rv.bed \
  "$RAW/Results/B_Extra_UsefulDataFiles/F_Defining_RLC_Regions/RLC_Regions.Plus.LowPmapK50E4.H37Rv.bed"

MB="https://mycobrowser.epfl.ch/releases"
F="Mycobacterium_tuberculosis_H37Rv"
curl -fsSL -o mycobrowser/H37Rv_r5.txt "$MB/5/get_file?dir=txt&file=$F.txt"
curl -fsSL -o mycobrowser/H37Rv_r5.gff "$MB/5/get_file?dir=gff&file=$F.gff"
curl -fsSL -o mycobrowser/H37Rv_r2.txt "$MB/2/get_file?dir=txt&file=$F.txt"

# Europe PMC, not PMC's own /bin/ paths (those return a bot-block page).
curl -fsSL -o essentiality/supp.zip \
  "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5241402/supplementaryFiles"
unzip -oq essentiality/supp.zip -d essentiality/dejesus_supp '*.xlsx'
cd essentiality
for n in 3 5 6; do
  case $n in
    3) out=dejesus_S3_orf_essentiality.tsv ;;
    5) out=dejesus_S5_feature_essentiality.tsv ;;
    6) out=dejesus_S6_unannotated_essential.tsv ;;
  esac
  "$PY" -I ../scripts/xlsx2tsv.py "dejesus_supp/mbo002173137st${n}.xlsx" "$out" 1
done
echo "done"
