# Uncalled-base (N) check

17942 isolates; per-drug loci (DRUG_TO_LOCI). Source = phenotype table `category`.

## A. N burden by source collection

```
                            isolates  any N (share)  mean N per isolate  median
category                                                                       
set1_original_10202             9274          0.198               0.809     0.0
set2                            4765          0.191               3.183     0.0
set3_matthias_in_rollingdb       810          0.073               1.074     0.0
set4_matthias_new_vcf           3093          0.216               3.544     0.0
```

## B. Does N predict resistance, and is it the source?

Batch predicts AUC(N) ~ AUC(source) and within-source AUC(N) ~ 0.5.

```
        drug           loci     n  R share  isolates with any N  R share | any N  R share | no N  AUC(N count)  AUC(source only, CV)  AUC(N count) within source
   ISONIAZID      inhA+katG 17436    0.334                  829            0.287           0.336         0.495                 0.598                       0.491
  RIFAMPICIN      rpoB+rpoC 17582    0.277                  364            0.632           0.269         0.518                 0.580                       0.518
  ETHAMBUTOL embC+embA+embB 15111    0.195                  415            0.277           0.193         0.507                 0.600                       0.508
PYRAZINAMIDE           pncA 12904    0.165                  200            0.770           0.156         0.534                 0.579                       0.532
STREPTOMYCIN   rpsL+rrs+gid  7682    0.315                  298            0.372           0.312         0.505                 0.600                       0.504
   KANAMYCIN            rrs  3576    0.197                  100            0.480           0.188         0.525                 0.621                       0.535
    AMIKACIN        rrs+eis  3705    0.191                  131            0.405           0.183         0.524                 0.612                       0.548
 CAPREOMYCIN   rrs+rrl+tlyA  3854    0.175                  173            0.266           0.171         0.514                 0.602                       0.514
LEVOFLOXACIN      gyrB+gyrA   269    0.283                   12            0.833           0.257         0.561                 0.528                       0.562
MOXIFLOXACIN      gyrB+gyrA  2868    0.135                   74            0.514           0.125         0.542                 0.630                       0.539
 ETHIONAMIDE inhA+ethA+ethR  2153    0.320                  111            0.414           0.315         0.511                 0.497                       0.511
```

## C. Columns where N is most enriched in resistant isolates (>= 20 N isolates, top 5 per drug by Fisher p)

`h37rv` = (CDS nucleotide, codon), or (upstream/downstream, offset in nt).

```
        drug locus  column  N isolates  R share | N here  R share | called  odds ratio  fisher p       h37rv
    AMIKACIN   rrs    1658          40            0.9250            0.1828     55.1318    0.0000        None
   KANAMYCIN   rrs    1658          35            0.9429            0.1892     70.7037    0.0000        None
  RIFAMPICIN  rpoB    1506          52            0.8077            0.2754     11.0530    0.0000 (1349, 450)
MOXIFLOXACIN  gyrA     280          28            0.7857            0.1289     24.7851    0.0000   (281, 94)
  RIFAMPICIN  rpoC    1453          22            0.9545            0.2761     55.0644    0.0000 (1448, 483)
  ETHAMBUTOL  embB     926          42            0.6429            0.1936      7.4955    0.0000  (918, 306)
 CAPREOMYCIN   rrs    1658          41            0.6098            0.1705      7.6034    0.0000        None
  RIFAMPICIN  rpoB    1446          21            0.9048            0.2762     24.8978    0.0000 (1304, 435)
   ISONIAZID  katG    1073          55            0.7091            0.3329      4.8847    0.0000  (944, 315)
PYRAZINAMIDE  pncA     383          24            0.6250            0.1643      8.4783    0.0000   (195, 65)
  ETHAMBUTOL  embB     924          30            0.6000            0.1941      6.2286    0.0000  (916, 306)
STREPTOMYCIN   rrs    1658          37            0.7027            0.3126      5.1970    0.0000        None
  ETHAMBUTOL  embB    1225          21            0.5714            0.1944      5.5265    0.0001 (1217, 406)
STREPTOMYCIN  rpsL     239          23            0.6522            0.3135      4.1061    0.0010   (128, 43)
  RIFAMPICIN  rpoC    1631          36            0.4722            0.2765      2.3408    0.0140 (1626, 542)
  RIFAMPICIN  rpoB     415          31            0.4839            0.2766      2.4523    0.0147  (309, 103)
  ETHAMBUTOL  embB     514          22            0.4091            0.1946      2.8657    0.0255  (515, 172)
   ISONIAZID  katG    1536          54            0.4630            0.3337      1.7215    0.0591 (1388, 463)
  ETHAMBUTOL  embA     227          31            0.2903            0.1947      1.6921    0.1766   (228, 76)
STREPTOMYCIN   gid     447          29            0.4138            0.3141      1.5413    0.3155   (276, 92)
   KANAMYCIN   rrs    1105          22            0.2273            0.1964      1.2034    0.7869        None
STREPTOMYCIN   gid     891          49            0.3265            0.3144      1.0572    0.8777  (615, 205)
```

