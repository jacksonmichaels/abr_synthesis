# locus_sets sweep status (2026-11-01 09:54)

data (14 re-extracted loci): **ready**

| arm | folder | done | in queue | gave up | gated on |
|---|---|---|---|---|---|
| canary_md | locus_sets/canary_who_t12_ | 2/2 | 0 | 0 | - |
| canary_who_sd | locus_sets/canary_who_t12_sd_ | 2/2 | 0 | 0 | - |
| who_sd_lf | locus_sets/who_t12_sd_ | 55/55 | 0 | 0 | who_lf |
| who_sd_cnn | locus_sets/who_t12_sd_ | 55/55 | 0 | 0 | who_cnn |
| md_bigtb19 | locus_sets/bigtb19_ | 10/10 | 0 | 0 | md,md_cnn |
| md_who_lf | locus_sets/who_t12_ | 5/5 | 0 | 0 | md |
| md_who_cnn | locus_sets/who_t12_ | 5/5 | 0 | 0 | md_cnn |
| lab_sd_lf | locus_sets/lab64_sd_ | 55/55 | 0 | 0 | lab_lf |
| lab_sd_cnn | locus_sets/lab64_sd_ | 55/55 | 0 | 0 | lab_cnn |
| md_lab_lf | locus_sets/lab64_ | 5/5 | 0 | 0 | md |
| md_lab_cnn | locus_sets/lab64_ | 5/5 | 0 | 0 | md_cnn |
| lasso_who | lasso_onehot_who_t12 | 11/11 | 0 | 0 | - |
| lasso_lab | lasso_onehot_lab64 | 11/11 | 0 | 0 | - |

## Canaries
- who_lf: {'job': '66380596', 'rss': 42.0, 'hours': 2.25, 'req': 128}
- who_cnn: {'job': '66380597', 'rss': 36.6, 'hours': 3.24, 'req': 128}
- md: {'job': '66380594', 'rss': 57.9, 'hours': 7.19, 'req': 240}
- md_cnn: {'job': '66380595', 'rss': 48.7, 'hours': 10.1, 'req': 240}
- lab_cnn: {'job': '66382697', 'rss': 158.3, 'hours': 10.07, 'req': 240}
- lab_lf: {'job': '66382696', 'rss': 189.1, 'hours': 11.48, 'req': 240}

