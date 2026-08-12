# Final manuscript result summary

All MolTrans/BCAG-DTI values below are five-seed test means ± sample standard deviation using matched seeds 1–5.

## A1 MolTrans vs A8 BCAG-DTI

| Dataset | Model | AUROC | AUPRC | F1 |
|---|---|---:|---:|---:|
| BindingDB | A1 | 0.8815 ± 0.0028 | 0.5231 ± 0.0074 | 0.5571 ± 0.0089 |
| BindingDB | A8 | 0.9063 ± 0.0011 | 0.6062 ± 0.0141 | 0.6015 ± 0.0056 |
| BIOSNAP | A1 | 0.8631 ± 0.0053 | 0.8646 ± 0.0139 | 0.7988 ± 0.0061 |
| BIOSNAP | A8 | 0.8891 ± 0.0078 | 0.8992 ± 0.0067 | 0.8178 ± 0.0094 |
| DAVIS | A1 | 0.8808 ± 0.0128 | 0.2926 ± 0.0512 | 0.3913 ± 0.0302 |
| DAVIS | A8 | 0.8955 ± 0.0060 | 0.3545 ± 0.0303 | 0.4176 ± 0.0233 |

## Controlled ablation conclusion

A5 (enhanced classifier only) has the highest mean AUROC on BindingDB and BIOSNAP and the highest mean AUPRC and F1 on all three datasets. A7 (cross-attention + pooling + enhanced training) has the highest mean AUROC on DAVIS. A8 improves A1 across all reported main-dataset metrics, but A8 does not dominate the controlled ablation.

## BIOSNAP robustness

| Setting | A1 AUROC | A8 AUROC | A1 AUPRC | A8 AUPRC | A1 F1 | A8 F1 |
|---|---:|---:|---:|---:|---:|---:|
| Unseen drug | 0.8245 | 0.8349 | 0.8468 | 0.8648 | 0.7563 | 0.7582 |
| Unseen protein | 0.6482 | 0.6639 | 0.6566 | 0.6820 | 0.4911 | 0.5029 |
| 70% missing | 0.8349 | 0.8543 | 0.8431 | 0.8672 | 0.7668 | 0.7824 |
| 80% missing | 0.8212 | 0.8322 | 0.8313 | 0.8506 | 0.7558 | 0.7576 |
| 90% missing | 0.7809 | 0.7905 | 0.7900 | 0.8119 | 0.7221 | 0.7235 |
| 95% missing | 0.7510 | 0.7491 | 0.7658 | 0.7755 | 0.6986 | 0.6933 |

At 95% missing interactions, A8 retains higher AUPRC but has slightly lower AUROC and F1 than A1; no universal robustness claim is made.

## CPI-GGS same-split external reference

On the fixed BIOSNAP partitions, reproduced CPI-GGS obtains AUROC `0.861854 ± 0.002316`, AUPRC `0.864542 ± 0.004239`, and F1 `0.793816 ± 0.002842` across five deterministic seeds. CPI-GGS contains 725,260 trainable parameters compared with 106,261,023 for A8. Because preprocessing and model capacity differ, this is interpreted as a same-split external reference rather than a fully protocol-identical comparison.
