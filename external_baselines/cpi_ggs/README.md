# CPI-GGS exact-split reproduction

**Method:** CPI-GGS: A deep learning model for predicting compound-protein interaction based on graphs and sequences. Hou et al., *Computational Biology and Chemistry* 115 (2025), 108326. DOI: 10.1016/j.compbiolchem.2024.108326.

Official source: `xingjie321/CPI-GGS`; inspected source commit `1c4df174c03c4d9c359db3b4cc2c37cc363b994b`.

## Why this method was selected

CPI-GGS is a recent interaction-aware CPI method with public code. Its architecture combines molecular graph features, protein sequence features and multi-head interaction attention.

## Exact-split adaptation

The official script performs an internal random 80/10/10 split. `run_fixed_splits.py` preserves the published model components and core optimiser schedule but replaces that split with the fixed BIOSNAP train/validation/test CSV files used by the MolTrans/BCAG-DTI experiments. It additionally provides deterministic seeds 1–5, batching, validation-AUROC checkpoint selection, validation-derived F1 thresholding, early stopping and machine-readable outputs.

## Preprocessing differences

CPI-GGS converts molecules to RDKit graphs with radius-2 Weisfeiler–Lehman fingerprints and proteins to overlapping 3-grams. MolTrans/BCAG-DTI use FCS-derived substructure tokenisation implemented with the MolTrans ESPF vocabulary assets. This is therefore an adapted exact-split reproduction, not a byte-identical rerun of the authors' original script.

## Reproduced BIOSNAP result

Across five deterministic seeds: AUROC `0.861854 ± 0.002316`, AUPRC `0.864542 ± 0.004239`, F1 `0.793816 ± 0.002842`. The reproduced model contains 725,260 trainable parameters.

These values are used as an external same-split reference only; no significance or fully protocol-identical comparison is claimed.
