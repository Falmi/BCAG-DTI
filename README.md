# BCAG-DTI

Reproducibility repository for the manuscript **“Controlled Evaluation of Architectural, Classifier, and Training Refinements in MolTrans-Based Drug–Target Interaction Prediction.”**

BCAG-DTI is a MolTrans-based drug–target interaction framework used to evaluate, under a controlled protocol, the separate and combined effects of bidirectional cross-attention, global average/max aggregation, classifier design, and training strategy.

## Repository contents

```text
BCAG-DTI/
├── ESPF/                         # MolTrans substructure vocabulary assets
├── dataset/                      # Fixed BindingDB, BIOSNAP, and DAVIS splits
├── docs/                         # Verified implementation and split documentation
├── external_baselines/
│   └── cpi_ggs/                  # CPI-GGS fixed-split reproduction
├── results/                      # Compact manuscript result summary
├── ablation.py                   # A1–A8 definitions and training settings
├── config.py                     # MolTrans model configuration
├── evaluate.py                   # Checkpoint evaluation
├── experiment.py                 # Data loading, metrics, checkpoint and threshold logic
├── improved_models.py            # BCAG-DTI architectural/classifier components
├── models.py                     # MolTrans baseline implementation
├── stream.py                     # FCS-derived/ESPF token processing
├── train.py                      # Training entry point
└── requirements.txt
```

The repository deliberately keeps the already-published `dataset/` and `ESPF/` directories as the canonical data/tokenisation locations instead of adding duplicate copies under new folder names.

## A1–A8 controlled configurations

| ID | Configuration | Cross-attention | Global pooling | Enhanced classifier | Enhanced training |
|---|---|---:|---:|---:|---:|
| A1 | MolTrans baseline | No | No | No | No |
| A2 | Training strategy | No | No | No | Yes |
| A3 | Cross-attention | Yes | No | No | No |
| A4 | Global pooling | No | Yes | No | No |
| A5 | Enhanced classifier | No | No | Yes | No |
| A6 | Cross-attention + pooling | Yes | Yes | No | No |
| A7 | Cross-attention + pooling + training | Yes | Yes | No | Yes |
| A8 | Complete BCAG-DTI | Yes | Yes | Yes | Yes |

The machine-readable mapping is also provided in `docs/ablation_mapping_confirmation.csv`.

## Training and model selection

A1, A3, A4, A5, and A6 use Adam, learning rate `1e-4`, global batch size 16, dropout 0.10, and a fixed 50-epoch training duration. They do **not** use early stopping, but validation metrics are evaluated after every epoch and the checkpoint with the highest validation AUROC is retained and reloaded for testing.

A2, A7, and A8 use the enhanced training strategy. BindingDB/BIOSNAP use AdamW with learning rate `1e-4`, weight decay `1e-5`, dropout 0.15, three warm-up epochs followed by cosine decay, AMP, gradient clipping at 1.0, and early-stopping patience 15. DAVIS uses learning rate `1.2e-4`, weight decay `5e-6`, dropout 0.10, five warm-up epochs, a maximum of 60 epochs, patience 25, and label smoothing 0.1.

For every A1–A8 run, **checkpoint selection is based on validation AUROC**. The F1 decision threshold is selected only from validation data at the selected checkpoint and is applied unchanged to the test set. The test set is not used for checkpoint selection or threshold optimisation.

See `docs/final_implementation_details.csv` for the verified implementation settings.

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-capable PyTorch environment is recommended for training. The core repository dependencies are listed in `requirements.txt`; the CPI-GGS external reproduction additionally requires RDKit and PyTorch Geometric.

## Single-run reproduction

All A1–A8 configurations are exposed through `train.py`. For example:

```bash
python train.py \
  --task biosnap_full \
  --variant a8_bcag_dti \
  --seed 1 \
  --batch-size 16
```

Replace `--variant` with one of:

```text
a1_original
a2_training_strategy
a3_cross_attention
a4_multiscale_pooling
a5_enhanced_classifier
a6_cross_pooling
a7_cross_pooling_training
a8_bcag_dti
```

Canonical tasks are:

```text
biosnap_full
bindingdb
davis
biosnap_unseen_drug
biosnap_unseen_protein
biosnap_missing_70
biosnap_missing_80
biosnap_missing_90
biosnap_missing_95
```

A1 and A8 are used for the BIOSNAP robustness tasks; the principal A1–A8 controlled comparison uses BindingDB, BIOSNAP full, and DAVIS.

## Fixed data partitions

No runtime 80/10/10 resplitting is performed for the controlled MolTrans/BCAG-DTI experiments. The fixed split sizes are:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| BindingDB | 12,668 | 6,644 | 13,289 |
| BIOSNAP | 19,238 | 2,748 | 5,497 |
| DAVIS | 2,086 | 3,006 | 6,011 |

Unseen-entity and missing-interaction split counts are provided in `docs/data_split_confirmation.csv`.

## External CPI-GGS reference

`external_baselines/cpi_ggs/run_fixed_splits.py` evaluates CPI-GGS using the same fixed BIOSNAP train/validation/test partitions. Five deterministic runs use seeds 1–5. The manuscript experiment uses a maximum of 20 epochs, batch size 32, early-stopping patience 5, checkpoint selection by validation AUROC, and a validation-derived F1 threshold reused unchanged on test.

CPI-GGS uses molecular graphs and protein 3-grams, whereas MolTrans/BCAG-DTI use FCS-derived substructure tokenisation. The CPI-GGS experiment is therefore interpreted as an external **same-split reference**, not a fully protocol-identical comparison.

Example:

```bash
python external_baselines/cpi_ggs/run_fixed_splits.py \
  --moltrans-root /path/to/BCAG-DTI \
  --dataset biosnap_full \
  --seeds 1 2 3 4 5 \
  --epochs 20 \
  --batch-size 32 \
  --patience 5 \
  --output-dir results/cpi_ggs
```

## Attention case provenance

`attention_cases.json` records the selected BIOSNAP seed-3 TP/TN/FP/FN cases and the structural/pharmacological context used in the manuscript. Attention weights are treated as model-internal allocation patterns rather than experimentally validated binding contacts or causal binding determinants.

## Results

`results/FINAL_RESULTS.md` contains a compact audit-friendly summary of the final A1/A8 comparison, controlled-ablation conclusions, robustness results, and CPI-GGS same-split reference.

## Reproducibility notes

- Large model checkpoints are intentionally excluded from version control.
- `docs/REPRODUCTION_QUICKSTART.md` provides a concise reproduction guide.
- `docs/validation_summary_explanation.txt` explains checkpoint and validation-threshold semantics.
- The exact fixed CSV files remain under `dataset/`.
- The MolTrans vocabulary assets remain under `ESPF/`.

## Citation

Citation information will be updated after publication. Until then, please cite the accompanying manuscript when using this code or the fixed experimental protocol.

## Acknowledgement

This work builds on the publicly released MolTrans implementation and retains its corresponding substructure vocabulary assets and baseline components where required for controlled comparison.
