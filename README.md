# BCAG-DTI

Reproducibility repository for the manuscript **“Controlled Evaluation of Architectural, Classifier, and Training Refinements in MolTrans-Based Drug–Target Interaction Prediction.”**

BCAG-DTI is a MolTrans-based drug–target interaction framework used to evaluate, under a controlled protocol, the separate and combined effects of bidirectional cross-attention, global average/max aggregation, classifier design, and training strategy.

## What is included

- Final A1–A8 implementation and experiment registry.
- Fixed benchmark datasets already stored under `dataset/`.
- MolTrans ESPF vocabulary assets under `ESPF/`.
- Five-seed experiment runners and validation/test evaluation logic.
- Code-verified checkpoint selection and validation-derived F1 thresholding.
- Robustness experiments for unseen entities and 70–95% missing interactions.
- CPI-GGS same-split reproduction code and aggregate results.
- Attention-case analysis code and selected case definitions.
- Final configuration tables, run commands, summary results, and manuscript figure assets.

## Repository layout

```text
BCAG-DTI/
├── ESPF/                         # MolTrans substructure vocabulary assets
├── dataset/                      # Fixed BindingDB, BIOSNAP, and DAVIS splits
├── tests/                        # Unit and distributed smoke/regression tests
├── configs/                      # Final commands and per-run configurations
├── docs/                         # Reproducibility and implementation details
├── results/                      # Compact manuscript result summaries
├── external_baselines/
│   └── cpi_ggs/                  # CPI-GGS fixed-split reproduction
├── figures/                      # Manuscript/reproducibility figures
├── ablation.py                   # A1–A8 definitions and training settings
├── train.py                      # Single-run training entry point
├── run_experiments.py            # MolTrans benchmark runner
├── run_ablation_experiments.py   # Controlled ablation/robustness runner
├── experiment.py                 # Data loading, evaluation, checkpoint logic
├── models.py                     # MolTrans baseline components
├── improved_models.py            # BCAG-DTI components
├── attention_analysis.py         # Attention-case analysis
├── evaluate.py                   # Checkpoint evaluation
└── requirements.txt
```

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

## Training and model selection

A1, A3, A4, A5, and A6 use Adam, learning rate `1e-4`, global batch size 16, dropout 0.10, and a fixed 50-epoch training duration. They do **not** use early stopping, but the checkpoint with the highest validation AUROC is retained and reloaded for testing.

A2, A7, and A8 use the enhanced training strategy. BindingDB/BIOSNAP use AdamW with learning rate `1e-4`, weight decay `1e-5`, dropout 0.15, three warm-up epochs, cosine decay, AMP, gradient clipping at 1.0, and early-stopping patience 15. DAVIS uses learning rate `1.2e-4`, weight decay `5e-6`, dropout 0.10, five warm-up epochs, patience 25, and label smoothing 0.1, with a maximum of 60 epochs.

For every A1–A8 run, checkpoint selection is based on validation AUROC. The F1 decision threshold is selected exclusively on validation data at the selected checkpoint and is then applied unchanged to the test set.

## Quick start

```bash
pip install -r requirements.txt
python train.py --task biosnap_full --variant a8_bcag_dti --seed 1 --batch-size 16
```

For the full controlled suite:

```bash
python run_ablation_experiments.py \
  --suite-name moltrans_ablation_REPRO \
  --gpus 0,1,2,3 \
  --batch-size 16 \
  --monitor auroc
```

The canonical tasks are `biosnap_full`, `bindingdb`, `davis`, `biosnap_unseen_drug`, `biosnap_unseen_protein`, `biosnap_missing_70`, `biosnap_missing_80`, `biosnap_missing_90`, and `biosnap_missing_95`.

See [`docs/REPRODUCTION_QUICKSTART.md`](docs/REPRODUCTION_QUICKSTART.md) and [`configs/RUN_COMMANDS.txt`](configs/RUN_COMMANDS.txt) for full instructions.

## Fixed data partitions

The controlled experiments use the fixed CSV partitions in `dataset/`; no runtime 80/10/10 resplitting is performed. Main split sizes are:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| BindingDB | 12,668 | 6,644 | 13,289 |
| BIOSNAP | 19,238 | 2,748 | 5,497 |
| DAVIS | 2,086 | 3,006 | 6,011 |

Robustness split details are recorded in `docs/data_split_confirmation.csv`.

## External CPI-GGS reference

`external_baselines/cpi_ggs/` contains the adapter used to evaluate CPI-GGS on the same fixed BIOSNAP train/validation/test partitions. Five deterministic runs use seeds 1–5, a maximum of 20 epochs, batch size 32, early-stopping patience 5, checkpoint selection by validation AUROC, and a validation-derived F1 threshold.

This is a same-split external reference rather than a fully protocol-identical comparison because CPI-GGS uses molecular graphs and protein 3-grams, whereas MolTrans/BCAG-DTI use FCS-derived substructure tokenisation.

## Reproducibility notes

- Large model checkpoints are intentionally excluded from version control.
- Final compact result tables are under `results/`.
- Exact historical per-run configurations are under `configs/all_run_configurations.csv`.
- Code-verified implementation settings are under `docs/final_implementation_details.csv`.
- The multi-GPU suite runners use POSIX file locking and are intended for Linux/CUDA environments.

## Citation

Citation information will be updated after publication. If you use this repository before publication, please cite the accompanying manuscript.

## Acknowledgement

This work builds on the publicly released MolTrans implementation and retains the corresponding substructure vocabulary assets and baseline components where required for controlled comparison.
