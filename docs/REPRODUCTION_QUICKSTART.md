# Reproduction quick start

Run the commands below from the repository root.

## 1. Environment

```bash
pip install -r requirements.txt
```

A CUDA-capable PyTorch environment is recommended for model training.

## 2. Data layout

The repository uses the fixed processed split files stored under:

```text
dataset/BindingDB/{train,val,test}.csv
dataset/DAVIS/{train,val,test}.csv
dataset/BIOSNAP/full_data/{train,val,test}.csv
dataset/BIOSNAP/unseen_drug/{train,val,test}.csv
dataset/BIOSNAP/unseen_protein/{train,val,test}.csv
dataset/BIOSNAP/missing_data/{70,80,90,95}/{train,val,test}.csv
```

An alternative data location can be selected with the `MOLTRANS_DATA_ROOT` environment variable.

## 3. Run one MolTrans/BCAG-DTI configuration

```bash
python train.py \
  --task biosnap_full \
  --variant a8_bcag_dti \
  --seed 1 \
  --batch-size 16
```

Available variants are:

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

Canonical tasks are `biosnap_full`, `bindingdb`, `davis`, `biosnap_unseen_drug`, `biosnap_unseen_protein`, `biosnap_missing_70`, `biosnap_missing_80`, `biosnap_missing_90`, and `biosnap_missing_95`.

For the principal A1–A8 comparison, run seeds 1–5 on `biosnap_full`, `bindingdb`, and `davis`. For robustness, run A1 and A8 with seeds 1–5 on the unseen-drug, unseen-protein, and four missing-interaction tasks.

The variant-specific training defaults are defined in `ablation.py` and summarised in `docs/final_implementation_details.csv`.

## 4. Evaluate a saved checkpoint

```bash
python evaluate.py \
  --checkpoint /path/to/best_checkpoint.pt \
  --split test \
  --batch-size 16
```

The checkpoint stores the validation-derived F1 threshold associated with the best-validation-AUROC epoch; evaluation applies that threshold unchanged to test predictions.

## 5. CPI-GGS exact-split reproduction

The CPI-GGS adapter additionally requires RDKit and PyTorch Geometric.

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

See `external_baselines/cpi_ggs/README.md` for preprocessing and protocol differences relative to MolTrans/BCAG-DTI.

## 6. Attention case provenance

The selected seed-3 TP/TN/FP/FN cases and evidence context are recorded in `attention_cases.json`. Large model checkpoints and generated attention images are intentionally not stored in version control.

## Result selection

For all final MolTrans/BCAG-DTI configurations, the selected model is the checkpoint with the highest validation AUROC. Test F1 uses the decision threshold selected from validation data at that selected checkpoint. The test set is never used to choose the checkpoint or optimise the threshold.
