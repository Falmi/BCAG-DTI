# Reproduction quick start

Run the commands below from the repository root.

## 1. Environment

```bash
pip install -r requirements.txt
```

A CUDA-capable PyTorch environment is recommended. The four-GPU suite runners use POSIX file locking and should be launched on Linux.

## 2. Data layout

The repository uses the fixed processed split files already stored under:

```text
dataset/BindingDB/{train,val,test}.csv
dataset/DAVIS/{train,val,test}.csv
dataset/BIOSNAP/full_data/{train,val,test}.csv
dataset/BIOSNAP/unseen_drug/{train,val,test}.csv
dataset/BIOSNAP/unseen_protein/{train,val,test}.csv
dataset/BIOSNAP/missing_data/{70,80,90,95}/{train,val,test}.csv
```

An alternative data location can be selected with the `MOLTRANS_DATA_ROOT` environment variable.

## 3. Single run

```bash
python train.py --task biosnap_full --variant a8_bcag_dti --seed 1 --batch-size 16
```

Canonical task names are `biosnap_full`, `bindingdb`, `davis`, `biosnap_unseen_drug`, `biosnap_unseen_protein`, and `biosnap_missing_70`, `biosnap_missing_80`, `biosnap_missing_90`, `biosnap_missing_95`.

## 4. Main MolTrans benchmark

```bash
python run_experiments.py \
  --suite-name moltrans_benchmark_REPRO \
  --gpus 0,1,2,3 \
  --epochs 50 \
  --batch-size 16 \
  --monitor auroc
```

## 5. A1–A8 controlled ablations and robustness

```bash
python run_ablation_experiments.py \
  --suite-name moltrans_ablation_REPRO \
  --gpus 0,1,2,3 \
  --batch-size 16 \
  --monitor auroc
```

The exact historical commands and per-run hyperparameters are recorded in `configs/RUN_COMMANDS.txt` and `configs/all_run_configurations.csv`.

## 6. CPI-GGS exact-split reproduction

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

Its distinct preprocessing and adaptation details are documented in `external_baselines/cpi_ggs/README.txt` and `external_baselines/cpi_ggs/RESULT_ANALYSIS.txt`.

## 7. Attention analysis

The source script is `attention_analysis.py` and the selected cases are defined in `attention_cases.json`. Attention analysis requires the selected A1/A8 checkpoints; large checkpoints are intentionally not versioned.

## Result selection

For all final MolTrans/BCAG-DTI configurations, the selected model is the checkpoint with the highest validation AUROC. Test F1 uses the decision threshold selected from the validation data associated with that checkpoint. The test set is never used to optimise the threshold.
