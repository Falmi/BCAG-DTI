# BCAG-DTI: Bidirectional Cross-Attention and Global Semantic Aggregation for Robust Drug–Target Interaction Prediction

## Overview

BCAG-DTI is a Transformer-based framework for drug–target interaction (DTI) prediction. Building upon the MolTrans architecture, BCAG-DTI introduces three key enhancements:

* **Bidirectional Cross-Attention** for explicit modelling of interactions between drug substructures and protein residues.
* **Hybrid Local–Global Representation Learning** that combines CNN-based interaction modelling with multi-scale global semantic aggregation.
* **Enhanced Training and Classification Strategy** using a residual-enhanced MLP classifier, BCEWithLogitsLoss, AdamW optimisation, and cosine learning-rate scheduling.

The proposed framework aims to improve predictive performance, training stability, and interpretability for DTI prediction, particularly in small-data and imbalanced settings.

---

## Datasets

This repository contains the processed benchmark datasets used in our experiments:

* BindingDB
* BIOSNAP
* DAVIS

For BIOSNAP, we additionally provide:

* Unseen Drug split
* Unseen Protein split
* Missing-data settings (70%, 80%, 90%, and 95%)

Please refer to the original dataset publications for dataset descriptions and licensing information.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

Example training configuration:

| Parameter           | Value |
| ------------------- | ----- |
| Embedding Dimension | 128   |
| Transformer Layers  | 2     |
| Attention Heads     | 8     |
| Batch Size          | 16    |
| Optimiser           | AdamW |
| Learning Rate       | 1e-4  |
| Epochs              | 60    |

Run training using:

```bash
python train.py
```

---

## Evaluation

Evaluate a trained model using:

```bash
python evaluate.py
```

---

## Repository Structure

```text
BCAG-DTI/
├── datasets/
│   ├── BindingDB/
│   ├── BIOSNAP/
│   └── DAVIS/
├── models/
├── scripts/
├── notebooks/
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## Experimental Results

The proposed BCAG-DTI framework was evaluated on three widely used DTI benchmark datasets:

* BindingDB
* BIOSNAP
* DAVIS

The model consistently improves prediction performance over the MolTrans baseline through explicit cross-modal interaction modelling and hybrid local–global representation learning.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{pang2026bcagdti,
  title={Bidirectional Cross-Attention and Global Semantic Aggregation for Robust Drug--Target Interaction Prediction},
  author={Hao Panga, Fiseha Berhanu Tesemaa, Tianxiang Cuia and Yuan Chenga},
  journal={BMC Bioinformatics},
  year={2026},
  note={Under Review}
}
```

---
