"""Reproduce CPI-GGS on the exact MolTrans CSV splits.

This adapter preserves the published repository's CPI-GGS network components and
hyperparameters while adding batched training, fixed-split loading, validation
checkpoint selection, F1 reporting, deterministic seeds, and machine-readable
outputs required for manuscript revision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence

RDLogger.DisableLog("rdApp.*")
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GCNConv, global_max_pool, global_mean_pool


SPLIT_PATHS = {
    "bindingdb": Path("dataset/BindingDB"),
    "biosnap_full": Path("dataset/BIOSNAP/full_data"),
    "davis": Path("dataset/DAVIS"),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Vocabulary:
    def __init__(self):
        self.mapping = {}

    def id(self, item):
        if item not in self.mapping:
            self.mapping[item] = len(self.mapping)
        return self.mapping[item]

    def __len__(self):
        return len(self.mapping)


def atom_features(mol, atom_vocab: Vocabulary):
    return np.asarray([
        atom_vocab.id((a.GetSymbol(), a.GetTotalNumHs(), a.GetImplicitValence(), a.GetIsAromatic()))
        for a in mol.GetAtoms()
    ], dtype=np.int64)


def adjacency_with_bonds(mol, bond_vocab: Vocabulary):
    graph = defaultdict(list)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_id = bond_vocab.id(str(bond.GetBondType()))
        graph[i].append((j, bond_id))
        graph[j].append((i, bond_id))
    return graph


def wl_fingerprints(atoms, graph, radius, fingerprint_vocab: Vocabulary, edge_vocab: Vocabulary):
    if len(atoms) == 1 or radius == 0:
        return np.asarray([fingerprint_vocab.id(int(a)) for a in atoms], dtype=np.int64)
    nodes = list(map(int, atoms))
    edges = graph
    for _ in range(radius):
        new_nodes = []
        for i in range(len(nodes)):
            neighbors = tuple(sorted((nodes[j], edge) for j, edge in edges.get(i, [])))
            new_nodes.append(fingerprint_vocab.id((nodes[i], neighbors)))
        new_edges = defaultdict(list)
        for i, neighbors in edges.items():
            for j, edge in neighbors:
                new_edge = edge_vocab.id((tuple(sorted((new_nodes[i], new_nodes[j]))), edge))
                new_edges[i].append((j, new_edge))
        nodes, edges = new_nodes, new_edges
    return np.asarray(nodes, dtype=np.int64)


def edge_index(mol):
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((i, j), (j, i)))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def protein_ngrams(sequence: str, ngram: int, word_vocab: Vocabulary):
    padded = "-" + str(sequence).strip().upper() + "="
    return torch.tensor(
        [word_vocab.id(padded[i : i + ngram]) for i in range(len(padded) - ngram + 1)],
        dtype=torch.long,
    )


class FixedSplitDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate(samples):
    graphs, proteins, labels = zip(*samples)
    batch = Batch.from_data_list(graphs)
    lengths = torch.tensor([len(p) for p in proteins], dtype=torch.long)
    proteins = pad_sequence(proteins, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return batch, proteins, lengths, labels


def preprocess(csv_paths, radius=2, ngram=3):
    atom_vocab, bond_vocab = Vocabulary(), Vocabulary()
    fingerprint_vocab, edge_vocab, word_vocab = Vocabulary(), Vocabulary(), Vocabulary()
    frames = {split: pd.read_csv(path) for split, path in csv_paths.items()}
    processed = {}
    invalid = []
    for split in ("train", "val", "test"):
        samples = []
        for row_index, row in frames[split].iterrows():
            smiles = str(row["SMILES"])
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid.append((split, int(row_index), smiles))
                continue
            mol = Chem.AddHs(mol)
            atoms = atom_features(mol, atom_vocab)
            graph = adjacency_with_bonds(mol, bond_vocab)
            fingerprints = wl_fingerprints(atoms, graph, radius, fingerprint_vocab, edge_vocab)
            pyg_graph = Data(x=torch.from_numpy(fingerprints), edge_index=edge_index(mol))
            protein = protein_ngrams(row["Target Sequence"], ngram, word_vocab)
            label = int(float(row["Label"]))
            samples.append((pyg_graph, protein, label))
        processed[split] = FixedSplitDataset(samples)
    meta = {
        "fingerprint_count": len(fingerprint_vocab),
        "protein_ngram_count": len(word_vocab),
        "invalid_rows": invalid,
        "rows": {key: len(value) for key, value in processed.items()},
    }
    return processed, meta


class CPIGGS(nn.Module):
    def __init__(self, n_fingerprint, n_word, dim=10, window=11, layer_cnn=3, dropout=0.05):
        super().__init__()
        self.dim = dim
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.conv1 = GATv2Conv(dim, dim, heads=5)
        self.conv2 = GCNConv(dim, dim * 5)
        self.conv3 = GCNConv(dim * 5, dim * 5)
        self.fc_g1 = nn.Linear(dim * 10 * 2, 1600)
        self.fc_g2 = nn.Linear(1600, 128)
        self.drug_projection = nn.Linear(128, 10)
        self.protein_convs = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=2 * window + 1, stride=1, padding=window)
            for _ in range(layer_cnn)
        ])
        self.bigru = nn.GRU(dim, 5, 1, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=2 * dim, num_heads=1, batch_first=True)
        self.output = nn.Linear(2 * dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_batch, proteins, protein_lengths):
        fingerprints = self.embed_fingerprint(graph_batch.x)
        gat = F.relu(self.conv1(fingerprints, graph_batch.edge_index))
        gcn = F.relu(self.conv2(fingerprints, graph_batch.edge_index))
        gcn = F.relu(self.conv3(gcn, graph_batch.edge_index))
        nodes = torch.cat((gat, gcn), dim=1)
        drug = torch.cat(
            (global_max_pool(nodes, graph_batch.batch), global_mean_pool(nodes, graph_batch.batch)), dim=1
        )
        drug = F.relu(self.fc_g1(drug))
        drug = self.drug_projection(self.fc_g2(self.dropout(drug)))

        protein = self.embed_word(proteins).unsqueeze(1)
        for conv in self.protein_convs:
            protein = F.relu(conv(protein))
        protein = protein.squeeze(1)
        protein, _ = self.bigru(protein)
        mask = torch.arange(protein.shape[1], device=protein.device)[None, :] < protein_lengths[:, None]
        protein = (protein * mask.unsqueeze(-1)).sum(1) / protein_lengths.clamp_min(1).unsqueeze(1)

        joint = torch.cat((drug, protein), dim=1).unsqueeze(1)
        joint, _ = self.attention(joint, joint, joint, need_weights=False)
        return self.output(joint.squeeze(1))


def threshold_for_f1(labels, scores):
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    best = (0.5, -1.0)
    for threshold in candidates:
        value = f1_score(labels, scores >= threshold, zero_division=0)
        if value > best[1]:
            best = (float(threshold), float(value))
    return best[0]


@torch.no_grad()
def evaluate(model, loader, device, threshold=None):
    model.eval()
    labels, scores = [], []
    for graph, protein, lengths, y in loader:
        graph, protein, lengths = graph.to(device), protein.to(device), lengths.to(device)
        probability = model(graph, protein, lengths).softmax(1)[:, 1]
        labels.extend(y.numpy().tolist())
        scores.extend(probability.cpu().numpy().tolist())
    labels, scores = np.asarray(labels), np.asarray(scores)
    threshold = threshold_for_f1(labels, scores) if threshold is None else threshold
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, scores >= threshold, zero_division=0)),
        "threshold": float(threshold),
        "num_samples": int(len(labels)),
    }


@dataclass
class RunConfig:
    dataset: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    lr_decay: float
    decay_interval: int
    patience: int
    radius: int = 2
    ngram: int = 3
    embedding_dim: int = 10
    cnn_window: int = 11
    cnn_layers: int = 3
    dropout: float = 0.05


def run_one(config, datasets, meta, output_dir, device):
    seed_everything(config.seed)
    loaders = {
        key: DataLoader(
            value,
            batch_size=config.batch_size,
            shuffle=(key == "train"),
            num_workers=0,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
        for key, value in datasets.items()
    }
    model = CPIGGS(
        meta["fingerprint_count"],
        meta["protein_ngram_count"],
        dim=config.embedding_dim,
        window=config.cnn_window,
        layer_cnn=config.cnn_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    parameter_count = sum(p.numel() for p in model.parameters())
    run_dir = output_dir / f"seed_{config.seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_epoch, best_auroc, best_state, best_validation = 0, -math.inf, None, None
    no_improvement, history = 0, []
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        if epoch % config.decay_interval == 0:
            for group in optimizer.param_groups:
                group["lr"] *= config.lr_decay
        model.train()
        loss_total, examples = 0.0, 0
        for graph, protein, lengths, labels in loaders["train"]:
            graph = graph.to(device)
            protein, lengths, labels = protein.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(graph, protein, lengths)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(labels)
            examples += len(labels)
        validation = evaluate(model, loaders["val"], device)
        history.append({
            "epoch": epoch,
            "train_loss": loss_total / max(examples, 1),
            "validation_auroc": validation["auroc"],
            "validation_auprc": validation["auprc"],
            "validation_f1": validation["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(json.dumps({"seed": config.seed, **history[-1]}), flush=True)
        if validation["auroc"] > best_auroc:
            best_epoch, best_auroc = epoch, validation["auroc"]
            best_validation = validation
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= config.patience:
            break
    model.load_state_dict(best_state)
    test = evaluate(model, loaders["test"], device, threshold=best_validation["threshold"])
    duration = time.perf_counter() - start
    torch.save(best_state, run_dir / "best_checkpoint.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    summary = {
        "method": "CPI-GGS fixed-split reproduction",
        "result_type": "reproduced",
        "configuration": asdict(config),
        "best_epoch": best_epoch,
        "checkpoint_selection_metric": "validation AUROC",
        "best_validation": best_validation,
        "test": test,
        "parameter_count": parameter_count,
        "training_time_seconds": duration,
        "epochs_completed": len(history),
        "stopping_rule": f"validation AUROC early stopping, patience={config.patience}",
        "preprocessing": meta,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_geometric_version": __import__("torch_geometric").__version__,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def aggregate(summaries, output_dir):
    metrics = ["auroc", "auprc", "f1"]
    row = {"method": "CPI-GGS fixed-split reproduction", "runs": len(summaries)}
    for metric in metrics:
        values = np.asarray([summary["test"][metric] for summary in summaries], dtype=float)
        row[f"test_{metric}_mean"] = float(values.mean())
        row[f"test_{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    row["parameter_count"] = summaries[0]["parameter_count"]
    durations = np.asarray([summary["training_time_seconds"] for summary in summaries])
    row["training_time_seconds_mean"] = float(durations.mean())
    row["training_time_seconds_std"] = float(durations.std(ddof=1)) if len(durations) > 1 else 0.0
    with (output_dir / "aggregate.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moltrans-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(SPLIT_PATHS), default="biosnap_full")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--decay-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    split_root = args.moltrans_root / SPLIT_PATHS[args.dataset]
    csv_paths = {split: split_root / f"{split}.csv" for split in ("train", "val", "test")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, meta = preprocess(csv_paths)
    (args.output_dir / "preprocessing.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = []
    for seed in args.seeds:
        config = RunConfig(
            dataset=args.dataset,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_decay=args.lr_decay,
            decay_interval=args.decay_interval,
            patience=args.patience,
        )
        summaries.append(run_one(config, datasets, meta, args.output_dir, device))
    print(json.dumps(aggregate(summaries, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
