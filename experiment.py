from __future__ import annotations

import json
import logging
import os
import random
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from stream import BIN_Data_Encoder


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "dataset"
PACKAGE_DATA_ROOT = REPO_ROOT.parent / "03_data_splits"
DATA_ROOT = Path(os.environ.get("MOLTRANS_DATA_ROOT", "")).expanduser() if os.environ.get("MOLTRANS_DATA_ROOT") else (
    DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else PACKAGE_DATA_ROOT
)
CHECKPOINT_SCHEMA_VERSION = 1
TASK_DATA_DIRS = {
    "biosnap_full": Path("BIOSNAP") / "full_data",
    "bindingdb": Path("BindingDB"),
    "davis": Path("DAVIS"),
    "biosnap_unseen_drug": Path("BIOSNAP") / "unseen_drug",
    "biosnap_unseen_protein": Path("BIOSNAP") / "unseen_protein",
    "biosnap_missing_70": Path("BIOSNAP") / "missing_data" / "70",
    "biosnap_missing_80": Path("BIOSNAP") / "missing_data" / "80",
    "biosnap_missing_90": Path("BIOSNAP") / "missing_data" / "90",
    "biosnap_missing_95": Path("BIOSNAP") / "missing_data" / "95",
}
EXPERIMENT_TASKS = tuple(TASK_DATA_DIRS)
TASK_ALIASES = {"biosnap": "biosnap_full"}
TASK_CHOICES = tuple(sorted((*TASK_DATA_DIRS, *TASK_ALIASES)))


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    result_dir: Path
    checkpoint: Path
    history: Path
    summary: Path
    predictions: Path
    log: Path
    picture: Path


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return (index, *self.dataset[index])


class LabelSmoothingBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("Label smoothing must be in [0, 1).")
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)


def init_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed training requires CUDA.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return DistributedContext(
            rank=dist.get_rank(),
            local_rank=local_rank,
            world_size=dist.get_world_size(),
            device=torch.device("cuda", local_rank),
        )

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return DistributedContext(rank=0, local_rank=0, world_size=1, device=device)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(context: DistributedContext) -> None:
    if context.is_distributed:
        dist.barrier()


def distributed_max(value: float, context: DistributedContext) -> float:
    value_tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    if context.is_distributed:
        dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX)
    return float(value_tensor.item())


def validate_global_batch_size(global_batch_size: int, world_size: int) -> int:
    if global_batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by world size {world_size}."
        )
    local_batch_size = global_batch_size // world_size
    if local_batch_size < 2:
        raise ValueError(
            "Each process must receive at least two samples because MolTrans uses BatchNorm; "
            f"got global batch size {global_batch_size} with world size {world_size}."
        )
    return local_batch_size


def seed_everything(context: DistributedContext, seed: int = 1) -> None:
    if seed < 0:
        raise ValueError("Seed must be non-negative.")
    rank_seed = seed + context.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def normalize_task_name(task: str) -> str:
    normalized = task.strip().lower().replace("-", "_")
    normalized = TASK_ALIASES.get(normalized, normalized)
    if normalized not in TASK_DATA_DIRS:
        raise ValueError(f"Unsupported task: {task}")
    return normalized


def get_task_data_dir(task: str) -> Path:
    return DATA_ROOT / TASK_DATA_DIRS[normalize_task_name(task)]


def load_split(task: str, split: str) -> pd.DataFrame:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    path = get_task_data_dir(task) / f"{split}.csv"
    frame = pd.read_csv(path)
    required_columns = {"SMILES", "Target Sequence", "Label"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} is empty.")
    if frame["Label"].nunique() < 2:
        raise ValueError(f"{path} must contain both binary classes.")
    return frame


def make_dataset(frame: pd.DataFrame) -> BIN_Data_Encoder:
    return BIN_Data_Encoder(frame.index.to_numpy(), frame["Label"].to_numpy(), frame)


def _loader_options(workers: int, use_cuda: bool) -> Dict[str, Any]:
    if workers < 0:
        raise ValueError("--workers cannot be negative.")
    return {
        "num_workers": workers,
        "pin_memory": use_cuda,
        "persistent_workers": workers > 0,
    }


def build_train_loader(
    frame: pd.DataFrame,
    local_batch_size: int,
    workers: int,
    context: DistributedContext,
    seed: int = 1,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = make_dataset(frame)
    sampler = None
    if context.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        **_loader_options(workers, context.device.type == "cuda"),
    )
    if len(loader) == 0:
        raise ValueError("The training loader has no full batches.")
    return loader, sampler


def build_eval_loader(
    frame: pd.DataFrame,
    local_batch_size: int,
    workers: int,
    context: DistributedContext,
) -> DataLoader:
    dataset = IndexedDataset(make_dataset(frame))
    sampler = None
    if context.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            drop_last=False,
        )
    return DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=False,
        **_loader_options(workers, context.device.type == "cuda"),
    )


def _to_device(value: torch.Tensor, context: DistributedContext, dtype: torch.dtype) -> torch.Tensor:
    return value.to(device=context.device, dtype=dtype, non_blocking=True)


BatchNormBufferSnapshot = List[Tuple[torch.nn.Module, Dict[str, torch.Tensor]]]


def snapshot_batch_norm_buffers(model: torch.nn.Module) -> BatchNormBufferSnapshot:
    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    snapshots: BatchNormBufferSnapshot = []
    for module in base_model.modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        buffers = {
            name: buffer.detach().clone()
            for name in ("running_mean", "running_var", "num_batches_tracked")
            if (buffer := getattr(module, name, None)) is not None
        }
        if buffers:
            snapshots.append((module, buffers))
    return snapshots


@torch.no_grad()
def restore_batch_norm_buffers(snapshots: BatchNormBufferSnapshot) -> None:
    for module, buffers in snapshots:
        for name, value in buffers.items():
            getattr(module, name).copy_(value)


def _batch_norm_floating_buffers(
    snapshots: BatchNormBufferSnapshot,
) -> List[torch.Tensor]:
    buffers: List[torch.Tensor] = []
    for module, saved_buffers in snapshots:
        for name in saved_buffers:
            current = getattr(module, name)
            if current.is_floating_point():
                buffers.append(current)
    return buffers


def tensors_are_globally_finite(
    tensors: List[torch.Tensor], context: DistributedContext
) -> bool:
    finite = torch.ones((), dtype=torch.int32, device=context.device)
    for tensor in tensors:
        if tensor.is_floating_point() or tensor.is_complex():
            finite.mul_(torch.isfinite(tensor.detach()).all().to(dtype=torch.int32))
    if context.is_distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    context: DistributedContext,
    logger: Optional[logging.Logger] = None,
    loss_function: Optional[torch.nn.Module] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
    max_grad_norm: Optional[float] = None,
    recovery_stats: Optional[Dict[str, int]] = None,
) -> float:
    model.train()
    loss_function = loss_function or LabelSmoothingBCEWithLogitsLoss()
    amp_enabled = use_amp and context.device.type == "cuda"
    loss_sum = 0.0
    sample_count = 0
    if recovery_stats is not None:
        recovery_stats.setdefault("fp32_fallback_batches", 0)

    for step, (drug, protein, drug_mask, protein_mask, label) in enumerate(loader):
        drug = _to_device(drug, context, torch.long)
        protein = _to_device(protein, context, torch.long)
        drug_mask = _to_device(drug_mask, context, torch.long)
        protein_mask = _to_device(protein_mask, context, torch.long)
        label = _to_device(label, context, torch.float32).reshape(-1)

        batch_norm_snapshot = snapshot_batch_norm_buffers(model) if amp_enabled else []
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(drug, protein, drug_mask, protein_mask).reshape(-1)
            loss = loss_function(logits, label)

        use_fp32_fallback = amp_enabled and not tensors_are_globally_finite(
            [logits, loss, *_batch_norm_floating_buffers(batch_norm_snapshot)],
            context,
        )
        if use_fp32_fallback:
            previous_scale = float(scaler.get_scale()) if scaler is not None else 1.0
            if logger is not None and context.is_main:
                logger.warning(
                    "non-finite AMP forward at training step=%d; restored BatchNorm "
                    "buffers and retrying the same batch in FP32 (scale=%.1f)",
                    step,
                    previous_scale,
                )
            if isinstance(model, DistributedDataParallel):
                torch.autograd.backward(logits, grad_tensors=torch.zeros_like(logits))
            optimizer.zero_grad(set_to_none=True)
            restore_batch_norm_buffers(batch_norm_snapshot)
            del logits, loss
            with torch.cuda.amp.autocast(enabled=False):
                logits = model(drug, protein, drug_mask, protein_mask).reshape(-1)
                loss = loss_function(logits, label)
            fallback_forward_is_finite = tensors_are_globally_finite(
                [logits, loss, *_batch_norm_floating_buffers(batch_norm_snapshot)],
                context,
            )
            if not fallback_forward_is_finite:
                restore_batch_norm_buffers(batch_norm_snapshot)
                raise RuntimeError(
                    f"Non-finite values remained after FP32 recovery at training step {step}."
                )
            loss.backward()
            if max_grad_norm is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                if not tensors_are_globally_finite([gradient_norm], context):
                    restore_batch_norm_buffers(batch_norm_snapshot)
                    optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError(
                        f"Non-finite gradients remained after FP32 recovery at training step {step}."
                    )
            optimizer.step()
            if scaler is not None and scaler.is_enabled():
                scaler.scale(torch.zeros((), device=context.device))
                scaler.update(new_scale=max(previous_scale / 2.0, 1.0))
            if recovery_stats is not None:
                recovery_stats["fp32_fallback_batches"] += 1
        elif scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        batch_size = label.numel()
        loss_sum += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        if logger is not None and context.is_main and step % 1000 == 0:
            logger.info("training step=%d local_loss=%.6f", step, float(loss.detach().item()))

    totals = torch.tensor([loss_sum, sample_count], dtype=torch.float64, device=context.device)
    if context.is_distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if totals[1].item() == 0:
        raise RuntimeError("No training samples were processed.")
    return float((totals[0] / totals[1]).item())


def choose_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    numerator = 2.0 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1_values))])


def compute_binary_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.size != logits.size:
        raise ValueError("Labels and logits must be non-empty arrays of equal length.")
    if np.unique(labels).size < 2:
        raise ValueError("Evaluation metrics require both binary classes.")

    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    if threshold is None:
        threshold = choose_f1_threshold(labels, probabilities)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity_denominator = tn + fp
    loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)

    metrics = {
        "num_samples": int(labels.size),
        "loss": float(loss),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "threshold": float(threshold),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / specificity_denominator) if specificity_denominator else 0.0,
        "accuracy": float(accuracy_score(labels, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, probabilities, predictions


def _all_gather_equal_tensor(tensor: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if not context.is_distributed:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(context.world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    context: DistributedContext,
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    model.eval()
    local_indices = []
    local_logits = []
    local_labels = []

    with torch.inference_mode():
        for index, drug, protein, drug_mask, protein_mask, label in loader:
            drug = _to_device(drug, context, torch.long)
            protein = _to_device(protein, context, torch.long)
            drug_mask = _to_device(drug_mask, context, torch.long)
            protein_mask = _to_device(protein_mask, context, torch.long)
            logits = model(drug, protein, drug_mask, protein_mask).reshape(-1)

            local_indices.append(index.to(device=context.device, dtype=torch.long, non_blocking=True))
            local_logits.append(logits.detach().to(dtype=torch.float64))
            local_labels.append(label.to(device=context.device, dtype=torch.float64, non_blocking=True).reshape(-1))

    if not local_indices:
        raise RuntimeError("The evaluation loader produced no samples.")

    indices_tensor = _all_gather_equal_tensor(torch.cat(local_indices), context)
    logits_tensor = _all_gather_equal_tensor(torch.cat(local_logits), context)
    labels_tensor = _all_gather_equal_tensor(torch.cat(local_labels), context)

    indices = indices_tensor.cpu().numpy()
    logits = logits_tensor.cpu().numpy()
    labels = labels_tensor.cpu().numpy()
    order = np.argsort(indices, kind="stable")
    indices = indices[order]
    logits = logits[order]
    labels = labels[order]
    unique_mask = np.ones(indices.shape[0], dtype=bool)
    unique_mask[1:] = indices[1:] != indices[:-1]
    indices = indices[unique_mask]
    logits = logits[unique_mask]
    labels = labels[unique_mask]

    expected_samples = len(loader.dataset)
    if indices.size != expected_samples:
        raise RuntimeError(
            f"Distributed evaluation produced {indices.size} unique samples; expected {expected_samples}."
        )

    metrics, probabilities, predictions = compute_binary_metrics(labels, logits, threshold)
    prediction_frame = None
    if context.is_main:
        prediction_frame = pd.DataFrame(
            {
                "sample_index": indices.astype(np.int64),
                "label": labels.astype(np.int64),
                "logit": logits,
                "probability": probabilities,
                "prediction": predictions,
            }
        )
    return metrics, prediction_frame


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def model_state_dict_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in unwrap_model(model).state_dict().items()
    }


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    required = {
        "schema_version",
        "task",
        "epoch",
        "monitor",
        "best_metric",
        "threshold",
        "model_config",
        "validation_metrics",
        "model_state_dict",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint {path} is missing fields: {sorted(missing)}")
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema {checkpoint['schema_version']}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    return checkpoint


def load_model_state(model: torch.nn.Module, checkpoint: Dict[str, Any]) -> None:
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=True)


def _atomic_text_write(path: Path, write_callback) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write_callback(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    _atomic_text_write(path, write)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_text_write(path, lambda temporary: frame.to_csv(temporary, index=False))


def write_text(path: Path, content: str) -> None:
    def write(temporary: Path) -> None:
        temporary.write_text(content, encoding="utf-8")

    _atomic_text_write(path, write)


def save_history_plot(history: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["epoch"], history["val_auroc"], label="validation AUROC")
    axes[1].plot(history["epoch"], history["val_auprc"], label="validation AUPRC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=160)
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def validate_output_name(name: str, option: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name):
        raise ValueError(
            f"{option} must use lower_snake_case and start with a letter; got: {name!r}."
        )


def create_run_paths(
    task: str,
    requested_run_name: Optional[str],
    context: DistributedContext,
    suite_name: Optional[str] = None,
    seed: Optional[int] = None,
    variant: Optional[str] = None,
) -> RunPaths:
    shared = [None]
    if context.is_main:
        try:
            task = normalize_task_name(task)
            if suite_name is not None:
                if requested_run_name is not None:
                    raise ValueError("--run-name cannot be combined with --suite-name.")
                if seed is None or seed < 0:
                    raise ValueError("Suite runs require a non-negative --seed.")
                validate_output_name(suite_name, "--suite-name")
                leaf_name = f"seed_{seed:03d}"
                components = [suite_name]
                if variant is not None:
                    validate_output_name(variant, "--variant")
                    components.append(variant)
                components.extend((task, leaf_name))
                relative_run = Path(*components)
                run_id = relative_run.as_posix()
                result_dir = REPO_ROOT / "results" / relative_run
                log_path = REPO_ROOT / "log" / relative_run.parent / f"{leaf_name}.log"
                picture_path = REPO_ROOT / "picture" / relative_run.parent / (
                    f"{leaf_name}_training_curves.png"
                )
            else:
                leaf_name = requested_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                validate_output_name(leaf_name, "--run-name")
                components = ["standalone"]
                if variant is not None:
                    validate_output_name(variant, "--variant")
                    components.append(variant)
                components.extend((task, leaf_name))
                relative_run = Path(*components)
                run_id = relative_run.as_posix()
                result_dir = REPO_ROOT / "results" / relative_run
                log_path = REPO_ROOT / "log" / relative_run.parent / f"{leaf_name}.log"
                picture_path = REPO_ROOT / "picture" / relative_run.parent / (
                    f"{leaf_name}_training_curves.png"
                )
            if result_dir.exists() or log_path.exists() or picture_path.exists():
                raise FileExistsError(f"Run output already exists: {run_id}")
            result_dir.mkdir(parents=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            picture_path.parent.mkdir(parents=True, exist_ok=True)
            paths = RunPaths(
                run_id=run_id,
                result_dir=result_dir,
                checkpoint=result_dir / "best_checkpoint.pt",
                history=result_dir / "history.csv",
                summary=result_dir / "summary.json",
                predictions=result_dir / "test_predictions.csv",
                log=log_path,
                picture=picture_path,
            )
            shared[0] = {"paths": {key: str(value) for key, value in asdict(paths).items()}}
        except Exception as exc:
            shared[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if context.is_distributed:
        dist.broadcast_object_list(shared, src=0)
    if "error" in shared[0]:
        raise RuntimeError(shared[0]["error"])
    payload = shared[0]["paths"]
    return RunPaths(
        run_id=payload["run_id"],
        result_dir=Path(payload["result_dir"]),
        checkpoint=Path(payload["checkpoint"]),
        history=Path(payload["history"]),
        summary=Path(payload["summary"]),
        predictions=Path(payload["predictions"]),
        log=Path(payload["log"]),
        picture=Path(payload["picture"]),
    )


def setup_logger(name: str, log_path: Optional[Path], context: DistributedContext) -> logging.Logger:
    logger = logging.getLogger(f"{name}.rank{context.rank}")
    logger.handlers.clear()
    logger.propagate = False
    if not context.is_main:
        logger.addHandler(logging.NullHandler())
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_path is not None:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def git_metadata() -> Dict[str, Any]:
    def run_git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run_git("rev-parse", "HEAD"),
            "branch": run_git("branch", "--show-current"),
            "dirty": bool(run_git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def runtime_metadata(context: DistributedContext) -> Dict[str, Any]:
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
        "world_size": context.world_size,
        "command": shlex.join(sys.argv),
        "git": git_metadata(),
    }


def format_metrics(metrics: Dict[str, Any]) -> str:
    keys = ["loss", "auroc", "auprc", "f1", "precision", "recall", "specificity", "accuracy"]
    return " ".join(f"{key}={metrics[key]:.6f}" for key in keys)
