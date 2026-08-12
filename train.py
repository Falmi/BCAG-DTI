from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel

from ablation import (
    A1_ORIGINAL,
    VARIANT_CHOICES,
    build_model_config,
    get_variant_spec,
    normalize_variant_name,
    resolve_training_settings,
)
from config import BIN_config_DBPE
from experiment import (
    CHECKPOINT_SCHEMA_VERSION,
    LabelSmoothingBCEWithLogitsLoss,
    REPO_ROOT,
    TASK_CHOICES,
    DistributedContext,
    RunPaths,
    atomic_torch_save,
    build_eval_loader,
    build_train_loader,
    cleanup_distributed,
    create_run_paths,
    distributed_barrier,
    distributed_max,
    evaluate_model,
    format_metrics,
    init_distributed,
    load_checkpoint,
    load_model_state,
    load_split,
    model_state_dict_cpu,
    normalize_task_name,
    runtime_metadata,
    save_history_plot,
    seed_everything,
    setup_logger,
    train_one_epoch,
    validate_global_batch_size,
    write_csv,
    write_json,
)
from improved_models import build_moltrans_model


AMP_NONFINITE_POLICY = "restore_batch_norm_and_retry_same_batch_fp32"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MolTrans training with optional distributed execution.")
    parser.add_argument(
        "-b",
        "--batch-size",
        default=16,
        type=int,
        metavar="N",
        help="global mini-batch size across all processes (default: 16)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="data loading workers per process (default: 0)",
    )
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        metavar="N",
        help="maximum training epochs; defaults to the variant/task protocol",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=TASK_CHOICES,
        help="dataset task",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        metavar="N",
        help="base random seed recorded with the run (default: 1)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=None,
        type=float,
        metavar="LR",
        dest="lr",
        help="learning rate; defaults to the variant/task protocol",
    )
    parser.add_argument(
        "--variant",
        choices=VARIANT_CHOICES,
        default=A1_ORIGINAL,
        help="registered A1-A8 ablation variant (default: a1_original)",
    )
    parser.add_argument(
        "--monitor",
        choices=["auroc", "auprc"],
        default="auroc",
        help="validation metric used to select the best checkpoint (default: auroc)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="standalone lower_snake_case run name; defaults to run_YYYYmmdd_HHMMSS",
    )
    parser.add_argument(
        "--suite-name",
        default=None,
        help="lower_snake_case suite name; stores output under <suite>/<task>/seed_NNN",
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return parser


def apply_variant_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args.variant = normalize_variant_name(args.variant)
    settings = resolve_training_settings(args.variant, args.task)
    if args.epochs is None:
        args.epochs = settings.max_epochs
    if args.lr is None:
        args.lr = settings.learning_rate
    args.training_settings = settings.to_dict()
    args.training_settings["max_epochs"] = args.epochs
    args.training_settings["learning_rate"] = args.lr
    args.training_settings["amp_nonfinite_policy"] = AMP_NONFINITE_POLICY
    return args


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
):
    if warmup_epochs < 0 or warmup_epochs > total_epochs:
        raise ValueError("warmup epochs must be between zero and total epochs.")

    def multiplier(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)
        cosine_epochs = total_epochs - warmup_epochs
        if cosine_epochs <= 0:
            return 1.0
        progress = float(epoch_index - warmup_epochs) / float(cosine_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    optimized: bool,
    learning_rate: float,
    weight_decay: float,
    warmup_epochs: int,
    total_epochs: int,
):
    if optimized:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            min(warmup_epochs, total_epochs),
            total_epochs,
        )
        return optimizer, scheduler
    return torch.optim.Adam(model.parameters(), lr=learning_rate), None


def validate_arguments(args: argparse.Namespace, context: DistributedContext) -> int:
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")
    if args.lr <= 0:
        raise ValueError("--lr must be greater than zero.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.suite_name is not None and args.run_name is not None:
        raise ValueError("--run-name cannot be combined with --suite-name.")
    return validate_global_batch_size(args.batch_size, context.world_size)


def checkpoint_payload(
    model: torch.nn.Module,
    config: Dict[str, Any],
    args: argparse.Namespace,
    epoch: int,
    validation_metrics: Dict[str, Any],
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task": args.task,
        "variant": getattr(args, "variant", A1_ORIGINAL),
        "epoch": epoch,
        "monitor": args.monitor,
        "best_metric": float(validation_metrics[args.monitor]),
        "threshold": float(validation_metrics["threshold"]),
        "model_config": dict(config),
        "train_args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"local_rank"}
        },
        "validation_metrics": dict(validation_metrics),
        "runtime": dict(runtime),
        "model_state_dict": model_state_dict_cpu(model),
    }


def history_row(
    epoch: int,
    train_loss: float,
    validation_metrics: Dict[str, Any],
    epoch_seconds: float,
    is_best: bool,
    learning_rate: float | None = None,
    early_stopping_counter: int | None = None,
    amp_fp32_fallback_batches: int = 0,
) -> Dict[str, Any]:
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "epoch_seconds": epoch_seconds,
        "is_best": is_best,
        "amp_fp32_fallback_batches": amp_fp32_fallback_batches,
    }
    if learning_rate is not None:
        row["learning_rate"] = learning_rate
    if early_stopping_counter is not None:
        row["early_stopping_counter"] = early_stopping_counter
    row.update({f"val_{key}": value for key, value in validation_metrics.items()})
    return row


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_training(
    args: argparse.Namespace,
    context: DistributedContext,
    paths: RunPaths,
    logger,
) -> Dict[str, Any]:
    local_batch_size = validate_arguments(args, context)
    seed_everything(context, args.seed)
    runtime = runtime_metadata(context)
    variant_spec = get_variant_spec(args.variant)
    settings = resolve_training_settings(args.variant, args.task)

    config = build_model_config(BIN_config_DBPE(), args.variant, args.task, settings)
    config["batch_size"] = args.batch_size

    train_frame = load_split(args.task, "train")
    validation_frame = load_split(args.task, "val")
    test_frame = load_split(args.task, "test")
    train_loader, train_sampler = build_train_loader(
        train_frame, local_batch_size, args.workers, context, seed=args.seed
    )
    validation_loader = build_eval_loader(
        validation_frame, local_batch_size, args.workers, context
    )
    test_loader = build_eval_loader(test_frame, local_batch_size, args.workers, context)

    model = build_moltrans_model(config, args.variant).to(context.device)
    if context.is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=True,
        )
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        settings.optimized,
        args.lr,
        settings.weight_decay,
        settings.warmup_epochs,
        args.epochs,
    )
    loss_function = LabelSmoothingBCEWithLogitsLoss(settings.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(
        enabled=settings.use_amp and context.device.type == "cuda"
    )
    torch.backends.cudnn.benchmark = True

    logger.info(
        "run_id=%s variant=%s task=%s seed=%d monitor=%s",
        paths.run_id,
        args.variant,
        args.task,
        args.seed,
        args.monitor,
    )
    logger.info(
        "world_size=%d global_batch_size=%d local_batch_size=%d learning_rate=%g",
        context.world_size,
        args.batch_size,
        local_batch_size,
        args.lr,
    )
    logger.info(
        "training_strategy optimizer=%s weight_decay=%g scheduler=%s warmup_epochs=%d "
        "amp=%s label_smoothing=%g max_grad_norm=%s patience=%s",
        type(optimizer).__name__,
        settings.weight_decay,
        type(scheduler).__name__ if scheduler is not None else "none",
        settings.warmup_epochs,
        settings.use_amp,
        settings.label_smoothing,
        settings.max_grad_norm,
        settings.patience,
    )
    logger.info("variant_components=%s", variant_spec)
    logger.info(
        "dataset_sizes train=%d validation=%d test=%d",
        len(train_frame),
        len(validation_frame),
        len(test_frame),
    )
    logger.info("runtime=%s", runtime)
    logger.info("checkpoint=%s", paths.checkpoint)

    history = []
    best_metric = float("-inf")
    early_stopping_counter = 0
    stopped_early = False
    epochs_completed = 0
    args.amp_fp32_fallback_batches_total = 0
    training_started = time.perf_counter()

    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_index)
        epoch_started = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        recovery_stats = {"fp32_fallback_batches": 0}
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            context,
            logger,
            loss_function=loss_function,
            scaler=scaler,
            use_amp=settings.use_amp,
            max_grad_norm=settings.max_grad_norm,
            recovery_stats=recovery_stats,
        )
        epoch_fp32_fallback_batches = recovery_stats["fp32_fallback_batches"]
        args.amp_fp32_fallback_batches_total += epoch_fp32_fallback_batches
        validation_metrics, _ = evaluate_model(model, validation_loader, context)
        epoch_seconds = distributed_max(time.perf_counter() - epoch_started, context)
        monitored_value = float(validation_metrics[args.monitor])
        is_best = monitored_value > best_metric

        if is_best:
            best_metric = monitored_value
            early_stopping_counter = 0
            if context.is_main:
                payload = checkpoint_payload(
                    model,
                    config,
                    args,
                    epoch,
                    validation_metrics,
                    runtime,
                )
                atomic_torch_save(payload, paths.checkpoint)
                logger.info(
                    "saved best checkpoint epoch=%d %s=%.6f path=%s",
                    epoch,
                    args.monitor,
                    monitored_value,
                    paths.checkpoint,
                )
            distributed_barrier(context)
        else:
            early_stopping_counter += 1

        row = history_row(
            epoch,
            train_loss,
            validation_metrics,
            epoch_seconds,
            is_best,
            learning_rate=learning_rate,
            early_stopping_counter=early_stopping_counter,
            amp_fp32_fallback_batches=epoch_fp32_fallback_batches,
        )
        if context.is_main:
            history.append(row)
            write_csv(paths.history, pd.DataFrame(history))
            logger.info(
                "epoch=%d/%d lr=%.8g train_loss=%.6f %s seconds=%.2f best=%s "
                "early_stopping_counter=%d amp_fp32_fallback_batches=%d",
                epoch,
                args.epochs,
                learning_rate,
                train_loss,
                format_metrics(validation_metrics),
                epoch_seconds,
                is_best,
                early_stopping_counter,
                epoch_fp32_fallback_batches,
            )
        epochs_completed = epoch
        if scheduler is not None:
            scheduler.step()
        if settings.patience is not None and early_stopping_counter >= settings.patience:
            stopped_early = True
            logger.info(
                "early stopping at epoch=%d after %d epochs without strict %s improvement",
                epoch,
                early_stopping_counter,
                args.monitor,
            )
            break

    distributed_barrier(context)
    if not paths.checkpoint.exists():
        raise RuntimeError("Training completed without producing a best checkpoint.")

    best_checkpoint = load_checkpoint(paths.checkpoint, torch.device("cpu"))
    load_model_state(model, best_checkpoint)
    best_checkpoint.pop("model_state_dict")
    distributed_barrier(context)
    test_metrics, test_predictions = evaluate_model(
        model,
        test_loader,
        context,
        threshold=float(best_checkpoint["threshold"]),
    )
    total_seconds = distributed_max(time.perf_counter() - training_started, context)

    summary = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": paths.run_id,
        "suite_name": args.suite_name,
        "variant": args.variant,
        "task": args.task,
        "seed": args.seed,
        "monitor": args.monitor,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_metric": float(best_checkpoint["best_metric"]),
        "best_validation": best_checkpoint["validation_metrics"],
        "test": test_metrics,
        "checkpoint": _relative(paths.checkpoint),
        "history": _relative(paths.history),
        "predictions": _relative(paths.predictions),
        "picture": _relative(paths.picture),
        "epochs": args.epochs,
        "epochs_requested": args.epochs,
        "epochs_completed": epochs_completed,
        "stopped_early": stopped_early,
        "global_batch_size": args.batch_size,
        "local_batch_size": local_batch_size,
        "world_size": context.world_size,
        "learning_rate": args.lr,
        "optimizer": type(optimizer).__name__,
        "training_settings": dict(args.training_settings),
        "amp_fp32_fallback_batches_total": args.amp_fp32_fallback_batches_total,
        "duration_seconds": total_seconds,
        "runtime": runtime,
    }
    if context.is_main:
        if test_predictions is None:
            raise RuntimeError("Rank 0 did not receive test predictions.")
        write_csv(paths.predictions, test_predictions)
        history_frame = pd.DataFrame(history)
        write_json(paths.summary, summary)
        save_history_plot(history_frame, paths.picture)
        logger.info("test %s", format_metrics(test_metrics))
        logger.info("completed duration_seconds=%.2f summary=%s", total_seconds, paths.summary)
    distributed_barrier(context)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    args.task = normalize_task_name(args.task)
    args = apply_variant_defaults(args)
    context = None
    logger = None
    try:
        context = init_distributed()
        validate_arguments(args, context)
        paths = create_run_paths(
            args.task,
            args.run_name,
            context,
            suite_name=args.suite_name,
            seed=args.seed,
            variant=args.variant if args.variant != A1_ORIGINAL else None,
        )
        logger = setup_logger("moltrans.train", paths.log, context)
        run_training(args, context, paths, logger)
    except Exception:
        if logger is not None:
            logger.exception("training failed")
        raise
    finally:
        if context is not None:
            cleanup_distributed(context)


if __name__ == "__main__":
    main()
