from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from ablation import checkpoint_variant
from experiment import (
    REPO_ROOT,
    build_eval_loader,
    cleanup_distributed,
    distributed_barrier,
    evaluate_model,
    format_metrics,
    init_distributed,
    load_checkpoint,
    load_model_state,
    load_split,
    normalize_task_name,
    runtime_metadata,
    seed_everything,
    setup_logger,
    validate_global_batch_size,
    write_csv,
    write_json,
)
from improved_models import build_moltrans_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a MolTrans best checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="path to best_checkpoint.pt")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="global evaluation batch size; defaults to the checkpoint training batch size",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="data loading workers per process (default: 0)",
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return parser


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = build_parser().parse_args()
    context = None
    logger = None
    try:
        context = init_distributed()
        checkpoint_path = args.checkpoint.expanduser().resolve()
        checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
        variant = checkpoint_variant(checkpoint)
        seed = int(checkpoint.get("train_args", {}).get("seed", 1))
        seed_everything(context, seed)
        checkpoint_batch_size = int(checkpoint.get("train_args", {}).get("batch_size", 16))
        global_batch_size = args.batch_size or checkpoint_batch_size
        local_batch_size = validate_global_batch_size(global_batch_size, context.world_size)

        evaluation_id = f"evaluation_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        try:
            run_path = checkpoint_path.parent.relative_to(REPO_ROOT / "results")
        except ValueError:
            run_path = Path("external") / checkpoint_path.parent.name
        evaluation_dir = checkpoint_path.parent / "evaluations" / evaluation_id
        log_path = REPO_ROOT / "log" / run_path / "evaluations" / f"{evaluation_id}.log"
        logger = setup_logger("moltrans.evaluate", log_path, context)

        task = normalize_task_name(str(checkpoint["task"]))
        frame = load_split(task, args.split)
        loader = build_eval_loader(frame, local_batch_size, args.workers, context)
        model = build_moltrans_model(checkpoint["model_config"], variant).to(context.device)
        load_model_state(model, checkpoint)
        checkpoint.pop("model_state_dict")

        metrics, predictions = evaluate_model(
            model,
            loader,
            context,
            threshold=float(checkpoint["threshold"]),
        )
        metrics_path = evaluation_dir / "metrics.json"
        predictions_path = evaluation_dir / "predictions.csv"
        payload = {
            "checkpoint": _relative(checkpoint_path),
            "task": task,
            "variant": variant,
            "seed": seed,
            "split": args.split,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "threshold_source": "checkpoint_validation",
            "metrics": metrics,
            "global_batch_size": global_batch_size,
            "local_batch_size": local_batch_size,
            "world_size": context.world_size,
            "runtime": runtime_metadata(context),
        }

        if context.is_main:
            if predictions is None:
                raise RuntimeError("Rank 0 did not receive evaluation predictions.")
            write_json(metrics_path, payload)
            write_csv(predictions_path, predictions)
            logger.info(
                "checkpoint=%s variant=%s task=%s split=%s world_size=%d "
                "global_batch_size=%d local_batch_size=%d",
                checkpoint_path,
                variant,
                task,
                args.split,
                context.world_size,
                global_batch_size,
                local_batch_size,
            )
            logger.info("evaluation %s", format_metrics(metrics))
            logger.info("metrics=%s predictions=%s", metrics_path, predictions_path)
        distributed_barrier(context)
    except Exception:
        if logger is not None:
            logger.exception("evaluation failed")
        raise
    finally:
        if context is not None:
            cleanup_distributed(context)


if __name__ == "__main__":
    main()
