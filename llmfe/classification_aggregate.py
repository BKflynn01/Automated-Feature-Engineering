"""Aggregation and W&B logging helpers for classification sample-order comparisons."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

_REQUIRED_INPUT_KEYS = (
    "global_step",
    "split_id",
    "accuracy",
    "f1",
    "n_eval_points",
    "tn",
    "fp",
    "fn",
    "tp",
)


def aggregate_sample_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-split sample metrics into weighted sample-order metrics."""
    accum: dict[int, dict[str, Any]] = {}

    for row in rows:
        if any(k not in row for k in _REQUIRED_INPUT_KEYS):
            continue
        try:
            step = int(row["global_step"])
            split_id = int(row["split_id"])
            n_eval_points = int(row["n_eval_points"])
            accuracy = float(row["accuracy"])
            f1 = float(row["f1"])
            tn = int(row["tn"])
            fp = int(row["fp"])
            fn = int(row["fn"])
            tp = int(row["tp"])
        except (TypeError, ValueError):
            continue

        if n_eval_points <= 0:
            continue

        entry = accum.setdefault(
            step,
            {
                "global_step": step,
                "weighted_accuracy_sum": 0.0,
                "weighted_f1_sum": 0.0,
                "total_n_eval_points": 0,
                "split_ids": set(),
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
            },
        )
        entry["weighted_accuracy_sum"] += accuracy * n_eval_points
        entry["weighted_f1_sum"] += f1 * n_eval_points
        entry["total_n_eval_points"] += n_eval_points
        entry["split_ids"].add(split_id)
        entry["tn"] += tn
        entry["fp"] += fp
        entry["fn"] += fn
        entry["tp"] += tp

    aggregated_rows: list[dict[str, Any]] = []
    for step in sorted(accum):
        entry = accum[step]
        total_n_eval_points = int(entry["total_n_eval_points"])
        if total_n_eval_points <= 0:
            continue
        aggregated_rows.append(
            {
                "global_step": int(entry["global_step"]),
                "accuracy": float(entry["weighted_accuracy_sum"] / total_n_eval_points),
                "f1": float(entry["weighted_f1_sum"] / total_n_eval_points),
                "n_eval_points": total_n_eval_points,
                "num_splits_contributing": int(len(entry["split_ids"])),
                "tn": int(entry["tn"]),
                "fp": int(entry["fp"]),
                "fn": int(entry["fn"]),
                "tp": int(entry["tp"]),
            }
        )
    return aggregated_rows


def _build_confusion_payload(
    *, wandb_api: Any, tn: int, fp: int, fn: int, tp: int
) -> Any:
    """Build a confusion-matrix visual when possible, otherwise a count table."""
    total = int(tn + fp + fn + tp)
    if total <= 0:
        return None

    if (
        total <= 10000
        and hasattr(wandb_api, "plot")
        and hasattr(wandb_api.plot, "confusion_matrix")
    ):
        y_true = ["down"] * tn + ["down"] * fp + ["up"] * fn + ["up"] * tp
        y_pred = ["down"] * tn + ["up"] * fp + ["down"] * fn + ["up"] * tp
        try:
            return wandb_api.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=["down", "up"],
                title="Aggregate Confusion Matrix",
            )
        except Exception:
            pass

    cm_table = wandb_api.Table(columns=["actual", "predicted", "count"])
    cm_table.add_data("down (0)", "down (0)", int(tn))
    cm_table.add_data("down (0)", "up (1)", int(fp))
    cm_table.add_data("up (1)", "down (0)", int(fn))
    cm_table.add_data("up (1)", "up (1)", int(tp))
    return cm_table


def log_aggregate_run(
    *,
    wandb_api: Any,
    project: str,
    group: str | None,
    run_name: str,
    run_config: Mapping[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
    run_type: str = "llmfe_classification_aggregate",
) -> str | None:
    """Log aggregated sample-order metrics into a dedicated W&B run."""
    if not rows:
        return None

    config_payload = dict(run_config or {})
    config_payload["run_type"] = run_type

    wandb_api.init(
        project=project,
        group=group,
        name=run_name,
        config=config_payload,
        tags=[run_type],
        reinit=True,
    )
    wandb_api.define_metric("global_step")
    wandb_api.define_metric("*", step_metric="global_step")
    if getattr(wandb_api, "run", None) is not None:
        wandb_api.run.summary["run_type"] = run_type

    summary_table = wandb_api.Table(
        columns=[
            "global_step",
            "accuracy",
            "f1",
            "num_splits_contributing",
            "n_eval_points",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    )

    for row in rows:
        global_step = int(row["global_step"])
        accuracy = float(row["accuracy"])
        f1 = float(row["f1"])
        num_splits_contributing = int(row["num_splits_contributing"])
        n_eval_points = int(row["n_eval_points"])
        tn = int(row["tn"])
        fp = int(row["fp"])
        fn = int(row["fn"])
        tp = int(row["tp"])

        log_data: dict[str, Any] = {
            "global_step": global_step,
            "Aggregate/Accuracy": accuracy,
            "Aggregate/F1": f1,
            "Aggregate/num_splits_contributing": num_splits_contributing,
            "Aggregate/n_eval_points": n_eval_points,
        }

        confusion_payload = _build_confusion_payload(
            wandb_api=wandb_api, tn=tn, fp=fp, fn=fn, tp=tp
        )
        if confusion_payload is not None:
            log_data["Aggregate/Confusion_Matrix"] = confusion_payload

        wandb_api.log(log_data, step=global_step)
        summary_table.add_data(
            global_step,
            accuracy,
            f1,
            num_splits_contributing,
            n_eval_points,
            tn,
            fp,
            fn,
            tp,
        )

    wandb_api.log({"Aggregate/Per_Sample_Table": summary_table})
    if getattr(wandb_api, "run", None) is not None:
        wandb_api.run.summary["aggregate_num_steps"] = int(len(rows))
    run_id = getattr(getattr(wandb_api, "run", None), "id", None)
    wandb_api.finish()
    return run_id
