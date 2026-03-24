from __future__ import annotations

from llmfe.classification_aggregate import aggregate_sample_metrics, log_aggregate_run


class _FakeRun:
    def __init__(self, run_id: str):
        self.id = run_id
        self.summary: dict[str, float | int | str] = {}


class _FakeTable:
    def __init__(self, columns):
        self.columns = list(columns)
        self.rows = []

    def add_data(self, *args):
        self.rows.append(tuple(args))


class _FakePlot:
    @staticmethod
    def confusion_matrix(*, y_true, preds, class_names, title):
        return {
            "kind": "confusion_matrix",
            "n": len(y_true),
            "class_names": class_names,
            "title": title,
            "pred_len": len(preds),
        }


class _FakeWandb:
    def __init__(self):
        self.init_calls = []
        self.define_metric_calls = []
        self.log_calls = []
        self.finish_calls = 0
        self.plot = _FakePlot()
        self.run = None

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.run = _FakeRun(run_id=f"fake-{len(self.init_calls)}")
        return self.run

    def define_metric(self, *args, **kwargs):
        self.define_metric_calls.append((args, kwargs))

    def Table(self, columns):  # noqa: N802 - mimics wandb API
        return _FakeTable(columns)

    def log(self, data, step=None):
        self.log_calls.append({"data": data, "step": step})

    def finish(self):
        self.finish_calls += 1
        self.run = None


def test_aggregate_sample_metrics_weighted_and_confusion_sum():
    rows = [
        {
            "global_step": 1,
            "split_id": 1,
            "accuracy": 0.8,
            "f1": 0.7,
            "n_eval_points": 10,
            "tn": 3,
            "fp": 1,
            "fn": 2,
            "tp": 4,
        },
        {
            "global_step": 1,
            "split_id": 2,
            "accuracy": 0.5,
            "f1": 0.4,
            "n_eval_points": 30,
            "tn": 11,
            "fp": 3,
            "fn": 6,
            "tp": 10,
        },
        {
            "global_step": 2,
            "split_id": 1,
            "accuracy": 0.9,
            "f1": 0.85,
            "n_eval_points": 20,
            "tn": 6,
            "fp": 2,
            "fn": 1,
            "tp": 11,
        },
    ]

    out = aggregate_sample_metrics(rows)
    assert [row["global_step"] for row in out] == [1, 2]

    first = out[0]
    assert first["accuracy"] == (0.8 * 10 + 0.5 * 30) / 40
    assert first["f1"] == (0.7 * 10 + 0.4 * 30) / 40
    assert first["n_eval_points"] == 40
    assert first["num_splits_contributing"] == 2
    assert first["tn"] == 14
    assert first["fp"] == 4
    assert first["fn"] == 8
    assert first["tp"] == 14

    second = out[1]
    assert second["n_eval_points"] == 20
    assert second["num_splits_contributing"] == 1


def test_aggregate_sample_metrics_skips_invalid_rows_and_missing_steps():
    rows = [
        {
            "global_step": 3,
            "split_id": 1,
            "accuracy": 0.77,
            "f1": 0.73,
            "n_eval_points": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        },
        {
            "global_step": 4,
            "split_id": 1,
            "accuracy": 0.81,
            "f1": 0.79,
            "n_eval_points": 25,
            "tn": 10,
            "fp": 2,
            "fn": 3,
            "tp": 10,
        },
    ]

    out = aggregate_sample_metrics(rows)
    assert len(out) == 1
    assert out[0]["global_step"] == 4


def test_log_aggregate_run_logs_expected_series_and_table():
    fake_wandb = _FakeWandb()
    rows = [
        {
            "global_step": 1,
            "accuracy": 0.61,
            "f1": 0.58,
            "n_eval_points": 40,
            "num_splits_contributing": 2,
            "tn": 13,
            "fp": 5,
            "fn": 7,
            "tp": 15,
        },
        {
            "global_step": 2,
            "accuracy": 0.64,
            "f1": 0.6,
            "n_eval_points": 38,
            "num_splits_contributing": 2,
            "tn": 12,
            "fp": 6,
            "fn": 6,
            "tp": 14,
        },
    ]

    run_id = log_aggregate_run(
        wandb_api=fake_wandb,
        project="proj",
        group="grp",
        run_name="aggregate_run",
        run_config={"problem_name": "btc-classification"},
        rows=rows,
    )

    assert run_id == "fake-1"
    assert len(fake_wandb.init_calls) == 1
    assert fake_wandb.finish_calls == 1
    assert fake_wandb.init_calls[0]["tags"] == ["llmfe_classification_aggregate"]
    assert (
        fake_wandb.init_calls[0]["config"]["run_type"]
        == "llmfe_classification_aggregate"
    )

    step_logs = [entry for entry in fake_wandb.log_calls if entry["step"] is not None]
    assert len(step_logs) == 2
    for entry in step_logs:
        payload = entry["data"]
        assert "Aggregate/Accuracy" in payload
        assert "Aggregate/F1" in payload
        assert "Aggregate/Confusion_Matrix" in payload

    table_logs = [
        entry
        for entry in fake_wandb.log_calls
        if "Aggregate/Per_Sample_Table" in entry["data"]
    ]
    assert len(table_logs) == 1
