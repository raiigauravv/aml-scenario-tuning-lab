"""Tests for Phase 6 ML alert prioritization (prioritize.py).

Three things the task explicitly asked to guard: no test-period rows
leak into training, the saved metrics JSON has the expected schema, and
the whole pipeline is seed-reproducible (byte-identical on rerun - this
project already hit a real non-determinism bug here: DuckDB doesn't
guarantee row order without ORDER BY, which silently shuffled the
alert table, and therefore the bootstrap resample indices and XGBoost's
histogram-building order, between runs despite every seed being fixed).
"""

import copy
import json

import pytest

from aml_lab.features import compute_features_from_frames
from aml_lab.generate import generate_dataset, load_config
from aml_lab.prioritize import run_prioritization
from aml_lab.scenarios import build_context_from_frames, load_scenarios_config
from aml_lab.segment import (
    build_customer_profile,
    fit_segments,
    load_segments_config,
    transform_features,
)
from aml_lab.tuning import load_capacity_config, sweep_scenario


@pytest.fixture(scope="module")
def pipeline_context(tmp_path_factory):
    gen_config = copy.deepcopy(load_config())
    scenarios_config = load_scenarios_config()
    capacity_cfg = load_capacity_config()
    data = generate_dataset(gen_config, n_customers=4000)
    features = compute_features_from_frames(data, gen_config)

    seg_config = load_segments_config()
    profile = build_customer_profile(features, data["accounts"], seg_config)
    X = transform_features(profile, seg_config)
    seg_labels = fit_segments(X, seg_config["k"], seg_config["seed"])
    segments = profile.reset_index()[["customer_id"]]
    segments["segment_assigned"] = seg_labels

    metrics_dir = tmp_path_factory.mktemp("metrics")
    con = build_context_from_frames(data, gen_config)
    for scenario_id, spec in scenarios_config.items():
        result = sweep_scenario(scenario_id, con, spec, gen_config, segments, data["labels"], capacity_cfg)
        with open(metrics_dir / f"tuning_{scenario_id}.json", "w") as f:
            json.dump(result, f, default=str)

    return {
        "gen_config": gen_config,
        "scenarios_config": scenarios_config,
        "segments": segments,
        "data": data,
        "features": features,
        "metrics_dir": metrics_dir,
    }


@pytest.fixture(scope="module")
def prioritization_out(pipeline_context):
    con = build_context_from_frames(pipeline_context["data"], pipeline_context["gen_config"])
    return run_prioritization(
        con,
        pipeline_context["gen_config"],
        pipeline_context["scenarios_config"],
        pipeline_context["segments"],
        pipeline_context["data"]["labels"],
        pipeline_context["features"],
        pipeline_context["metrics_dir"],
    )


def test_no_test_period_rows_in_training(prioritization_out):
    train_df = prioritization_out["train_df"]
    test_df = prioritization_out["test_df"]
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert (train_df["period"] != "test").all()
    assert set(train_df["period"].unique()) <= {"tuning", "validation"}
    assert (test_df["period"] == "test").all()
    # No customer-month-scenario triple appears in both splits.
    train_keys = set(zip(train_df["customer_id"], train_df["month"], train_df["scenario_id"]))
    test_keys = set(zip(test_df["customer_id"], test_df["month"], test_df["scenario_id"]))
    assert train_keys.isdisjoint(test_keys)


def test_metrics_json_schema(prioritization_out):
    r = prioritization_out["result"]
    assert set(r.keys()) == {
        "seed", "train", "test", "pr_auc", "review_budget", "by_scenario", "model_params", "rules",
    }
    assert isinstance(r["seed"], int)

    for split in ("train", "test"):
        assert set(r[split].keys()) == {"n_alerts", "n_positives"}
        assert isinstance(r[split]["n_alerts"], int)
        assert isinstance(r[split]["n_positives"], int)
        assert r[split]["n_positives"] <= r[split]["n_alerts"]

    assert set(r["pr_auc"].keys()) == {"xgboost", "logistic_regression", "xgboost_beats_baseline"}
    for model_key in ("xgboost", "logistic_regression"):
        m = r["pr_auc"][model_key]
        assert set(m.keys()) == {"mean", "ci_lower", "ci_upper", "n_bootstrap_samples_used"}
        assert 0.0 <= m["ci_lower"] <= m["mean"] <= m["ci_upper"] <= 1.0
    assert isinstance(r["pr_auc"]["xgboost_beats_baseline"], bool)

    assert set(r["review_budget"].keys()) == {"top_20pct", "top_40pct", "top_60pct"}
    for budget_stats in r["review_budget"].values():
        assert set(budget_stats.keys()) == {
            "n_reviewed", "ml_productive_captured", "fifo_productive_captured",
            "random_productive_captured_mean", "fps_avoided_vs_fifo_at_equal_detection",
        }
        assert isinstance(budget_stats["n_reviewed"], int)
        assert isinstance(budget_stats["ml_productive_captured"], int)
        # The exact type bug this guards: fps_avoided used to serialize as
        # a numpy int64 -> str() -> a JSON string like "38" instead of 38.
        if budget_stats["fps_avoided_vs_fifo_at_equal_detection"] is not None:
            assert isinstance(budget_stats["fps_avoided_vs_fifo_at_equal_detection"], int)

    for stats in r["by_scenario"].values():
        assert "n_alerts" in stats and "n_positives" in stats and "pr_auc" in stats
        if stats["pr_auc"] is not None:
            assert 0.0 <= stats["pr_auc"] <= 1.0

    assert "xgboost" in r["model_params"] and "logistic_regression" in r["model_params"]
    assert isinstance(r["rules"], list) and len(r["rules"]) >= 2


def test_seed_reproducibility(pipeline_context, prioritization_out):
    con2 = build_context_from_frames(pipeline_context["data"], pipeline_context["gen_config"])
    rerun = run_prioritization(
        con2,
        pipeline_context["gen_config"],
        pipeline_context["scenarios_config"],
        pipeline_context["segments"],
        pipeline_context["data"]["labels"],
        pipeline_context["features"],
        pipeline_context["metrics_dir"],
    )
    assert rerun["result"]["pr_auc"] == prioritization_out["result"]["pr_auc"]
    assert rerun["result"]["review_budget"] == prioritization_out["result"]["review_budget"]
    assert rerun["result"]["by_scenario"] == prioritization_out["result"]["by_scenario"]
    assert rerun["test_df"]["xgb_score"].tolist() == prioritization_out["test_df"]["xgb_score"].tolist()
