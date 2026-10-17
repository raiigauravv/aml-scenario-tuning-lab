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

import pandas as pd
import pytest

from aml_lab.features import compute_features_from_frames
from aml_lab.generate import generate_dataset, load_config
from aml_lab.prioritize import review_budget_analysis, run_prioritization
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
        "seed", "train", "test", "pr_auc", "scenario_prior_train_rates", "review_budget",
        "by_scenario", "productive_alerts_outside_top_60pct", "model_params", "rules",
    }
    assert isinstance(r["seed"], int)

    for split in ("train", "test"):
        assert set(r[split].keys()) == {"n_alerts", "n_positives"}
        assert isinstance(r[split]["n_alerts"], int)
        assert isinstance(r[split]["n_positives"], int)
        assert r[split]["n_positives"] <= r[split]["n_alerts"]

    assert set(r["pr_auc"].keys()) == {"xgboost", "logistic_regression", "scenario_prior", "xgboost_beats_baseline"}
    for model_key in ("xgboost", "logistic_regression", "scenario_prior"):
        m = r["pr_auc"][model_key]
        assert set(m.keys()) == {"mean", "ci_lower", "ci_upper", "n_bootstrap_samples_used"}
        assert 0.0 <= m["ci_lower"] <= m["mean"] <= m["ci_upper"] <= 1.0
    assert isinstance(r["pr_auc"]["xgboost_beats_baseline"], bool)

    # Computed from TRAIN scenario groups, so its keys need not match
    # by_scenario's TEST-derived keys exactly (a scenario can have train
    # alerts with no test alerts, or vice versa, especially at small n).
    assert isinstance(r["scenario_prior_train_rates"], dict)
    assert len(r["scenario_prior_train_rates"]) > 0
    for rate in r["scenario_prior_train_rates"].values():
        assert 0.0 <= rate <= 1.0

    assert set(r["review_budget"].keys()) == {"top_20pct", "top_40pct", "top_60pct"}
    for budget_stats in r["review_budget"].values():
        assert set(budget_stats.keys()) == {
            "n_reviewed", "productive_captured", "fifo_alerts_needed_to_match_xgboost",
            "fps_avoided_vs_fifo_at_equal_detection", "random_alerts_needed_to_match_xgboost_mean",
            "fps_avoided_vs_random_at_equal_detection",
        }
        assert isinstance(budget_stats["n_reviewed"], int)
        assert set(budget_stats["productive_captured"].keys()) == {
            "xgboost", "fifo", "random_mean", "logistic_regression", "scenario_prior",
        }
        assert isinstance(budget_stats["productive_captured"]["xgboost"], int)
        # The exact type bug this guarded originally: fps_avoided used to
        # serialize as a numpy int64 -> str() -> "38" instead of 38.
        if budget_stats["fps_avoided_vs_fifo_at_equal_detection"] is not None:
            assert isinstance(budget_stats["fps_avoided_vs_fifo_at_equal_detection"], int)
        # Equal-detection sanity: FIFO/random need AT LEAST as many alerts
        # as XGBoost reviewed to match XGBoost's own capture count, so the
        # avoided counts should never be negative.
        if budget_stats["fps_avoided_vs_fifo_at_equal_detection"] is not None:
            assert budget_stats["fps_avoided_vs_fifo_at_equal_detection"] >= 0
        if budget_stats["fps_avoided_vs_random_at_equal_detection"] is not None:
            assert budget_stats["fps_avoided_vs_random_at_equal_detection"] >= -1e-6

    for stats in r["by_scenario"].values():
        assert "n_alerts" in stats and "n_positives" in stats and "pr_auc" in stats
        if stats["pr_auc"] is not None:
            assert 0.0 <= stats["pr_auc"] <= 1.0

    assert isinstance(r["productive_alerts_outside_top_60pct"], list)
    k_60 = max(1, round(r["test"]["n_alerts"] * 0.60))
    for item in r["productive_alerts_outside_top_60pct"]:
        assert set(item.keys()) == {"rank", "customer_id", "month", "scenario_id", "xgb_score"}
        assert item["rank"] > k_60

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


# ---------------------------------------------------------------------------
# Hand-built check of the corrected equal-detection metric (Phase 6 fix 1).
#
# 10 alerts, FIFO order = alphabetical customer_id (A..J), 3 productive
# ones placed at FIFO positions 4, 7, 10 (D, G, J). The "ML" score ranks
# all 3 productive alerts into the top 3 positions. At budget k=3, ML
# captures all 3 productive alerts; FIFO needs to review all 10 alerts to
# find the same 3 (the last one, J, is FIFO's very last alert) - so FIFO
# needs 10 alerts where ML needed 3: fps_avoided_vs_fifo = 10 - 3 = 7.
# This is the exact direction the original (buggy) metric got backwards.
# ---------------------------------------------------------------------------

def test_review_budget_equal_detection_metric_is_correctly_directioned():
    customer_ids = list("ABCDEFGHIJ")
    is_productive = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]  # D, G, J productive
    # ML score ranks D, G, J (the productive ones) into the top 3.
    ml_score = {"D": 0.9, "G": 0.8, "J": 0.7, "A": 0.6, "B": 0.5, "C": 0.4, "E": 0.3, "F": 0.2, "H": 0.1, "I": 0.05}
    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "month": [1] * 10,
            "is_productive": is_productive,
            "xgb_score": [ml_score[c] for c in customer_ids],
        }
    )

    results, order_primary = review_budget_analysis(df, "xgb_score", budgets=(0.3,), n_shuffles=50, seed=1)
    stats = results["top_30pct"]

    assert stats["n_reviewed"] == 3
    assert stats["productive_captured"]["xgboost"] == 3  # all 3 productive alerts in ML's top 3
    assert stats["productive_captured"]["fifo"] == 0  # FIFO's first 3 (A,B,C) are all negative
    assert stats["fifo_alerts_needed_to_match_xgboost"] == 10  # J (the last productive) is FIFO's last alert
    assert stats["fps_avoided_vs_fifo_at_equal_detection"] == 7  # 10 - 3

    # Random order needs, on average, noticeably more than 3 alerts too
    # (strictly bounded by n=10) - not an exact value since it's
    # shuffle-based, but the direction and bounds must hold.
    assert stats["random_alerts_needed_to_match_xgboost_mean"] is not None
    assert 3 < stats["random_alerts_needed_to_match_xgboost_mean"] <= 10
    assert stats["fps_avoided_vs_random_at_equal_detection"] > 0

    assert order_primary["customer_id"].iloc[0] == "D"
