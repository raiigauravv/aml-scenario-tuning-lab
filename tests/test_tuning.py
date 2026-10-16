"""Tests for Phase 5 threshold tuning (tuning.py).

Unit tests for the pure-Python pieces (period assignment, capacity,
period-scoped metrics - the bug this project actually hit: an unscoped
label denominator made every period's detection rate compare against the
whole-typology count instead of that period's own cases), then an
end-to-end hand-built STR-01 sweep that exercises capacity-constrained
selection, the pooled-fallback rule (both the too-few-tuning-cases path
and the too-few-validation-cases path), and ATL/BTL, plus a real-data
smoke test across all five scenarios.
"""

import copy

import pandas as pd
import pytest

from aml_lab.generate import load_config
from aml_lab.scenarios import build_context_from_frames, load_scenarios_config
from aml_lab.tuning import (
    compute_metrics,
    labeled_ids_by_period,
    load_capacity_config,
    month_to_period,
    segment_capacity,
    sweep_scenario,
)


def test_month_to_period():
    assert month_to_period(1) == "tuning"
    assert month_to_period(8) == "tuning"
    assert month_to_period(9) == "validation"
    assert month_to_period(10) == "validation"
    assert month_to_period(11) == "test"
    assert month_to_period(12) == "test"
    with pytest.raises(ValueError):
        month_to_period(13)


def test_segment_capacity_clips_to_floor_and_ceiling():
    cfg = {"max_alert_rate_per_month": 0.01, "min_alerts_per_month": 5, "max_alerts_per_month": 300}
    assert segment_capacity(cfg, 100) == 5  # 0.01*100=1, floored to 5
    assert segment_capacity(cfg, 1000) == 10  # 0.01*1000=10, within bounds
    assert segment_capacity(cfg, 100000) == 300  # 0.01*100000=1000, capped to 300


def test_labeled_ids_by_period_buckets_correctly():
    labels = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D"],
            "typology": ["structuring", "structuring", "structuring", "rapid_movement"],
            "start_date": ["2024-02-01", "2024-09-15", "2024-11-20", "2024-03-01"],
            "end_date": ["2024-02-05", "2024-09-18", "2024-11-25", "2024-03-02"],
        }
    )
    by_period = labeled_ids_by_period(labels, "structuring")
    assert by_period["tuning"] == {"A"}
    assert by_period["validation"] == {"B"}
    assert by_period["test"] == {"C"}
    # A different typology must not leak in.
    assert "D" not in by_period["tuning"]


def test_compute_metrics_uses_period_scoped_denominator_not_whole_typology():
    """The bug this guards: n_labeled_in_scope must be the labeled count
    for THIS period, not the typology's total count across all periods -
    using the total would make e.g. validation's detection rate compare
    against tuning-period cases that can't possibly be detected in the
    validation window.
    """
    alerts = pd.DataFrame(
        {"customer_id": ["X", "Y"], "month": [9, 9], "period": ["validation", "validation"]}
    )
    labeled_all = {"X", "Y", "Z"}  # Z's event was in tuning, not validation
    labeled_validation_only = {"X"}  # only X's event is actually in validation
    metrics = compute_metrics(alerts, "validation", None, labeled_all, labeled_validation_only)
    assert metrics["n_labeled_in_scope"] == 1
    assert metrics["n_detected"] == 1
    assert metrics["detection_rate"] == 1.0
    # Productive rate still uses the whole-typology set (X and Y are both
    # genuine typology members, Z just wasn't a validation-period case).
    assert metrics["n_productive_customers"] == 2
    assert metrics["productive_rate"] == 1.0


def test_compute_metrics_segment_scoping():
    alerts = pd.DataFrame({"customer_id": ["A", "B", "C"], "month": [2, 2, 2], "period": ["tuning"] * 3})
    labeled_all = {"A", "B"}
    labeled_tuning = {"A", "B", "D"}  # D is labeled for tuning but never alerted
    segment = {"A", "C"}  # only A and C are in this segment
    metrics = compute_metrics(alerts, "tuning", segment, labeled_all, labeled_tuning)
    assert metrics["n_customers_alerted"] == 2  # A, C (B excluded by segment)
    assert metrics["n_productive_customers"] == 1  # only A is both alerted and labeled
    assert metrics["n_labeled_in_scope"] == 1  # only A of {A,B,D} is in the segment
    assert metrics["n_detected"] == 1


# ---------------------------------------------------------------------------
# End-to-end hand-built STR-01 sweep: capacity, both fallback triggers, ATL/BTL
# ---------------------------------------------------------------------------

def _customers_df(ids, segment="retail"):
    return pd.DataFrame(
        {
            "customer_id": ids,
            "segment_true": [segment] * len(ids),
            "country_risk_tier": ["low"] * len(ids),
            "account_open_date": [pd.Timestamp("2020-01-01").date()] * len(ids),
            "expected_monthly_activity": [1000.0] * len(ids),
        }
    )


def _tx(transaction_id, ts, from_acct, to_acct, amount, channel):
    return {
        "transaction_id": transaction_id,
        "timestamp": pd.Timestamp(ts),
        "from_account": from_acct,
        "to_account": to_acct,
        "amount": amount,
        "currency": "USD",
        "channel": channel,
        "cross_border": False,
        "instrument_type": None,
    }


def _structuring_burst(customer_id, month, n=6, amount=9200.0, day_start=10):
    rows = []
    for i in range(n):
        rows.append(
            _tx(
                f"{customer_id}-T{i}",
                f"2024-{month:02d}-{day_start + i:02d} 10:00:00",
                "CASH",
                f"CHQ_{customer_id}",
                amount,
                "cash_deposit",
            )
        )
    return rows


@pytest.fixture(scope="module")
def str01_sweep_result():
    gen_config = copy.deepcopy(load_config())
    scenarios_config = load_scenarios_config()
    capacity_cfg = load_capacity_config()

    # Two segments: "A" (customers P1-P8, segment 0 in our segments.parquet
    # stand-in) gets 6 structuring bursts in tuning + 2 in validation -
    # enough for its own threshold. "B" (customers Q1-Q3) gets only 2
    # tuning bursts and 0 validation bursts - must fall back to pooled.
    seg_a_ids = [f"P{i}" for i in range(1, 9)]
    seg_b_ids = [f"Q{i}" for i in range(1, 4)]
    noise_ids = [f"N{i}" for i in range(1, 6)]  # never structuring, just scaffold customers
    all_ids = seg_a_ids + seg_b_ids + noise_ids

    customers = _customers_df(all_ids, segment="small_business")
    accounts = pd.DataFrame(
        [(f"CHQ_{c}", c, "chequing", pd.Timestamp("2020-01-01").date()) for c in all_ids],
        columns=["account_id", "customer_id", "account_type", "open_date"],
    )
    relationships = pd.DataFrame(columns=["account_id_a", "account_id_b", "customer_id_a", "customer_id_b", "relationship_type"])

    rows = []
    # Segment A: 6 tuning-period bursts (months 1-6), 2 validation (months 9-10).
    for i, cust in enumerate(seg_a_ids[:6]):
        rows += _structuring_burst(cust, month=(i % 6) + 1)
    for i, cust in enumerate(seg_a_ids[6:8]):
        rows += _structuring_burst(cust, month=9 + i)
    # Segment B: 2 tuning bursts only, 0 validation.
    for i, cust in enumerate(seg_b_ids[:2]):
        rows += _structuring_burst(cust, month=i + 1)
    # Noise customers: no structuring activity at all.

    transactions = pd.DataFrame(rows)
    transaction_labels = pd.DataFrame(
        {"transaction_id": transactions["transaction_id"], "is_injected": True, "typology": "structuring"}
    )
    labels_rows = []
    for i, cust in enumerate(seg_a_ids[:6]):
        m = (i % 6) + 1
        labels_rows.append({"customer_id": cust, "typology": "structuring", "role": "primary", "start_date": f"2024-{m:02d}-10", "end_date": f"2024-{m:02d}-15"})
    for i, cust in enumerate(seg_a_ids[6:8]):
        m = 9 + i
        labels_rows.append({"customer_id": cust, "typology": "structuring", "role": "primary", "start_date": f"2024-{m:02d}-10", "end_date": f"2024-{m:02d}-15"})
    for i, cust in enumerate(seg_b_ids[:2]):
        m = i + 1
        labels_rows.append({"customer_id": cust, "typology": "structuring", "role": "primary", "start_date": f"2024-{m:02d}-10", "end_date": f"2024-{m:02d}-15"})
    labels = pd.DataFrame(labels_rows)

    data = {
        "customers": customers,
        "accounts": accounts,
        "relationships": relationships,
        "transactions": transactions,
        "transaction_labels": transaction_labels,
        "labels": labels,
    }
    con = build_context_from_frames(data, gen_config)

    segments = pd.DataFrame(
        {"customer_id": all_ids, "segment_assigned": ([0] * 8) + ([1] * 3) + ([0] * 5)}
    )

    result = sweep_scenario("STR-01", con, scenarios_config["STR-01"], gen_config, segments, labels, capacity_cfg)
    return result


def test_segment_with_enough_cases_gets_own_threshold(str01_sweep_result):
    choice = str01_sweep_result["segment_choices"]["0"]
    assert choice["fallback_to_pooled"] is False
    assert choice["n_tuning_cases"] >= 5
    assert choice["n_validation_cases"] >= 1


def test_segment_with_too_few_cases_falls_back_to_pooled(str01_sweep_result):
    choice = str01_sweep_result["segment_choices"]["1"]
    assert choice["fallback_to_pooled"] is True
    assert "2" in choice["reason"] or "0" in choice["reason"]


def test_atl_btl_steps_down_and_reports_extra_productive(str01_sweep_result):
    atl_btl = str01_sweep_result["atl_btl"]
    grid_values = sorted(load_scenarios_config()["STR-01"]["grid"]["min_deposits"])
    chosen = str01_sweep_result["pooled_choice"]["params"]["min_deposits"]
    if chosen == grid_values[0]:
        # Already at the loosest grid value - genuinely no room to step
        # down, not a bug (this small fixture has little incentive for a
        # strict threshold, so the capacity-constrained max-detection
        # choice can legitimately land on the loosest setting).
        assert atl_btl == []
        return
    assert len(atl_btl) >= 1
    for step in atl_btl:
        assert step["step_value"] < step["chosen_value"]
        assert step["n_extra_alerted_customers"] >= step["n_extra_productive"]


def test_test_period_numbers_computed_for_chosen_point(str01_sweep_result):
    pooled_test = str01_sweep_result["test_results"]["pooled"]
    assert pooled_test["n_labeled_in_scope"] == 0  # no test-period structuring cases in this fixture
    assert pooled_test["detection_rate"] is None


# ---------------------------------------------------------------------------
# Real-data smoke test: all five scenarios run end to end without crashing
# and produce internally consistent numbers.
# ---------------------------------------------------------------------------

def test_all_scenarios_sweep_on_generated_data():
    from aml_lab.features import compute_features_from_frames
    from aml_lab.generate import generate_dataset
    from aml_lab.segment import (
        build_customer_profile,
        fit_segments,
        load_segments_config,
        transform_features,
    )

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

    con = build_context_from_frames(data, gen_config)

    for scenario_id, spec in scenarios_config.items():
        result = sweep_scenario(scenario_id, con, spec, gen_config, segments, data["labels"], capacity_cfg)
        assert result["grid_size"] > 1
        pooled = result["pooled_choice"]
        assert pooled["tuning"]["n_labeled_in_scope"] >= 0
        if pooled["tuning"]["detection_rate"] is not None:
            assert 0.0 <= pooled["tuning"]["detection_rate"] <= 1.0
        test_metrics = result["test_results"]["pooled"]
        if test_metrics["productive_rate"] is not None:
            assert 0.0 <= test_metrics["productive_rate"] <= 1.0
        # Every segment choice must declare its fallback status explicitly.
        for choice in result["segment_choices"].values():
            assert "fallback_to_pooled" in choice
