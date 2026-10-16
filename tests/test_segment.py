"""Tests for customer segmentation (segment.py).

A hand-built dataset with three obviously distinct behaviour archetypes
(two customers each) checks that: (1) only tuning-period rows feed the
profile - a validation-period row with wildly different values must not
move the result, and (2) KMeans actually recovers the three archetypes as
three separate clusters, not an artifact of the real dataset.
"""

import copy

import pandas as pd
import pytest

from aml_lab.segment import (
    build_customer_profile,
    build_segment_profile_table,
    fit_segments,
    load_segments_config,
    plot_segment_profile,
    transform_features,
)


def _tuning_rows(customer_id, volume, cash, distinct, cross_border, months=(1, 2, 3)):
    return [
        {
            "customer_id": customer_id,
            "period": "tuning",
            "month": m,
            "total_flow_volume_30d": volume,
            "cash_deposit_sum_30d": cash,
            "distinct_counterparties_out_30d": distinct,
            "cross_border_share_30d": cross_border,
        }
        for m in months
    ]


@pytest.fixture(scope="module")
def config():
    return load_segments_config()


@pytest.fixture(scope="module")
def toy_data():
    rows = []
    # Retail-like: low volume, no cash, no cross-border.
    rows += _tuning_rows("A", 1000, 0, 3, 0.0)
    rows += _tuning_rows("B", 1200, 0, 4, 0.0)
    # Small-business-like: high volume driven by heavy cash deposits.
    rows += _tuning_rows("C", 20000, 15000, 5, 0.0)
    rows += _tuning_rows("D", 22000, 16000, 6, 0.0)
    # HNW-like: very high volume, some cross-border, investment account.
    rows += _tuning_rows("E", 150000, 0, 3, 0.05)
    rows += _tuning_rows("F", 140000, 0, 4, 0.04)

    # A validation-period row with values nothing like A's tuning
    # behaviour - must be ignored by build_customer_profile.
    rows.append(
        {
            "customer_id": "A",
            "period": "validation",
            "month": 9,
            "total_flow_volume_30d": 999999,
            "cash_deposit_sum_30d": 999999,
            "distinct_counterparties_out_30d": 50,
            "cross_border_share_30d": 1.0,
        }
    )
    features = pd.DataFrame(rows)

    accounts = pd.DataFrame(
        [
            ("CHQ_E", "E", "investment"),
            ("CHQ_F", "F", "investment"),
            ("CHQ_A", "A", "chequing"),
            ("CHQ_B", "B", "chequing"),
            ("CHQ_C", "C", "chequing"),
            ("CHQ_D", "D", "chequing"),
        ],
        columns=["account_id", "customer_id", "account_type"],
    )
    return features, accounts


def test_profile_ignores_non_tuning_rows(toy_data, config):
    features, accounts = toy_data
    profile = build_customer_profile(features, accounts, config)
    assert profile.loc["A", "avg_cash_deposit_sum_30d"] == 0
    assert profile.loc["A", "avg_monthly_volume"] == pytest.approx(1000)
    assert profile.loc["A", "avg_distinct_counterparties_out_30d"] == pytest.approx(3)


def test_has_investment_account_flag(toy_data, config):
    features, accounts = toy_data
    profile = build_customer_profile(features, accounts, config)
    assert profile.loc["E", "has_investment_account"] == 1
    assert profile.loc["F", "has_investment_account"] == 1
    assert profile.loc["A", "has_investment_account"] == 0
    assert profile.loc["C", "has_investment_account"] == 0


def test_kmeans_recovers_three_archetypes(toy_data, config):
    features, accounts = toy_data
    profile = build_customer_profile(features, accounts, config)
    X = transform_features(profile, config)
    labels = fit_segments(X, k=3, seed=config["seed"])
    label_of = dict(zip(profile.index, labels))

    assert label_of["A"] == label_of["B"]
    assert label_of["C"] == label_of["D"]
    assert label_of["E"] == label_of["F"]
    assert len({label_of["A"], label_of["C"], label_of["E"]}) == 3


def test_segment_profile_table_structure(toy_data, config):
    features, accounts = toy_data
    profile = build_customer_profile(features, accounts, config)
    X = transform_features(profile, config)
    labels = fit_segments(X, k=3, seed=config["seed"])
    segment_true = pd.Series(
        {"A": "retail", "B": "retail", "C": "small_business", "D": "small_business", "E": "hnw_investment", "F": "hnw_investment"}
    )
    table = build_segment_profile_table(profile, labels, segment_true)

    assert len(table) == 3
    assert table["n_customers"].sum() == 6
    assert {"retail", "small_business", "hnw_investment"} >= set(table["dominant_segment_true"])
    # Every true segment should be the dominant label for exactly one
    # assigned cluster, since the archetypes are perfectly separable here.
    assert sorted(table["dominant_segment_true"]) == ["hnw_investment", "retail", "small_business"]


def test_plot_segment_profile_smoke(toy_data, config, tmp_path):
    features, accounts = toy_data
    profile = build_customer_profile(features, accounts, config)
    X = transform_features(profile, config)
    labels = fit_segments(X, k=3, seed=config["seed"])
    segment_true = pd.Series(
        {"A": "retail", "B": "retail", "C": "small_business", "D": "small_business", "E": "hnw_investment", "F": "hnw_investment"}
    )
    table = build_segment_profile_table(profile, labels, segment_true)
    out_path = tmp_path / "segment_profile.png"
    plot_segment_profile(table, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Real (generated) dataset: sanity-check the actual pipeline end to end.
# ---------------------------------------------------------------------------

def test_real_dataset_segmentation_aligns_with_true_segments():
    from aml_lab.features import compute_features_from_frames
    from aml_lab.generate import generate_dataset, load_config

    gen_config = copy.deepcopy(load_config())
    data = generate_dataset(gen_config, n_customers=3000)
    features = compute_features_from_frames(data, gen_config)

    seg_config = load_segments_config()
    profile = build_customer_profile(features, data["accounts"], seg_config)
    X = transform_features(profile, seg_config)
    labels = fit_segments(X, k=seg_config["k"], seed=seg_config["seed"])

    segment_true = data["customers"].set_index("customer_id")["segment_true"]
    table = build_segment_profile_table(profile, labels, segment_true)

    # HNW customers should be concentrated in a small number of clusters
    # dominated by hnw_investment - not scattered evenly, and not mixed
    # with retail as the dominant label.
    hnw_dominant = table[table["dominant_segment_true"] == "hnw_investment"]
    assert len(hnw_dominant) >= 1
    assert hnw_dominant["pct_hnw_investment"].max() > 80

    sb_dominant = table[table["dominant_segment_true"] == "small_business"]
    assert len(sb_dominant) >= 1
    assert sb_dominant["pct_small_business"].max() > 70
