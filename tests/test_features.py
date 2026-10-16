"""Tests for the DuckDB feature layer (features.sql / features.py).

Two kinds of tests:
  1. A tiny, hand-built dataset with known answers, worked out by hand in
     the comments below, for every feature family.
  2. A check that features computed on a real (generated) dataset show a
     genuine, sizeable gap between labeled and unlabeled customers for
     each typology - not just "present", but actually separating signal
     from noise, with the underlying case counts reported.
"""

import copy

import pandas as pd
import pytest

from aml_lab.features import compute_features_from_frames
from aml_lab.generate import generate_dataset, load_config

# ---------------------------------------------------------------------------
# Hand-built dataset
# ---------------------------------------------------------------------------

def _customers_df(ids):
    return pd.DataFrame(
        {
            "customer_id": ids,
            "segment_true": ["retail"] * len(ids),
            "country_risk_tier": ["low"] * len(ids),
            "account_open_date": [pd.Timestamp("2020-01-01").date()] * len(ids),
            "expected_monthly_activity": [1000.0] * len(ids),
        }
    )


def _tx(transaction_id, ts, from_acct, to_acct, amount, channel, cross_border=False, instrument_type=None):
    return {
        "transaction_id": transaction_id,
        "timestamp": pd.Timestamp(ts),
        "from_account": from_acct,
        "to_account": to_acct,
        "amount": amount,
        "currency": "USD",
        "channel": channel,
        "cross_border": cross_border,
        "instrument_type": instrument_type,
    }


@pytest.fixture(scope="module")
def hand_built_result():
    config = copy.deepcopy(load_config())
    config["months"] = 3  # Jan/Feb/Mar 2024; enough history for the z-score test

    customers = _customers_df(["A", "D", "E", "X", "Z", "C", "F1", "F2", "F3", "G"])

    accounts = pd.DataFrame(
        [
            ("CHQ_A", "A", "chequing", pd.Timestamp("2020-01-01").date()),
            ("CHQ_D", "D", "chequing", pd.Timestamp("2020-01-01").date()),
            ("CHQ_E", "E", "chequing", pd.Timestamp("2020-01-01").date()),
            ("CHQ_X", "X", "chequing", pd.Timestamp("2020-01-01").date()),
            ("CHQ_Z", "Z", "chequing", pd.Timestamp("2020-01-01").date()),
            ("INV1", "C", "investment", pd.Timestamp("2020-01-01").date()),
            ("INV2", "C", "investment", pd.Timestamp("2020-01-01").date()),
            ("INV3", "C", "investment", pd.Timestamp("2020-01-01").date()),
            ("CHQ_F1", "F1", "chequing", pd.Timestamp("2020-01-01").date()),
            ("CHQ_F2", "F2", "chequing", pd.Timestamp("2020-01-01").date()),
            ("CHQ_F3", "F3", "chequing", pd.Timestamp("2020-01-01").date()),
            ("CHQ_G", "G", "chequing", pd.Timestamp("2020-01-01").date()),
        ],
        columns=["account_id", "customer_id", "account_type", "open_date"],
    )

    # Only INV1-INV2 is a declared relationship; INV1-INV3 is deliberately
    # NOT listed, to prove the SQL consults relationships.parquet rather
    # than just checking "same customer on both sides".
    relationships = pd.DataFrame(
        [("INV1", "INV2", "C", "C", "same_owner")],
        columns=["account_id_a", "account_id_b", "customer_id_a", "customer_id_b", "relationship_type"],
    )

    tx = [
        # --- A: cash deposits, 7d vs 30d, near-threshold -------------
        _tx("T1", "2024-01-30 10:00:00", "CASH", "CHQ_A", 500.0, "cash_deposit"),
        _tx("T2", "2024-01-05 09:00:00", "CASH", "CHQ_A", 9500.0, "cash_deposit"),
        # --- D: rapid movement within 48h -----------------------------
        _tx("T3", "2024-01-10 10:00:00", "EXT-HOME-000001", "CHQ_D", 20000.0, "wire"),
        _tx("T4", "2024-01-11 12:00:00", "CHQ_D", "EXT-HOME-000002", 18000.0, "wire"),
        _tx("T5", "2024-01-15 00:00:00", "EXT-HOME-000003", "CHQ_D", 500.0, "e_transfer"),  # below $1k floor
        # --- E: distinct counterparties in/out -------------------------
        _tx("T6", "2024-01-03 00:00:00", "CHQ_E", "EXT-HOME-100001", 50.0, "card"),
        _tx("T7", "2024-01-10 00:00:00", "CHQ_E", "EXT-HOME-100002", 60.0, "card"),
        _tx("T8", "2024-01-20 00:00:00", "CHQ_E", "EXT-HOME-100003", 70.0, "card"),
        _tx("T9", "2024-01-02 00:00:00", "EXT-HOME-200001", "CHQ_E", 1000.0, "ach"),
        _tx("T10", "2024-01-25 00:00:00", "EXT-HOME-200002", "CHQ_E", 200.0, "e_transfer"),
        # --- X: cross-border share / high-risk exposure ---------------
        _tx("T11", "2024-01-05 00:00:00", "CHQ_X", "EXT-HOME-300001", 700.0, "card", cross_border=False),
        _tx("T12", "2024-01-06 00:00:00", "CHQ_X", "EXT-ALD-300002", 300.0, "wire", cross_border=True),
        _tx("T13", "2024-01-07 00:00:00", "CHQ_X", "EXT-DRK-300003", 200.0, "wire", cross_border=True),
        # --- Z: activity z-score across 3 months -----------------------
        _tx("T14", "2024-01-15 00:00:00", "EXT-HOME-400001", "CHQ_Z", 1000.0, "ach"),
        _tx("T15", "2024-02-15 00:00:00", "EXT-HOME-400001", "CHQ_Z", 2000.0, "ach"),
        _tx("T16", "2024-03-15 00:00:00", "EXT-HOME-400001", "CHQ_Z", 5000.0, "ach"),
        # --- C: journal transfers, related vs not-related --------------
        _tx("T17", "2024-01-10 00:00:00", "INV1", "INV2", 5000.0, "internal_journal"),
        _tx("T18", "2024-01-12 00:00:00", "INV1", "INV3", 7000.0, "internal_journal"),
        # --- F1 -> F2 -> F3: a same-bank chain; G is isolated -----------
        _tx("T19", "2024-01-05 00:00:00", "CHQ_F1", "CHQ_F2", 1000.0, "e_transfer"),
        _tx("T20", "2024-01-06 00:00:00", "CHQ_F2", "CHQ_F3", 800.0, "e_transfer"),
    ]
    transactions = pd.DataFrame(tx)

    transaction_labels = pd.DataFrame(
        {"transaction_id": transactions["transaction_id"], "is_injected": False, "typology": None}
    )
    labels = pd.DataFrame(columns=["customer_id", "typology", "role", "start_date", "end_date"])

    data = {
        "customers": customers,
        "accounts": accounts,
        "relationships": relationships,
        "transactions": transactions,
        "transaction_labels": transaction_labels,
        "labels": labels,
    }
    features = compute_features_from_frames(data, config)
    return features.set_index(["customer_id", "month"])


def _row(result, customer_id, month=1):
    return result.loc[(customer_id, month)]


def test_cash_deposit_windows(hand_built_result):
    row = _row(hand_built_result, "A")
    assert row["cash_deposit_count_7d"] == 1
    assert row["cash_deposit_sum_7d"] == pytest.approx(500.0)
    assert row["cash_deposit_count_30d"] == 2
    assert row["cash_deposit_sum_30d"] == pytest.approx(10000.0)
    assert row["near_threshold_deposit_count_30d"] == 1


def test_rapid_movement_48h(hand_built_result):
    row = _row(hand_built_result, "D")
    assert row["rapid_movement_ratio_48h"] == pytest.approx(18000.0 / 20000.0)
    assert row["residual_balance_48h_min"] == pytest.approx(2000.0)


def test_distinct_counterparties(hand_built_result):
    row = _row(hand_built_result, "E")
    assert row["distinct_counterparties_out_30d"] == 3
    assert row["distinct_counterparties_in_30d"] == 2


def test_cross_border_and_high_risk_exposure(hand_built_result):
    row = _row(hand_built_result, "X")
    assert row["total_flow_volume_30d"] == pytest.approx(1200.0)
    assert row["cross_border_share_30d"] == pytest.approx(500.0 / 1200.0)
    assert row["high_risk_country_exposure_30d"] == pytest.approx(200.0 / 1200.0)


def test_activity_zscore_baseline(hand_built_result):
    assert pd.isna(_row(hand_built_result, "Z", month=1)["activity_zscore_90d"])
    assert _row(hand_built_result, "Z", month=2)["activity_zscore_90d"] == pytest.approx(10.0)
    assert _row(hand_built_result, "Z", month=3)["activity_zscore_90d"] == pytest.approx(3500.0 / 707.1067811865476)


def test_journal_transfer_only_counts_related_accounts(hand_built_result):
    row = _row(hand_built_result, "C")
    assert row["journal_transfer_volume_30d_related"] == pytest.approx(5000.0)


def test_graph_degree_and_component_size(hand_built_result):
    f1 = _row(hand_built_result, "F1")
    f2 = _row(hand_built_result, "F2")
    f3 = _row(hand_built_result, "F3")
    g = _row(hand_built_result, "G")

    assert f1["graph_out_degree_30d"] == 1 and f1["graph_in_degree_30d"] == 0
    assert f2["graph_in_degree_30d"] == 1 and f2["graph_out_degree_30d"] == 1
    assert f3["graph_in_degree_30d"] == 1 and f3["graph_out_degree_30d"] == 0

    assert f1["component_size_30d"] == f2["component_size_30d"] == f3["component_size_30d"] == 3
    assert f1["community_size_30d"] == f2["community_size_30d"] == f3["community_size_30d"] == 3

    assert g["graph_in_degree_30d"] == 0 and g["graph_out_degree_30d"] == 0
    assert g["component_size_30d"] == 1 and g["community_size_30d"] == 1


def test_no_future_leakage_in_scaffold(hand_built_result):
    # A customer with zero transactions after their account opens must
    # still get a defined (zero-ish) row for every month - and a month-1
    # feature must not depend on month-2/3 data (checked implicitly by
    # the z-score test above: month 1's baseline is NULL, not borrowed
    # from later months).
    row = _row(hand_built_result, "G", month=3)
    assert row["cash_deposit_count_30d"] == 0
    assert row["total_flow_volume_30d"] == 0


# ---------------------------------------------------------------------------
# Labeled vs. unlabeled separation on a real generated dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_features():
    config = copy.deepcopy(load_config())
    data = generate_dataset(config, n_customers=4000)
    features = compute_features_from_frames(data, config)
    return features, data["labels"]


def _per_customer_max(features: pd.DataFrame, column: str) -> pd.Series:
    return features.groupby("customer_id")[column].max()


def _report_gap(name, labeled_vals, unlabeled_vals):
    print(
        f"\n{name}: labeled mean={labeled_vals.mean():.3f} (n={len(labeled_vals)}), "
        f"unlabeled mean={unlabeled_vals.mean():.3f} (n={len(unlabeled_vals)})"
    )


def test_structuring_separates_on_near_threshold_deposits(real_features):
    features, labels = real_features
    labeled_ids = set(labels.loc[labels["typology"] == "structuring", "customer_id"])
    assert len(labeled_ids) > 0
    per_cust = _per_customer_max(features, "near_threshold_deposit_count_30d")
    labeled_vals = per_cust[per_cust.index.isin(labeled_ids)]
    unlabeled_vals = per_cust[~per_cust.index.isin(labeled_ids)]
    _report_gap("structuring / near_threshold_deposit_count_30d", labeled_vals, unlabeled_vals)
    assert labeled_vals.mean() > unlabeled_vals.mean() * 3
    assert labeled_vals.mean() > 3  # structuring instances have >= 5 near-threshold deposits


def test_rapid_movement_separates_on_ratio(real_features):
    # Mean-of-max is noisy here: a normal customer's rare large one-off
    # transfer (config: ~3%/month) can coincide with an unrelated outflow
    # within 48h purely by chance, over 12 months of observations - that
    # is deliberate "legitimate activity sometimes looks suspicious"
    # noise, not a bug. A threshold-crossing comparison (closer to how a
    # scenario rule actually fires) is the fairer signal check.
    #
    # The unlabeled hit rate here is higher than it used to be because
    # the legitimate FAN-02 pass-through look-alikes (rent collectors,
    # small-business supplier payments - see generate.py) also produce a
    # fast inflow-then-outflow pattern; that's a real, intentional
    # cross-typology overlap, not a bug, so the margin below is looser
    # than before rather than tightened to hide it.
    features, labels = real_features
    labeled_ids = set(labels.loc[labels["typology"] == "rapid_movement", "customer_id"])
    assert len(labeled_ids) > 0
    per_cust = _per_customer_max(features, "rapid_movement_ratio_48h").fillna(0)
    labeled_vals = per_cust[per_cust.index.isin(labeled_ids)]
    unlabeled_vals = per_cust[~per_cust.index.isin(labeled_ids)]
    _report_gap("rapid_movement / rapid_movement_ratio_48h", labeled_vals, unlabeled_vals)

    threshold = 0.8
    labeled_hit = (labeled_vals >= threshold).sum()
    unlabeled_hit = (unlabeled_vals >= threshold).sum()
    print(
        f"  >= {threshold} ratio: {labeled_hit} of {len(labeled_vals)} labeled "
        f"({labeled_hit / len(labeled_vals):.0%}) vs {unlabeled_hit} of {len(unlabeled_vals)} unlabeled "
        f"({unlabeled_hit / len(unlabeled_vals):.1%})"
    )
    assert labeled_hit / len(labeled_vals) > 0.8
    assert (unlabeled_hit / len(unlabeled_vals)) * 2 < (labeled_hit / len(labeled_vals))


def test_median_unlabeled_customer_has_nonzero_graph_degree_some_month(real_features):
    """Guards against the fan-in/fan-out signal being unrealistically
    clean: with ordinary peer-to-peer activity (family/friends/
    roommates/small-business purchases - see generate.py), a typical
    unlabeled customer should show nonzero same-bank graph degree in at
    least some month, not a flat 0 all year.
    """
    features, labels = real_features
    labeled_ids = set(labels["customer_id"])
    unl = features[~features["customer_id"].isin(labeled_ids)].copy()
    unl["total_degree"] = unl["graph_in_degree_30d"] + unl["graph_out_degree_30d"]
    per_cust_any_nonzero = unl.groupby("customer_id")["total_degree"].max() > 0
    share = per_cust_any_nonzero.mean()
    print(
        f"\nshare of unlabeled customers with >= 1 month of nonzero same-bank graph degree: "
        f"{share:.1%} (n={len(per_cust_any_nonzero)})"
    )
    assert share > 0.5


def test_fan_in_fan_out_separates_on_degree_at_burst_month_by_role(real_features):
    """Raw same-bank degree maxed over a whole year is no longer a clean
    separator once ordinary peer-to-peer noise exists (every customer
    eventually gets a nonzero-degree month by chance). The fair
    comparison is each labeled customer's degree AT THEIR ACTUAL BURST
    MONTH (not their best month out of 12 tries) against the unlabeled
    population pooled across all customer-months, restricted to the
    same segments the fan-in/fan-out typology actually drew from - this
    mirrors how a scenario evaluates month by month in production, and
    is reported by role since the signal genuinely differs: only the
    hub shows a large in-degree spike; senders/downstream look close to
    ambient by degree alone (see the caveat in features.py).
    """
    features, labels = real_features
    fan_labels = labels[labels["typology"] == "fan_in_fan_out"].copy()
    assert len(fan_labels) > 0
    fan_labels["month"] = pd.to_datetime(fan_labels["start_date"]).dt.month

    burst = fan_labels.merge(features, on=["customer_id", "month"])
    burst["total_degree"] = burst["graph_in_degree_30d"] + burst["graph_out_degree_30d"]

    fan_segments = set(burst["segment_true"].unique())
    unl = features[
        (~features["customer_id"].isin(set(fan_labels["customer_id"])))
        & (features["segment_true"].isin(fan_segments))
    ].copy()
    unl["total_degree"] = unl["graph_in_degree_30d"] + unl["graph_out_degree_30d"]

    print(f"\nfan_in_fan_out segments involved: {sorted(fan_segments)}")
    for role in ["hub", "sender", "downstream"]:
        role_vals = burst.loc[burst["role"] == role, "total_degree"]
        if len(role_vals) == 0:
            continue
        print(
            f"  {role}: labeled mean degree at burst month = {role_vals.mean():.2f} (n={len(role_vals)}) vs "
            f"unlabeled pooled mean = {unl['total_degree'].mean():.2f} (n={len(unl)}, same segments)"
        )

    hub_vals = burst.loc[burst["role"] == "hub", "total_degree"]
    assert len(hub_vals) > 0
    assert hub_vals.mean() > unl["total_degree"].mean() * 3

    sender_downstream_vals = burst.loc[burst["role"].isin(["sender", "downstream"]), "total_degree"]
    assert sender_downstream_vals.mean() > unl["total_degree"].mean()


def test_wealth_layering_separates_on_journal_transfer_volume(real_features):
    # Restricted to HNW customers only (including the legit journal-transfer
    # look-alikes), since wealth_layering can only ever be injected into
    # HNW customers - comparing against the full population (mostly
    # retail/small_business, who never do investment journal transfers at
    # all) would overstate the gap.
    features, labels = real_features
    labeled_ids = set(labels.loc[labels["typology"] == "wealth_layering", "customer_id"])
    assert len(labeled_ids) > 0
    hnw_ids = set(features.loc[features["segment_true"] == "hnw_investment", "customer_id"])

    per_cust = _per_customer_max(features, "journal_transfer_volume_30d_related")
    labeled_vals = per_cust[per_cust.index.isin(labeled_ids)]
    unlabeled_vals = per_cust[per_cust.index.isin(hnw_ids - labeled_ids)]
    _report_gap("wealth_layering / journal_transfer_volume_30d_related (HNW only)", labeled_vals, unlabeled_vals)
    assert labeled_vals.mean() > unlabeled_vals.mean() * 2
    assert labeled_vals.mean() > 10000
