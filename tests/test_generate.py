import copy
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from aml_lab.generate import (
    PERIOD_BANDS,
    assign_stratified_months,
    generate_dataset,
    load_config,
    month_bounds,
    sim_window,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def small_dataset(config):
    # Small enough to run in well under a second, large enough that every
    # typology (rate ~0.0015-0.003) produces several instances.
    return generate_dataset(copy.deepcopy(config), n_customers=4000)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_generation_is_deterministic_given_seed(config):
    data_a = generate_dataset(copy.deepcopy(config), n_customers=200)
    data_b = generate_dataset(copy.deepcopy(config), n_customers=200)
    pd.testing.assert_frame_equal(data_a["customers"], data_b["customers"])
    pd.testing.assert_frame_equal(data_a["transactions"], data_b["transactions"])
    pd.testing.assert_frame_equal(data_a["labels"], data_b["labels"])


def test_different_seed_gives_different_output(config):
    cfg_other = copy.deepcopy(config)
    cfg_other["seed"] = 999
    data_a = generate_dataset(copy.deepcopy(config), n_customers=200)
    data_b = generate_dataset(cfg_other, n_customers=200)
    assert not data_a["transactions"]["amount"].equals(data_b["transactions"]["amount"])


# ---------------------------------------------------------------------------
# Typology injection rates and pattern presence
# ---------------------------------------------------------------------------

def test_typology_injection_rate_within_configured_bounds(small_dataset):
    labels = small_dataset["labels"]
    n_customers = len(small_dataset["customers"])
    labeled_share = labels["customer_id"].nunique() / n_customers
    # Spec target: roughly 0.5-1% of customers carry an injected typology.
    assert 0.003 <= labeled_share <= 0.02


def test_all_four_typologies_present(small_dataset):
    labels = small_dataset["labels"]
    assert set(labels["typology"].unique()) == {
        "structuring",
        "rapid_movement",
        "fan_in_fan_out",
        "wealth_layering",
    }


def test_structuring_pattern_actually_present(small_dataset, config):
    threshold = config["reporting_threshold"]
    labels = small_dataset["labels"]
    tx = small_dataset["transactions"].merge(small_dataset["transaction_labels"], on="transaction_id")
    accounts = small_dataset["accounts"]

    structuring_customers = labels.loc[labels["typology"] == "structuring", "customer_id"].unique()
    assert len(structuring_customers) > 0
    for cust in structuring_customers:
        acct_ids = accounts.loc[accounts["customer_id"] == cust, "account_id"]
        cust_tx = tx[(tx["typology"] == "structuring") & (tx["to_account"].isin(acct_ids))]
        assert len(cust_tx) >= 5
        assert (cust_tx["channel"] == "cash_deposit").all()
        assert (cust_tx["amount"] < threshold).all()
        assert (cust_tx["amount"] >= 0.8 * threshold).all()
        span_days = (cust_tx["timestamp"].max() - cust_tx["timestamp"].min()).days
        assert span_days <= 14


def test_rapid_movement_pattern_actually_present(small_dataset):
    labels = small_dataset["labels"]
    tx = small_dataset["transactions"].merge(small_dataset["transaction_labels"], on="transaction_id")

    rmv_customers = labels.loc[labels["typology"] == "rapid_movement", "customer_id"].unique()
    assert len(rmv_customers) > 0
    for cust in rmv_customers:
        cust_tx = tx[(tx["typology"] == "rapid_movement")].merge(
            small_dataset["accounts"][small_dataset["accounts"]["customer_id"] == cust][["account_id"]],
            left_on="to_account",
            right_on="account_id",
            how="inner",
        )
        # at least the inflow should match this customer's account
        assert len(cust_tx) >= 1


# ---------------------------------------------------------------------------
# Point 2: typology start dates spread across all three time-split periods
# ---------------------------------------------------------------------------

def test_assign_stratified_months_covers_every_band():
    rng = np.random.default_rng(0)
    months = assign_stratified_months(rng, n_instances=10, months=12)
    assert len(months) == 10
    for lo, hi in PERIOD_BANDS.values():
        assert any(lo <= m <= hi for m in months)


def test_assign_stratified_months_handles_fewer_instances_than_bands():
    rng = np.random.default_rng(0)
    months = assign_stratified_months(rng, n_instances=2, months=12)
    assert len(months) == 2
    assert all(1 <= m <= 12 for m in months)


def test_injected_typologies_cover_tuning_validation_and_test_periods(small_dataset, config):
    start, _end, _months = sim_window(config)
    labels = small_dataset["labels"]

    def month_of(d):
        return (d.year - start.year) * 12 + (d.month - start.month) + 1

    for typology in ["structuring", "rapid_movement", "fan_in_fan_out", "wealth_layering"]:
        start_months = labels.loc[labels["typology"] == typology, "start_date"].map(month_of)
        for band_lo, band_hi in PERIOD_BANDS.values():
            assert any(band_lo <= m <= band_hi for m in start_months), (
                f"{typology} has no instance starting in band ({band_lo}, {band_hi})"
            )


# ---------------------------------------------------------------------------
# Point 3: legitimate look-alikes exist among NON-injected customers
# ---------------------------------------------------------------------------

def test_small_business_lookalikes_exist_near_threshold(small_dataset, config):
    threshold = config["reporting_threshold"]
    customers = small_dataset["customers"]
    labels = small_dataset["labels"]
    accounts = small_dataset["accounts"]
    tx = small_dataset["transactions"].merge(small_dataset["transaction_labels"], on="transaction_id")

    labeled = set(labels["customer_id"])
    unlabeled_small_biz = customers[
        (customers["segment_true"] == "small_business") & (~customers["customer_id"].isin(labeled))
    ]["customer_id"]

    near_threshold = tx[
        (tx["channel"] == "cash_deposit") & (tx["is_injected"] == False) & (tx["amount"] >= 0.85 * threshold) & (tx["amount"] < threshold)
    ]
    chequing = accounts[accounts["account_type"] == "chequing"][["account_id", "customer_id"]]
    near_threshold = near_threshold.merge(chequing, left_on="to_account", right_on="account_id")
    customers_with_lookalike_deposits = near_threshold[
        near_threshold["customer_id"].isin(unlabeled_small_biz)
    ]["customer_id"].nunique()

    assert customers_with_lookalike_deposits > 0


def test_hnw_lookalike_journal_transfers_exist(small_dataset):
    customers = small_dataset["customers"]
    labels = small_dataset["labels"]
    tx = small_dataset["transactions"].merge(small_dataset["transaction_labels"], on="transaction_id")
    accounts = small_dataset["accounts"]

    labeled = set(labels["customer_id"])
    unlabeled_hnw = set(
        customers[(customers["segment_true"] == "hnw_investment") & (~customers["customer_id"].isin(labeled))][
            "customer_id"
        ]
    )

    invest_accounts = accounts[accounts["account_type"] == "investment"]
    acct_to_cust = dict(zip(invest_accounts["account_id"], invest_accounts["customer_id"]))

    journal = tx[(tx["channel"] == "internal_journal") & (tx["is_injected"] == False)]
    journal_customers = journal["from_account"].map(acct_to_cust).dropna()

    lookalike_customers_found = set(journal_customers) & unlabeled_hnw
    assert len(lookalike_customers_found) > 0


def test_hnw_wire_lookalikes_exist_and_overlap_with_rebalancing(small_dataset):
    """Some unlabeled HNW customers legitimately send outbound cross-border
    wires too (property abroad, foreign investments, family support), and
    some of those overlap in month with a rebalancing journal transfer -
    this is what makes WLT-01 need a real threshold instead of just
    "journal transfers + any cross-border wire" (see generate.py).
    """
    customers = small_dataset["customers"]
    labels = small_dataset["labels"]
    tx = small_dataset["transactions"].merge(small_dataset["transaction_labels"], on="transaction_id")
    accounts = small_dataset["accounts"]

    labeled = set(labels["customer_id"])
    unlabeled_hnw = set(
        customers[(customers["segment_true"] == "hnw_investment") & (~customers["customer_id"].isin(labeled))][
            "customer_id"
        ]
    )
    invest_accounts = accounts[accounts["account_type"] == "investment"]
    chequing_accounts = accounts[accounts["account_type"] == "chequing"]
    acct_to_cust = dict(zip(invest_accounts["account_id"], invest_accounts["customer_id"]))
    acct_to_cust.update(zip(chequing_accounts["account_id"], chequing_accounts["customer_id"]))

    wires_out = tx[(tx["channel"] == "wire") & (tx["cross_border"]) & (~tx["is_injected"])]
    wire_customers = set(wires_out["from_account"].map(acct_to_cust).dropna()) & unlabeled_hnw
    assert len(wire_customers) > 0

    journal = tx[(tx["channel"] == "internal_journal") & (~tx["is_injected"])]
    journal_customers = set(journal["from_account"].map(acct_to_cust).dropna()) & unlabeled_hnw
    assert len(wire_customers & journal_customers) > 0

    wires_out = wires_out.assign(month=wires_out["timestamp"].dt.to_period("M"))
    journal = journal.assign(month=journal["timestamp"].dt.to_period("M"))
    wire_months = wires_out.assign(cust=wires_out["from_account"].map(acct_to_cust)).dropna(subset=["cust"])
    journal_months = journal.assign(cust=journal["from_account"].map(acct_to_cust)).dropna(subset=["cust"])
    same_month_pairs = wire_months.merge(journal_months, on=["cust", "month"], suffixes=("_wire", "_journal"))
    assert len(same_month_pairs) > 0


# ---------------------------------------------------------------------------
# Point 6: every customer in a fan-in/fan-out network is labeled
# ---------------------------------------------------------------------------

def test_fan_in_fan_out_labels_every_participant(small_dataset):
    labels = small_dataset["labels"]
    tx = small_dataset["transactions"].merge(small_dataset["transaction_labels"], on="transaction_id")
    accounts = small_dataset["accounts"]
    acct_to_cust = dict(zip(accounts["account_id"], accounts["customer_id"]))

    fan_tx = tx[tx["typology"] == "fan_in_fan_out"]
    assert len(fan_tx) > 0

    participants = set(fan_tx["from_account"].map(acct_to_cust).dropna()) | set(
        fan_tx["to_account"].map(acct_to_cust).dropna()
    )
    fan_labels = labels[labels["typology"] == "fan_in_fan_out"]
    labeled_participants = set(fan_labels["customer_id"])

    assert participants == labeled_participants
    assert {"sender", "hub", "downstream"}.issubset(set(fan_labels["role"].unique()))


# ---------------------------------------------------------------------------
# Point 8: equity/fixed-income buy/sell transactions exist, both as normal
# investment activity and as part of the wealth-layering typology.
# ---------------------------------------------------------------------------

def test_normal_investment_accounts_have_buy_sell_activity(small_dataset):
    tx = small_dataset["transactions"].merge(small_dataset["transaction_labels"], on="transaction_id")
    normal_trades = tx[(tx["channel"].isin(["security_buy", "security_sell"])) & (tx["is_injected"] == False)]
    assert len(normal_trades) > 0
    assert set(normal_trades["instrument_type"].dropna().unique()) <= {"equity", "fixed_income"}


def test_wealth_layering_uses_buy_sell_trades(small_dataset):
    tx = small_dataset["transactions"].merge(small_dataset["transaction_labels"], on="transaction_id")
    layering_trades = tx[(tx["typology"] == "wealth_layering") & (tx["channel"].isin(["security_buy", "security_sell"]))]
    assert len(layering_trades) > 0

    labels = small_dataset["labels"]
    wealth_customers = labels.loc[labels["typology"] == "wealth_layering", "customer_id"].unique()
    accounts = small_dataset["accounts"]
    invest_accounts = accounts[accounts["account_type"] == "investment"]
    acct_to_cust = dict(zip(invest_accounts["account_id"], invest_accounts["customer_id"]))

    trade_customers = set(layering_trades["from_account"].map(acct_to_cust).dropna()) | set(
        layering_trades["to_account"].map(acct_to_cust).dropna()
    )
    assert trade_customers.issubset(set(wealth_customers))

    # wealth layering must still end with an outbound cross-border wire
    wires = tx[(tx["typology"] == "wealth_layering") & (tx["channel"] == "wire")]
    assert len(wires) == len(wealth_customers)
    assert (wires["cross_border"]).all()


# ---------------------------------------------------------------------------
# Schema / sanity checks
# ---------------------------------------------------------------------------

def test_transaction_labels_align_with_transactions(small_dataset):
    tx = small_dataset["transactions"]
    tx_labels = small_dataset["transaction_labels"]
    assert len(tx) == len(tx_labels)
    assert set(tx["transaction_id"]) == set(tx_labels["transaction_id"])
    assert tx_labels["is_injected"].sum() > 0
    assert tx_labels.loc[~tx_labels["is_injected"], "typology"].isna().all()
    assert tx_labels.loc[tx_labels["is_injected"], "typology"].notna().all()


def test_relationships_link_hnw_multi_investment_accounts(small_dataset):
    relationships = small_dataset["relationships"]
    same_owner = relationships[relationships["relationship_type"] == "same_owner"]
    assert len(same_owner) > 0
    assert (same_owner["customer_id_a"] == same_owner["customer_id_b"]).all()


def test_distinct_counterparties_per_customer_per_month_is_realistic(small_dataset):
    """Counterparties must come from a persistent per-customer roster
    (occasional new ones aside), or every transaction looks "new" and any
    distinct-counterparty feature built on this data in Phase 2 would be
    meaningless.
    """
    tx = small_dataset["transactions"]
    accounts = small_dataset["accounts"]
    chequing = accounts[accounts["account_type"] == "chequing"][["account_id", "customer_id"]]

    from_is_ext = tx["from_account"].astype(str).str.startswith("EXT-")
    to_is_ext = tx["to_account"].astype(str).str.startswith("EXT-")
    ext_tx = tx[from_is_ext | to_is_ext].copy()
    ext_tx["counterparty"] = np.where(from_is_ext[ext_tx.index], ext_tx["from_account"], ext_tx["to_account"])
    ext_tx["account_id"] = np.where(from_is_ext[ext_tx.index], ext_tx["to_account"], ext_tx["from_account"])
    ext_tx = ext_tx.merge(chequing, on="account_id")
    ext_tx["period"] = pd.to_datetime(ext_tx["timestamp"]).dt.to_period("M")

    distinct_counts = ext_tx.groupby(["customer_id", "period"])["counterparty"].nunique()
    assert len(distinct_counts) > 0

    median = distinct_counts.median()
    assert median < 20, f"median distinct counterparties/customer/month = {median}, expected a roster-driven value"

    # Most of the mass should be using the persistent roster, not a flood of
    # one-off counterparties.
    assert distinct_counts.quantile(0.95) < 40


def test_month_bounds_and_sim_window(config):
    start, end, months = sim_window(config)
    assert months == 12
    assert start == dt.date(2024, 1, 1)
    assert end == dt.date(2024, 12, 31)
    jan_start, jan_end = month_bounds(start, 1)
    assert jan_start == dt.date(2024, 1, 1)
    assert jan_end == dt.date(2024, 1, 31)
