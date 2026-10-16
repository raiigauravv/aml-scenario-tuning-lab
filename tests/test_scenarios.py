"""Tests for the Phase 4 rule-based scenarios (scenarios.py).

Hand-built datasets with known answers for each scenario's core logic
and evidence, plus an integration check against a generated dataset that
reports recall with counts (as requested) - including the honest
FAN-01-vs-FAN-02 comparison: the simple per-customer degree rule can
only ever catch a hub, never a sender or downstream participant, while
the ring-level scenario can catch a whole component.
"""

import copy
import json

import pandas as pd
import pytest

from aml_lab.generate import load_config
from aml_lab.scenarios import (
    build_context_from_frames,
    evaluate_fan_ring,
    evaluate_fan_simple,
    evaluate_rapid_movement,
    evaluate_structuring,
    evaluate_wealth_layering,
    load_scenarios_config,
)


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


def _empty_labels():
    return pd.DataFrame(columns=["customer_id", "typology", "role", "start_date", "end_date"])


def _empty_tx_labels(transactions):
    return pd.DataFrame({"transaction_id": transactions["transaction_id"], "is_injected": False, "typology": None})


@pytest.fixture(scope="module")
def gen_config():
    cfg = copy.deepcopy(load_config())
    cfg["months"] = 2
    return cfg


@pytest.fixture(scope="module")
def scenarios_config():
    return load_scenarios_config()


def _build_con(customers, accounts, relationships, tx_rows, gen_config):
    transactions = pd.DataFrame(tx_rows)
    data = {
        "customers": customers,
        "accounts": accounts,
        "relationships": relationships,
        "transactions": transactions,
        "transaction_labels": _empty_tx_labels(transactions),
        "labels": _empty_labels(),
    }
    return build_context_from_frames(data, gen_config)


# ---------------------------------------------------------------------------
# STR-01
# ---------------------------------------------------------------------------

def test_structuring_fires_with_evidence(gen_config, scenarios_config):
    customers = _customers_df(["A"])
    accounts = pd.DataFrame([("CHQ_A", "A", "chequing", pd.Timestamp("2020-01-01").date())],
                            columns=["account_id", "customer_id", "account_type", "open_date"])
    # Dated well inside the month (not near month-start): the scenario's
    # trailing-30-day lookback is relative to month_end, so a deposit on
    # the 1st of the month can fall just outside a later month's window
    # by less than a day - see the no-future-leakage boundary discussion
    # in features.sql. Mid-month dates avoid that edge case here.
    rows = [
        _tx(f"T{i}", f"2024-01-1{i+1} 10:00:00", "CASH", "CHQ_A", 9200.0, "cash_deposit") for i in range(5)
    ]
    con = _build_con(customers, accounts, pd.DataFrame(columns=["account_id_a","account_id_b","customer_id_a","customer_id_b","relationship_type"]), rows, gen_config)

    params = scenarios_config["STR-01"]["params"]
    alerts = evaluate_structuring(con, params, gen_config["reporting_threshold"])
    assert len(alerts) == 1
    assert alerts.iloc[0]["customer_id"] == "A"
    evidence = json.loads(alerts.iloc[0]["evidence"])
    assert evidence["n_deposits"] == 5
    assert len(evidence["deposit_transaction_ids"]) == 5


def test_structuring_does_not_fire_below_min_deposits(gen_config, scenarios_config):
    customers = _customers_df(["B"])
    accounts = pd.DataFrame([("CHQ_B", "B", "chequing", pd.Timestamp("2020-01-01").date())],
                            columns=["account_id", "customer_id", "account_type", "open_date"])
    rows = [
        _tx(f"T{i}", f"2024-01-0{i+1} 10:00:00", "CASH", "CHQ_B", 9200.0, "cash_deposit") for i in range(4)
    ]
    con = _build_con(customers, accounts, pd.DataFrame(columns=["account_id_a","account_id_b","customer_id_a","customer_id_b","relationship_type"]), rows, gen_config)

    params = scenarios_config["STR-01"]["params"]
    alerts = evaluate_structuring(con, params, gen_config["reporting_threshold"])
    assert alerts.empty


# ---------------------------------------------------------------------------
# RMV-01
# ---------------------------------------------------------------------------

def test_rapid_movement_fires_with_evidence(gen_config, scenarios_config):
    customers = _customers_df(["C"])
    accounts = pd.DataFrame([("CHQ_C", "C", "chequing", pd.Timestamp("2020-01-01").date())],
                            columns=["account_id", "customer_id", "account_type", "open_date"])
    rows = [
        _tx("T1", "2024-01-10 10:00:00", "EXT-HOME-000001", "CHQ_C", 20000.0, "wire"),
        _tx("T2", "2024-01-11 12:00:00", "CHQ_C", "EXT-HOME-000002", 18000.0, "wire"),
    ]
    con = _build_con(customers, accounts, pd.DataFrame(columns=["account_id_a","account_id_b","customer_id_a","customer_id_b","relationship_type"]), rows, gen_config)

    params = scenarios_config["RMV-01"]["params"]
    alerts = evaluate_rapid_movement(con, params)
    assert len(alerts) == 1
    evidence = json.loads(alerts.iloc[0]["evidence"])
    assert evidence["ratio"] == pytest.approx(0.9)
    assert evidence["outflow_transaction_ids"] == ["T2"]


def test_rapid_movement_does_not_fire_if_outflow_too_slow(gen_config, scenarios_config):
    customers = _customers_df(["D"])
    accounts = pd.DataFrame([("CHQ_D", "D", "chequing", pd.Timestamp("2020-01-01").date())],
                            columns=["account_id", "customer_id", "account_type", "open_date"])
    rows = [
        _tx("T1", "2024-01-10 10:00:00", "EXT-HOME-000001", "CHQ_D", 20000.0, "wire"),
        _tx("T2", "2024-01-13 12:00:00", "CHQ_D", "EXT-HOME-000002", 18000.0, "wire"),  # 74h later
    ]
    con = _build_con(customers, accounts, pd.DataFrame(columns=["account_id_a","account_id_b","customer_id_a","customer_id_b","relationship_type"]), rows, gen_config)

    params = scenarios_config["RMV-01"]["params"]
    alerts = evaluate_rapid_movement(con, params)
    assert alerts.empty


# ---------------------------------------------------------------------------
# FAN-01 vs FAN-02: hub-only rule vs whole-ring detection
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ring_context(gen_config):
    members = ["HUB", "S1", "S2", "S3", "S4", "D1"]
    customers = _customers_df(members)
    accounts = pd.DataFrame(
        [(f"CHQ_{m}", m, "chequing", pd.Timestamp("2020-01-01").date()) for m in members],
        columns=["account_id", "customer_id", "account_type", "open_date"],
    )
    rows = []
    # 4 senders -> hub, all within minutes of each other, material amounts.
    for i, s in enumerate(["S1", "S2", "S3", "S4"]):
        rows.append(_tx(f"IN{i}", f"2024-01-05 09:0{i}:00", f"CHQ_{s}", "CHQ_HUB", 5000.0, "e_transfer"))
    # Hub forwards ~90% onward to D1 within the hour - well inside every
    # sender edge's own 48h pass-through window (FAN-02 evaluates
    # pass-through PER EDGE, anchored on that edge's own first transfer;
    # D1 itself never forwards further, so the hub->D1 edge is the chain's
    # final leg and is NOT expected to pass the filter - D1 is a cash-out
    # destination here, not a pass-through node, which is realistic.
    rows.append(_tx("OUT1", "2024-01-05 10:00:00", "CHQ_HUB", "CHQ_D1", 18000.0, "e_transfer"))
    relationships = pd.DataFrame(columns=["account_id_a", "account_id_b", "customer_id_a", "customer_id_b", "relationship_type"])
    con = _build_con(customers, accounts, relationships, rows, gen_config)
    return con, members


def test_fan_simple_catches_hub_only(ring_context, scenarios_config):
    con, _members = ring_context
    params = scenarios_config["FAN-01"]["params"]
    alerts = evaluate_fan_simple(con, params)
    alerted = set(alerts["customer_id"]) if not alerts.empty else set()
    assert "HUB" in alerted
    assert alerted.isdisjoint({"S1", "S2", "S3", "S4", "D1"})


def test_fan_ring_catches_senders_and_hub_not_just_hub(ring_context, scenarios_config, gen_config):
    con, _members = ring_context
    params = dict(scenarios_config["FAN-02"]["params"])
    params["min_edge_amount"] = 1000
    alerts = evaluate_fan_ring(con, params, gen_config)
    alerted = set(alerts["customer_id"]) if not alerts.empty else set()
    # The whole sender+hub accumulation side of the ring is caught - this
    # is the point of FAN-02 vs. FAN-01, which only ever catches HUB.
    assert alerted == {"HUB", "S1", "S2", "S3", "S4"}
    evidence = json.loads(alerts.iloc[0]["evidence"])
    assert evidence["hub_candidate"] == "HUB"
    assert evidence["hub_in_degree"] == 4


def test_fan_ring_requires_pass_through_not_just_degree(gen_config, scenarios_config):
    # Same fan-in shape, but the hub never forwards the money onward -
    # high in-degree alone must NOT trigger a ring alert.
    members = ["HUB2", "S1", "S2", "S3", "S4"]
    customers = _customers_df(members)
    accounts = pd.DataFrame(
        [(f"CHQ_{m}", m, "chequing", pd.Timestamp("2020-01-01").date()) for m in members],
        columns=["account_id", "customer_id", "account_type", "open_date"],
    )
    rows = [
        _tx(f"IN{i}", f"2024-01-0{5+i} 10:00:00", f"CHQ_{s}", "CHQ_HUB2", 5000.0, "e_transfer")
        for i, s in enumerate(["S1", "S2", "S3", "S4"])
    ]
    relationships = pd.DataFrame(columns=["account_id_a", "account_id_b", "customer_id_a", "customer_id_b", "relationship_type"])
    con = _build_con(customers, accounts, relationships, rows, gen_config)

    params = dict(scenarios_config["FAN-02"]["params"])
    params["min_edge_amount"] = 1000
    alerts = evaluate_fan_ring(con, params, gen_config)
    assert alerts.empty


# ---------------------------------------------------------------------------
# WLT-01
# ---------------------------------------------------------------------------

def test_wealth_layering_fires_with_evidence(gen_config, scenarios_config):
    customers = _customers_df(["E"])
    accounts = pd.DataFrame(
        [("INV1", "E", "investment", pd.Timestamp("2020-01-01").date()), ("INV2", "E", "investment", pd.Timestamp("2020-01-01").date())],
        columns=["account_id", "customer_id", "account_type", "open_date"],
    )
    relationships = pd.DataFrame([("INV1", "INV2", "E", "E", "same_owner")],
                                  columns=["account_id_a", "account_id_b", "customer_id_a", "customer_id_b", "relationship_type"])
    rows = [
        _tx("J1", "2024-01-05 10:00:00", "INV1", "INV2", 20000.0, "internal_journal"),
        _tx("J2", "2024-01-07 10:00:00", "INV2", "INV1", 20000.0, "internal_journal"),
        _tx("J3", "2024-01-09 10:00:00", "INV1", "INV2", 20000.0, "internal_journal"),
        _tx("W1", "2024-01-10 10:00:00", "INV2", "EXT-DRK-000001", 55000.0, "wire", cross_border=True),
    ]
    con = _build_con(customers, accounts, relationships, rows, gen_config)

    params = scenarios_config["WLT-01"]["params"]
    alerts = evaluate_wealth_layering(con, params)
    assert len(alerts) == 1
    evidence = json.loads(alerts.iloc[0]["evidence"])
    assert evidence["n_journal"] == 3
    assert evidence["journal_volume"] == pytest.approx(60000.0)
    assert evidence["wire_transaction_id"] == "W1"


# ---------------------------------------------------------------------------
# Integration: recall on a generated dataset, reported with counts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_run():
    from aml_lab.generate import generate_dataset
    from aml_lab.scenarios import build_context_from_frames, run_all_scenarios

    full_gen_config = copy.deepcopy(load_config())
    data = generate_dataset(full_gen_config, n_customers=6000)
    scenarios_cfg = load_scenarios_config()
    con = build_context_from_frames(data, full_gen_config)
    results = run_all_scenarios(con, scenarios_cfg, full_gen_config)
    return results, data["labels"], con, full_gen_config, scenarios_cfg


@pytest.mark.parametrize("scenario_id,typology,min_recall", [
    ("STR-01", "structuring", 0.8),
    ("RMV-01", "rapid_movement", 0.8),
    ("WLT-01", "wealth_layering", 0.8),
])
def test_scenario_recall_on_generated_data(real_run, scenario_id, typology, min_recall):
    results, labels, _con, _gen_config, _scenarios_cfg = real_run
    alerts = results[scenario_id]
    labeled_ids = set(labels.loc[labels["typology"] == typology, "customer_id"])
    assert len(labeled_ids) > 0
    alerted_ids = set(alerts["customer_id"]) if not alerts.empty else set()
    caught = labeled_ids & alerted_ids
    recall = len(caught) / len(labeled_ids)
    print(f"\n{scenario_id}: {len(caught)} of {len(labeled_ids)} labeled {typology} customers caught ({recall:.0%})")
    assert recall >= min_recall


def test_fan01_vs_fan02_comparison(real_run):
    """Honest comparison, reported rather than asserted into looking
    better than it is: FAN-01 (simple degree rule) can only ever catch a
    hub. FAN-02 (ring detection) can catch the whole accumulation side of
    a ring, but only once pass_through_window_hours is wide enough to
    span the hub's actual accumulate-then-forward delay - the generator
    lets that delay run up to several days (generator.yaml window_days
    3-10), so at the scenario's literal 48h default, FAN-02 can
    under-perform FAN-01 on this dataset (0 vs 2 caught at n=6000 is
    typical). The real, meaningful assertion is that widening the window
    - already in the Phase 5 grid, not invented here - recovers
    substantially more ring members than FAN-01 ever could.
    """
    results, labels, con, gen_config, scenarios_cfg = real_run
    fan_labeled = set(labels.loc[labels["typology"] == "fan_in_fan_out", "customer_id"])
    fan01_caught = fan_labeled & set(results["FAN-01"]["customer_id"]) if not results["FAN-01"].empty else set()
    fan02_default_caught = fan_labeled & set(results["FAN-02"]["customer_id"]) if not results["FAN-02"].empty else set()

    wide_params = dict(scenarios_cfg["FAN-02"]["params"])
    wide_params["pass_through_window_hours"] = 240
    fan02_wide_alerts = evaluate_fan_ring(con, wide_params, gen_config)
    fan02_wide_caught = fan_labeled & set(fan02_wide_alerts["customer_id"]) if not fan02_wide_alerts.empty else set()

    print(f"\nFAN-01 (simple degree rule): {len(fan01_caught)} of {len(fan_labeled)} labeled fan_in_fan_out caught")
    print(
        f"FAN-02 (ring detection, default 48h pass-through window): "
        f"{len(fan02_default_caught)} of {len(fan_labeled)} caught"
    )
    print(
        f"FAN-02 (ring detection, 240h pass-through window, from the Phase 5 grid): "
        f"{len(fan02_wide_caught)} of {len(fan_labeled)} caught"
    )
    assert len(fan02_wide_caught) > len(fan01_caught)
