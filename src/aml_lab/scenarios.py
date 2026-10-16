"""Rule-based AML scenarios (Phase 4).

Each scenario is a transparent, named-parameter rule, evaluated per
customer-month against the same point-in-time convention as the feature
layer: every window is trailing and capped at month_end, so a scenario
can never fire on data from after the month it alerts in. Every alert
carries its triggering evidence (transaction ids and the values that
crossed the threshold) in an `evidence` JSON column.

Five scenarios:
  STR-01  Structuring
  RMV-01  Rapid movement
  FAN-01  Fan-in/fan-out, simple per-customer degree rule (baseline)
  FAN-02  Fan-in/fan-out, ring detection on a filtered pass-through
          subgraph (see below)
  WLT-01  Investment / wealth layering

FAN-02 note (read before changing pass_through_window_hours): the
generator's fan-in/fan-out hub accumulates from several senders over a
multi-day window (generator.yaml: window_days 3-10) before forwarding
onward in one lump, with a random delay up to window_days*24 hours after
the LAST sender's inflow. A strict 48h pass-through window (the
scenario's literal default) structurally misses rings where that delay
exceeds 48h - empirically about 3 of the 5 labeled rings in the default
dataset. This is reported honestly by `report_fan02_recovery` rather
than silently widened; `pass_through_window_hours` is in the FAN-02 grid
in scenarios.yaml precisely so Phase 5 can pick a window that actually
recovers rings instead of guessing one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import networkx as nx
import pandas as pd
import yaml

from aml_lab.features import (
    load_tables_from_dir,
    load_tables_from_frames,
    register_scaffold_tables,
    run_features_sql,
)
from aml_lab.generate import DEFAULT_CONFIG_PATH, load_config, month_bounds, sim_window

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS_CONFIG_PATH = REPO_ROOT / "config" / "scenarios.yaml"


def load_scenarios_config(path: Path = DEFAULT_SCENARIOS_CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Context: reuse features.py's table loading + features.sql (tx_resolved,
# tx_flows, scaffold, relationships are all scenario building blocks too).
# ---------------------------------------------------------------------------

def build_context_from_dir(data_dir: Path, gen_config: dict) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    load_tables_from_dir(con, data_dir)
    register_scaffold_tables(con, gen_config)
    run_features_sql(con)
    return con


def build_context_from_frames(data: dict[str, pd.DataFrame], gen_config: dict) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    load_tables_from_frames(con, data)
    register_scaffold_tables(con, gen_config)
    run_features_sql(con)
    return con


# ---------------------------------------------------------------------------
# STR-01: Structuring
# ---------------------------------------------------------------------------

def evaluate_structuring(con: duckdb.DuckDBPyConnection, params: dict, reporting_threshold: float) -> pd.DataFrame:
    lower = (1 - params["pct_below_threshold"]) * reporting_threshold

    sql = """
    WITH qualifying AS (
        SELECT to_customer AS customer_id, transaction_id, timestamp, amount
        FROM tx_resolved
        WHERE channel = 'cash_deposit' AND amount >= ? AND amount < ? AND to_customer IS NOT NULL
    ),
    candidates AS (
        SELECT s.customer_id, s.month, s.month_end, q.transaction_id, q.timestamp,
            (
                SELECT COUNT(*) FROM qualifying o
                WHERE o.customer_id = q.customer_id
                  AND o.timestamp >= q.timestamp
                  AND o.timestamp < LEAST(q.timestamp + INTERVAL (?) DAY, s.month_end + INTERVAL 1 SECOND)
            ) AS count_in_window
        FROM scaffold s
        JOIN qualifying q
          ON q.customer_id = s.customer_id
         AND q.timestamp > s.month_end - INTERVAL 30 DAY
         AND q.timestamp <= s.month_end
    )
    SELECT * FROM candidates
    """
    candidates = con.execute(sql, [lower, reporting_threshold, params["window_days"]]).df()
    if candidates.empty:
        return _empty_alert_frame()

    alerts = []
    for (customer_id, month), group in candidates.groupby(["customer_id", "month"]):
        best = group.loc[group["count_in_window"].idxmax()]
        if best["count_in_window"] < params["min_deposits"]:
            continue
        window_end = min(best["timestamp"] + pd.Timedelta(days=params["window_days"]), best["month_end"])
        evidence_tx = group[(group["timestamp"] >= best["timestamp"]) & (group["timestamp"] < window_end)]
        alerts.append(
            {
                "customer_id": customer_id,
                "month": month,
                "evidence": json.dumps(
                    {
                        "deposit_transaction_ids": sorted(evidence_tx["transaction_id"].tolist()),
                        "n_deposits": int(best["count_in_window"]),
                        "window_start": str(best["timestamp"]),
                        "window_days": params["window_days"],
                    }
                ),
            }
        )
    return pd.DataFrame(alerts) if alerts else _empty_alert_frame()


# ---------------------------------------------------------------------------
# RMV-01: Rapid movement
# ---------------------------------------------------------------------------

def evaluate_rapid_movement(con: duckdb.DuckDBPyConnection, params: dict) -> pd.DataFrame:
    sql = """
    WITH candidates AS (
        SELECT s.customer_id, s.month, i.transaction_id AS inflow_transaction_id,
               i.timestamp AS inflow_time, i.amount AS inflow_amount,
            (
                SELECT COALESCE(SUM(o.amount), 0)
                FROM tx_flows o
                WHERE o.customer_id = i.customer_id
                  AND o.direction = 'out'
                  AND o.timestamp > i.timestamp
                  AND o.timestamp <= LEAST(i.timestamp + INTERVAL (?) HOUR, s.month_end)
            ) AS outflow_followed
        FROM scaffold s
        JOIN tx_flows i
            ON i.customer_id = s.customer_id
           AND i.direction = 'in'
           AND i.amount >= ?
           AND i.timestamp > s.month_end - INTERVAL 30 DAY
           AND i.timestamp <= s.month_end
    )
    SELECT *, outflow_followed / inflow_amount AS ratio FROM candidates
    """
    df = con.execute(sql, [params["window_hours"], params["min_inflow"]]).df()
    if df.empty:
        return _empty_alert_frame()

    hits = df[df["ratio"] >= params["min_outflow_pct"]]
    if hits.empty:
        return _empty_alert_frame()

    alerts = []
    for (customer_id, month), group in hits.groupby(["customer_id", "month"]):
        best = group.loc[group["ratio"].idxmax()]
        window_end = best["inflow_time"] + pd.Timedelta(hours=params["window_hours"])
        out_sql = """
        SELECT transaction_id FROM tx_flows
        WHERE customer_id = ? AND direction = 'out' AND timestamp > ? AND timestamp <= ?
        """
        out_tx = con.execute(out_sql, [customer_id, best["inflow_time"], window_end]).df()
        alerts.append(
            {
                "customer_id": customer_id,
                "month": month,
                "evidence": json.dumps(
                    {
                        "inflow_transaction_id": best["inflow_transaction_id"],
                        "inflow_amount": round(float(best["inflow_amount"]), 2),
                        "outflow_transaction_ids": sorted(out_tx["transaction_id"].tolist()),
                        "outflow_amount": round(float(best["outflow_followed"]), 2),
                        "ratio": round(float(best["ratio"]), 4),
                    }
                ),
            }
        )
    return pd.DataFrame(alerts)


# ---------------------------------------------------------------------------
# FAN-01: simple per-customer degree rule (baseline, no network structure)
# ---------------------------------------------------------------------------

def evaluate_fan_simple(con: duckdb.DuckDBPyConnection, params: dict) -> pd.DataFrame:
    sql = """
    WITH inflow_events AS (
        SELECT s.customer_id, s.month, s.month_end, f.timestamp AS anchor_time
        FROM scaffold s
        JOIN tx_flows f
            ON f.customer_id = s.customer_id
           AND f.direction = 'in'
           AND f.counterparty_customer IS NOT NULL
           AND f.timestamp > s.month_end - INTERVAL 30 DAY
           AND f.timestamp <= s.month_end
    ),
    windowed AS (
        SELECT DISTINCT customer_id, month, month_end, anchor_time,
            (SELECT COUNT(DISTINCT o.counterparty_customer) FROM tx_flows o
             WHERE o.customer_id = inflow_events.customer_id AND o.direction = 'in'
               AND o.counterparty_customer IS NOT NULL
               AND o.timestamp >= anchor_time
               AND o.timestamp < LEAST(anchor_time + INTERVAL (?) DAY, month_end + INTERVAL 1 SECOND)
            ) AS distinct_senders,
            (SELECT COALESCE(SUM(o.amount), 0) FROM tx_flows o
             WHERE o.customer_id = inflow_events.customer_id AND o.direction = 'in'
               AND o.counterparty_customer IS NOT NULL
               AND o.timestamp >= anchor_time
               AND o.timestamp < LEAST(anchor_time + INTERVAL (?) DAY, month_end + INTERVAL 1 SECOND)
            ) AS total_in,
            (SELECT COALESCE(SUM(o.amount), 0) FROM tx_flows o
             WHERE o.customer_id = inflow_events.customer_id AND o.direction = 'out'
               AND o.timestamp >= anchor_time
               AND o.timestamp < LEAST(anchor_time + INTERVAL (?) DAY, month_end + INTERVAL 1 SECOND)
            ) AS total_out
        FROM inflow_events
    )
    SELECT * FROM windowed
    """
    wd = params["window_days"]
    df = con.execute(sql, [wd, wd, wd]).df()
    if df.empty:
        return _empty_alert_frame()
    df["forwarded_share"] = df["total_out"] / df["total_in"]

    alerts = []
    for (customer_id, month), group in df.groupby(["customer_id", "month"]):
        eligible = group[group["distinct_senders"] >= params["min_distinct_senders"]]
        eligible = eligible[eligible["forwarded_share"] >= params["min_forwarded_share"]]
        if eligible.empty:
            continue
        best = eligible.loc[eligible["forwarded_share"].idxmax()]
        alerts.append(
            {
                "customer_id": customer_id,
                "month": month,
                "evidence": json.dumps(
                    {
                        "distinct_senders": int(best["distinct_senders"]),
                        "total_in": round(float(best["total_in"]), 2),
                        "total_out": round(float(best["total_out"]), 2),
                        "forwarded_share": round(float(best["forwarded_share"]), 4),
                        "window_start": str(best["anchor_time"]),
                        "window_days": wd,
                    }
                ),
            }
        )
    return pd.DataFrame(alerts) if alerts else _empty_alert_frame()


# ---------------------------------------------------------------------------
# FAN-02: ring detection on a filtered pass-through subgraph
# ---------------------------------------------------------------------------

def evaluate_fan_ring(con: duckdb.DuckDBPyConnection, params: dict, gen_config: dict) -> pd.DataFrame:
    start, _end, months = sim_window(gen_config)

    edge_sql = """
    WITH edges AS (
        SELECT customer_id AS src, counterparty_customer AS dst, SUM(amount) AS edge_amount, MIN(timestamp) AS first_ts
        FROM tx_flows
        WHERE direction = 'out' AND counterparty_customer IS NOT NULL
          AND timestamp > ? AND timestamp <= ?
        GROUP BY customer_id, counterparty_customer
        HAVING SUM(amount) >= ?
    )
    SELECT e.*,
        (
            SELECT COALESCE(SUM(o.amount), 0) FROM tx_flows o
            WHERE o.customer_id = e.dst AND o.direction = 'out'
              AND o.timestamp > e.first_ts
              AND o.timestamp <= LEAST(e.first_ts + INTERVAL (?) HOUR, ?)
        ) AS dst_outflow_followed
    FROM edges e
    """

    alerts = []
    for m in range(1, months + 1):
        _month_start, month_end = month_bounds(start, m)
        month_end_ts = pd.Timestamp(month_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        window_start_ts = month_end_ts - pd.Timedelta(days=params["window_days"])

        edges = con.execute(
            edge_sql,
            [window_start_ts, month_end_ts, params["min_edge_amount"], params["pass_through_window_hours"], month_end_ts],
        ).df()
        if edges.empty:
            continue
        edges["pass_through_ratio"] = edges["dst_outflow_followed"] / edges["edge_amount"]
        kept = edges[edges["pass_through_ratio"] >= params["pass_through_ratio"]]
        if kept.empty:
            continue

        dg = nx.DiGraph()
        for row in kept.itertuples():
            dg.add_edge(row.src, row.dst, amount=row.edge_amount, pass_through_ratio=row.pass_through_ratio)
        undirected = dg.to_undirected()

        for component in nx.connected_components(undirected):
            if len(component) < params["min_component_size"]:
                continue
            in_degrees = {n: dg.in_degree(n) for n in component}
            hub = max(in_degrees, key=in_degrees.get)
            if in_degrees[hub] < params["min_hub_in_degree"]:
                continue

            sub_edges = [
                {"src": u, "dst": v, "amount": round(float(d["amount"]), 2), "pass_through_ratio": round(float(d["pass_through_ratio"]), 4)}
                for u, v, d in dg.subgraph(component).edges(data=True)
            ]
            evidence = json.dumps(
                {
                    "component_members": sorted(component),
                    "hub_candidate": hub,
                    "hub_in_degree": int(in_degrees[hub]),
                    "component_size": len(component),
                    "edges": sub_edges,
                }
            )
            for member in component:
                alerts.append({"customer_id": member, "month": m, "evidence": evidence})

    return pd.DataFrame(alerts) if alerts else _empty_alert_frame()


# ---------------------------------------------------------------------------
# WLT-01: Investment / wealth layering
# ---------------------------------------------------------------------------

def evaluate_wealth_layering(con: duckdb.DuckDBPyConnection, params: dict) -> pd.DataFrame:
    sql = """
    WITH wires AS (
        SELECT s.customer_id, s.month, t.transaction_id AS wire_transaction_id,
               t.timestamp AS wire_time, t.amount AS wire_amount
        FROM scaffold s
        JOIN tx_resolved t
            ON t.from_customer = s.customer_id
           AND t.channel = 'wire' AND t.cross_border
           AND t.amount >= ?
           AND t.timestamp > s.month_end - INTERVAL 30 DAY
           AND t.timestamp <= s.month_end
    ),
    journal AS (
        SELECT from_customer AS customer_id, transaction_id, timestamp, amount
        FROM tx_resolved
        WHERE channel = 'internal_journal' AND from_customer = to_customer
          AND EXISTS (
            SELECT 1 FROM relationships r
            WHERE r.relationship_type = 'same_owner'
              AND ((r.account_id_a = tx_resolved.from_account AND r.account_id_b = tx_resolved.to_account)
                OR (r.account_id_a = tx_resolved.to_account AND r.account_id_b = tx_resolved.from_account))
          )
    )
    SELECT w.*,
        (SELECT COUNT(*) FROM journal j WHERE j.customer_id = w.customer_id
           AND j.timestamp >= w.wire_time - INTERVAL (?) DAY AND j.timestamp < w.wire_time) AS n_journal,
        (SELECT COALESCE(SUM(j.amount), 0) FROM journal j WHERE j.customer_id = w.customer_id
           AND j.timestamp >= w.wire_time - INTERVAL (?) DAY AND j.timestamp < w.wire_time) AS journal_volume
    FROM wires w
    """
    wd = params["window_days"]
    df = con.execute(sql, [params["min_outbound_wire"], wd, wd]).df()
    if df.empty:
        return _empty_alert_frame()

    hits = df[(df["n_journal"] >= params["min_journal_transfers"]) & (df["journal_volume"] >= params["min_journal_volume"])]
    if hits.empty:
        return _empty_alert_frame()

    alerts = []
    for (customer_id, month), group in hits.groupby(["customer_id", "month"]):
        best = group.loc[group["journal_volume"].idxmax()]
        journal_sql = """
        SELECT transaction_id FROM tx_resolved
        WHERE from_customer = ? AND to_customer = ? AND channel = 'internal_journal'
          AND timestamp >= ? AND timestamp < ?
        """
        journal_tx = con.execute(
            journal_sql, [customer_id, customer_id, best["wire_time"] - pd.Timedelta(days=wd), best["wire_time"]]
        ).df()
        alerts.append(
            {
                "customer_id": customer_id,
                "month": month,
                "evidence": json.dumps(
                    {
                        "journal_transaction_ids": sorted(journal_tx["transaction_id"].tolist()),
                        "journal_volume": round(float(best["journal_volume"]), 2),
                        "n_journal": int(best["n_journal"]),
                        "wire_transaction_id": best["wire_transaction_id"],
                        "wire_amount": round(float(best["wire_amount"]), 2),
                    }
                ),
            }
        )
    return pd.DataFrame(alerts)


def _empty_alert_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["customer_id", "month", "evidence"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

EVALUATORS = {
    "STR-01": lambda con, params, gen_config: evaluate_structuring(con, params, gen_config["reporting_threshold"]),
    "RMV-01": lambda con, params, gen_config: evaluate_rapid_movement(con, params),
    "FAN-01": lambda con, params, gen_config: evaluate_fan_simple(con, params),
    "FAN-02": lambda con, params, gen_config: evaluate_fan_ring(con, params, gen_config),
    "WLT-01": lambda con, params, gen_config: evaluate_wealth_layering(con, params),
}


def run_all_scenarios(
    con: duckdb.DuckDBPyConnection, scenarios_config: dict, gen_config: dict
) -> dict[str, pd.DataFrame]:
    results = {}
    for scenario_id, spec in scenarios_config.items():
        alerts = EVALUATORS[scenario_id](con, spec["params"], gen_config)
        alerts = alerts.copy()
        alerts["scenario_id"] = scenario_id
        alerts["typology"] = spec["typology"]
        results[scenario_id] = alerts
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run rule-based AML scenarios at their default parameters.")
    parser.add_argument("--generator-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--scenarios-config", type=Path, default=DEFAULT_SCENARIOS_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "artifacts" / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "scenarios")
    args = parser.parse_args(argv)

    gen_config = load_config(args.generator_config)
    scenarios_config = load_scenarios_config(args.scenarios_config)

    con = build_context_from_dir(args.data_dir, gen_config)
    results = run_all_scenarios(con, scenarios_config, gen_config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(args.data_dir / "labels.parquet")
    for scenario_id, alerts in results.items():
        alerts.to_parquet(args.out_dir / f"alerts_{scenario_id}.parquet", index=False)
        typology = scenarios_config[scenario_id]["typology"]
        labeled_ids = set(labels.loc[labels["typology"] == typology, "customer_id"])
        alerted_ids = set(alerts["customer_id"])
        caught = len(labeled_ids & alerted_ids)
        print(
            f"{scenario_id} ({scenarios_config[scenario_id]['name']}): "
            f"{len(alerts):,} alert rows, {alerts['customer_id'].nunique():,} distinct customers alerted, "
            f"{caught} of {len(labeled_ids)} labeled {typology} customers caught"
        )

    print(f"\nWrote alert tables to {args.out_dir}")


if __name__ == "__main__":
    main()
