"""Threshold tuning with ATL/BTL testing (Phase 5).

Time split (months 1-8 tuning, 9-10 validation, 11-12 test) is the same
one used throughout the project. Methodology, stated explicitly because
it's a design choice that needs to be defensible in an interview:

  1. For each scenario, sweep the full parameter grid from scenarios.yaml,
     computing alert volume, productive rate, and detection rate on the
     TUNING period (months 1-8) - the largest window, giving the most
     statistical power to characterize the volume/accuracy tradeoff
     curve for every grid point.
  2. Among grid points whose tuning-period average monthly alert volume
     fits the segment's analyst capacity (capacity.yaml), SELECT the
     operating point that maximizes VALIDATION-period (months 9-10)
     detection - this is "choosing thresholds on a validation period."
  3. ATL/BTL: step the scenario's primary threshold parameter down 1-2
     grid increments from the chosen point and recompute on the TUNING
     period (again, more cases to work with) - the extra alerts this
     would add, and how many are productive, estimate what the chosen
     line misses.
  4. Report the chosen point's performance on the TEST period (months
     11-12), untouched - the only place test data is used, and only for
     the already-chosen point, never for selection.

Where a segment has fewer than
capacity.yaml:min_tuning_cases_for_segment_specific_threshold labeled
tuning-period cases for a scenario's typology, that segment falls back to
the scenario's POOLED (whole-population) chosen point instead of its own
- a threshold "chosen" from 0-4 cases isn't a real decision.

Limitations, carried into every tuning_<scenario>.json and worth
repeating in any write-up: (1) month-boundary effects - a typology
instance whose burst straddles a month-end can be partially invisible to
BOTH adjacent months' trailing-window evaluation (see RMV-01's one
documented miss in scenarios.py); point-in-time, no-future-leakage
evaluation has this cost structurally, not as a bug. (2) FAN-02's
detection-rate numbers rest on only 5 labeled ring instances in the whole
dataset - any ratio computed from that (e.g. "60% of rings recovered") is
really "3 of 5", and should be read as directional, not a precise rate.

Performance note: most of a scenario's grid parameters are cheap
pandas-level filters once a handful of expensive, correlated-subquery SQL
candidate tables are computed (one per distinct value of the parameters
that actually change the SQL - typically a window size). See
`*_raw_candidates` vs `*_apply_thresholds` below.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml

from aml_lab.generate import DEFAULT_CONFIG_PATH, load_config, month_bounds, sim_window
from aml_lab.scenarios import (
    DEFAULT_SCENARIOS_CONFIG_PATH,
    REPO_ROOT,
    build_context_from_dir,
    load_scenarios_config,
)

DEFAULT_CAPACITY_CONFIG_PATH = REPO_ROOT / "config" / "capacity.yaml"

PERIOD_BANDS = {"tuning": (1, 8), "validation": (9, 10), "test": (11, 12)}


def load_capacity_config(path: Path = DEFAULT_CAPACITY_CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def month_to_period(month: int) -> str:
    for period, (lo, hi) in PERIOD_BANDS.items():
        if lo <= month <= hi:
            return period
    raise ValueError(f"month {month} outside 1-12")


def segment_capacity(capacity_cfg: dict, n_customers: int) -> int:
    raw = capacity_cfg["max_alert_rate_per_month"] * n_customers
    return int(np.clip(round(raw), capacity_cfg["min_alerts_per_month"], capacity_cfg["max_alerts_per_month"]))


# ---------------------------------------------------------------------------
# Raw candidates: one SQL run per distinct value of the parameters that
# actually change the query (window sizes, materiality floors). Everything
# else (count/ratio/share thresholds) is applied afterward in pandas.
# ---------------------------------------------------------------------------

def str01_raw_candidates(con, window_days: int, pct_below_threshold: float, reporting_threshold: float) -> pd.DataFrame:
    lower = (1 - pct_below_threshold) * reporting_threshold
    sql = """
    WITH qualifying AS (
        SELECT to_customer AS customer_id, transaction_id, timestamp, amount
        FROM tx_resolved
        WHERE channel = 'cash_deposit' AND amount >= ? AND amount < ? AND to_customer IS NOT NULL
    ),
    candidates AS (
        SELECT s.customer_id, s.month, q.timestamp,
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
    SELECT customer_id, month, MAX(count_in_window) AS max_count FROM candidates GROUP BY customer_id, month
    """
    df = con.execute(sql, [lower, reporting_threshold, window_days]).df()
    df["period"] = df["month"].map(month_to_period)
    return df


def str01_apply_thresholds(raw: pd.DataFrame, min_deposits: int) -> pd.DataFrame:
    return raw.loc[raw["max_count"] >= min_deposits, ["customer_id", "month", "period"]]


def rmv01_raw_candidates(con, window_hours: int, min_inflow: float) -> pd.DataFrame:
    sql = """
    WITH candidates AS (
        SELECT s.customer_id, s.month, i.timestamp AS inflow_time, i.amount AS inflow_amount,
            (
                SELECT COALESCE(SUM(o.amount), 0) FROM tx_flows o
                WHERE o.customer_id = i.customer_id AND o.direction = 'out'
                  AND o.timestamp > i.timestamp
                  AND o.timestamp <= LEAST(i.timestamp + INTERVAL (?) HOUR, s.month_end)
            ) AS outflow_followed
        FROM scaffold s
        JOIN tx_flows i
            ON i.customer_id = s.customer_id AND i.direction = 'in' AND i.amount >= ?
           AND i.timestamp > s.month_end - INTERVAL 30 DAY AND i.timestamp <= s.month_end
    )
    SELECT customer_id, month, MAX(outflow_followed / inflow_amount) AS max_ratio FROM candidates GROUP BY customer_id, month
    """
    df = con.execute(sql, [window_hours, min_inflow]).df()
    df["period"] = df["month"].map(month_to_period)
    return df


def rmv01_apply_thresholds(raw: pd.DataFrame, min_outflow_pct: float) -> pd.DataFrame:
    return raw.loc[raw["max_ratio"] >= min_outflow_pct, ["customer_id", "month", "period"]]


def fan01_raw_candidates(con, window_days: int) -> pd.DataFrame:
    sql = """
    WITH inflow_events AS (
        SELECT s.customer_id, s.month, s.month_end, f.timestamp AS anchor_time
        FROM scaffold s
        JOIN tx_flows f
            ON f.customer_id = s.customer_id AND f.direction = 'in' AND f.counterparty_customer IS NOT NULL
           AND f.timestamp > s.month_end - INTERVAL 30 DAY AND f.timestamp <= s.month_end
    ),
    windowed AS (
        SELECT DISTINCT customer_id, month, anchor_time,
            (SELECT COUNT(DISTINCT o.counterparty_customer) FROM tx_flows o
             WHERE o.customer_id = inflow_events.customer_id AND o.direction = 'in' AND o.counterparty_customer IS NOT NULL
               AND o.timestamp >= anchor_time AND o.timestamp < LEAST(anchor_time + INTERVAL (?) DAY, month_end + INTERVAL 1 SECOND)
            ) AS distinct_senders,
            (SELECT COALESCE(SUM(o.amount), 0) FROM tx_flows o
             WHERE o.customer_id = inflow_events.customer_id AND o.direction = 'in' AND o.counterparty_customer IS NOT NULL
               AND o.timestamp >= anchor_time AND o.timestamp < LEAST(anchor_time + INTERVAL (?) DAY, month_end + INTERVAL 1 SECOND)
            ) AS total_in,
            (SELECT COALESCE(SUM(o.amount), 0) FROM tx_flows o
             WHERE o.customer_id = inflow_events.customer_id AND o.direction = 'out'
               AND o.timestamp >= anchor_time AND o.timestamp < LEAST(anchor_time + INTERVAL (?) DAY, month_end + INTERVAL 1 SECOND)
            ) AS total_out
        FROM inflow_events
    )
    SELECT * FROM windowed
    """
    df = con.execute(sql, [window_days, window_days, window_days]).df()
    df["forwarded_share"] = df["total_out"] / df["total_in"]
    df["period"] = df["month"].map(month_to_period)
    return df


def fan01_apply_thresholds(raw: pd.DataFrame, min_distinct_senders: int, min_forwarded_share: float) -> pd.DataFrame:
    hits = raw[(raw["distinct_senders"] >= min_distinct_senders) & (raw["forwarded_share"] >= min_forwarded_share)]
    return hits[["customer_id", "month", "period"]].drop_duplicates()


def wlt01_raw_candidates(con, window_days: int, min_outbound_wire: float) -> pd.DataFrame:
    sql = """
    WITH wires AS (
        SELECT s.customer_id, s.month, t.timestamp AS wire_time, t.amount AS wire_amount
        FROM scaffold s
        JOIN tx_resolved t
            ON t.from_customer = s.customer_id AND t.channel = 'wire' AND t.cross_border AND t.amount >= ?
           AND t.timestamp > s.month_end - INTERVAL 30 DAY AND t.timestamp <= s.month_end
    ),
    journal AS (
        SELECT from_customer AS customer_id, timestamp, amount FROM tx_resolved
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
    df = con.execute(sql, [min_outbound_wire, window_days, window_days]).df()
    df["period"] = df["month"].map(month_to_period)
    return df


def wlt01_apply_thresholds(raw: pd.DataFrame, min_journal_transfers: int, min_journal_volume: float) -> pd.DataFrame:
    hits = raw[(raw["n_journal"] >= min_journal_transfers) & (raw["journal_volume"] >= min_journal_volume)]
    return hits.groupby(["customer_id", "month", "period"]).size().reset_index()[["customer_id", "month", "period"]]


def fan02_raw_edges(con, start, window_days: int, window_hours: int, min_edge_amount: float, months: int) -> pd.DataFrame:
    sql = """
    WITH edges AS (
        SELECT customer_id AS src, counterparty_customer AS dst, SUM(amount) AS edge_amount, MIN(timestamp) AS first_ts
        FROM tx_flows
        WHERE direction = 'out' AND counterparty_customer IS NOT NULL AND timestamp > ? AND timestamp <= ?
        GROUP BY customer_id, counterparty_customer
        HAVING SUM(amount) >= ?
    )
    SELECT e.*,
        (SELECT COALESCE(SUM(o.amount), 0) FROM tx_flows o
         WHERE o.customer_id = e.dst AND o.direction = 'out'
           AND o.timestamp > e.first_ts AND o.timestamp <= LEAST(e.first_ts + INTERVAL (?) HOUR, ?)
        ) AS dst_outflow_followed
    FROM edges e
    """
    frames = []
    for m in range(1, months + 1):
        _month_start, month_end = month_bounds(start, m)
        month_end_ts = pd.Timestamp(month_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        window_start_ts = month_end_ts - pd.Timedelta(days=window_days)
        e = con.execute(sql, [window_start_ts, month_end_ts, min_edge_amount, window_hours, month_end_ts]).df()
        if e.empty:
            continue
        e["month"] = m
        frames.append(e)
    edges = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["src", "dst", "edge_amount", "first_ts", "dst_outflow_followed", "month"]
    )
    edges["pass_through_ratio"] = edges["dst_outflow_followed"] / edges["edge_amount"]
    return edges


def fan02_apply_thresholds(
    edges: pd.DataFrame, pass_through_ratio: float, min_hub_in_degree: int, min_component_size: int, months: int
) -> pd.DataFrame:
    kept = edges[edges["pass_through_ratio"] >= pass_through_ratio]
    rows = []
    for m in range(1, months + 1):
        month_edges = kept[kept["month"] == m]
        if month_edges.empty:
            continue
        dg = nx.DiGraph()
        dg.add_edges_from(zip(month_edges["src"], month_edges["dst"]))
        undirected = dg.to_undirected()
        for component in nx.connected_components(undirected):
            if len(component) < min_component_size:
                continue
            in_degrees = {n: dg.in_degree(n) for n in component}
            if max(in_degrees.values()) < min_hub_in_degree:
                continue
            period = month_to_period(m)
            rows.extend({"customer_id": c, "month": m, "period": period} for c in component)
    return pd.DataFrame(rows, columns=["customer_id", "month", "period"])


# ---------------------------------------------------------------------------
# Metrics given an alert table (customer_id, month, period)
# ---------------------------------------------------------------------------

def labeled_ids_by_period(labels: pd.DataFrame, typology: str) -> dict[str, set[str]]:
    """Labeled customers for a typology, grouped by which period their
    injected event's start_date falls in - NOT the whole-typology set,
    which would let a customer whose burst was in test inflate test's
    denominator when evaluating validation's detection rate.
    """
    sub = labels[labels["typology"] == typology].copy()
    if sub.empty:
        return {p: set() for p in PERIOD_BANDS}
    sub["month"] = pd.to_datetime(sub["start_date"]).dt.month
    sub["period"] = sub["month"].map(month_to_period)
    return {p: set(sub.loc[sub["period"] == p, "customer_id"]) for p in PERIOD_BANDS}


def compute_metrics(
    alerts: pd.DataFrame,
    period: str,
    segment_customer_ids: set[str] | None,
    labeled_customer_ids_all: set[str],
    labeled_customer_ids_period: set[str],
) -> dict:
    period_alerts = alerts[alerts["period"] == period]
    if segment_customer_ids is not None:
        period_alerts = period_alerts[period_alerts["customer_id"].isin(segment_customer_ids)]
        labeled_in_scope = labeled_customer_ids_period & segment_customer_ids
    else:
        labeled_in_scope = labeled_customer_ids_period

    n_months = PERIOD_BANDS[period][1] - PERIOD_BANDS[period][0] + 1
    alert_rows = len(period_alerts)
    alerted_customers = set(period_alerts["customer_id"])
    n_alerted = len(alerted_customers)
    productive_customers = alerted_customers & labeled_customer_ids_all
    detected = alerted_customers & labeled_in_scope

    return {
        "alert_rows": int(alert_rows),
        "avg_alerts_per_month": round(alert_rows / n_months, 2),
        "n_customers_alerted": n_alerted,
        "n_productive_customers": len(productive_customers),
        "productive_rate": round(len(productive_customers) / n_alerted, 4) if n_alerted else None,
        "n_labeled_in_scope": len(labeled_in_scope),
        "n_detected": len(detected),
        "detection_rate": round(len(detected) / len(labeled_in_scope), 4) if labeled_in_scope else None,
    }


# ---------------------------------------------------------------------------
# Generic per-scenario sweep + selection + ATL/BTL + test evaluation
# ---------------------------------------------------------------------------

SQL_PARAM_KEYS = {
    "STR-01": ("window_days", "pct_below_threshold"),
    "RMV-01": ("window_hours", "min_inflow"),
    "FAN-01": ("window_days",),
    "WLT-01": ("window_days", "min_outbound_wire"),
    "FAN-02": ("pass_through_window_hours",),
}

PRIMARY_PARAM = {
    "STR-01": "min_deposits",
    "RMV-01": "min_outflow_pct",
    "FAN-01": "min_distinct_senders",
    "WLT-01": "min_journal_transfers",
    "FAN-02": "pass_through_ratio",
}


def _grid_combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*[grid[k] for k in keys])]


def _apply(scenario_id: str, raw_cache: dict, params: dict, extra: dict) -> pd.DataFrame:
    if scenario_id == "STR-01":
        key = (params["window_days"], params["pct_below_threshold"])
        raw = raw_cache[key]
        return str01_apply_thresholds(raw, params["min_deposits"])
    if scenario_id == "RMV-01":
        key = (params["window_hours"], params["min_inflow"])
        raw = raw_cache[key]
        return rmv01_apply_thresholds(raw, params["min_outflow_pct"])
    if scenario_id == "FAN-01":
        key = (params["window_days"],)
        raw = raw_cache[key]
        return fan01_apply_thresholds(raw, params["min_distinct_senders"], params["min_forwarded_share"])
    if scenario_id == "WLT-01":
        key = (params["window_days"], params["min_outbound_wire"])
        raw = raw_cache[key]
        return wlt01_apply_thresholds(raw, params["min_journal_transfers"], params["min_journal_volume"])
    if scenario_id == "FAN-02":
        key = (params["pass_through_window_hours"],)
        raw = raw_cache[key]
        return fan02_apply_thresholds(
            raw, params["pass_through_ratio"], params["min_hub_in_degree"], params["min_component_size"], extra["months"]
        )
    raise ValueError(scenario_id)


def _build_raw_cache(scenario_id: str, con, spec: dict, gen_config: dict, extra: dict) -> dict:
    grid = spec["grid"]
    sql_keys = SQL_PARAM_KEYS[scenario_id]
    combos = {tuple(v[k] for k in sql_keys) for v in _grid_combinations(grid)}
    cache = {}
    for combo in combos:
        kwargs = dict(zip(sql_keys, combo))
        if scenario_id == "STR-01":
            cache[combo] = str01_raw_candidates(con, kwargs["window_days"], kwargs["pct_below_threshold"], gen_config["reporting_threshold"])
        elif scenario_id == "RMV-01":
            cache[combo] = rmv01_raw_candidates(con, kwargs["window_hours"], kwargs["min_inflow"])
        elif scenario_id == "FAN-01":
            cache[combo] = fan01_raw_candidates(con, kwargs["window_days"])
        elif scenario_id == "WLT-01":
            cache[combo] = wlt01_raw_candidates(con, kwargs["window_days"], kwargs["min_outbound_wire"])
        elif scenario_id == "FAN-02":
            fixed = spec["params"]
            cache[combo] = fan02_raw_edges(
                con, extra["start"], fixed["window_days"], kwargs["pass_through_window_hours"], fixed["min_edge_amount"], extra["months"]
            )
    return cache


def sweep_scenario(
    scenario_id: str,
    con,
    spec: dict,
    gen_config: dict,
    segments: pd.DataFrame,
    labels: pd.DataFrame,
    capacity_cfg: dict,
) -> dict:
    typology = spec["typology"]
    labeled_customer_ids = set(labels.loc[labels["typology"] == typology, "customer_id"])
    by_period = labeled_ids_by_period(labels, typology)

    start, _end, months = sim_window(gen_config)
    extra = {"start": start, "months": months}
    raw_cache = _build_raw_cache(scenario_id, con, spec, gen_config, extra)

    segment_ids = sorted(segments["segment_assigned"].unique())
    segment_customers = {s: set(segments.loc[segments["segment_assigned"] == s, "customer_id"]) for s in segment_ids}
    all_customers = set(segments["customer_id"])

    combos = _grid_combinations(spec["grid"])
    grid_results = []
    for params in combos:
        alerts = _apply(scenario_id, raw_cache, params, extra)
        row = {"params": params}
        row["pooled_tuning"] = compute_metrics(alerts, "tuning", None, labeled_customer_ids, by_period["tuning"])
        row["pooled_validation"] = compute_metrics(alerts, "validation", None, labeled_customer_ids, by_period["validation"])
        for s in segment_ids:
            row[f"segment_{s}_tuning"] = compute_metrics(
                alerts, "tuning", segment_customers[s], labeled_customer_ids, by_period["tuning"]
            )
            row[f"segment_{s}_validation"] = compute_metrics(
                alerts, "validation", segment_customers[s], labeled_customer_ids, by_period["validation"]
            )
        grid_results.append(row)

    # Labeled tuning-period case counts, to decide the pooled-fallback rule.
    tuning_label_months = labels.copy()
    tuning_label_months["month"] = pd.to_datetime(tuning_label_months["start_date"]).dt.month
    tuning_label_months = tuning_label_months[
        (tuning_label_months["typology"] == typology) & (tuning_label_months["month"].between(1, 8))
    ]
    min_cases = capacity_cfg["min_tuning_cases_for_segment_specific_threshold"]

    def select_operating_point(scope_key_prefix: str, customer_ids: set[str] | None) -> dict:
        cap = segment_capacity(capacity_cfg, len(customer_ids) if customer_ids is not None else len(all_customers))
        feasible = [
            r
            for r in grid_results
            if r[f"{scope_key_prefix}_tuning"]["avg_alerts_per_month"] <= cap
        ]
        pool = feasible if feasible else grid_results
        best = max(
            pool,
            key=lambda r: (
                r[f"{scope_key_prefix}_validation"]["n_detected"],
                r[f"{scope_key_prefix}_validation"]["productive_rate"] or 0,
                -r[f"{scope_key_prefix}_tuning"]["avg_alerts_per_month"],
            ),
        )
        return {
            "params": best["params"],
            "capacity": cap,
            "capacity_constraint_satisfied": bool(feasible),
            "tuning": best[f"{scope_key_prefix}_tuning"],
            "validation": best[f"{scope_key_prefix}_validation"],
        }

    pooled_choice = select_operating_point("pooled", None)

    segment_choices = {}
    for s in segment_ids:
        n_tuning_cases = len(tuning_label_months[tuning_label_months["customer_id"].isin(segment_customers[s])])
        n_validation_cases = len(by_period["validation"] & segment_customers[s])
        # Both checks matter: too few tuning cases means there's nothing
        # to sweep the grid against; zero validation cases means there's
        # nothing to SELECT a segment-specific point against (the
        # selection would degenerate to "whichever grid point has the
        # lowest volume," which isn't a real decision either).
        if n_tuning_cases < min_cases or n_validation_cases < 1:
            choice = dict(pooled_choice)
            choice["fallback_to_pooled"] = True
            choice["reason"] = (
                f"{n_tuning_cases} labeled tuning-period case(s) (need >= {min_cases}) and "
                f"{n_validation_cases} labeled validation-period case(s) (need >= 1) in this segment"
            )
            # Re-derive tuning/validation metrics for THIS segment under the pooled params for comparability.
            alerts = _apply(scenario_id, raw_cache, pooled_choice["params"], extra)
            choice["tuning"] = compute_metrics(
                alerts, "tuning", segment_customers[s], labeled_customer_ids, by_period["tuning"]
            )
            choice["validation"] = compute_metrics(
                alerts, "validation", segment_customers[s], labeled_customer_ids, by_period["validation"]
            )
        else:
            choice = select_operating_point(f"segment_{s}", segment_customers[s])
            choice["fallback_to_pooled"] = False
            choice["n_tuning_cases"] = n_tuning_cases
            choice["n_validation_cases"] = n_validation_cases
        segment_choices[str(s)] = choice

    # --- ATL/BTL on the pooled chosen point (tuning period) -------------
    primary = PRIMARY_PARAM[scenario_id]
    grid_values = sorted(spec["grid"][primary])
    chosen_value = pooled_choice["params"][primary]
    idx = grid_values.index(chosen_value)
    # "Looser" direction catches more: index-1 for count/share mins is
    # lower-is-looser for every scenario here (min_deposits,
    # min_outflow_pct, min_distinct_senders, min_journal_transfers,
    # pass_through_ratio all loosen going down the sorted list).
    steps = [grid_values[idx - k] for k in (1, 2) if idx - k >= 0]

    chosen_alerts = _apply(scenario_id, raw_cache, pooled_choice["params"], extra)
    chosen_tuning_customers = set(chosen_alerts.loc[chosen_alerts["period"] == "tuning", "customer_id"])

    atl_btl = []
    for step_value in steps:
        step_params = dict(pooled_choice["params"])
        step_params[primary] = step_value
        step_alerts = _apply(scenario_id, raw_cache, step_params, extra)
        step_tuning_customers = set(step_alerts.loc[step_alerts["period"] == "tuning", "customer_id"])
        extra_customers = step_tuning_customers - chosen_tuning_customers
        extra_productive = extra_customers & labeled_customer_ids
        atl_btl.append(
            {
                "primary_param": primary,
                "chosen_value": chosen_value,
                "step_value": step_value,
                "n_extra_alerted_customers": len(extra_customers),
                "n_extra_productive": len(extra_productive),
                "extra_productive_rate": round(len(extra_productive) / len(extra_customers), 4) if extra_customers else None,
            }
        )

    # --- Final, untouched test-period numbers for the chosen points -----
    test_results = {"pooled": compute_metrics(chosen_alerts, "test", None, labeled_customer_ids, by_period["test"])}
    for s in segment_ids:
        choice = segment_choices[str(s)]
        alerts_for_seg = chosen_alerts if choice["fallback_to_pooled"] else _apply(scenario_id, raw_cache, choice["params"], extra)
        test_results[str(s)] = compute_metrics(
            alerts_for_seg, "test", segment_customers[s], labeled_customer_ids, by_period["test"]
        )

    # Did the chosen point's ACTUAL volume (not just the tuning-period
    # estimate used for selection) stay within capacity in validation/test?
    cap = pooled_choice["capacity"]
    pooled_choice["validation"]["capacity_satisfied"] = pooled_choice["validation"]["avg_alerts_per_month"] <= cap
    test_results["pooled"]["capacity_satisfied"] = test_results["pooled"]["avg_alerts_per_month"] <= cap

    result = {
        "scenario_id": scenario_id,
        "name": spec["name"],
        "typology": typology,
        "grid_size": len(combos),
        "pooled_choice": pooled_choice,
        "segment_choices": segment_choices,
        "atl_btl": atl_btl,
        "test_results": test_results,
        "grid_results": grid_results,
    }

    if scenario_id == "FAN-02":
        result["ring_report"] = fan02_ring_report(labels, chosen_alerts)

    return result


def fan02_ring_report(labels: pd.DataFrame, chosen_alerts: pd.DataFrame) -> dict:
    """Ring-level view for FAN-02: the per-customer detection numbers
    elsewhere conflate "how many of 5 rings did we find" with "how many
    total members did we catch," which is a very different, more
    actionable question for an investigator (one spotted member is often
    enough to open a case on a whole ring). A ring counts as RECOVERED if
    its hub - the structural anchor the detector actually looks for - is
    among the alerted customers for that ring's period; member recall is
    reported separately alongside it.
    """
    fan_labels = labels[labels["typology"] == "fan_in_fan_out"].copy()
    if fan_labels.empty:
        return {"rings": [], "by_period": {}}
    fan_labels["month"] = pd.to_datetime(fan_labels["start_date"]).dt.month
    fan_labels["period"] = fan_labels["month"].map(month_to_period)

    alerted_by_period = {
        p: set(chosen_alerts.loc[chosen_alerts["period"] == p, "customer_id"]) for p in PERIOD_BANDS
    }

    rings = []
    for (start_date, end_date), group in fan_labels.groupby(["start_date", "end_date"]):
        period = group["period"].iloc[0]
        members = set(group["customer_id"])
        hub_rows = group[group["role"] == "hub"]
        hub = hub_rows["customer_id"].iloc[0] if not hub_rows.empty else None
        caught_members = members & alerted_by_period[period]
        rings.append(
            {
                "start_date": str(start_date),
                "end_date": str(end_date),
                "period": period,
                "hub": hub,
                "hub_caught": hub in caught_members if hub else False,
                "n_members": len(members),
                "n_members_caught": len(caught_members),
            }
        )

    by_period = {}
    for p in PERIOD_BANDS:
        period_rings = [r for r in rings if r["period"] == p]
        by_period[p] = {
            "n_rings": len(period_rings),
            "n_rings_recovered": sum(1 for r in period_rings if r["hub_caught"]),
            "n_members_total": sum(r["n_members"] for r in period_rings),
            "n_members_caught": sum(r["n_members_caught"] for r in period_rings),
        }

    return {"rings": rings, "by_period": by_period}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_scenario(result: dict, out_path: Path) -> None:
    primary = PRIMARY_PARAM[result["scenario_id"]]
    rows = []
    for r in result["grid_results"]:
        rows.append(
            {
                "value": r["params"][primary],
                "avg_alerts_per_month": r["pooled_tuning"]["avg_alerts_per_month"],
                "productive_rate": r["pooled_tuning"]["productive_rate"],
                "detection_rate": r["pooled_tuning"]["detection_rate"],
            }
        )
    df = pd.DataFrame(rows).groupby("value").mean().reset_index().sort_values("value")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(df["value"], df["avg_alerts_per_month"], marker="o")
    axes[0].axvline(result["pooled_choice"]["params"][primary], color="red", linestyle="--", label="chosen")
    axes[0].set_xlabel(primary)
    axes[0].set_ylabel("avg alerts / month (tuning, pooled)")
    axes[0].set_title("Alert volume vs. threshold")
    axes[0].legend()

    axes[1].plot(df["value"], df["productive_rate"], marker="o", color="green")
    axes[1].axvline(result["pooled_choice"]["params"][primary], color="red", linestyle="--")
    axes[1].set_xlabel(primary)
    axes[1].set_ylabel("productive rate (tuning, pooled)")
    axes[1].set_title("Productive rate vs. threshold")

    chosen_val = result["pooled_choice"]["params"][primary]
    chosen_productive = result["pooled_choice"]["tuning"]["n_productive_customers"]
    # True cumulative total (chosen + extra), not just the marginal "extra"
    # figure, so the bars tell a monotonic "loosen the line, catch more
    # (with diminishing returns)" story at a glance.
    bars = [(a["step_value"], chosen_productive + a["n_extra_productive"], False) for a in result["atl_btl"]]
    bars.append((chosen_val, chosen_productive, True))
    bars.sort(key=lambda b: b[0])
    atl_labels = [f"{v} (chosen)" if is_chosen else str(v) for v, _, is_chosen in bars]
    atl_totals = [n for _, n, _ in bars]
    axes[2].bar(atl_labels, atl_totals, color=["red" if is_chosen else "orange" for _, _, is_chosen in bars])
    axes[2].set_xlabel(f"{primary} (stepped down = ATL/BTL)")
    axes[2].set_ylabel("productive customers (cumulative, tuning)")
    axes[2].set_title("ATL/BTL: productive cases at/below the line")

    fig.suptitle(f"SYNTHETIC DATA - {result['scenario_id']} ({result['name']}) threshold tuning", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

LIMITATIONS = [
    (
        "Month-boundary effects: a typology instance whose burst straddles a month-end can be partially invisible "
        "to BOTH adjacent months' trailing-window evaluation (documented case: a rapid_movement instance spanning "
        "Aug31-Sep1 in the default dataset; confirmed to also affect FAN-02 - a January ring with 6 senders "
        "Jan23-29 and hub forward-outs Feb1-7 is invisible in BOTH January (outflows fall after month-end, "
        "excluded) and February (inflows fall before the 30-day lookback start, excluded)). This is a structural "
        "cost of point-in-time, no-future-leakage evaluation, not a bug - see scenarios.py."
    ),
    (
        "FAN-01/FAN-02's tuning-period detection (0 of 17) looks alarming next to validation (2/17, 7/17) and "
        "test (2/11, 9/11), but is fully explained, not a bug: the 2 tuning-period rings are the dataset's worst-"
        "case instances - one is the month-boundary case above, the other has only 4 senders. The capacity "
        "constraint rules out any threshold loose enough to catch a 4-sender ring (min_hub_in_degree 3-4 produces "
        "355-1046 alerts/month against a 200 cap, driven by the new legitimate pass-through look-alikes), leaving "
        "only min_hub_in_degree>=5 as capacity-feasible - which the 4-sender ring can never clear regardless of "
        "timing. Validation/test's 3 rings (6-7 senders each) aren't affected by either limitation. Verified "
        "directly against every grid point, not just the chosen one - see tuning_FAN-02.json's grid_results."
    ),
    (
        "FAN-02's detection-rate numbers rest on only 5 labeled ring instances in the whole dataset. Any recovery "
        "ratio (e.g. 'X of 5 rings') should be read as a small, directional count, not a precise rate - with n=5, "
        "one more or fewer ring recovered swings the rate by 20 points. See ring_report for the per-ring, "
        "per-period breakdown (rings recovered vs. member recall) rather than reading detection_rate alone."
    ),
    (
        "STR-01's test-period dip (4/8 vs. validation's 4/4) is a genuine threshold-generalization gap, not drift: "
        "the 4 missed test-period instances all have deposits spread across 7-10 days, which the chosen "
        "window_days=7 structurally cannot fully capture in any single 7-day slide. window_days=10 and 14 tie "
        "window_days=7 on validation (4/4 each, since validation's 4 cases all happened to be tightly spread) and "
        "score higher on tuning (23/28 vs 22/28); the tie-break toward minimum alert volume - defensible on its "
        "own, but under-informed with only 4 validation cases to discriminate on - picked the narrowest window, "
        "which doesn't generalize to wider-spread test-period bursts. A larger validation set would likely have "
        "broken the tie differently."
    ),
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Threshold tuning with ATL/BTL testing (Phase 5).")
    parser.add_argument("--generator-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--scenarios-config", type=Path, default=DEFAULT_SCENARIOS_CONFIG_PATH)
    parser.add_argument("--capacity-config", type=Path, default=DEFAULT_CAPACITY_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "artifacts" / "data")
    parser.add_argument("--segments-path", type=Path, default=REPO_ROOT / "artifacts" / "segments" / "customer_segments.parquet")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "metrics")
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "artifacts" / "figures")
    args = parser.parse_args(argv)

    gen_config = load_config(args.generator_config)
    scenarios_config = load_scenarios_config(args.scenarios_config)
    capacity_cfg = load_capacity_config(args.capacity_config)

    segments = pd.read_parquet(args.segments_path)[["customer_id", "segment_assigned"]]
    labels = pd.read_parquet(args.data_dir / "labels.parquet")
    con = build_context_from_dir(args.data_dir, gen_config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for scenario_id, spec in scenarios_config.items():
        result = sweep_scenario(scenario_id, con, spec, gen_config, segments, labels, capacity_cfg)
        result["limitations"] = LIMITATIONS
        out_path = args.out_dir / f"tuning_{scenario_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        plot_scenario(result, args.figures_dir / f"tuning_{scenario_id}.png")

        pc = result["pooled_choice"]
        te = result["test_results"]["pooled"]
        print(f"{scenario_id}: chosen pooled params={pc['params']}")
        for label, m in [("tuning", pc["tuning"]), ("validation", pc["validation"]), ("test", te)]:
            tp = m["n_productive_customers"]
            fp = m["n_customers_alerted"] - tp
            cap_note = f" (capacity_satisfied={m['capacity_satisfied']})" if "capacity_satisfied" in m else ""
            print(
                f"  {label}: {m['n_detected']}/{m['n_labeled_in_scope']} detected, TP={tp} FP={fp}, "
                f"productive_rate={m['productive_rate']}, {m['avg_alerts_per_month']} alerts/mo{cap_note}"
            )
        if "ring_report" in result:
            print("  ring_report (rings recovered / members caught, per period):")
            for period, stats in result["ring_report"]["by_period"].items():
                print(
                    f"    {period}: {stats['n_rings_recovered']} of {stats['n_rings']} rings, "
                    f"{stats['n_members_caught']} of {stats['n_members_total']} members"
                )
        print("  segments:", {
            s: ("pooled" if c["fallback_to_pooled"] else f"own ({c['params']})")
            for s, c in result["segment_choices"].items()
        })
        if result["atl_btl"]:
            print("  ATL/BTL (tuning period):", result["atl_btl"])
        print(f"  Wrote {out_path}")

    print("\nLimitations:")
    for item in LIMITATIONS:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
