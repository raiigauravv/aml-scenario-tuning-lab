"""DuckDB feature layer orchestration for the AML Scenario Tuning Lab.

Most feature logic lives in `features.sql` (readable SQL, trailing
point-in-time windows). This module:
  1. registers the generated parquet tables (or in-memory DataFrames, for
     tests) plus small scaffold tables (months, params, country_risk) that
     the SQL needs,
  2. runs features.sql,
  3. computes the graph features (in/out degree, connected-component size,
     community size) that SQL alone can't express, using the same
     trailing-30-day same-bank-customer edge definition, via networkx,
  4. merges everything into one customer-month feature table.

An optional PySpark engine for the SQL portion is a possible future
addition (see the project README/addendum) but is intentionally not
implemented here; DuckDB is the only engine this module uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import networkx as nx
import pandas as pd

from aml_lab.generate import DEFAULT_CONFIG_PATH, load_config, month_bounds, sim_window

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_SQL_PATH = Path(__file__).resolve().parent / "features.sql"

TABLE_NAMES = ["customers", "accounts", "relationships", "transactions", "transaction_labels", "labels"]


# ---------------------------------------------------------------------------
# Scaffold tables that features.sql expects
# ---------------------------------------------------------------------------

def months_table(config: dict) -> pd.DataFrame:
    start, _end, months = sim_window(config)
    rows = []
    for m in range(1, months + 1):
        month_start, month_end = month_bounds(start, m)
        rows.append(
            {
                "month": m,
                "month_start": pd.Timestamp(month_start),
                "month_end": pd.Timestamp(month_end) + pd.Timedelta(hours=23, minutes=59, seconds=59),
            }
        )
    return pd.DataFrame(rows)


def country_risk_table(config: dict) -> pd.DataFrame:
    cr = config["country_risk"]
    rows = [{"code": cr["domestic_code"], "risk_tier": "domestic"}]
    rows += [{"code": c["code"], "risk_tier": c["risk_tier"]} for c in cr["foreign_countries"]]
    return pd.DataFrame(rows)


def register_scaffold_tables(con: duckdb.DuckDBPyConnection, config: dict) -> None:
    con.register("_months_df", months_table(config))
    con.execute("CREATE OR REPLACE TABLE months AS SELECT * FROM _months_df")
    con.register("_country_risk_df", country_risk_table(config))
    con.execute("CREATE OR REPLACE TABLE country_risk AS SELECT * FROM _country_risk_df")
    con.execute(
        f"CREATE OR REPLACE TABLE params AS SELECT {float(config['reporting_threshold'])} AS reporting_threshold"
    )


def load_tables_from_dir(con: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    for name in TABLE_NAMES:
        path = data_dir / f"{name}.parquet"
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")


def load_tables_from_frames(con: duckdb.DuckDBPyConnection, data: dict[str, pd.DataFrame]) -> None:
    for name in TABLE_NAMES:
        con.register(f"_{name}_df", data[name])
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM _{name}_df")


def run_features_sql(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(FEATURES_SQL_PATH.read_text())


# ---------------------------------------------------------------------------
# Graph features (networkx) - same-bank customer-to-customer edges only,
# trailing 30 days ending at each month_end. This is the piece SQL can't
# express: in/out degree, connected-component size, and community size.
#
# Caveat for Phase 4 (the planned ring-level scenario): once ordinary
# peer-to-peer activity exists (family/friends/roommates - see generate.py),
# the same-bank graph is small-world enough that connected_components and
# louvain_communities mostly return one giant component/community covering
# most of the customer base, which swamps any individual mule ring. Degree
# is a much cleaner signal, and even then mainly for the fan-in/fan-out
# HUB specifically, evaluated in its actual burst month rather than
# "best month in a year" - see the labeled-vs-unlabeled comparison in
# tests/test_features.py. A real ring detector will need a filtered or
# edge-weighted subgraph (e.g. only edges above some amount/recency), not
# components/communities over the full raw graph.
# ---------------------------------------------------------------------------

def compute_graph_features(con: duckdb.DuckDBPyConnection, config: dict) -> pd.DataFrame:
    start, _end, months = sim_window(config)
    customer_ids = con.execute("SELECT customer_id FROM customers").df()["customer_id"].tolist()

    rows = []
    for m in range(1, months + 1):
        _month_start, month_end = month_bounds(start, m)
        month_end_ts = pd.Timestamp(month_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        window_start_ts = month_end_ts - pd.Timedelta(days=30)

        edges = con.execute(
            """
            SELECT customer_id AS src, counterparty_customer AS dst, SUM(amount) AS amount
            FROM tx_flows
            WHERE direction = 'out'
              AND counterparty_customer IS NOT NULL
              AND timestamp > ?
              AND timestamp <= ?
            GROUP BY customer_id, counterparty_customer
            """,
            [window_start_ts, month_end_ts],
        ).df()

        g = nx.DiGraph()
        g.add_nodes_from(customer_ids)
        for src, dst, amount in zip(edges["src"], edges["dst"], edges["amount"]):
            g.add_edge(src, dst, weight=float(amount))

        undirected = g.to_undirected()

        comp_size: dict[str, int] = {}
        for comp in nx.connected_components(undirected):
            size = len(comp)
            for node in comp:
                comp_size[node] = size

        community_size: dict[str, int] = {}
        for community in nx.algorithms.community.louvain_communities(undirected, weight="weight", seed=42):
            size = len(community)
            for node in community:
                community_size[node] = size

        in_degree = dict(g.in_degree())
        out_degree = dict(g.out_degree())

        for cust in customer_ids:
            rows.append(
                {
                    "customer_id": cust,
                    "month": m,
                    "graph_in_degree_30d": in_degree.get(cust, 0),
                    "graph_out_degree_30d": out_degree.get(cust, 0),
                    "component_size_30d": comp_size.get(cust, 1),
                    "community_size_30d": community_size.get(cust, 1),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compute_features_from_con(con: duckdb.DuckDBPyConnection, config: dict) -> pd.DataFrame:
    register_scaffold_tables(con, config)
    run_features_sql(con)
    sql_features = con.execute("SELECT * FROM customer_month_features").df()
    graph_features = compute_graph_features(con, config)
    merged = sql_features.merge(graph_features, on=["customer_id", "month"], how="left")
    for col in ["graph_in_degree_30d", "graph_out_degree_30d", "component_size_30d", "community_size_30d"]:
        default = 1 if col.endswith("size_30d") else 0
        merged[col] = merged[col].fillna(default).astype(int)
    return merged


def compute_features_from_frames(data: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    con = duckdb.connect()
    load_tables_from_frames(con, data)
    result = compute_features_from_con(con, config)
    con.close()
    return result


def compute_features(data_dir: Path, config: dict) -> pd.DataFrame:
    con = duckdb.connect()
    load_tables_from_dir(con, data_dir)
    result = compute_features_from_con(con, config)
    con.close()
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute the customer-month feature table (DuckDB).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "artifacts" / "data")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "features" / "customer_month_features.parquet")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    features = compute_features(args.data_dir, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.out, index=False)
    print(f"Computed {len(features):,} customer-month feature rows -> {args.out}")


if __name__ == "__main__":
    main()
