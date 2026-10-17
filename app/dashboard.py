"""Streamlit dashboard (Phase 8).

Reads only precomputed results from artifacts/metrics, artifacts/papers,
and artifacts/figures - the app never recomputes a sweep, a model
prediction, or a SHAP value. Every number here already exists in a JSON
or CSV written by an earlier phase.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "artifacts" / "metrics"
PAPERS_DIR = REPO_ROOT / "artifacts" / "papers"
FIGURES_DIR = REPO_ROOT / "artifacts" / "figures"

SEGMENT_LABELS = {0: "retail", 1: "mixed", 2: "small_business", 3: "HNW"}
SCENARIO_IDS = ["STR-01", "RMV-01", "FAN-01", "FAN-02", "WLT-01"]

st.set_page_config(page_title="AML Scenario Tuning Lab", layout="wide")


# ---------------------------------------------------------------------------
# Cached, read-only loaders - no recomputation anywhere in this file.
# ---------------------------------------------------------------------------

@st.cache_data
def load_tuning_json(scenario_id: str) -> dict | None:
    path = METRICS_DIR / f"tuning_{scenario_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_alert_queue() -> pd.DataFrame | None:
    path = METRICS_DIR / "alert_queue_test.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.sort_values("xgb_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


@st.cache_data
def list_papers() -> list[str]:
    if not PAPERS_DIR.exists():
        return []
    return sorted(p.stem for p in PAPERS_DIR.glob("*.md"))


@st.cache_data
def load_paper(name: str) -> str:
    return (PAPERS_DIR / f"{name}.md").read_text()


def fmt_rate(numerator, denominator) -> str:
    if not denominator:
        return "n/a (0/0)"
    return f"{numerator / denominator:.2f} ({numerator}/{denominator})"


def find_grid_row(grid_results: list[dict], params: dict) -> dict | None:
    for row in grid_results:
        if row["params"] == params:
            return row
    return None


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

st.warning(
    "**SYNTHETIC DATA** - every customer, account, transaction, alert, and score shown below is generated, "
    "not real. No real customers, investigations, or regulatory filings are referenced anywhere in this app.",
    icon="⚠️",
)
st.title("AML Scenario Tuning Lab")

tab_whatif, tab_queue, tab_papers = st.tabs(["Threshold What-If", "Alert Queue", "Papers"])


# ---------------------------------------------------------------------------
# Tab 1: Threshold what-if (precomputed grid sweep only)
# ---------------------------------------------------------------------------

with tab_whatif:
    st.header("Threshold what-if")
    st.caption(
        "Pick a scenario and segment, move the sliders across Phase 5's precomputed parameter grid. "
        "Tuning/validation metrics update instantly from the saved sweep - nothing is recomputed. "
        "Test-period performance and the ATL/BTL table below are shown only for Phase 5's actual chosen "
        "threshold, since that is the only point evaluated on test data."
    )

    scenario_id = st.selectbox("Scenario", SCENARIO_IDS, key="whatif_scenario")
    result = load_tuning_json(scenario_id)

    if result is None:
        st.error(f"No tuning results found for {scenario_id}. Run `make tuning` first.")
    else:
        segment_id = st.selectbox(
            "Segment",
            sorted(int(s) for s in result["segment_choices"]),
            format_func=lambda s: f"{s} ({SEGMENT_LABELS.get(s, '?')})",
            key="whatif_segment",
        )

        grid = result["grid_results"][0]["params"].keys()
        spec_grid = {}
        for row in result["grid_results"]:
            for k, v in row["params"].items():
                spec_grid.setdefault(k, set()).add(v)
        spec_grid = {k: sorted(v) for k, v in spec_grid.items()}

        chosen_params = result["pooled_choice"]["params"]
        st.caption(f"Phase 5 chose (pooled): {chosen_params}")

        selected_params = {}
        cols = st.columns(len(spec_grid))
        for col, (param, values) in zip(cols, spec_grid.items()):
            with col:
                selected_params[param] = st.select_slider(
                    param, options=values, value=chosen_params.get(param, values[0]), key=f"whatif_{scenario_id}_{param}"
                )

        row = find_grid_row(result["grid_results"], selected_params)
        if row is None:
            st.error("No precomputed result for this exact combination (shouldn't happen if sliders stay on grid values).")
        else:
            scope_key = f"segment_{segment_id}"
            tuning_m = row.get(f"{scope_key}_tuning", row["pooled_tuning"])
            validation_m = row.get(f"{scope_key}_validation", row["pooled_validation"])

            st.subheader(f"Segment {segment_id} ({SEGMENT_LABELS.get(segment_id, '?')}) - selected parameters")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Alert volume (tuning, /mo)", tuning_m["avg_alerts_per_month"])
            m2.metric(
                "Productive rate (tuning)",
                fmt_rate(tuning_m["n_productive_customers"], tuning_m["n_customers_alerted"]),
            )
            m3.metric(
                "Detection rate (tuning)",
                fmt_rate(tuning_m["n_detected"], tuning_m["n_labeled_in_scope"]),
            )
            m4.metric(
                "Detection rate (validation)",
                fmt_rate(validation_m["n_detected"], validation_m["n_labeled_in_scope"]),
            )

            capacity = result["segment_choices"][str(segment_id)]["capacity"]
            if tuning_m["avg_alerts_per_month"] > capacity:
                st.error(f"This combination exceeds the segment's analyst capacity ({capacity} alerts/month).")
            else:
                st.success(f"Within the segment's analyst capacity ({capacity} alerts/month).")

        st.divider()
        st.subheader("Phase 5's actual chosen threshold: test-period performance")
        test_pooled = result["test_results"]["pooled"]
        t1, t2, t3 = st.columns(3)
        t1.metric("Test alerts/month", test_pooled["avg_alerts_per_month"])
        t2.metric("Test detection", fmt_rate(test_pooled["n_detected"], test_pooled["n_labeled_in_scope"]))
        t3.metric("Test productive rate", fmt_rate(test_pooled["n_productive_customers"], test_pooled["n_customers_alerted"]))

        st.subheader("ATL/BTL: what stepping the chosen threshold down finds (tuning period)")
        if result["atl_btl"]:
            atl_df = pd.DataFrame(
                [
                    {
                        "Parameter": a["primary_param"],
                        "Chosen value": a["chosen_value"],
                        "Stepped to": a["step_value"],
                        "Extra alerts": a["n_extra_alerted_customers"],
                        "Extra productive": fmt_rate(a["n_extra_productive"], a["n_extra_alerted_customers"]),
                    }
                    for a in result["atl_btl"]
                ]
            )
            st.dataframe(atl_df, hide_index=True, width="stretch")
        else:
            st.info("The chosen threshold is already the loosest grid value swept; no BTL step exists.")

        if "ring_report" in result:
            st.subheader("FAN-02 ring-level view")
            ring_df = pd.DataFrame(
                [
                    {
                        "Period": p,
                        "Rings recovered": fmt_rate(s["n_rings_recovered"], s["n_rings"]),
                        "Members caught": fmt_rate(s["n_members_caught"], s["n_members_total"]),
                    }
                    for p, s in result["ring_report"]["by_period"].items()
                ]
            )
            st.dataframe(ring_df, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tab 2: Alert queue (precomputed XGBoost scores + SHAP drivers)
# ---------------------------------------------------------------------------

with tab_queue:
    st.header("Alert queue (test period)")
    st.caption(
        "The 207 test-period alerts ordered by the trained XGBoost prioritizer's score, with each alert's "
        "top-3 SHAP drivers - read straight from artifacts/metrics/alert_queue_test.csv, not recomputed here."
    )

    queue = load_alert_queue()
    if queue is None:
        st.error("No alert queue found. Run `make model` first.")
    else:
        scenario_filter = st.multiselect(
            "Filter by scenario", SCENARIO_IDS, default=SCENARIO_IDS, key="queue_scenario_filter"
        )
        filtered = queue[queue["scenario_id"].isin(scenario_filter)]

        c1, c2 = st.columns(2)
        c1.metric("Alerts shown", len(filtered))
        c2.metric("Productive alerts shown", int(filtered["is_productive"].sum()))

        display_cols = [
            "rank", "customer_id", "month", "scenario_id", "typology", "is_productive", "xgb_score",
            "driver_1_feature", "driver_1_shap_value", "driver_2_feature", "driver_2_shap_value",
            "driver_3_feature", "driver_3_shap_value",
        ]
        st.dataframe(filtered[display_cols], hide_index=True, width="stretch", height=600)


# ---------------------------------------------------------------------------
# Tab 3: Papers
# ---------------------------------------------------------------------------

with tab_papers:
    st.header("Model white papers")
    papers = list_papers()
    if not papers:
        st.error("No papers found. Run `make papers` first.")
    else:
        paper_name = st.selectbox("Paper", papers, key="paper_select")
        st.markdown(load_paper(paper_name))
