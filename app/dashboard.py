"""Streamlit dashboard (Phase 8, redesigned).

Reads only precomputed results from artifacts/metrics, artifacts/papers,
and artifacts/figures - the app never recomputes a sweep, a model
prediction, or a SHAP value. Every number and image here already exists
in a JSON, CSV, or PNG written by an earlier phase. Charts are a
presentation layer over those same numbers (simple grouping/formatting),
not new modeling.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from aml_lab.papers import alert_count_units_note, baseline_comparison_note

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "artifacts" / "metrics"
PAPERS_DIR = REPO_ROOT / "artifacts" / "papers"
FIGURES_DIR = REPO_ROOT / "artifacts" / "figures"

SEGMENT_LABELS = {0: "retail", 1: "mixed", 2: "small_business", 3: "HNW"}
SCENARIO_IDS = ["STR-01", "RMV-01", "FAN-01", "FAN-02", "WLT-01"]
SCENARIO_NAMES = {
    "STR-01": "Structuring",
    "RMV-01": "Rapid movement",
    "FAN-01": "Fan-in/out (degree rule)",
    "FAN-02": "Fan-in/out (ring detection)",
    "WLT-01": "Wealth layering",
}
PAPER_FIGURES = {
    "STR-01": "tuning_STR-01.png",
    "RMV-01": "tuning_RMV-01.png",
    "FAN-01": "tuning_FAN-01.png",
    "FAN-02": "tuning_FAN-02.png",
    "WLT-01": "tuning_WLT-01.png",
    "prioritization": "shap_global.png",
}
MODEL_COLORS = {
    "XGBoost": "#0f766e",
    "Logistic regression": "#2563eb",
    "Scenario prior": "#d97706",
    "FIFO": "#64748b",
    "Random (mean)": "#94a3b8",
}

st.set_page_config(page_title="AML Scenario Tuning Lab", page_icon="shield", layout="wide")


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
def load_prioritization_json() -> dict | None:
    path = METRICS_DIR / "prioritization.json"
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


@st.cache_data
def load_scenario_table(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for scenario_id in SCENARIO_IDS:
        tr = load_tuning_json(scenario_id)
        if tr is None:
            continue
        pc = tr["pooled_choice"]
        test = tr["test_results"]["pooled"]
        test_productive_rate = (
            test["n_productive_customers"] / test["n_customers_alerted"] if test["n_customers_alerted"] else 0.0
        )
        test_detection_rate = test["n_detected"] / test["n_labeled_in_scope"] if test["n_labeled_in_scope"] else 0.0
        rows.append(
            {
                "Scenario": scenario_id,
                "Name": SCENARIO_NAMES.get(scenario_id, ""),
                "Validation detection": fmt_rate(pc["validation"]["n_detected"], pc["validation"]["n_labeled_in_scope"]),
                "Test detection": fmt_rate(test["n_detected"], test["n_labeled_in_scope"]),
                "Test detection rate": test_detection_rate,
                "Test productive rate": fmt_rate(test["n_productive_customers"], test["n_customers_alerted"]),
                "Test productive rate (num)": test_productive_rate,
                "Test alerts/month": test["avg_alerts_per_month"],
            }
        )
    return pd.DataFrame(rows)


def fmt_rate(numerator, denominator) -> str:
    if not denominator:
        return "n/a (0/0)"
    return f"{numerator / denominator:.2f} ({numerator}/{denominator})"


def find_grid_row(grid_results: list[dict], params: dict) -> dict | None:
    for row in grid_results:
        if row["params"] == params:
            return row
    return None


def show_figure(name: str, caption: str) -> None:
    path = FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure `{name}` not found - run `make tuning`/`make model` to regenerate it.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### AML Scenario Tuning Lab")
    st.caption(
        "A synthetic, end-to-end reproduction of a bank's AML transaction-monitoring tuning workflow: "
        "generation, feature engineering, segmentation, scenario tuning, and ML alert prioritization."
    )
    st.divider()
    st.markdown("**Segment numbering**")
    st.table(pd.DataFrame({"id": list(SEGMENT_LABELS.keys()), "segment": list(SEGMENT_LABELS.values())}).set_index("id"))
    st.divider()
    st.caption("Source: [github.com/raiigauravv/aml-scenario-tuning-lab](https://github.com/raiigauravv/aml-scenario-tuning-lab)")


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

st.warning(
    "**SYNTHETIC DATA** - every customer, account, transaction, alert, and score shown below is generated, "
    "not real. No real customers, investigations, or regulatory filings are referenced anywhere in this app.",
    icon=":material/warning:",
)
st.title("AML Scenario Tuning Lab")

tab_overview, tab_whatif, tab_queue, tab_papers = st.tabs(
    [":material/space_dashboard: Overview", ":material/tune: Threshold What-If", ":material/list_alt: Alert Queue", ":material/description: Papers"]
)


# ---------------------------------------------------------------------------
# Tab 0: Overview (headline KPIs, model comparison, segments)
# ---------------------------------------------------------------------------

with tab_overview:
    st.header("Headline results")
    st.caption("Every figure below is read from `artifacts/metrics/*.json` - nothing on this page is hand-typed.")

    pr = load_prioritization_json()
    scenario_df = load_scenario_table(METRICS_DIR)

    if pr is None or scenario_df.empty:
        st.error("No metrics found. Run `make all` first.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Scenarios tuned", len(scenario_df))
        k2.metric(
            "Test-period alerts reviewed",
            pr["test"]["n_alerts"],
            help="Alert rows: one per customer-month-scenario triggering event in the test period.",
        )
        k3.metric(
            "Productive alerts found",
            pr["test"]["n_positives"],
            help=f"{fmt_rate(pr['test']['n_positives'], pr['test']['n_alerts'])} of test-period alert rows.",
        )
        xgb = pr["pr_auc"]["xgboost"]
        k4.metric(
            "XGBoost PR-AUC (test)",
            f"{xgb['mean']:.3f}",
            help=f"95% bootstrap CI: [{xgb['ci_lower']:.3f}, {xgb['ci_upper']:.3f}] ({xgb['n_bootstrap_samples_used']} samples).",
        )

        st.subheader("Scenario performance (test period, pooled chosen threshold)")
        st.dataframe(
            scenario_df,
            hide_index=True,
            width="stretch",
            column_order=[
                "Scenario", "Name", "Validation detection", "Test detection",
                "Test detection rate", "Test productive rate", "Test productive rate (num)", "Test alerts/month",
            ],
            column_config={
                "Test detection rate": st.column_config.ProgressColumn(
                    "Test detection (bar)", min_value=0, max_value=1, format="%.2f"
                ),
                "Test productive rate (num)": st.column_config.ProgressColumn(
                    "Test productive (bar)", min_value=0, max_value=1, format="%.2f"
                ),
            },
        )

        note = alert_count_units_note(METRICS_DIR)
        if note:
            st.caption(note)

        st.divider()
        st.subheader("Alert prioritization: XGBoost vs. baselines")
        c1, c2 = st.columns([1, 1])

        with c1:
            pr_auc_df = pd.DataFrame(
                [
                    {"Model": "XGBoost", "mean": pr["pr_auc"]["xgboost"]["mean"],
                     "lo": pr["pr_auc"]["xgboost"]["ci_lower"], "hi": pr["pr_auc"]["xgboost"]["ci_upper"]},
                    {"Model": "Logistic regression", "mean": pr["pr_auc"]["logistic_regression"]["mean"],
                     "lo": pr["pr_auc"]["logistic_regression"]["ci_lower"], "hi": pr["pr_auc"]["logistic_regression"]["ci_upper"]},
                    {"Model": "Scenario prior", "mean": pr["pr_auc"]["scenario_prior"]["mean"],
                     "lo": pr["pr_auc"]["scenario_prior"]["ci_lower"], "hi": pr["pr_auc"]["scenario_prior"]["ci_upper"]},
                ]
            )
            color_scale = alt.Scale(domain=list(MODEL_COLORS.keys()), range=list(MODEL_COLORS.values()))
            bars = (
                alt.Chart(pr_auc_df)
                .mark_bar()
                .encode(
                    x=alt.X("Model", sort=None, title=None),
                    y=alt.Y("mean", title="PR-AUC (mean)", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Model", scale=color_scale, legend=None),
                    tooltip=["Model", alt.Tooltip("mean", format=".3f"), alt.Tooltip("lo", format=".3f"), alt.Tooltip("hi", format=".3f")],
                )
            )
            errors = alt.Chart(pr_auc_df).mark_errorbar(color="#0f172a").encode(
                x=alt.X("Model", sort=None), y=alt.Y("lo", title=""), y2="hi"
            )
            st.altair_chart((bars + errors).properties(title="PR-AUC, 1,000-sample bootstrap 95% CI", height=320), width="stretch")

        with c2:
            budget_rows = []
            model_keys = {"xgboost": "XGBoost", "logistic_regression": "Logistic regression",
                          "scenario_prior": "Scenario prior", "fifo": "FIFO", "random_mean": "Random (mean)"}
            for key, stats in pr["review_budget"].items():
                pct = key.replace("top_", "").replace("pct", "")
                for mk, label in model_keys.items():
                    budget_rows.append({"Budget": f"Top {pct}%", "Model": label, "Captured": stats["productive_captured"][mk]})
            budget_df = pd.DataFrame(budget_rows)
            budget_order = [f"Top {k.replace('top_', '').replace('pct', '')}%" for k in pr["review_budget"]]
            chart = (
                alt.Chart(budget_df)
                .mark_bar()
                .encode(
                    x=alt.X("Model", sort=list(MODEL_COLORS.keys()), title=None, axis=None),
                    y=alt.Y("Captured", title="Productive alerts captured"),
                    color=alt.Color("Model", scale=alt.Scale(domain=list(MODEL_COLORS.keys()), range=list(MODEL_COLORS.values()))),
                    column=alt.Column("Budget", sort=budget_order, title=None),
                    tooltip=["Model", "Budget", "Captured"],
                )
                .properties(title="Review-budget capture, all baselines", height=320, width=110)
            )
            st.altair_chart(chart, width="content")

        budgets_for_note = []
        for key, stats in pr["review_budget"].items():
            budgets_for_note.append({"captured": stats["productive_captured"]})
        st.caption(baseline_comparison_note(pr["pr_auc"], budgets_for_note, pr["test"]["n_positives"]))

        with st.expander("Customer segmentation (k=4, see sidebar for numbering)"):
            show_figure("segment_profile.png", "Segment profile: feature distributions by cluster, dataset-derived (not hardcoded).")


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

    scenario_id = st.selectbox(
        "Scenario", SCENARIO_IDS, format_func=lambda s: f"{s} - {SCENARIO_NAMES.get(s, '')}", key="whatif_scenario"
    )
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

        with st.expander("Grid sweep visualization (all swept combinations, tuning period)"):
            show_figure(f"tuning_{scenario_id}.png", f"{scenario_id} grid sweep, generated by src/aml_lab/tuning.py.")


# ---------------------------------------------------------------------------
# Tab 2: Alert queue (precomputed XGBoost scores + SHAP drivers)
# ---------------------------------------------------------------------------

with tab_queue:
    st.header("Alert queue (test period)")
    st.caption(
        "Test-period alerts ordered by the trained XGBoost prioritizer's score, with each alert's "
        "top-3 SHAP drivers - read straight from artifacts/metrics/alert_queue_test.csv, not recomputed here."
    )

    queue = load_alert_queue()
    if queue is None:
        st.error("No alert queue found. Run `make model` first.")
    else:
        f1, f2 = st.columns([3, 1])
        with f1:
            scenario_filter = st.multiselect(
                "Filter by scenario", SCENARIO_IDS, default=SCENARIO_IDS, key="queue_scenario_filter"
            )
        with f2:
            productive_only = st.checkbox("Productive only", value=False, key="queue_productive_only")

        filtered = queue[queue["scenario_id"].isin(scenario_filter)]
        if productive_only:
            filtered = filtered[filtered["is_productive"].astype(bool)]

        c1, c2, c3 = st.columns(3)
        c1.metric("Alerts shown", len(filtered))
        c2.metric("Productive alerts shown", int(filtered["is_productive"].sum()))
        c3.metric(
            "Productive rate shown",
            fmt_rate(int(filtered["is_productive"].sum()), len(filtered)) if len(filtered) else "n/a (0/0)",
        )

        productive_scale = alt.Scale(domain=["Not productive", "Productive"], range=["#94a3b8", "#0f766e"])
        chart_col, breakdown_col = st.columns(2)
        with chart_col:
            hist = (
                alt.Chart(filtered)
                .transform_calculate(productive_label="datum.is_productive ? 'Productive' : 'Not productive'")
                .mark_bar()
                .encode(
                    x=alt.X("xgb_score", bin=alt.Bin(maxbins=30), title="XGBoost score"),
                    y=alt.Y("count()", title="Alerts"),
                    color=alt.Color("productive_label:N", title=None, scale=productive_scale),
                )
                .properties(title="Score distribution", height=280)
            )
            st.altair_chart(hist, width="stretch")
        with breakdown_col:
            breakdown = (
                alt.Chart(filtered)
                .transform_calculate(productive_label="datum.is_productive ? 'Productive' : 'Not productive'")
                .mark_bar()
                .encode(
                    x=alt.X("scenario_id", title=None, sort=SCENARIO_IDS),
                    y=alt.Y("count()", title="Alerts"),
                    color=alt.Color("productive_label:N", title=None, scale=productive_scale),
                )
                .properties(title="Alerts by scenario", height=280)
            )
            st.altair_chart(breakdown, width="stretch")

        display_cols = [
            "rank", "customer_id", "month", "scenario_id", "typology", "is_productive", "xgb_score",
            "driver_1_feature", "driver_1_shap_value", "driver_2_feature", "driver_2_shap_value",
            "driver_3_feature", "driver_3_shap_value",
        ]
        st.dataframe(
            filtered[display_cols],
            hide_index=True,
            width="stretch",
            height=600,
            column_config={
                "xgb_score": st.column_config.ProgressColumn(
                    "xgb_score", min_value=0, max_value=1, format="%.3f"
                ),
                "is_productive": st.column_config.CheckboxColumn("is_productive", disabled=True),
            },
        )
        st.download_button(
            "Download filtered queue as CSV",
            filtered[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="alert_queue_filtered.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Tab 3: Papers
# ---------------------------------------------------------------------------

with tab_papers:
    st.header("Model white papers")
    papers = list_papers()
    if not papers:
        st.error("No papers found. Run `make papers` first.")
    else:
        paper_name = st.selectbox(
            "Paper",
            papers,
            format_func=lambda p: f"{p} - {SCENARIO_NAMES.get(p, 'Alert prioritization (ML reordering)')}",
            key="paper_select",
        )
        figure_name = PAPER_FIGURES.get(paper_name)
        if figure_name:
            with st.expander("Supporting figure", expanded=False):
                show_figure(figure_name, f"Referenced by the {paper_name} paper.")
        st.markdown(load_paper(paper_name))
