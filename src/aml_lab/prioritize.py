"""ML alert prioritization (Phase 6).

Reconstructs the actual DEPLOYED alert queue (Phase 5's chosen thresholds
- segment-specific where selected, pooled fallback elsewhere - applied via
the evidence-bearing evaluators from scenarios.py, not tuning.py's fast
sweep functions, since we need evidence columns here), then trains a model
to RANK that queue by how likely each alert is to be productive (hits a
labeled typology).

Train = alerts from the tuning + validation periods (months 1-10). Test =
alerts from the test period (months 11-12), touched only for final
evaluation - no fitting, tuning, or model selection ever sees it.

The model reorders the queue; it never suppresses a scenario hit. A
customer flagged by a rule stays flagged regardless of what the model
thinks - the model only changes review ORDER, the same way an analyst
triage queue is sorted without discarding tickets.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from aml_lab.generate import DEFAULT_CONFIG_PATH, load_config
from aml_lab.scenarios import (
    DEFAULT_SCENARIOS_CONFIG_PATH,
    EVALUATORS,
    REPO_ROOT,
    build_context_from_dir,
    load_scenarios_config,
)
from aml_lab.tuning import month_to_period

SEED = 42
N_BOOTSTRAP = 1000
N_SHUFFLES = 1000
REVIEW_BUDGETS = (0.20, 0.40, 0.60)

SQL_FEATURE_COLS = [
    "cash_deposit_count_7d",
    "cash_deposit_sum_7d",
    "cash_deposit_count_30d",
    "cash_deposit_sum_30d",
    "near_threshold_deposit_count_30d",
    "rapid_movement_ratio_48h",
    "residual_balance_48h_min",
    "distinct_counterparties_in_30d",
    "distinct_counterparties_out_30d",
    "cross_border_share_30d",
    "high_risk_country_exposure_30d",
    "total_flow_volume_30d",
    "activity_zscore_90d",
    "journal_transfer_volume_30d_related",
    "graph_in_degree_30d",
    "graph_out_degree_30d",
    "component_size_30d",
    "community_size_30d",
]
EVIDENCE_COLS = ["evidence_count", "evidence_amount", "evidence_ratio"]


# ---------------------------------------------------------------------------
# Reconstruct the deployed alert queue at Phase 5's chosen thresholds
# ---------------------------------------------------------------------------

def deployed_alerts_for_scenario(
    scenario_id: str, con, gen_config: dict, segments: pd.DataFrame, tuning_result: dict, spec: dict
) -> pd.DataFrame:
    """Every alert the DEPLOYED system (segment-specific threshold where
    Phase 5 chose one, pooled fallback otherwise) would actually raise,
    with evidence - using scenarios.py's evidence-bearing evaluators, not
    tuning.py's evidence-free fast-sweep ones.

    tuning.py's chosen "params" only carry the SWEPT grid parameters; the
    evidence-bearing evaluators need the full parameter set, so each
    choice is merged onto the scenario's fixed defaults (scenarios.yaml
    params not in the grid, e.g. FAN-02's window_days/min_edge_amount).
    """
    typology = spec["typology"]
    pooled_params = {**spec["params"], **tuning_result["pooled_choice"]["params"]}
    cache: dict[tuple, pd.DataFrame] = {}

    def get_alerts(params: dict) -> pd.DataFrame:
        key = tuple(sorted(params.items()))
        if key not in cache:
            cache[key] = EVALUATORS[scenario_id](con, params, gen_config)
        return cache[key]

    frames = []
    for seg_id_str, choice in tuning_result["segment_choices"].items():
        seg_id = int(seg_id_str)
        seg_customers = set(segments.loc[segments["segment_assigned"] == seg_id, "customer_id"])
        params = pooled_params if choice["fallback_to_pooled"] else {**spec["params"], **choice["params"]}
        alerts = get_alerts(params)
        if alerts.empty:
            continue
        seg_alerts = alerts[alerts["customer_id"].isin(seg_customers)].copy()
        if not seg_alerts.empty:
            frames.append(seg_alerts)

    if not frames:
        return pd.DataFrame(columns=["customer_id", "month", "evidence", "scenario_id", "typology", "period"])
    combined = pd.concat(frames, ignore_index=True)
    combined["scenario_id"] = scenario_id
    combined["typology"] = typology
    combined["period"] = combined["month"].map(month_to_period)
    return combined


def build_deployed_alert_table(
    con, gen_config: dict, scenarios_config: dict, segments: pd.DataFrame, metrics_dir: Path
) -> pd.DataFrame:
    frames = []
    for scenario_id, spec in scenarios_config.items():
        with open(metrics_dir / f"tuning_{scenario_id}.json") as f:
            tuning_result = json.load(f)
        frames.append(
            deployed_alerts_for_scenario(scenario_id, con, gen_config, segments, tuning_result, spec)
        )
    return pd.concat(frames, ignore_index=True)


def label_productive(alerts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labeled_pairs = set(zip(labels["customer_id"], labels["typology"]))
    alerts = alerts.copy()
    alerts["is_productive"] = [
        int((c, t) in labeled_pairs) for c, t in zip(alerts["customer_id"], alerts["typology"])
    ]
    return alerts


# ---------------------------------------------------------------------------
# Evidence -> a small, consistent set of numeric columns across scenarios
# ---------------------------------------------------------------------------

def extract_evidence_features(scenario_id: str, evidence_json: str) -> tuple[float | None, float | None, float | None]:
    """(count, amount, ratio): a generic 3-column mapping of each
    scenario's evidence onto a shared numeric shape, so one model can be
    trained across all five scenario queues without a sparse one-hot of
    every possible evidence field.
    """
    ev = json.loads(evidence_json)
    if scenario_id == "STR-01":
        return ev["n_deposits"], None, None
    if scenario_id == "RMV-01":
        return None, ev["inflow_amount"], ev["ratio"]
    if scenario_id == "FAN-01":
        return ev["distinct_senders"], ev["total_in"], ev["forwarded_share"]
    if scenario_id == "FAN-02":
        edges = ev.get("edges", [])
        amount = sum(e["amount"] for e in edges) if edges else None
        ratio = (sum(e["pass_through_ratio"] for e in edges) / len(edges)) if edges else None
        return ev["hub_in_degree"], amount, ratio
    if scenario_id == "WLT-01":
        return ev["n_journal"], ev["journal_volume"], None
    raise ValueError(scenario_id)


def build_dataset(alerts: pd.DataFrame, features: pd.DataFrame, scenario_ids: list[str]) -> pd.DataFrame:
    ev = alerts.apply(
        lambda r: extract_evidence_features(r["scenario_id"], r["evidence"]), axis=1, result_type="expand"
    )
    ev.columns = EVIDENCE_COLS
    alerts = pd.concat([alerts.reset_index(drop=True), ev.reset_index(drop=True)], axis=1)

    feat = features.drop(columns=["segment_true", "period"], errors="ignore")
    merged = alerts.merge(feat, on=["customer_id", "month"], how="left")

    for sid in scenario_ids:
        merged[f"scenario_{sid}"] = (merged["scenario_id"] == sid).astype(int)
    return merged


def design_matrix(df: pd.DataFrame, scenario_ids: list[str]) -> pd.DataFrame:
    cols = SQL_FEATURE_COLS + EVIDENCE_COLS + [f"scenario_{s}" for s in scenario_ids]
    return df[cols].astype(float)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def fit_xgboost(X_train: pd.DataFrame, y_train: np.ndarray) -> xgb.XGBClassifier:
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=1,
        eval_metric="aucpr",
    )
    model.fit(X_train, y_train)
    return model


def fit_logistic_baseline(X_train: pd.DataFrame, y_train: np.ndarray):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imputed)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    model.fit(X_scaled, y_train)
    return model, imputer, scaler


def bootstrap_pr_auc(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores.append(average_precision_score(yt, ys))
    scores = np.array(scores)
    return {
        "mean": round(float(scores.mean()), 4),
        "ci_lower": round(float(np.percentile(scores, 2.5)), 4),
        "ci_upper": round(float(np.percentile(scores, 97.5)), 4),
        "n_bootstrap_samples_used": len(scores),
    }


# ---------------------------------------------------------------------------
# Review-budget analysis
# ---------------------------------------------------------------------------

def _alerts_needed_for_target(cum_tp: np.ndarray, target_tp: int) -> int | None:
    """Smallest prefix length (1-indexed) whose cumulative true-positive
    count reaches target_tp, or None if it never does within the array.
    """
    if target_tp <= 0:
        return 0
    reach = np.where(cum_tp >= target_tp)[0]
    return int(reach[0] + 1) if len(reach) else None


def review_budget_analysis(
    test_df: pd.DataFrame,
    primary_score_col: str,
    other_score_cols: dict[str, list[str]] | None = None,
    budgets=REVIEW_BUDGETS,
    n_shuffles: int = N_SHUFFLES,
    seed: int = SEED,
) -> tuple[dict, pd.DataFrame]:
    """Review-budget capture for the primary ("ML") score, reported next
    to FIFO, random, and any other named baselines (logistic regression,
    scenario prior). "FPs avoided at equal detection" is computed only
    for the primary score, against FIFO and against random:

        reference = how many productive alerts the PRIMARY score
                    captures within its top-k (k = budget * n)
        fifo_needed = smallest FIFO-order prefix that also captures
                      `reference` productive alerts
        fps_avoided_vs_fifo = fifo_needed - k

    i.e. how many fewer alerts (so, since both reach the same number of
    true positives, how many fewer false positives) a reviewer following
    the primary score's order needs to inspect than one following FIFO
    to find the same number of productive alerts. Same idea for random,
    using the mean alerts-needed over n_shuffles shuffles.
    """
    other_score_cols = other_score_cols or {}
    n = len(test_df)
    order_primary = test_df.sort_values(primary_score_col, ascending=False).reset_index(drop=True)
    order_fifo = test_df.sort_values(["month", "customer_id"]).reset_index(drop=True)
    cum_tp_fifo = order_fifo["is_productive"].cumsum().to_numpy()
    y = test_df["is_productive"].to_numpy()

    rng = np.random.default_rng(seed)
    idx_all = np.arange(n)

    results = {}
    for b in budgets:
        k = max(1, round(n * b))
        primary_tp = int(order_primary["is_productive"].iloc[:k].sum())
        fifo_tp = int(order_fifo["is_productive"].iloc[:k].sum())

        fifo_needed = _alerts_needed_for_target(cum_tp_fifo, primary_tp)
        fps_avoided_vs_fifo = (fifo_needed - k) if fifo_needed is not None else None

        rand_tp_at_k = np.empty(n_shuffles)
        rand_needed = []
        for i in range(n_shuffles):
            rng.shuffle(idx_all)
            cum_tp_rand = y[idx_all].cumsum()
            rand_tp_at_k[i] = cum_tp_rand[k - 1]
            needed = _alerts_needed_for_target(cum_tp_rand, primary_tp)
            if needed is not None:
                rand_needed.append(needed)
        random_needed_mean = round(float(np.mean(rand_needed)), 2) if rand_needed else None
        fps_avoided_vs_random = (
            round(random_needed_mean - k, 2) if random_needed_mean is not None else None
        )

        productive_captured = {"xgboost": primary_tp, "fifo": fifo_tp, "random_mean": round(float(rand_tp_at_k.mean()), 2)}
        for name, cols in other_score_cols.items():
            order_other = test_df.sort_values(cols, ascending=False).reset_index(drop=True)
            productive_captured[name] = int(order_other["is_productive"].iloc[:k].sum())

        results[f"top_{round(b * 100)}pct"] = {
            "n_reviewed": k,
            "productive_captured": productive_captured,
            "fifo_alerts_needed_to_match_xgboost": fifo_needed,
            "fps_avoided_vs_fifo_at_equal_detection": fps_avoided_vs_fifo,
            "random_alerts_needed_to_match_xgboost_mean": random_needed_mean,
            "fps_avoided_vs_random_at_equal_detection": fps_avoided_vs_random,
        }
    return results, order_primary


# ---------------------------------------------------------------------------
# Scenario-prior baseline: no ML at all, just each scenario's own
# train-period productive rate (ties broken by evidence amount)
# ---------------------------------------------------------------------------

def add_scenario_prior_score(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train_rate = train_df.groupby("scenario_id")["is_productive"].mean().to_dict()
    test_df = test_df.copy()
    test_df["scenario_prior_score"] = test_df["scenario_id"].map(train_rate)
    # Tie-break column for ranking (not for the score itself, which stays
    # the plain per-scenario rate so PR-AUC reflects exactly what's being
    # claimed: "rank by scenario prior"). NaN evidence_amount (scenarios
    # where that evidence field doesn't apply) sorts last.
    test_df["scenario_prior_tiebreak"] = test_df["evidence_amount"].fillna(-np.inf)
    return test_df, {k: round(float(v), 4) for k, v in train_rate.items()}


# ---------------------------------------------------------------------------
# Per-scenario breakdown
# ---------------------------------------------------------------------------

def by_scenario_breakdown(test_df: pd.DataFrame, score_col: str) -> dict:
    out = {}
    for scenario_id, group in test_df.groupby("scenario_id"):
        y = group["is_productive"].to_numpy()
        n_pos = int(y.sum())
        entry = {"n_alerts": len(group), "n_positives": n_pos}
        if 0 < n_pos < len(group):
            entry["pr_auc"] = round(float(average_precision_score(y, group[score_col])), 4)
            fifo_order = group.sort_values(["month", "customer_id"])
            ml_order = group.sort_values(score_col, ascending=False)
            k = max(1, round(len(group) * 0.4))
            entry["top_40pct_ml_captured"] = int(ml_order["is_productive"].iloc[:k].sum())
            entry["top_40pct_fifo_captured"] = int(fifo_order["is_productive"].iloc[:k].sum())
            entry["n_reviewed_at_40pct"] = k
        else:
            entry["pr_auc"] = None
        out[scenario_id] = entry
    return out


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap(model: xgb.XGBClassifier, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    return explanation


def save_global_shap_plot(explanation, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 8))
    shap.summary_plot(explanation, plot_type="bar", show=False)
    plt.suptitle("SYNTHETIC DATA - SHAP global feature importance", fontsize=12, x=0.55)
    plt.xlabel("mean(|SHAP value|)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def top_drivers_csv(test_df: pd.DataFrame, X_test: pd.DataFrame, explanation, scores: np.ndarray, out_path: Path) -> None:
    values = explanation.values
    feature_names = np.array(X_test.columns)
    rows = []
    for i, (_, alert) in enumerate(test_df.iterrows()):
        order = np.argsort(-np.abs(values[i]))[:3]
        row = {
            "customer_id": alert["customer_id"],
            "month": int(alert["month"]),
            "scenario_id": alert["scenario_id"],
            "typology": alert["typology"],
            "is_productive": int(alert["is_productive"]),
            "xgb_score": round(float(scores[i]), 4),
        }
        for rank, j in enumerate(order, start=1):
            row[f"driver_{rank}_feature"] = feature_names[j]
            row[f"driver_{rank}_shap_value"] = round(float(values[i, j]), 4)
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_prioritization(
    con, gen_config: dict, scenarios_config: dict, segments: pd.DataFrame, labels: pd.DataFrame,
    features: pd.DataFrame, metrics_dir: Path,
) -> dict:
    scenario_ids = sorted(scenarios_config.keys())

    alerts = build_deployed_alert_table(con, gen_config, scenarios_config, segments, metrics_dir)
    # DuckDB doesn't guarantee row order without ORDER BY, so without this
    # sort the alert table's row order (and therefore train/test row
    # order, XGBoost's histogram-building, and the bootstrap resample
    # indices) varies run to run despite every seed being fixed. Sorting
    # on stable keys makes the whole pipeline byte-for-byte reproducible.
    alerts = alerts.sort_values(["customer_id", "month", "scenario_id"]).reset_index(drop=True)
    alerts = label_productive(alerts, labels)
    dataset = build_dataset(alerts, features, scenario_ids)

    train_df = dataset[dataset["period"].isin(["tuning", "validation"])].reset_index(drop=True)
    test_df = dataset[dataset["period"] == "test"].reset_index(drop=True)

    X_train = design_matrix(train_df, scenario_ids)
    y_train = train_df["is_productive"].to_numpy()
    X_test = design_matrix(test_df, scenario_ids)
    y_test = test_df["is_productive"].to_numpy()

    xgb_model = fit_xgboost(X_train, y_train)
    lr_model, imputer, scaler = fit_logistic_baseline(X_train, y_train)

    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
    lr_scores = lr_model.predict_proba(scaler.transform(imputer.transform(X_test)))[:, 1]

    xgb_pr_auc = bootstrap_pr_auc(y_test, xgb_scores)
    lr_pr_auc = bootstrap_pr_auc(y_test, lr_scores)
    xgb_beats_baseline = xgb_pr_auc["mean"] > lr_pr_auc["mean"]

    test_df = test_df.copy()
    test_df["xgb_score"] = xgb_scores
    test_df["lr_score"] = lr_scores
    test_df, scenario_prior_rates = add_scenario_prior_score(train_df, test_df)
    scenario_prior_pr_auc = bootstrap_pr_auc(y_test, test_df["scenario_prior_score"].to_numpy())

    review_budget, ml_order = review_budget_analysis(
        test_df,
        "xgb_score",
        other_score_cols={
            "logistic_regression": ["lr_score"],
            "scenario_prior": ["scenario_prior_score", "scenario_prior_tiebreak"],
        },
    )
    by_scenario = by_scenario_breakdown(test_df, "xgb_score")

    # The one productive alert (if any) XGBoost ranks outside the top 60%.
    ml_order_ranked = ml_order.reset_index(drop=True)
    ml_order_ranked["rank"] = np.arange(1, len(ml_order_ranked) + 1)
    k_60 = max(1, round(len(ml_order_ranked) * 0.60))
    missed_outside_top_60pct = [
        {
            "rank": int(row["rank"]),
            "customer_id": row["customer_id"],
            "month": int(row["month"]),
            "scenario_id": row["scenario_id"],
            "xgb_score": round(float(row["xgb_score"]), 4),
        }
        for _, row in ml_order_ranked[(ml_order_ranked["rank"] > k_60) & (ml_order_ranked["is_productive"] == 1)].iterrows()
    ]

    explanation = compute_shap(xgb_model, X_test)

    result = {
        "seed": SEED,
        "train": {"n_alerts": len(train_df), "n_positives": int(y_train.sum())},
        "test": {"n_alerts": len(test_df), "n_positives": int(y_test.sum())},
        "pr_auc": {
            "xgboost": xgb_pr_auc,
            "logistic_regression": lr_pr_auc,
            "scenario_prior": scenario_prior_pr_auc,
            "xgboost_beats_baseline": xgb_beats_baseline,
        },
        "scenario_prior_train_rates": scenario_prior_rates,
        "review_budget": review_budget,
        "by_scenario": by_scenario,
        "productive_alerts_outside_top_60pct": missed_outside_top_60pct,
        "model_params": {
            "xgboost": {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.1, "scale_pos_weight": float(xgb_model.get_params()["scale_pos_weight"])},
            "logistic_regression": {"class_weight": "balanced", "max_iter": 2000},
        },
        "rules": [
            ("The model reorders alerts within the queue it is given; it never suppresses a scenario hit - "
            "every rule-based alert stays in the queue regardless of model score."),
            ("Test-period data (months 11-12) was used only for the numbers above, never for fitting, "
            "hyperparameter choice, or model selection."),
        ],
    }
    if not xgb_beats_baseline:
        result["rules"].append(
            "XGBoost did NOT beat the logistic regression baseline on test PR-AUC. Reported plainly, not tuned "
            "away by touching test data."
        )

    return {
        "result": result,
        "xgb_model": xgb_model,
        "X_test": X_test,
        "train_df": train_df,
        "test_df": test_df,
        "explanation": explanation,
        "xgb_scores": xgb_scores,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ML alert prioritization with SHAP (Phase 6).")
    parser.add_argument("--generator-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--scenarios-config", type=Path, default=DEFAULT_SCENARIOS_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "artifacts" / "data")
    parser.add_argument("--segments-path", type=Path, default=REPO_ROOT / "artifacts" / "segments" / "customer_segments.parquet")
    parser.add_argument("--features-path", type=Path, default=REPO_ROOT / "artifacts" / "features" / "customer_month_features.parquet")
    parser.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "artifacts" / "metrics")
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "artifacts" / "figures")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "artifacts" / "models")
    args = parser.parse_args(argv)

    gen_config = load_config(args.generator_config)
    scenarios_config = load_scenarios_config(args.scenarios_config)
    segments = pd.read_parquet(args.segments_path)[["customer_id", "segment_assigned"]]
    labels = pd.read_parquet(args.data_dir / "labels.parquet")
    features = pd.read_parquet(args.features_path)
    con = build_context_from_dir(args.data_dir, gen_config)

    out = run_prioritization(con, gen_config, scenarios_config, segments, labels, features, args.metrics_dir)

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_dir / "prioritization.json", "w") as f:
        json.dump(out["result"], f, indent=2, default=str)

    save_global_shap_plot(out["explanation"], args.figures_dir / "shap_global.png")
    top_drivers_csv(out["test_df"], out["X_test"], out["explanation"], out["xgb_scores"], args.metrics_dir / "alert_queue_test.csv")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    with open(args.models_dir / "prioritizer.pkl", "wb") as f:
        pickle.dump(out["xgb_model"], f)

    r = out["result"]
    print(f"Train: {r['train']['n_alerts']} alerts, {r['train']['n_positives']} positive")
    print(f"Test:  {r['test']['n_alerts']} alerts, {r['test']['n_positives']} positive")
    for name in ("xgboost", "logistic_regression", "scenario_prior"):
        m = r["pr_auc"][name]
        print(f"PR-AUC {name}: {m['mean']} (CI {m['ci_lower']}-{m['ci_upper']})")
    print(f"XGBoost beats baseline: {r['pr_auc']['xgboost_beats_baseline']}")
    print(f"Scenario-prior train rates: {r['scenario_prior_train_rates']}")
    for budget, stats in r["review_budget"].items():
        pc = stats["productive_captured"]
        print(
            f"  {budget} (n={stats['n_reviewed']}): xgboost={pc['xgboost']} lr={pc['logistic_regression']} "
            f"scenario_prior={pc['scenario_prior']} fifo={pc['fifo']} random~{pc['random_mean']} | "
            f"FPs avoided vs fifo={stats['fps_avoided_vs_fifo_at_equal_detection']}, "
            f"vs random={stats['fps_avoided_vs_random_at_equal_detection']}"
        )
    print("By scenario:")
    for sid, stats in r["by_scenario"].items():
        print(f"  {sid}: {stats}")
    print(f"Productive alerts XGBoost ranks outside the top 60%: {r['productive_alerts_outside_top_60pct']}")
    print(f"\nWrote {args.metrics_dir / 'prioritization.json'}")
    print(f"Wrote {args.figures_dir / 'shap_global.png'}")
    print(f"Wrote {args.metrics_dir / 'alert_queue_test.csv'}")
    print(f"Wrote {args.models_dir / 'prioritizer.pkl'}")


if __name__ == "__main__":
    main()
