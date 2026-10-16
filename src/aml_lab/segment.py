"""Customer segmentation for the AML Scenario Tuning Lab (Phase 3).

Clusters customers on a handful of behaviour features (KMeans), compares
the result against the generator's `segment_true` ground truth, and saves
a segment profile table + figure. The clustering model is fit ONLY on
tuning-period (months 1-8) behaviour - never validation/test - for the
same no-leakage reason thresholds get chosen on validation and reported
on test: a segment label that "saw" later behaviour would leak information
forward into every segment-specific threshold downstream. The resulting
label is then held fixed for the customer for the rest of the study.

Why segment before tuning thresholds at all: a single global threshold
averages away real behavioural differences - a cash-heavy small business
and a low-activity retail customer do not share a sensible "normal" range
for cash-deposit volume, so one threshold is either too loose for one
group or too noisy for the other. Segment-specific thresholds (Phase 5)
fix that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTS_CONFIG_PATH = REPO_ROOT / "config" / "segments.yaml"


def load_segments_config(path: Path = DEFAULT_SEGMENTS_CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Customer-level behaviour profile
# ---------------------------------------------------------------------------

def build_customer_profile(features: pd.DataFrame, accounts: pd.DataFrame, config: dict) -> pd.DataFrame:
    fit_period = config["fit_period"]
    fit_rows = features[features["period"] == fit_period]

    profile = fit_rows.groupby("customer_id").agg(
        avg_monthly_volume=("total_flow_volume_30d", "mean"),
        avg_cash_deposit_sum_30d=("cash_deposit_sum_30d", "mean"),
        avg_distinct_counterparties_out_30d=("distinct_counterparties_out_30d", "mean"),
        avg_cross_border_share_30d=("cross_border_share_30d", "mean"),
    )

    has_investment = accounts.loc[accounts["account_type"] == "investment", "customer_id"].unique()
    profile["has_investment_account"] = profile.index.isin(has_investment).astype(int)

    # Customers with zero activity in the fit period (shouldn't normally
    # happen, but keeps the pipeline defined) get zero-filled behaviour.
    return profile.fillna(0.0)


def transform_features(profile: pd.DataFrame, config: dict) -> np.ndarray:
    columns = []
    for spec in config["features"]:
        col = profile[spec["name"]].to_numpy(dtype=float)
        if spec["log"]:
            col = np.log1p(np.clip(col, 0, None))
        columns.append(col)
    matrix = np.column_stack(columns)
    return StandardScaler().fit_transform(matrix)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def evaluate_k_candidates(X: np.ndarray, k_candidates: list[int], seed: int) -> dict[int, float]:
    scores = {}
    for k in k_candidates:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    return scores


def fit_segments(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)


# ---------------------------------------------------------------------------
# Profile table (comparison against segment_true)
# ---------------------------------------------------------------------------

def build_segment_profile_table(profile: pd.DataFrame, labels: np.ndarray, segment_true: pd.Series) -> pd.DataFrame:
    df = profile.copy()
    df["segment_assigned"] = labels
    df["segment_true"] = segment_true.reindex(df.index)

    feature_cols = [
        "avg_monthly_volume",
        "avg_cash_deposit_sum_30d",
        "avg_distinct_counterparties_out_30d",
        "avg_cross_border_share_30d",
        "has_investment_account",
    ]
    rows = []
    for seg_id, group in df.groupby("segment_assigned"):
        true_counts = group["segment_true"].value_counts()
        true_pct = (true_counts / len(group) * 100).round(1)
        row = {
            "segment_assigned": seg_id,
            "n_customers": len(group),
            **{col: round(group[col].mean(), 2) for col in feature_cols},
            "dominant_segment_true": true_counts.idxmax(),
        }
        for seg_true_name in ["retail", "small_business", "hnw_investment"]:
            row[f"pct_{seg_true_name}"] = float(true_pct.get(seg_true_name, 0.0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("segment_assigned").reset_index(drop=True)


def plot_segment_profile(profile_table: pd.DataFrame, out_path: Path) -> None:
    feature_cols = [
        "avg_monthly_volume",
        "avg_cash_deposit_sum_30d",
        "avg_distinct_counterparties_out_30d",
        "avg_cross_border_share_30d",
        "has_investment_account",
    ]
    segments = profile_table["segment_assigned"].astype(str).tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: standardized (z-scored across segments) feature means, as a
    # heatmap - shows which features drive each cluster apart.
    z = (profile_table[feature_cols] - profile_table[feature_cols].mean()) / profile_table[feature_cols].std().replace(
        0, 1
    )
    im = ax1.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax1.set_xticks(range(len(feature_cols)))
    ax1.set_xticklabels(feature_cols, rotation=30, ha="right")
    ax1.set_yticks(range(len(segments)))
    ax1.set_yticklabels([f"segment {s}" for s in segments])
    for i in range(len(segments)):
        for j in range(len(feature_cols)):
            ax1.text(j, i, f"{profile_table[feature_cols[j]].iloc[i]:.2g}", ha="center", va="center", fontsize=8)
    ax1.set_title("Feature means per assigned segment\n(color = z-score across segments)")
    fig.colorbar(im, ax=ax1, shrink=0.8)

    # Right: composition of each assigned segment by the generator's true
    # segment label.
    bottom = np.zeros(len(segments))
    for seg_true_name, color in [
        ("pct_retail", "#4C72B0"),
        ("pct_small_business", "#DD8452"),
        ("pct_hnw_investment", "#55A868"),
    ]:
        vals = profile_table[seg_true_name].to_numpy()
        ax2.bar(segments, vals, bottom=bottom, label=seg_true_name.replace("pct_", ""), color=color)
        bottom += vals
    ax2.set_xlabel("assigned segment")
    ax2.set_ylabel("% of segment (by true label)")
    ax2.set_title("Assigned-segment composition vs. true segment\n(synthetic ground truth, for validation only)")
    ax2.legend()

    fig.suptitle("SYNTHETIC DATA - AML Scenario Tuning Lab: customer segmentation (Phase 3)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_segmentation(
    features: pd.DataFrame, accounts: pd.DataFrame, customers: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = build_customer_profile(features, accounts, config)
    X = transform_features(profile, config)
    labels = fit_segments(X, config["k"], config["seed"])

    segment_true = customers.set_index("customer_id")["segment_true"]
    profile_table = build_segment_profile_table(profile, labels, segment_true)

    customer_segments = profile.copy()
    customer_segments["segment_assigned"] = labels
    customer_segments = customer_segments.reset_index()

    return customer_segments, profile_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Segment customers on tuning-period behaviour (KMeans).")
    parser.add_argument("--segments-config", type=Path, default=DEFAULT_SEGMENTS_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "artifacts" / "data")
    parser.add_argument(
        "--features-path", type=Path, default=REPO_ROOT / "artifacts" / "features" / "customer_month_features.parquet"
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "segments")
    parser.add_argument("--figure-path", type=Path, default=REPO_ROOT / "artifacts" / "figures" / "segment_profile.png")
    args = parser.parse_args(argv)

    config = load_segments_config(args.segments_config)
    features = pd.read_parquet(args.features_path)
    accounts = pd.read_parquet(args.data_dir / "accounts.parquet")
    customers = pd.read_parquet(args.data_dir / "customers.parquet")

    profile = build_customer_profile(features, accounts, config)
    X = transform_features(profile, config)

    k_scores = evaluate_k_candidates(X, config["k_candidates"], config["seed"])
    print("Silhouette score by k:", {k: round(v, 4) for k, v in k_scores.items()})
    print(f"Using k={config['k']} (set in {args.segments_config.name})")

    customer_segments, profile_table = run_segmentation(features, accounts, customers, config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    customer_segments.to_parquet(args.out_dir / "customer_segments.parquet", index=False)
    profile_table.to_csv(args.out_dir / "segment_profile.csv", index=False)
    plot_segment_profile(profile_table, args.figure_path)

    print(f"\nSegmented {len(customer_segments):,} customers into {config['k']} segments.")
    print(profile_table.to_string(index=False))
    print(f"\nWrote {args.out_dir / 'customer_segments.parquet'}")
    print(f"Wrote {args.out_dir / 'segment_profile.csv'}")
    print(f"Wrote {args.figure_path}")


if __name__ == "__main__":
    main()
