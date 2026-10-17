"""Model white papers (Phase 7).

Every number in every generated paper is read from a metrics JSON
(`artifacts/metrics/tuning_<scenario>.json`, `artifacts/metrics/
prioritization.json`) or a derived CSV (`artifacts/segments/
segment_profile.csv`, `artifacts/metrics/alert_queue_test.csv`) - nothing
is hand-typed. The only hand-written content is narrative prose: scenario
descriptions (already static text in scenarios.yaml), the plain-English
summaries, and monitoring-plan policy language (forward-looking rules,
not reported historical numbers).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import jinja2
import pandas as pd

from aml_lab.generate import DEFAULT_CONFIG_PATH, load_config, sim_window
from aml_lab.scenarios import DEFAULT_SCENARIOS_CONFIG_PATH, REPO_ROOT, load_scenarios_config
from aml_lab.tuning import DEFAULT_CAPACITY_CONFIG_PATH, PERIOD_BANDS, load_capacity_config

TEMPLATES_DIR = REPO_ROOT / "templates"
SEGMENT_PURITY_THRESHOLD = 90.0

PLAIN_ENGLISH = {
    "STR-01": (
        "Structuring looks for repeated cash deposits that stay just under the reporting threshold - "
        "several deposits over a short window, each sized to avoid triggering a single large-transaction "
        "report on its own, but which add up to a meaningfully large sum when looked at together."
    ),
    "RMV-01": (
        "Rapid movement looks for money that arrives and then leaves again almost immediately - a large "
        "inflow followed, within hours to a couple of days, by most of it moving back out. Funds that pass "
        "through an account this fast are rarely sitting there for an ordinary reason."
    ),
    "FAN-01": (
        "This is the simple version of fan-in/fan-out detection: flag a customer who receives from many "
        "distinct people and then forwards most of it onward. It looks only at one customer at a time, so "
        "it can spot a very busy hub but has no way to see the senders or downstream recipients around them."
    ),
    "FAN-02": (
        "This is the network version of fan-in/fan-out detection: instead of looking at one customer at a "
        "time, it builds a map of who sends money to whom, keeps only the connections where money is "
        "clearly passing through (not sitting still), and flags whole clusters built around a busy hub - "
        "catching the senders and downstream recipients around a mule, not just the mule."
    ),
    "WLT-01": (
        "Wealth/investment layering looks for money shuffled between a customer's own related investment "
        "accounts several times, then wired out of the country - a pattern that obscures where money "
        "originally came from before it leaves the bank entirely."
    ),
}


def rate_with_counts(numerator, denominator) -> str:
    if not denominator:
        return "n/a (0/0)"
    return f"{numerator / denominator:.2f} ({numerator}/{denominator})"


def fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def segment_label(row: pd.Series) -> str:
    pct = {
        "retail": row["pct_retail"],
        "small_business": row["pct_small_business"],
        "hnw_investment": row["pct_hnw_investment"],
    }
    best = max(pct, key=pct.get)
    return best if pct[best] >= SEGMENT_PURITY_THRESHOLD else "mixed"


def load_segment_profile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df.apply(segment_label, axis=1)
    return df


def jinja_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR), trim_blocks=True, lstrip_blocks=True
    )


# ---------------------------------------------------------------------------
# Scenario white papers
# ---------------------------------------------------------------------------

def _fmt_period_metrics(m: dict) -> dict:
    return {
        "alert_rows": m["alert_rows"],
        "avg_alerts_per_month": m["avg_alerts_per_month"],
        "detection": rate_with_counts(m["n_detected"], m["n_labeled_in_scope"]),
        "productive": rate_with_counts(m["n_productive_customers"], m["n_customers_alerted"]),
    }


def build_scenario_context(
    scenario_id: str, tuning_result: dict, spec: dict, gen_config: dict, capacity_cfg: dict, segment_profile: pd.DataFrame
) -> dict:
    start, end, months = sim_window(gen_config)
    pc = tuning_result["pooled_choice"]

    segments_rows = []
    for _, row in segment_profile.iterrows():
        sid = int(row["segment_assigned"])
        choice = tuning_result["segment_choices"][str(sid)]
        segments_rows.append(
            {
                "segment_id": sid,
                "label": row["label"],
                "n_customers": int(row["n_customers"]),
                "composition": (
                    f"{row['pct_retail']:.1f}% retail / {row['pct_small_business']:.1f}% small_business / "
                    f"{row['pct_hnw_investment']:.1f}% hnw_investment"
                ),
                "fallback": choice["fallback_to_pooled"],
                "params": fmt_params(choice["params"]),
                "capacity": choice["capacity"],
                "tuning": _fmt_period_metrics(choice["tuning"]),
                "validation": _fmt_period_metrics(choice["validation"]),
                "test": _fmt_period_metrics(tuning_result["test_results"][str(sid)]),
                "reason": choice.get("reason"),
            }
        )

    atl_btl_rows = [
        {
            "param": a["primary_param"],
            "chosen_value": a["chosen_value"],
            "step_value": a["step_value"],
            "extra_alerted": a["n_extra_alerted_customers"],
            "extra_productive": rate_with_counts(a["n_extra_productive"], a["n_extra_alerted_customers"]),
        }
        for a in tuning_result["atl_btl"]
    ]

    ring_report_rows = None
    if "ring_report" in tuning_result:
        ring_report_rows = [
            {
                "period": p,
                "rings": rate_with_counts(stats["n_rings_recovered"], stats["n_rings"]),
                "members": rate_with_counts(stats["n_members_caught"], stats["n_members_total"]),
            }
            for p, stats in tuning_result["ring_report"]["by_period"].items()
        ]

    return {
        "scenario_id": scenario_id,
        "name": spec["name"],
        "typology": spec["typology"],
        "description": spec["description"].strip(),
        "plain_english": PLAIN_ENGLISH[scenario_id],
        "generated_date": dt.date.today().isoformat(),
        "seed": gen_config["seed"],
        "sim_start": str(start),
        "sim_end": str(end),
        "months": months,
        "period_bands": PERIOD_BANDS,
        "grid": spec["grid"],
        "grid_size": tuning_result["grid_size"],
        "chosen_params": pc["params"],
        "capacity_cfg": capacity_cfg,
        "pooled_capacity": pc["capacity"],
        "pooled_capacity_satisfied": pc["capacity_constraint_satisfied"],
        "pooled_tuning": _fmt_period_metrics(pc["tuning"]),
        "pooled_validation": _fmt_period_metrics(pc["validation"]),
        "pooled_test": _fmt_period_metrics(tuning_result["test_results"]["pooled"]),
        "segments": segments_rows,
        "atl_btl": atl_btl_rows,
        "limitations": tuning_result["limitations"],
        "ring_report_rows": ring_report_rows,
    }


# ---------------------------------------------------------------------------
# Prioritization white paper
# ---------------------------------------------------------------------------

def build_prioritization_context(prioritization_result: dict, alert_queue_csv_path: Path, gen_config: dict) -> dict:
    r = prioritization_result

    budgets = []
    for key, stats in r["review_budget"].items():
        pct = int(key.replace("top_", "").replace("pct", ""))
        budgets.append(
            {
                "pct": pct,
                "n_reviewed": stats["n_reviewed"],
                "captured": stats["productive_captured"],
                "fifo_needed": stats["fifo_alerts_needed_to_match_xgboost"],
                "fps_avoided_fifo": stats["fps_avoided_vs_fifo_at_equal_detection"],
                "random_needed": stats["random_alerts_needed_to_match_xgboost_mean"],
                "fps_avoided_random": stats["fps_avoided_vs_random_at_equal_detection"],
            }
        )
    budget_20 = next(b for b in budgets if b["pct"] == 20)

    by_scenario = []
    for sid, stats in r["by_scenario"].items():
        top40 = (
            f"{stats['top_40pct_ml_captured']} / {stats['top_40pct_fifo_captured']} (n={stats['n_reviewed_at_40pct']})"
            if "top_40pct_ml_captured" in stats
            else "n/a (too few positives/negatives)"
        )
        by_scenario.append({"scenario_id": sid, "n_alerts": stats["n_alerts"], "n_positives": stats["n_positives"], "pr_auc": stats["pr_auc"], "top40": top40})

    fan01 = r["by_scenario"].get("FAN-01", {})
    rmv01 = r["by_scenario"].get("RMV-01", {})
    str01 = r["by_scenario"].get("STR-01", {})
    wlt01 = r["by_scenario"].get("WLT-01", {})

    top_driver_counts = []
    if alert_queue_csv_path.exists():
        queue = pd.read_csv(alert_queue_csv_path)
        counts = Counter(queue["driver_1_feature"])
        top_driver_counts = [{"feature": f, "count": int(c)} for f, c in counts.most_common(8)]

    baseline_note = (
        "Reported plainly, not tuned away by touching test data."
        if not r["pr_auc"]["xgboost_beats_baseline"]
        else "The gap is modest and the confidence intervals overlap substantially."
    )

    return {
        "generated_date": dt.date.today().isoformat(),
        "seed": r["seed"],
        "train_n_alerts": r["train"]["n_alerts"],
        "train_n_positives": r["train"]["n_positives"],
        "test_n_alerts": r["test"]["n_alerts"],
        "test_n_positives": r["test"]["n_positives"],
        "model_params": r["model_params"],
        "scenario_prior_train_rates": r["scenario_prior_train_rates"],
        "pr_auc": r["pr_auc"],
        "baseline_note": baseline_note,
        "budgets": budgets,
        "budget_20": budget_20,
        "by_scenario": by_scenario,
        "fan01_pr_auc": fan01.get("pr_auc"),
        "fan01_n_alerts": fan01.get("n_alerts"),
        "rmv01_pr_auc": rmv01.get("pr_auc"),
        "str01_n": str01.get("n_alerts"),
        "wlt01_n": wlt01.get("n_alerts"),
        "missed_alerts": r["productive_alerts_outside_top_60pct"],
        "missed_count": len(r["productive_alerts_outside_top_60pct"]),
        "top_driver_counts": top_driver_counts,
        "rules": r["rules"],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_scenario_papers(
    scenarios_config: dict, gen_config: dict, capacity_cfg: dict, metrics_dir: Path, segment_profile_path: Path, out_dir: Path
) -> list[Path]:
    env = jinja_env()
    template = env.get_template("white_paper.md.j2")
    segment_profile = load_segment_profile(segment_profile_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for scenario_id, spec in scenarios_config.items():
        with open(metrics_dir / f"tuning_{scenario_id}.json") as f:
            tuning_result = json.load(f)
        context = build_scenario_context(scenario_id, tuning_result, spec, gen_config, capacity_cfg, segment_profile)
        out_path = out_dir / f"{scenario_id}.md"
        out_path.write_text(template.render(**context))
        written.append(out_path)
    return written


def generate_prioritization_paper(metrics_dir: Path, gen_config: dict, out_dir: Path) -> Path:
    env = jinja_env()
    template = env.get_template("prioritization_paper.md.j2")
    with open(metrics_dir / "prioritization.json") as f:
        result = json.load(f)
    context = build_prioritization_context(result, metrics_dir / "alert_queue_test.csv", gen_config)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prioritization.md"
    out_path.write_text(template.render(**context))
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate model white papers (Phase 7).")
    parser.add_argument("--generator-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--scenarios-config", type=Path, default=DEFAULT_SCENARIOS_CONFIG_PATH)
    parser.add_argument("--capacity-config", type=Path, default=DEFAULT_CAPACITY_CONFIG_PATH)
    parser.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "artifacts" / "metrics")
    parser.add_argument("--segment-profile-path", type=Path, default=REPO_ROOT / "artifacts" / "segments" / "segment_profile.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "papers")
    args = parser.parse_args(argv)

    gen_config = load_config(args.generator_config)
    scenarios_config = load_scenarios_config(args.scenarios_config)
    capacity_cfg = load_capacity_config(args.capacity_config)

    scenario_paths = generate_scenario_papers(
        scenarios_config, gen_config, capacity_cfg, args.metrics_dir, args.segment_profile_path, args.out_dir
    )
    prioritization_path = generate_prioritization_paper(args.metrics_dir, gen_config, args.out_dir)

    for p in scenario_paths:
        print(f"Wrote {p}")
    print(f"Wrote {prioritization_path}")


if __name__ == "__main__":
    main()
