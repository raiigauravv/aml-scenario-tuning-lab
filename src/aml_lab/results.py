"""RESULTS.md generation (Phase 9).

Reads only artifacts/metrics/*.json and renders RESULTS.md via Jinja2 -
every number in the output traces back to a metrics file; nothing is
hand-typed. Run with `make results`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from aml_lab.generate import DEFAULT_CONFIG_PATH, load_config
from aml_lab.papers import jinja_env, rate_with_counts
from aml_lab.scenarios import DEFAULT_SCENARIOS_CONFIG_PATH, REPO_ROOT, load_scenarios_config

SCENARIO_ORDER = ["STR-01", "RMV-01", "FAN-01", "FAN-02", "WLT-01"]


def build_context(metrics_dir: Path, scenarios_config: dict, gen_config: dict) -> dict:
    scenarios = []
    atl_btl_rows = []
    fan02_ring_rows = []

    for scenario_id in SCENARIO_ORDER:
        path = metrics_dir / f"tuning_{scenario_id}.json"
        if not path.exists():
            continue
        with open(path) as f:
            tr = json.load(f)
        pc = tr["pooled_choice"]
        test = tr["test_results"]["pooled"]

        scenarios.append(
            {
                "scenario_id": scenario_id,
                "validation_detection": rate_with_counts(pc["validation"]["n_detected"], pc["validation"]["n_labeled_in_scope"]),
                "validation_productive": rate_with_counts(pc["validation"]["n_productive_customers"], pc["validation"]["n_customers_alerted"]),
                "test_detection": rate_with_counts(test["n_detected"], test["n_labeled_in_scope"]),
                "test_productive": rate_with_counts(test["n_productive_customers"], test["n_customers_alerted"]),
                "test_alerts_per_month": test["avg_alerts_per_month"],
            }
        )

        for a in tr["atl_btl"]:
            atl_btl_rows.append(
                {
                    "scenario_id": scenario_id,
                    "param": a["primary_param"],
                    "chosen_value": a["chosen_value"],
                    "step_value": a["step_value"],
                    "extra_alerted": a["n_extra_alerted_customers"],
                    "extra_productive": rate_with_counts(a["n_extra_productive"], a["n_extra_alerted_customers"]),
                }
            )

        if "ring_report" in tr:
            for period, stats in tr["ring_report"]["by_period"].items():
                fan02_ring_rows.append(
                    {
                        "period": period,
                        "rings": rate_with_counts(stats["n_rings_recovered"], stats["n_rings"]),
                        "members": rate_with_counts(stats["n_members_caught"], stats["n_members_total"]),
                    }
                )

    prioritization_path = metrics_dir / "prioritization.json"
    with open(prioritization_path) as f:
        pr = json.load(f)

    budgets = []
    for key, stats in pr["review_budget"].items():
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
    budgets.sort(key=lambda b: b["pct"])

    return {
        "generated_date": dt.date.today().isoformat(),
        "seed": gen_config["seed"],
        "scenarios": scenarios,
        "atl_btl_rows": atl_btl_rows,
        "fan02_ring_rows": fan02_ring_rows,
        "pr_auc": pr["pr_auc"],
        "budgets": budgets,
        "missed_alerts": pr["productive_alerts_outside_top_60pct"],
        "missed_count": len(pr["productive_alerts_outside_top_60pct"]),
        "test_n_alerts": pr["test"]["n_alerts"],
    }


def generate_results(metrics_dir: Path, scenarios_config: dict, gen_config: dict, out_path: Path) -> Path:
    env = jinja_env()
    template = env.get_template("results.md.j2")
    context = build_context(metrics_dir, scenarios_config, gen_config)
    out_path.write_text(template.render(**context))
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate RESULTS.md from artifacts/metrics (Phase 9).")
    parser.add_argument("--generator-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--scenarios-config", type=Path, default=DEFAULT_SCENARIOS_CONFIG_PATH)
    parser.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "artifacts" / "metrics")
    parser.add_argument("--out-path", type=Path, default=REPO_ROOT / "RESULTS.md")
    args = parser.parse_args(argv)

    gen_config = load_config(args.generator_config)
    scenarios_config = load_scenarios_config(args.scenarios_config)
    out_path = generate_results(args.metrics_dir, scenarios_config, gen_config, args.out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
