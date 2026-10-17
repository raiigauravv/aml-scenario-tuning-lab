"""Tests for Phase 7 white paper generation (papers.py).

Covers the formatting helpers directly, then an end-to-end check that
every number appearing in a generated paper traces back to the metrics
JSON it was built from - including a test that MUTATES a copy of a real
metrics file and confirms the new number (not a cached/hand-typed one)
shows up in the regenerated paper.
"""

import copy
import json

import pandas as pd
import pytest

from aml_lab.generate import load_config
from aml_lab.papers import (
    build_prioritization_context,
    build_scenario_context,
    fmt_params,
    jinja_env,
    load_segment_profile,
    rate_with_counts,
    segment_label,
)
from aml_lab.scenarios import load_scenarios_config
from aml_lab.tuning import load_capacity_config

REPO_ROOT = next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "artifacts").exists())
METRICS_DIR = REPO_ROOT / "artifacts" / "metrics"
SEGMENT_PROFILE_PATH = REPO_ROOT / "artifacts" / "segments" / "segment_profile.csv"

pytestmark = pytest.mark.skipif(
    not (METRICS_DIR / "tuning_WLT-01.json").exists() or not SEGMENT_PROFILE_PATH.exists(),
    reason="requires artifacts/metrics and artifacts/segments from a prior make tuning run",
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def test_rate_with_counts_formats_and_handles_zero_denominator():
    assert rate_with_counts(4, 6) == "0.67 (4/6)"
    assert rate_with_counts(0, 5) == "0.00 (0/5)"
    assert rate_with_counts(0, 0) == "n/a (0/0)"
    assert rate_with_counts(5, 5) == "1.00 (5/5)"


def test_segment_label_purity_threshold():
    pure_retail = pd.Series({"pct_retail": 100.0, "pct_small_business": 0.0, "pct_hnw_investment": 0.0})
    assert segment_label(pure_retail) == "retail"

    mostly_small_biz = pd.Series({"pct_retail": 2.2, "pct_small_business": 97.8, "pct_hnw_investment": 0.0})
    assert segment_label(mostly_small_biz) == "small_business"

    mixed = pd.Series({"pct_retail": 79.0, "pct_small_business": 19.2, "pct_hnw_investment": 1.8})
    assert segment_label(mixed) == "mixed"


def test_fmt_params():
    assert fmt_params({"min_deposits": 6, "window_days": 7}) == "min_deposits=6, window_days=7"


# ---------------------------------------------------------------------------
# End-to-end: every number in the rendered paper traces to the JSON
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gen_config():
    return load_config()


@pytest.fixture(scope="module")
def scenarios_config():
    return load_scenarios_config()


@pytest.fixture(scope="module")
def capacity_cfg():
    return load_capacity_config()


@pytest.fixture(scope="module")
def segment_profile():
    return load_segment_profile(SEGMENT_PROFILE_PATH)


def test_scenario_paper_contains_numbers_from_its_own_json(gen_config, scenarios_config, capacity_cfg, segment_profile):
    with open(METRICS_DIR / "tuning_WLT-01.json") as f:
        tuning_result = json.load(f)
    context = build_scenario_context("WLT-01", tuning_result, scenarios_config["WLT-01"], gen_config, capacity_cfg, segment_profile)

    env = jinja_env()
    rendered = env.get_template("white_paper.md.j2").render(**context)

    assert "SYNTHETIC DATA" in rendered
    assert "0 = retail, 1 = mixed, 2 = small_business, 3 = HNW" in rendered
    # The pooled test-period detection rate, read straight from the JSON.
    test_metrics = tuning_result["test_results"]["pooled"]
    expected = rate_with_counts(test_metrics["n_detected"], test_metrics["n_labeled_in_scope"])
    assert expected in rendered
    # Limitations copied verbatim.
    for item in tuning_result["limitations"]:
        assert item in rendered
    # Chosen params appear.
    for key, value in tuning_result["pooled_choice"]["params"].items():
        assert f"`{key}` = {value}" in rendered


def test_scenario_paper_reflects_mutated_json_not_a_cached_value(gen_config, scenarios_config, capacity_cfg, segment_profile):
    """Nothing is hand-typed or cached: change a number in a COPY of the
    JSON and the regenerated paper must show the new number, not the
    original one.
    """
    with open(METRICS_DIR / "tuning_WLT-01.json") as f:
        tuning_result = json.load(f)
    mutated = copy.deepcopy(tuning_result)
    mutated["test_results"]["pooled"]["n_detected"] = 1
    mutated["test_results"]["pooled"]["n_labeled_in_scope"] = 9

    context = build_scenario_context("WLT-01", mutated, scenarios_config["WLT-01"], gen_config, capacity_cfg, segment_profile)
    rendered = jinja_env().get_template("white_paper.md.j2").render(**context)

    # The mutated detection rate (not the original) appears in the
    # Performance section - proof the paper is rendered from whatever
    # JSON it's given, not a cached or hand-typed value. (Not asserting
    # the ORIGINAL rate is absent: "1.00 (5/5)" also legitimately appears
    # elsewhere as an unrelated, un-mutated productive-rate figure with
    # the same numerator/denominator by coincidence.)
    performance_section = rendered.split("Performance on Test Period")[1].split("Limitations")[0]
    assert rate_with_counts(1, 9) in performance_section


def test_fan02_paper_includes_ring_report(gen_config, scenarios_config, capacity_cfg, segment_profile):
    with open(METRICS_DIR / "tuning_FAN-02.json") as f:
        tuning_result = json.load(f)
    context = build_scenario_context("FAN-02", tuning_result, scenarios_config["FAN-02"], gen_config, capacity_cfg, segment_profile)
    rendered = jinja_env().get_template("white_paper.md.j2").render(**context)

    assert "Ring-level view" in rendered
    for stats in tuning_result["ring_report"]["by_period"].values():
        assert rate_with_counts(stats["n_rings_recovered"], stats["n_rings"]) in rendered
        assert rate_with_counts(stats["n_members_caught"], stats["n_members_total"]) in rendered


def test_non_ring_scenario_paper_omits_ring_section(gen_config, scenarios_config, capacity_cfg, segment_profile):
    with open(METRICS_DIR / "tuning_STR-01.json") as f:
        tuning_result = json.load(f)
    context = build_scenario_context("STR-01", tuning_result, scenarios_config["STR-01"], gen_config, capacity_cfg, segment_profile)
    rendered = jinja_env().get_template("white_paper.md.j2").render(**context)
    assert "Ring-level view" not in rendered


def test_prioritization_paper_covers_all_required_elements(gen_config):
    with open(METRICS_DIR / "prioritization.json") as f:
        result = json.load(f)
    context = build_prioritization_context(result, METRICS_DIR / "alert_queue_test.csv", gen_config)
    rendered = jinja_env().get_template("prioritization_paper.md.j2").render(**context)

    # All three baselines, PR-AUC with CIs.
    for model in ("xgboost", "logistic_regression", "scenario_prior"):
        m = result["pr_auc"][model]
        assert str(m["mean"]) in rendered
        assert str(m["ci_lower"]) in rendered

    # Corrected equal-detection table values.
    for stats in result["review_budget"].values():
        assert str(stats["fps_avoided_vs_fifo_at_equal_detection"]) in rendered

    # Logistic regression ties XGBoost claim.
    assert "logistic regression ties XGBoost" in rendered.lower() or "ties XGBoost" in rendered

    # FAN-01 weakness and the one missed alert.
    assert "FAN-01" in rendered and "does not mainly help the FAN-01 queue" in rendered
    for m in result["productive_alerts_outside_top_60pct"]:
        assert str(m["rank"]) in rendered
        assert m["scenario_id"] in rendered

    # Reorders-never-suppresses statement.
    assert result["rules"][0] in rendered

    assert "0 = retail, 1 = mixed, 2 = small_business, 3 = HNW" in rendered
