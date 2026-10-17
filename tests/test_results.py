"""Tests for RESULTS.md generation (results.py)."""

import json

import pytest

from aml_lab.generate import load_config
from aml_lab.papers import rate_with_counts
from aml_lab.results import build_context, generate_results
from aml_lab.scenarios import REPO_ROOT, load_scenarios_config

METRICS_DIR = REPO_ROOT / "artifacts" / "metrics"

pytestmark = pytest.mark.skipif(
    not (METRICS_DIR / "prioritization.json").exists(),
    reason="requires artifacts/metrics from a prior make tuning/model run",
)


@pytest.fixture(scope="module")
def gen_config():
    return load_config()


@pytest.fixture(scope="module")
def scenarios_config():
    return load_scenarios_config()


def test_build_context_scenario_table_matches_json(gen_config, scenarios_config):
    context = build_context(METRICS_DIR, scenarios_config, gen_config)
    with open(METRICS_DIR / "tuning_WLT-01.json") as f:
        wlt01 = json.load(f)

    row = next(s for s in context["scenarios"] if s["scenario_id"] == "WLT-01")
    test = wlt01["test_results"]["pooled"]
    assert row["test_detection"] == rate_with_counts(test["n_detected"], test["n_labeled_in_scope"])


def test_build_context_includes_fan02_ring_rows(gen_config, scenarios_config):
    context = build_context(METRICS_DIR, scenarios_config, gen_config)
    assert len(context["fan02_ring_rows"]) == 3
    assert {r["period"] for r in context["fan02_ring_rows"]} == {"tuning", "validation", "test"}


def test_generate_results_writes_file_with_every_section(gen_config, scenarios_config, tmp_path):
    out_path = tmp_path / "RESULTS.md"
    generate_results(METRICS_DIR, scenarios_config, gen_config, out_path)
    text = out_path.read_text()

    assert "SYNTHETIC DATA" in text
    assert "0 = retail, 1 = mixed, 2 = small_business, 3 = HNW" in text
    for scenario_id in ["STR-01", "RMV-01", "FAN-01", "FAN-02", "WLT-01"]:
        assert scenario_id in text
    assert "ATL/BTL" in text
    assert "FAN-02 ring-level report" in text

    with open(METRICS_DIR / "prioritization.json") as f:
        pr = json.load(f)
    assert str(pr["pr_auc"]["xgboost"]["mean"]) in text
    for stats in pr["review_budget"].values():
        assert str(stats["fps_avoided_vs_fifo_at_equal_detection"]) in text
