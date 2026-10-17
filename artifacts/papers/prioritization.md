# Model White Paper: Alert Prioritization (ML Reordering)

**SYNTHETIC DATA.** Every customer, account, and transaction referenced in this document is generated, not real.

Generated: 2026-10-17 | Dataset seed: 42

**Segment numbering used throughout this paper:** 0 = retail, 1 = mixed, 2 = small_business, 3 = HNW.

---

## Plain-English Summary

Analysts can only review a fraction of the alerts a rule-based monitoring system raises. This model does not decide which alerts fire - the five rule scenarios (STR-01, RMV-01, FAN-01, FAN-02, WLT-01) do that, at their Phase 5 chosen thresholds, unchanged. **The model reorders alerts within the queue it is given; it never suppresses a scenario hit - every rule-based alert stays in the queue regardless of model score.**

What the model does is rank the queue so the alerts most likely to be genuine (productive) are reviewed first. On the test period, reviewing the top 20% of the queue in model order catches 26 of 27 productive alerts - compared to 3 for first-in-first-out order and 5.37 for random order over the same budget.

## 1. Purpose and Scope

This model reorders the combined alert queue from all five scenarios; it is evaluated against three baselines: a logistic regression on the same features, a "scenario prior" (rank by each scenario's own historical productive rate, no per-alert learning at all), and the FIFO/random orders an analyst queue would otherwise use.

The model reorders alerts within the queue it is given; it never suppresses a scenario hit - every rule-based alert stays in the queue regardless of model score.

Test-period data (months 11-12) was used only for the numbers above, never for fitting, hyperparameter choice, or model selection.

## 2. Data

| Split | Alerts | Positives (productive) |
|---|---|---|
| Train (tuning + validation, months 1-10) | 921 | 76 |
| Test (months 11-12, untouched until final evaluation) | 207 | 27 |

Features: the Phase 2 SQL feature table (18 columns) joined to each alert's `(customer_id, month)`, plus 3 generic evidence columns (`evidence_count`, `evidence_amount`, `evidence_ratio` - a scenario-specific mapping of each rule's triggering evidence onto a shared numeric shape) and a one-hot scenario ID.

## 3. Model

| | XGBoost | Logistic regression (baseline) | Scenario prior (baseline) |
|---|---|---|---|
| Type | Gradient-boosted trees | Linear, `class_weight="balanced"` | No learning - each test alert scored by its own scenario's train-period productive rate (ties broken by evidence amount) |
| max_depth | 3 | - | - |
| n_estimators | 100 | - | - |
| learning_rate | 0.1 | - | - |
| scale_pos_weight | 11.118 | class_weight=balanced | - |
| Seed | 42 (fixed throughout: model fit, bootstrap resampling, shuffle baselines) | 42 | - |

**Scenario-prior train-period productive rates** (the baseline's entire "model"):

- FAN-01: 0.0048
- FAN-02: 0.2593
- RMV-01: 0.0773
- STR-01: 0.8966
- WLT-01: 1.0

## 4. Performance: PR-AUC (Test Period, 1,000-Sample Bootstrap 95% CI)

| Model | PR-AUC (mean) | 95% CI |
|---|---|---|
| XGBoost | 0.8603 | [0.7421, 0.9449] |
| Logistic regression | 0.8447 | [0.7151, 0.9413] |
| Scenario prior | 0.528 | [0.3544, 0.6868] |

**XGBoost beats the logistic regression baseline on PR-AUC: True.** The gap is modest and the confidence intervals overlap substantially.

The more informative comparison is against the scenario-prior baseline (no per-alert learning): 0.8603 vs 0.528, with non-overlapping confidence intervals - evidence that alert-level features add real ranking power beyond "trust each scenario's own historical hit rate."

**However: logistic regression ties XGBoost's productive-alert capture at every review budget tested** (Section 5) despite the lower PR-AUC - the two models rank the small number of true positives into essentially the same top slice, they just disagree on the exact ordering below it. XGBoost's edge in this run is therefore about ranking precision deeper in the queue, not about which alerts reach the top at the budgets analysts actually use.

## 5. Review-Budget Capture, All Baselines

| Budget | Reviewed | XGBoost | Logistic regression | Scenario prior | FIFO | Random (mean) |
|---|---|---|---|---|---|---|
| Top 20% | 41 | 26 | 26 | 18 | 3 | 5.37 |
| Top 40% | 83 | 26 | 26 | 26 | 9 | 10.78 |
| Top 60% | 124 | 26 | 26 | 27 | 16 | 16.09 |

### Corrected Equal-Detection Table

For each budget: the number of alerts FIFO (or random, mean of 1,000 shuffles) must review to find as many productive alerts as XGBoost found in its own top-k, minus XGBoost's k - i.e. alerts (and therefore false positives) avoided for the same number of true positives found.

| Budget | XGBoost alerts reviewed | FIFO alerts needed to match | **FPs avoided vs. FIFO** | Random alerts needed to match (mean) | **FPs avoided vs. random** |
|---|---|---|---|---|---|
| Top 20% | 41 | 191 | 150 | 193.05 | 152.05 |
| Top 40% | 83 | 191 | 108 | 193.32 | 110.32 |
| Top 60% | 124 | 191 | 67 | 193.26 | 69.26 |

## 6. Per-Scenario Breakdown

| Scenario | Test alerts | Positives | PR-AUC | Top-40% captured (ML / FIFO) |
|---|---|---|---|---|
| FAN-01 | 125 | 2 | 0.5169 | 1 / 1 (n=50) |
| FAN-02 | 35 | 9 | 0.5773 | 7 / 2 (n=14) |
| RMV-01 | 36 | 7 | 0.9683 | 7 / 4 (n=14) |
| STR-01 | 6 | 4 | 1.0 | 2 / 2 (n=2) |
| WLT-01 | 5 | 5 | None | n/a (too few positives/negatives) |

**The model does not mainly help the FAN-01 queue.** FAN-01 has the weakest PR-AUC of any scenario with enough signal to compute one (0.5169, barely above chance) despite carrying 125 of 207 test-period alerts - the largest single queue. RMV-01 (0.9683) benefits the most from ranking. This is consistent with Phase 5: FAN-01's rule-level detection economics were already weak (2 of 11 true positives buried in hundreds of false positives), and there is correspondingly little separable per-alert signal for any model to exploit yet.

## 7. The One Missed Alert

XGBoost places **1** productive alert(s) outside the top 60% of the ranked queue:

- Rank **137** of 207 (FAN-01, customer CUST010717, month 11, score 0.0005).

This sits squarely in the FAN-01 queue (Section 6), consistent with that scenario's weak overall signal.

## 8. SHAP: Global Feature Importance

See `artifacts/figures/shap_global.png` for the full ranking. Most frequent #1 driver across the 207 test alerts (from `artifacts/metrics/alert_queue_test.csv`):

| Feature | Times ranked #1 driver |
|---|---|
| evidence_amount | 191 |
| activity_zscore_90d | 11 |
| evidence_ratio | 4 |
| residual_balance_48h_min | 1 |

Per-alert top-3 SHAP drivers for every test alert are in `artifacts/metrics/alert_queue_test.csv`.

## 9. Rules

- The model reorders alerts within the queue it is given; it never suppresses a scenario hit - every rule-based alert stays in the queue regardless of model score.
- Test-period data (months 11-12) was used only for the numbers above, never for fitting, hyperparameter choice, or model selection.

## 10. Limitations and Assumptions

- Several scenarios have very small test sets (STR-01: 6 alerts; WLT-01: 5 alerts), making their individual PR-AUC figures (Section 6) statistically fragile - read as directional, not precise.
- WLT-01 has no negative test alerts, so PR-AUC is undefined there (reported as such, not imputed).
- The scenario-prior baseline can match or exceed XGBoost's raw capture count at the largest budget tested (60%) purely because of scenario alert-volume differences (Section 5) - a reminder that budget-capture and PR-AUC can disagree at a single operating point; PR-AUC (Section 4) is the more complete comparison.
- Train and test alerts come from the SAME deployed rule thresholds chosen in Phase 5; this model has not been evaluated against a different threshold regime.

## 11. Ongoing Monitoring Plan

- **PR-AUC drift:** investigate if a future test-period PR-AUC bootstrap CI for XGBoost drops to overlap the scenario-prior baseline's CI ([0.3544, 0.6868]) - that would mean the model has stopped adding value beyond trusting each scenario's own historical rate.
- **Review-budget capture drift:** investigate if top-20% capture (currently 26 of 27) falls for two consecutive monthly batches.
- **Re-tuning triggers:** retrain (a) whenever the underlying rule thresholds are re-tuned (Phase 5 re-run), since the alert population and evidence distributions shift with them, (b) on a fixed cadence as new labeled test-period-equivalent data accumulates, or (c) if either drift condition above is observed. Never retrain or select a model configuration using test-period data - re-run the full train/validation methodology on a newly rolled-forward split instead.

## 12. Change Log

| Version | Date | Change |
|---|---|---|
| v1 | 2026-10-17 | Initial paper, generated by `src/aml_lab/papers.py` from `artifacts/metrics/prioritization.json` (dataset seed 42). |