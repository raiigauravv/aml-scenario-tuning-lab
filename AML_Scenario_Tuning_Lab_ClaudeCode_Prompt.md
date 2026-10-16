# Claude Code Prompt: AML Scenario Tuning Lab

Paste everything below the line into Claude Code, run from an empty folder that will become the GitHub repo.

---

## Goal

Build `aml-scenario-tuning-lab`, a portfolio project that reproduces the core work of a bank's AML Transaction Monitoring data science team on fully synthetic data:

1. Rule-based monitoring scenarios with tunable parameters
2. Customer segmentation before tuning
3. Threshold calibration per segment with above-the-line / below-the-line (ATL/BTL) testing
4. An ML layer that prioritizes rule alerts (it does not replace the rules)
5. A model white paper per scenario, generated from committed results
6. A what-if dashboard for thresholds

Audience: a hiring manager on an AML Transaction Monitoring data science team. The project must read like work a model validator could review, not like a Kaggle notebook.

## Hard rules (non-negotiable)

- All data is synthetic. Say so in the README, every white paper, and the dashboard header.
- Never type a metric by hand. Every number in README, RESULTS.md, white papers and the dashboard must be produced by a script and saved to `artifacts/metrics/*.json`, then read from there.
- Fix all random seeds. `make all` must reproduce every artifact from scratch.
- Choose thresholds on a validation period. Report final numbers on a later, untouched test period. Never tune on test.
- State limitations plainly: synthetic data, injected typologies (so detection is easier than reality), no real regulatory context, no human-labeled investigation outcomes.
- Keep it simple enough that I can explain every design choice in an interview. Prefer clear code over clever code.
- Ask me before adding any new heavy dependency.

## Tech stack

Python 3.11+, pandas, numpy, DuckDB (SQL feature layer), scikit-learn, XGBoost, SHAP, networkx (only for the fan-in/fan-out typology features), Streamlit, matplotlib, pytest, ruff, GitHub Actions. Jinja2 for the white-paper template.

## Repo layout

```
aml-scenario-tuning-lab/
  README.md
  RESULTS.md                  # generated
  Makefile                    # make data / features / tune / model / papers / test / all
  pyproject.toml
  config/
    scenarios.yaml            # scenario definitions + parameter grids
    segments.yaml             # segmentation settings
    capacity.yaml             # analyst review capacity assumptions
  src/aml_lab/
    generate.py               # synthetic data + injected typologies
    features.sql              # DuckDB feature queries
    features.py
    segment.py
    scenarios.py              # rule engine
    tuning.py                 # threshold sweeps + ATL/BTL
    prioritize.py             # ML alert prioritization + SHAP
    papers.py                 # white paper generation
    metrics.py
  templates/white_paper.md.j2
  app/dashboard.py
  notebooks/01_eda.ipynb
  artifacts/                  # generated: data, metrics json, figures, papers
  tests/
  .github/workflows/ci.yml
```

## Phase 1: Synthetic data generator (`generate.py`)

Generate ~50k customers and ~2-5M transactions over 12 months (make size configurable; default small enough to run on a laptop in a few minutes).

Customer attributes: customer_id, segment_true (retail, small business, high-net-worth investment client), country risk tier, account open date, expected monthly activity.

Accounts: chequing/savings for all; investment accounts for high-net-worth clients.

Transactions: id, timestamp, from_account, to_account, amount, currency, channel (cash deposit, wire, e-transfer, ACH, card, internal journal), cross_border flag.

Normal behaviour must be realistic and noisy: payroll, bills, seasonal spikes, legitimate large one-off transfers, small businesses with many cash deposits. Legitimate behaviour must sometimes look suspicious (this is what makes false positives realistic).

Inject labeled typologies into a small fraction of customers (configurable, around 0.5-1% of customers), each with variation so they are not trivially separable:

1. Structuring: repeated cash deposits just below a reporting threshold (e.g. $10,000) across days or branches.
2. Rapid movement: funds in and out within a short window with little residual balance.
3. Fan-in / fan-out mule network: many accounts send to one, which forwards onward.
4. Wealth / investment layering: funds moved through internal journal transfers between related investment accounts, then wired out cross-border. (This one matters: the target role mentions expanding monitoring coverage to Capital Markets and Wealth Management.)

Output: parquet files plus `labels.parquet` (customer_id, typology, start/end dates). Write a short `artifacts/data_card.md` describing the generator, rates, and known simplifications.

Acceptance: deterministic with seed; data card generated; tests check typology rates and that injected patterns actually exist.

## Phase 2: SQL feature layer (`features.sql`, `features.py`)

Load parquet into DuckDB. Build features in SQL (keep the SQL readable, it is a selling point):

- Rolling 7/30-day cash deposit count and sum per customer
- Count of deposits within 10% below the reporting threshold
- In/out ratio and residual balance within 48h windows
- Number of distinct counterparties in/out per 30 days
- Share of cross-border volume, high-risk country exposure
- Activity vs the customer's own 90-day baseline (z-score)
- Journal-transfer volume between related investment accounts
- Simple graph features for fan-in/fan-out (in-degree, out-degree over 30 days) via networkx

Acceptance: features reproducible, unit tests on a tiny hand-built dataset with known answers.

## Phase 3: Segmentation (`segment.py`)

Segment customers using behaviour features (KMeans or rules + KMeans), compare against `segment_true`, and pick a small number of segments (3-5). Save a segment profile table and figure.

Explain in the white paper why tuning per segment beats one global threshold.

## Phase 4: Scenarios (`scenarios.py`, `config/scenarios.yaml`)

Implement 4 scenarios matching the typologies, each as a transparent rule with named parameters, e.g.:

- STR-01 Structuring: >= N cash deposits within D days, each between (1 - p) * threshold and threshold
- RMV-01 Rapid movement: inflow >= X and >= Y% moved out within H hours
- FAN-01 Fan-in/out: distinct senders >= S and forwarded share >= F within D days
- WLT-01 Investment layering: journal transfers between related accounts >= J and outbound cross-border wire >= W within D days

Each scenario produces alerts at the customer-period level with the triggering evidence attached (which transactions, which values).

## Phase 5: Threshold tuning with ATL/BTL (`tuning.py`)

Time split: months 1-8 tuning, 9-10 validation, 11-12 test.

For each scenario x segment:

1. Sweep the parameter grid from `scenarios.yaml`.
2. For each setting compute alert volume, productive alert rate (alerts that hit a labeled typology), and detection rate (labeled cases caught).
3. Choose the operating point under an explicit analyst capacity constraint from `capacity.yaml` (e.g. max alerts per month per segment), maximizing detection.
4. ATL/BTL testing: take the chosen threshold, step it down one or two increments, sample the extra alerts that would appear below the line, and report how many productive cases they contain. This estimates what the chosen threshold misses and justifies (or challenges) it.
5. Save everything to `artifacts/metrics/tuning_<scenario>.json` and plots: alert volume vs threshold, productive rate vs threshold, BTL findings.

Acceptance: chosen thresholds are defensible from the saved numbers alone; test-period results reported separately.

## Phase 6: ML alert prioritization (`prioritize.py`)

Train XGBoost on validation-period rule alerts (target = alert is productive), using the SQL features plus scenario evidence. Handle imbalance with `scale_pos_weight`. Evaluate on test-period alerts.

Report:

- PR-AUC
- At a fixed review budget (e.g. analysts review only the top K% of alerts), what share of productive alerts is still caught vs random or FIFO order
- False-positive alerts avoided at equal detection

Add SHAP: global feature importance and a per-alert explanation of the top drivers. Be explicit in the paper that the model reorders alerts and never suppresses a scenario hit by itself.

## Phase 7: Model white papers (`papers.py`, `templates/white_paper.md.j2`)

Generate one paper per scenario plus one for the prioritization model, all values pulled from metrics JSON. Sections:

1. Purpose and scope
2. Data (synthetic, data card reference, time periods)
3. Scenario logic and parameters
4. Segmentation and rationale
5. Threshold calibration: grid, capacity constraint, chosen values, why
6. ATL/BTL testing results and interpretation
7. Performance on test period
8. Limitations and assumptions
9. Ongoing monitoring plan (alert volume drift, productive rate drift, re-tuning triggers)
10. Change log

Export Markdown; optionally PDF.

## Phase 8: Dashboard (`app/dashboard.py`)

Streamlit app with a "SYNTHETIC DATA" banner:

- Pick scenario and segment, move threshold sliders, see alert volume, productive rate, detection rate, and estimated BTL misses update from precomputed sweep results
- Alert queue view ordered by ML priority with SHAP drivers per alert
- Link to the generated white paper

## Phase 9: Quality and docs

- pytest: generator rates, SQL features on known inputs, scenario rules on handcrafted cases, no test-period leakage in tuning, metrics JSON schema
- ruff, GitHub Actions running lint + tests on push
- README: what problem this mirrors, architecture diagram (Mermaid), how to run (`make all`, `streamlit run app/dashboard.py`), headline results read from RESULTS.md, limitations section
- RESULTS.md generated by a script from metrics JSON

## Build order and checkpoints

Work phase by phase. After each phase: run tests, show me the key outputs, and wait for my OK before moving on. Commit after each phase with a clear message.

## Addendum (approved after Phase 1 kickoff, 2026-10-16)

Changes to Phase 1 (`generate.py`), already incorporated into the build:

1. Default `--n-customers 20000`, plus a `--small` flag (~2000) for fast tests.
2. Spread typology start dates across all 12 months so tuning (1-8), validation (9-10) and test (11-12) periods each contain cases.
3. Deliberately generate legitimate look-alikes: small businesses with legit cash deposits near the threshold, HNW clients with legit journal transfers between their own accounts.
4. Add `relationships.parquet` linking related accounts (same household/owner) for the wealth layering typology.
5. Add transaction-level labels (`is_injected`, `typology`) in addition to customer-level labels.
6. For fan-in/fan-out, label every customer in the mule network (senders, hub, downstream).
7. Put the reporting threshold ($10,000) and injection rates in a config file (`config/generator.yaml`), not hard-coded.
8. Add equity and fixed-income buy/sell transactions for investment accounts (both as normal HNW activity and as part of the wealth layering typology, so the typology isn't trivially separable by trade activity alone).

Future phases — captured now, **not built yet**, to be applied when that phase starts:

- **Phase 2**: also run the feature SQL through PySpark (Spark SQL, Hive-compatible syntax) as an alternative engine; keep DuckDB as the default. Ask before adding the `pyspark` dependency.
- **Phase 2/4**: add network analysis — connected components or Louvain communities to detect whole mule rings — plus a ring-level scenario.
- **Phase 6**: add an unsupervised Isolation Forest anomaly score and compare what it catches vs. the rules and the supervised model.
- **Phase 7**: add a one-page plain-English summary per scenario for a non-technical audience.

## At the end, give me

1. A list of every metric with the file it comes from
2. A short plain-English explanation of ATL/BTL testing, segmentation, and the prioritization trade-off, so I can explain them in an interview
3. A list of known weaknesses an AML reviewer would point out
