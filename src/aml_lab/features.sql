-- =====================================================================
-- Feature layer for the AML Scenario Tuning Lab (DuckDB).
--
-- Produces one row per (customer_id, month): a point-in-time snapshot
-- "as of" that month's last instant (month_end). Every window below is
-- a TRAILING window ending at month_end - no feature here ever reads a
-- transaction timestamped after the month it describes. That is what
-- keeps the later tuning/validation/test split (months 1-8 / 9-10 /
-- 11-12) honest: a feature for month 5 cannot have seen month 6.
--
-- Run by features.py, which must first register these tables:
--   customers, accounts, relationships, transactions, transaction_labels
--   months        (month, month_start, month_end)        -- calendar scaffold
--   params        (reporting_threshold)                   -- one row
--   country_risk  (code, risk_tier)                        -- country code lookup
--
-- Graph features (in/out degree, connected-component size, community
-- size) are NOT computed here - they need actual graph algorithms
-- (networkx), so features.py computes them separately from the same
-- trailing-30-day edge definition used below and merges them in.
-- =====================================================================


-- ---------------------------------------------------------------------
-- 1. Resolve each transaction's two sides to a customer_id (NULL if the
--    side is external: CASH, MARKET, or an EXT-<country>-<seq> id), and
--    parse the fictional country code out of any EXT- id.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW tx_resolved AS
SELECT
    t.transaction_id,
    t.timestamp,
    t.from_account,
    t.to_account,
    t.amount,
    t.channel,
    t.cross_border,
    fa.customer_id AS from_customer,
    ta.customer_id AS to_customer,
    CASE WHEN t.from_account LIKE 'EXT-%' THEN regexp_extract(t.from_account, 'EXT-([A-Z]+)-', 1) END AS from_country,
    CASE WHEN t.to_account LIKE 'EXT-%' THEN regexp_extract(t.to_account, 'EXT-([A-Z]+)-', 1) END AS to_country
FROM transactions t
LEFT JOIN accounts fa ON fa.account_id = t.from_account
LEFT JOIN accounts ta ON ta.account_id = t.to_account;


-- ---------------------------------------------------------------------
-- 2. Flatten every transaction into one or two signed "flow" rows: an
--    outflow row for the paying customer, an inflow row for the
--    receiving customer. A transfer between a customer's OWN accounts
--    (from_customer = to_customer) is not a flow in or out of them, so
--    it is excluded here. CASH/MARKET legs keep counterparty_customer
--    NULL (they are not a real counterparty).
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW tx_flows AS
SELECT
    from_customer AS customer_id, timestamp, amount, channel, cross_border,
    'out' AS direction,
    to_customer AS counterparty_customer, to_account AS counterparty_account, to_country AS counterparty_country
FROM tx_resolved
WHERE from_customer IS NOT NULL AND (to_customer IS NULL OR to_customer != from_customer)
UNION ALL
SELECT
    to_customer AS customer_id, timestamp, amount, channel, cross_border,
    'in' AS direction,
    from_customer AS counterparty_customer, from_account AS counterparty_account, from_country AS counterparty_country
FROM tx_resolved
WHERE to_customer IS NOT NULL AND (from_customer IS NULL OR from_customer != to_customer);


-- ---------------------------------------------------------------------
-- 3. Scaffold: every (customer, month) pair we will produce a feature
--    row for.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW scaffold AS
SELECT c.customer_id, c.segment_true, m.month, m.month_start, m.month_end
FROM customers c
CROSS JOIN months m;


-- ---------------------------------------------------------------------
-- 4. Cash deposits: rolling 7d/30d count and sum, and a near-threshold
--    count (deposits within 10% below the reporting threshold) over
--    the trailing 30 days. STR-01 (structuring) is built on exactly
--    this shape of evidence.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW cash_deposit_features AS
SELECT
    s.customer_id,
    s.month,
    -- COUNT(d.transaction_id), not COUNT(*): the LEFT JOIN below still
    -- produces one (NULL) row for a customer with no deposits that
    -- month, and COUNT(*) would count that phantom row as 1.
    COUNT(d.transaction_id) FILTER (d.timestamp > s.month_end - INTERVAL 7 DAY)  AS cash_deposit_count_7d,
    SUM(d.amount) FILTER (d.timestamp > s.month_end - INTERVAL 7 DAY)  AS cash_deposit_sum_7d,
    COUNT(d.transaction_id) AS cash_deposit_count_30d,
    SUM(d.amount) AS cash_deposit_sum_30d,
    COUNT(d.transaction_id) FILTER (
        d.amount >= 0.90 * (SELECT reporting_threshold FROM params)
        AND d.amount < (SELECT reporting_threshold FROM params)
    ) AS near_threshold_deposit_count_30d
FROM scaffold s
LEFT JOIN tx_resolved d
    ON d.to_customer = s.customer_id
   AND d.channel = 'cash_deposit'
   AND d.timestamp > s.month_end - INTERVAL 30 DAY
   AND d.timestamp <= s.month_end
GROUP BY s.customer_id, s.month;


-- ---------------------------------------------------------------------
-- 5. Rapid movement: for every qualifying inflow (amount >= $1,000, a
--    materiality floor so this isn't dominated by routine small
--    payments) in the trailing 30 days, how much of it moved back out
--    within 48 hours - capped at month_end, so an inflow on the last
--    day of the month cannot "see" an outflow that happens next month.
--    The worst (highest-ratio / most-emptied) qualifying inflow in the
--    window is the customer-month's signal. RMV-01 is built on exactly
--    this shape of evidence.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW rapid_movement_candidates AS
SELECT
    s.customer_id,
    s.month,
    i.timestamp AS inflow_time,
    i.amount AS inflow_amount,
    (
        SELECT COALESCE(SUM(o.amount), 0)
        FROM tx_flows o
        WHERE o.customer_id = i.customer_id
          AND o.direction = 'out'
          AND o.timestamp > i.timestamp
          AND o.timestamp <= LEAST(i.timestamp + INTERVAL 48 HOUR, s.month_end)
    ) AS outflow_followed
FROM scaffold s
JOIN tx_flows i
    ON i.customer_id = s.customer_id
   AND i.direction = 'in'
   AND i.amount >= 1000
   AND i.timestamp > s.month_end - INTERVAL 30 DAY
   AND i.timestamp <= s.month_end;

CREATE OR REPLACE VIEW rapid_movement_features AS
SELECT
    customer_id,
    month,
    -- Capped at 3.0: the outflow in the 48h window can include money that
    -- did not come from this specific inflow (e.g. an unrelated large
    -- payment happens to land in the same window), so the raw ratio can
    -- run arbitrarily high. A cap keeps the feature interpretable while
    -- still clearly separating "fully emptied" (~1.0) from "far above
    -- normal" (capped) cases.
    MAX(LEAST(outflow_followed / inflow_amount, 3.0)) AS rapid_movement_ratio_48h,
    MIN(inflow_amount - outflow_followed) AS residual_balance_48h_min
FROM rapid_movement_candidates
GROUP BY customer_id, month;


-- ---------------------------------------------------------------------
-- 6. Distinct counterparties in/out over the trailing 30 days (CASH and
--    MARKET are internal bookkeeping accounts, not real counterparties,
--    so they are excluded).
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW counterparty_features AS
SELECT
    s.customer_id,
    s.month,
    COUNT(DISTINCT f.counterparty_account) FILTER (f.direction = 'in')  AS distinct_counterparties_in_30d,
    COUNT(DISTINCT f.counterparty_account) FILTER (f.direction = 'out') AS distinct_counterparties_out_30d
FROM scaffold s
LEFT JOIN tx_flows f
    ON f.customer_id = s.customer_id
   AND f.counterparty_account NOT IN ('CASH', 'MARKET')
   AND f.timestamp > s.month_end - INTERVAL 30 DAY
   AND f.timestamp <= s.month_end
GROUP BY s.customer_id, s.month;


-- ---------------------------------------------------------------------
-- 7. Cross-border volume share and high-risk-country exposure over the
--    trailing 30 days.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW risk_exposure_features AS
SELECT
    s.customer_id,
    s.month,
    SUM(f.amount) AS total_flow_volume_30d,
    SUM(f.amount) FILTER (f.cross_border) / NULLIF(SUM(f.amount), 0) AS cross_border_share_30d,
    SUM(f.amount) FILTER (cr.risk_tier = 'high') / NULLIF(SUM(f.amount), 0) AS high_risk_country_exposure_30d
FROM scaffold s
LEFT JOIN tx_flows f
    ON f.customer_id = s.customer_id
   AND f.timestamp > s.month_end - INTERVAL 30 DAY
   AND f.timestamp <= s.month_end
LEFT JOIN country_risk cr ON cr.code = f.counterparty_country
GROUP BY s.customer_id, s.month;


-- ---------------------------------------------------------------------
-- 8. Activity vs. the customer's own baseline: total flow volume this
--    month compared to the mean/stddev of their own prior 3 months
--    (~90 days). The window frame only looks at PRECEDING rows, so
--    month 5's baseline uses months 2-4, never month 5 or later.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW monthly_volume AS
SELECT
    s.customer_id,
    s.month,
    COALESCE(SUM(f.amount), 0) AS monthly_volume
FROM scaffold s
LEFT JOIN tx_flows f
    ON f.customer_id = s.customer_id
   AND f.timestamp > s.month_start
   AND f.timestamp <= s.month_end
GROUP BY s.customer_id, s.month;

CREATE OR REPLACE VIEW activity_baseline_raw AS
SELECT
    customer_id,
    month,
    monthly_volume,
    AVG(monthly_volume) OVER (
        PARTITION BY customer_id ORDER BY month ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) AS baseline_mean_90d,
    STDDEV_SAMP(monthly_volume) OVER (
        PARTITION BY customer_id ORDER BY month ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) AS baseline_std_90d
FROM monthly_volume;

CREATE OR REPLACE VIEW activity_baseline_features AS
SELECT
    customer_id,
    month,
    monthly_volume,
    baseline_mean_90d,
    baseline_std_90d,
    -- A 2-3 point baseline can have a tiny sample stddev just by chance,
    -- which blows up the z-score for a perfectly normal change in
    -- activity. Floor the denominator at 10% of the baseline mean (or
    -- $50, whichever is larger) so the z-score reflects a genuinely
    -- unusual swing, not sampling noise from a short history. DuckDB's
    -- GREATEST skips NULL args, so month 2 (a single-point baseline,
    -- where sample stddev is undefined) still gets a z-score using the
    -- floor alone; month 1 (zero prior months) still has no baseline
    -- mean at all and stays NULL.
    (monthly_volume - baseline_mean_90d)
        / NULLIF(GREATEST(baseline_std_90d, 0.10 * baseline_mean_90d, 50), 0) AS activity_zscore_90d
FROM activity_baseline_raw;


-- ---------------------------------------------------------------------
-- 9. Journal-transfer volume between RELATED investment accounts over
--    the trailing 30 days (relationships.parquet defines "related").
--    WLT-01 (wealth layering) is built on exactly this shape of
--    evidence.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW journal_transfer_features AS
SELECT
    s.customer_id,
    s.month,
    COALESCE(SUM(t.amount), 0) AS journal_transfer_volume_30d_related
FROM scaffold s
LEFT JOIN tx_resolved t
    ON t.channel = 'internal_journal'
   AND t.from_customer = s.customer_id
   AND t.to_customer = s.customer_id
   AND t.timestamp > s.month_end - INTERVAL 30 DAY
   AND t.timestamp <= s.month_end
   AND EXISTS (
        SELECT 1 FROM relationships r
        WHERE r.relationship_type = 'same_owner'
          AND ((r.account_id_a = t.from_account AND r.account_id_b = t.to_account)
               OR (r.account_id_a = t.to_account AND r.account_id_b = t.from_account))
   )
GROUP BY s.customer_id, s.month;


-- ---------------------------------------------------------------------
-- 10. Final assembly.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW customer_month_features AS
SELECT
    s.customer_id,
    s.segment_true,
    s.month,
    CASE WHEN s.month <= 8 THEN 'tuning' WHEN s.month <= 10 THEN 'validation' ELSE 'test' END AS period,
    COALESCE(cd.cash_deposit_count_7d, 0)  AS cash_deposit_count_7d,
    COALESCE(cd.cash_deposit_sum_7d, 0)    AS cash_deposit_sum_7d,
    COALESCE(cd.cash_deposit_count_30d, 0) AS cash_deposit_count_30d,
    COALESCE(cd.cash_deposit_sum_30d, 0)   AS cash_deposit_sum_30d,
    COALESCE(cd.near_threshold_deposit_count_30d, 0) AS near_threshold_deposit_count_30d,
    rm.rapid_movement_ratio_48h,
    rm.residual_balance_48h_min,
    COALESCE(cp.distinct_counterparties_in_30d, 0)  AS distinct_counterparties_in_30d,
    COALESCE(cp.distinct_counterparties_out_30d, 0) AS distinct_counterparties_out_30d,
    COALESCE(re.cross_border_share_30d, 0)       AS cross_border_share_30d,
    COALESCE(re.high_risk_country_exposure_30d, 0) AS high_risk_country_exposure_30d,
    COALESCE(re.total_flow_volume_30d, 0)        AS total_flow_volume_30d,
    ab.activity_zscore_90d,
    COALESCE(jt.journal_transfer_volume_30d_related, 0) AS journal_transfer_volume_30d_related
FROM scaffold s
LEFT JOIN cash_deposit_features      cd ON cd.customer_id = s.customer_id AND cd.month = s.month
LEFT JOIN rapid_movement_features    rm ON rm.customer_id = s.customer_id AND rm.month = s.month
LEFT JOIN counterparty_features      cp ON cp.customer_id = s.customer_id AND cp.month = s.month
LEFT JOIN risk_exposure_features     re ON re.customer_id = s.customer_id AND re.month = s.month
LEFT JOIN activity_baseline_features ab ON ab.customer_id = s.customer_id AND ab.month = s.month
LEFT JOIN journal_transfer_features  jt ON jt.customer_id = s.customer_id AND jt.month = s.month;
