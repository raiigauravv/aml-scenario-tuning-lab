# Data Card: AML Scenario Tuning Lab synthetic dataset

**All data on this page is synthetic.** No real customers, accounts, or transactions were used.

- Customers: 20,000
- Transactions: 2,257,220
- Simulation window: 2024-01-01 + 12 months
- Seed: 42
- Reporting threshold: $10,000

## Injected typologies (customer-level)

| Typology | Customers labeled |
|---|---|
| fan_in_fan_out | 56 |
| rapid_movement | 40 |
| structuring | 40 |
| wealth_layering | 30 |

- Total labeled customers: 166 (0.83% of all customers)
- Injected transactions: 899 (0.040% of all transactions)

## Legitimate look-alikes (deliberately NOT labeled)

- ~3% of small-business customers make some cash deposits clustered near the reporting threshold as part of normal revenue activity (no tight day-window burst, no label).
- ~10% of HNW customers with 2+ investment accounts make legitimate internal journal transfers between their own accounts (no outbound cross-border wire, no label).

## Known simplifications

- Single currency (USD); no FX modeling.
- Country codes are fictional (e.g. HOME/ALD/DRK); any resemblance to real countries is unintended.
- No branch-level modeling for structuring: all deposits are attributed to the account, not a branch.
- Each customer has a persistent roster of 8 recurring merchants and 3 recurring billers; card/ACH transactions draw from that roster (92% / 95% of the time respectively), with an occasional new counterparty, so distinct-counterparty-per-period features are meaningful rather than inflated by a fresh id on every transaction.
- Account balances are not modeled; only transaction flows.
- Investment trades do not model actual security positions or market prices; buy/sell amounts are independent random draws, not linked to a shared price series.
- Each injected customer carries at most one typology, and typologies are injected independently of each other. Real laundering is messier and less separable than this.
