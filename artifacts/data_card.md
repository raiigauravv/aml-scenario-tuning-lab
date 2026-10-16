# Data Card: AML Scenario Tuning Lab synthetic dataset

**All data on this page is synthetic.** No real customers, accounts, or transactions were used.

- Customers: 20,000
- Transactions: 2,518,896
- Simulation window: 2024-01-01 + 12 months
- Seed: 42
- Reporting threshold: $10,000

## Injected typologies (customer-level)

| Typology | Customers labeled |
|---|---|
| fan_in_fan_out | 45 |
| rapid_movement | 40 |
| structuring | 40 |
| wealth_layering | 30 |

- Total labeled customers: 155 (0.78% of all customers)
- Injected transactions: 870 (0.035% of all transactions)

## Legitimate look-alikes (deliberately NOT labeled)

- ~3% of small-business customers make some cash deposits clustered near the reporting threshold as part of normal revenue activity (no tight day-window burst, no label).
- ~10% of HNW customers with 2+ investment accounts make legitimate internal journal transfers between their own accounts (no label).
- ~12% of HNW customers make occasional legitimate outbound cross-border wires (property abroad, foreign investments, family support), some landing in the same month as a rebalancing transfer above - this is what makes WLT-01 need a real threshold, not just 'journal transfers + any cross-border wire' (no label).
- ~2% of retail/HNW customers periodically collect a shared bill from 2-4 peers (e-transfer/ACH) and pay a landlord/biller within a few days, and ~20% of small-business customers pay a supplier shortly after a batch of customer payments - legitimate same-bank pass-through activity that FAN-02 (ring detection) has to tell apart from an actual mule (no label).

## Known simplifications

- Single currency (USD); no FX modeling.
- Country codes are fictional (e.g. HOME/ALD/DRK); any resemblance to real countries is unintended.
- No branch-level modeling for structuring: all deposits are attributed to the account, not a branch.
- Each customer has a persistent roster of 8 recurring merchants and 3 recurring billers; card/ACH transactions draw from that roster (92% / 95% of the time respectively), with an occasional new counterparty, so distinct-counterparty-per-period features are meaningful rather than inflated by a fresh id on every transaction.
- Each customer also has a persistent roster of 3 peers (family/friends/roommates, with a bias toward small-business peers representing local purchases) for ordinary customer-to-customer e-transfers, so same-bank graph degree is nonzero for typical unlabeled customers too, not just the fan-in/fan-out typology.
- Account balances are not modeled; only transaction flows.
- Investment trades do not model actual security positions or market prices; buy/sell amounts are independent random draws, not linked to a shared price series.
- Each injected customer carries at most one typology, and typologies are injected independently of each other. Real laundering is messier and less separable than this.
