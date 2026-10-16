"""Synthetic data generator for the AML Scenario Tuning Lab.

Produces customers, accounts, relationships, transactions, and both
customer- and transaction-level typology labels, fully deterministic given a
seed. Running this module also writes `artifacts/data_card.md` describing
the generator's rates and known simplifications.

All thresholds and injection rates come from `config/generator.yaml`, not
from constants in this file.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "generator.yaml"

SEGMENTS = ["retail", "small_business", "hnw_investment"]

ACTIVITY_PARAMS = {
    "retail": (2500.0, 0.5),
    "small_business": (15000.0, 0.6),
    "hnw_investment": (40000.0, 0.7),
}

# Band logic in assign_stratified_months assumes a 12-month simulation
# (tuning = months 1-8, validation = 9-10, test = 11-12), matching the time
# split used later in tuning.py. Changing `months` in config requires
# revisiting this.
PERIOD_BANDS = {
    "tuning": (1, 8),
    "validation": (9, 10),
    "test": (11, 12),
}


# ---------------------------------------------------------------------------
# Config and time helpers
# ---------------------------------------------------------------------------

def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sim_window(config: dict) -> tuple[dt.date, dt.date, int]:
    start = dt.date.fromisoformat(config["sim_start"])
    months = config["months"]
    end = add_months(start, months) - dt.timedelta(days=1)
    return start, end, months


def add_months(d: dt.date, months: int) -> dt.date:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    return dt.date(year, month, d.day)


def month_bounds(start: dt.date, month_number: int) -> tuple[dt.date, dt.date]:
    """1-indexed month number -> (first day, last day) within the sim window."""
    month_start = add_months(start, month_number - 1)
    month_end = add_months(start, month_number) - dt.timedelta(days=1)
    return month_start, month_end


def random_datetimes_for_months(
    rng: np.random.Generator, start: dt.date, months_arr: np.ndarray, total_months: int
) -> np.ndarray:
    """Vectorized random datetime within each given 1-indexed month number."""
    month_starts = {m: month_bounds(start, m)[0] for m in range(1, total_months + 1)}
    month_ends = {m: month_bounds(start, m)[1] for m in range(1, total_months + 1)}
    starts = pd.to_datetime(pd.Series(months_arr).map(month_starts))
    ends = pd.to_datetime(pd.Series(months_arr).map(month_ends))
    day_span = (ends - starts).dt.days.values
    day_offset = rng.integers(0, day_span + 1)
    seconds_offset = rng.integers(0, 24 * 3600, size=len(months_arr))
    return (
        starts.values
        + pd.to_timedelta(day_offset, unit="D").values
        + pd.to_timedelta(seconds_offset, unit="s").values
    )


def assign_stratified_months(rng: np.random.Generator, n_instances: int, months: int = 12) -> np.ndarray:
    """Spread n_instances across 1..months so the tuning/validation/test
    period bands each get at least one instance whenever n_instances allows
    it, instead of letting all instances land early by chance.
    """
    if n_instances == 0:
        return np.array([], dtype=int)
    bands = list(PERIOD_BANDS.values())
    assigned = []
    remaining = n_instances
    for lo, hi in bands:
        if remaining <= 0:
            break
        assigned.append(int(rng.integers(lo, hi + 1)))
        remaining -= 1
    if remaining > 0:
        weights = np.array([hi - lo + 1 for lo, hi in bands], dtype=float)
        weights /= weights.sum()
        band_choices = rng.choice(len(bands), size=remaining, p=weights)
        for b in band_choices:
            lo, hi = bands[b]
            assigned.append(int(rng.integers(lo, hi + 1)))
    assigned = np.array(assigned[:n_instances], dtype=int)
    rng.shuffle(assigned)
    return assigned


# ---------------------------------------------------------------------------
# Customers, accounts, relationships
# ---------------------------------------------------------------------------

def generate_customers(rng: np.random.Generator, n_customers: int, start: dt.date, config: dict) -> pd.DataFrame:
    seg_probs = config["segments"]
    seg_names = list(seg_probs.keys())
    segments = rng.choice(seg_names, size=n_customers, p=list(seg_probs.values()))

    risk_probs = config["customer_country_risk"]
    risk_tiers = rng.choice(list(risk_probs.keys()), size=n_customers, p=list(risk_probs.values()))

    earliest_open = start - dt.timedelta(days=5 * 365)
    open_offsets = rng.integers(0, (start - earliest_open).days, size=n_customers)
    open_dates = np.array([earliest_open + dt.timedelta(days=int(o)) for o in open_offsets])

    activity = np.empty(n_customers)
    for seg in SEGMENTS:
        mask = segments == seg
        count = int(mask.sum())
        if count == 0:
            continue
        mu, sigma = ACTIVITY_PARAMS[seg]
        activity[mask] = rng.lognormal(mean=np.log(mu), sigma=sigma, size=count)

    customer_ids = np.array([f"CUST{i:06d}" for i in range(n_customers)])

    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "segment_true": segments,
            "country_risk_tier": risk_tiers,
            "account_open_date": open_dates,
            "expected_monthly_activity": np.round(activity, 2),
        }
    )


def generate_accounts(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seq = 0
    for cust_id, segment, open_date in zip(
        customers["customer_id"], customers["segment_true"], customers["account_open_date"]
    ):
        seq += 1
        rows.append((f"ACCT{seq:07d}", cust_id, "chequing", open_date))
        if segment == "retail" and rng.random() < 0.90:  # noqa: SIM114
            seq += 1
            rows.append((f"ACCT{seq:07d}", cust_id, "savings", open_date))
        elif segment == "small_business" and rng.random() < 0.50:
            seq += 1
            rows.append((f"ACCT{seq:07d}", cust_id, "savings", open_date))
        elif segment == "hnw_investment":
            seq += 1
            rows.append((f"ACCT{seq:07d}", cust_id, "savings", open_date))
            n_invest = int(rng.integers(1, 4))
            for _ in range(n_invest):
                seq += 1
                rows.append((f"ACCT{seq:07d}", cust_id, "investment", open_date))
    return pd.DataFrame(rows, columns=["account_id", "customer_id", "account_type", "open_date"])


def account_maps(accounts: pd.DataFrame) -> tuple[dict, dict]:
    chequing = accounts[accounts["account_type"] == "chequing"].drop_duplicates("customer_id", keep="first")
    chequing_map = dict(zip(chequing["customer_id"], chequing["account_id"]))
    invest = accounts[accounts["account_type"] == "investment"]
    invest_map = invest.groupby("customer_id")["account_id"].apply(list).to_dict()
    return chequing_map, invest_map


def generate_relationships(rng: np.random.Generator, customers: pd.DataFrame, accounts: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []

    invest_accounts = accounts[accounts["account_type"] == "investment"]
    for cust_id, group in invest_accounts.groupby("customer_id"):
        acct_ids = group["account_id"].tolist()
        for i in range(len(acct_ids)):
            for j in range(i + 1, len(acct_ids)):
                rows.append((acct_ids[i], acct_ids[j], cust_id, cust_id, "same_owner"))

    household_rate = config.get("household_rate", 0.04)
    eligible = customers["customer_id"].tolist()
    rng.shuffle(eligible)
    n_pairs = int(len(eligible) * household_rate / 2)
    acct_by_cust = accounts.groupby("customer_id")["account_id"].apply(list)
    for k in range(n_pairs):
        cust_a, cust_b = eligible[2 * k], eligible[2 * k + 1]
        for acct_a in acct_by_cust.get(cust_a, []):
            for acct_b in acct_by_cust.get(cust_b, []):
                rows.append((acct_a, acct_b, cust_a, cust_b, "household"))

    return pd.DataFrame(
        rows, columns=["account_id_a", "account_id_b", "customer_id_a", "customer_id_b", "relationship_type"]
    )


# ---------------------------------------------------------------------------
# External counterparties
# ---------------------------------------------------------------------------

CASH = "CASH"
MARKET = "MARKET"

# Recurring-counterparty roster sizes. Real customers overwhelmingly repeat
# transactions with the same small set of merchants/billers each month; a
# fresh random counterparty id per transaction would make any
# distinct-counterparty feature meaningless (every transaction would look
# "new"). A small persistent roster per customer, with an occasional new
# counterparty, keeps that feature realistic.
MERCHANT_ROSTER_SIZE = 8
ACH_ROSTER_SIZE = 3
CARD_NEW_COUNTERPARTY_PROB = 0.08
ACH_NEW_COUNTERPARTY_PROB = 0.05

# Peer-to-peer roster: family/friends e-transfers, roommates splitting
# bills, customers paying a local small business. Without this, same-bank
# customer-to-customer degree is only ever nonzero for the fan-in/fan-out
# typology, which makes that signal unrealistically clean.
PEER_ROSTER_SIZE = 3
PEER_TRANSFER_LAM = 1.0
PEER_SMALL_BUSINESS_BIAS = 0.3


def external_accounts(
    rng: np.random.Generator, config: dict, n: int, domestic_share_override: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized synthetic external counterparty ids, encoding a fictional
    country code in the id (EXT-<code>-<seq>) so later feature/SQL work can
    parse cross-border and country-risk exposure straight from the id.
    """
    if n == 0:
        return np.array([], dtype=object), np.array([], dtype=bool)
    cr = config["country_risk"]
    domestic_code = cr["domestic_code"]
    foreign_codes = [c["code"] for c in cr["foreign_countries"]]
    domestic_share = domestic_share_override if domestic_share_override is not None else cr["domestic_share"]
    is_domestic = rng.random(n) < domestic_share
    codes = np.where(is_domestic, domestic_code, rng.choice(foreign_codes, size=n))
    seqs = rng.integers(0, 999_999, size=n)
    ids = np.array([f"EXT-{c}-{s:06d}" for c, s in zip(codes, seqs)], dtype=object)
    cross_border = codes != domestic_code
    return ids, cross_border


def foreign_accounts(rng: np.random.Generator, config: dict, n: int, high_risk_bias: float = 0.5) -> np.ndarray:
    """Synthetic external ids guaranteed foreign, biased toward the
    fictional high-risk country tier (used for typology outbound wires)."""
    if n == 0:
        return np.array([], dtype=object)
    cr = config["country_risk"]
    foreign = cr["foreign_countries"]
    high_risk_codes = [c["code"] for c in foreign if c["risk_tier"] == "high"]
    other_codes = [c["code"] for c in foreign if c["risk_tier"] != "high"]
    use_high = rng.random(n) < high_risk_bias
    codes = np.where(use_high, rng.choice(high_risk_codes, size=n), rng.choice(other_codes, size=n))
    seqs = rng.integers(0, 999_999, size=n)
    return np.array([f"EXT-{c}-{s:06d}" for c, s in zip(codes, seqs)], dtype=object)


def _empty_tx_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "customer_id",
            "timestamp",
            "from_account",
            "to_account",
            "amount",
            "currency",
            "channel",
            "cross_border",
            "instrument_type",
        ]
    )


def _expand(df: pd.DataFrame, counts: np.ndarray) -> pd.DataFrame:
    counts = np.asarray(counts)
    idx = np.repeat(np.arange(len(df)), counts)
    return df.iloc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Normal (non-injected) transaction generation
# ---------------------------------------------------------------------------

def _gen_income(rng, cm, counts, start, months) -> pd.DataFrame:
    rows = _expand(cm, counts)
    if len(rows) == 0:
        return _empty_tx_frame()
    timestamps = random_datetimes_for_months(rng, start, rows["month"].values, months)
    amount = rng.lognormal(mean=np.log(np.maximum(rows["activity"].values, 100.0)), sigma=0.12)
    channel = np.where(rows["segment_true"].values == "small_business", "e_transfer", "ach")
    return pd.DataFrame(
        {
            "customer_id": rows["customer_id"].values,
            "timestamp": timestamps,
            "from_account": rows["employer_account"].values,
            "to_account": rows["chequing_account"].values,
            "amount": np.round(amount, 2),
            "currency": "USD",
            "channel": channel,
            "cross_border": False,
            "instrument_type": None,
        }
    )


def _gen_cash_deposits(rng, cm, config, start, months) -> pd.DataFrame:
    threshold = config["reporting_threshold"]
    segs = cm["segment_true"].values
    lam = np.where(segs == "small_business", 8.0, 0.3)
    counts = rng.poisson(lam=lam)
    rows = _expand(cm, counts)
    if len(rows) == 0:
        return _empty_tx_frame()
    timestamps = random_datetimes_for_months(rng, start, rows["month"].values, months)

    normal_amount = np.clip(rng.lognormal(mean=np.log(1500.0), sigma=0.9, size=len(rows)), 100.0, threshold * 0.99)
    lookalike_mask = rows["is_structuring_lookalike"].values & (rng.random(len(rows)) < 0.5)
    pct = rng.uniform(0.85, 0.98, size=len(rows))
    amount = np.where(lookalike_mask, threshold * pct, normal_amount)

    return pd.DataFrame(
        {
            "customer_id": rows["customer_id"].values,
            "timestamp": timestamps,
            "from_account": CASH,
            "to_account": rows["chequing_account"].values,
            "amount": np.round(amount, 2),
            "currency": "USD",
            "channel": "cash_deposit",
            "cross_border": False,
            "instrument_type": None,
        }
    )


def _roster_lookup(rng, cust_idx: np.ndarray, roster: np.ndarray, roster_cb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pick one persistent roster entry per row for the given customer index."""
    slot = rng.integers(0, roster.shape[1], size=len(cust_idx))
    return roster[cust_idx, slot], roster_cb[cust_idx, slot]


def _gen_card_spend(rng, cm, config, start, months, merchant_roster, merchant_roster_cb) -> pd.DataFrame:
    counts = rng.poisson(lam=4.0, size=len(cm))
    rows = _expand(cm, counts)
    if len(rows) == 0:
        return _empty_tx_frame()
    timestamps = random_datetimes_for_months(rng, start, rows["month"].values, months)
    amount = rng.lognormal(mean=np.log(45.0), sigma=0.6, size=len(rows))

    cust_idx = rows["cust_idx"].values
    roster_ids, roster_cb = _roster_lookup(rng, cust_idx, merchant_roster, merchant_roster_cb)
    new_ids, new_cb = external_accounts(rng, config, len(rows), domestic_share_override=0.95)
    use_new = rng.random(len(rows)) < CARD_NEW_COUNTERPARTY_PROB
    merchant_ids = np.where(use_new, new_ids, roster_ids)
    cross_border = np.where(use_new, new_cb, roster_cb)

    return pd.DataFrame(
        {
            "customer_id": rows["customer_id"].values,
            "timestamp": timestamps,
            "from_account": rows["chequing_account"].values,
            "to_account": merchant_ids,
            "amount": np.round(amount, 2),
            "currency": "USD",
            "channel": "card",
            "cross_border": cross_border,
            "instrument_type": None,
        }
    )


def _gen_ach_bills(rng, cm, config, start, months, ach_roster, ach_roster_cb) -> pd.DataFrame:
    counts = rng.poisson(lam=2.0, size=len(cm))
    rows = _expand(cm, counts)
    if len(rows) == 0:
        return _empty_tx_frame()
    timestamps = random_datetimes_for_months(rng, start, rows["month"].values, months)
    amount = rng.lognormal(mean=np.log(150.0), sigma=0.5, size=len(rows))

    cust_idx = rows["cust_idx"].values
    roster_ids, roster_cb = _roster_lookup(rng, cust_idx, ach_roster, ach_roster_cb)
    new_ids, new_cb = external_accounts(rng, config, len(rows), domestic_share_override=0.98)
    use_new = rng.random(len(rows)) < ACH_NEW_COUNTERPARTY_PROB
    payee_ids = np.where(use_new, new_ids, roster_ids)
    cross_border = np.where(use_new, new_cb, roster_cb)

    return pd.DataFrame(
        {
            "customer_id": rows["customer_id"].values,
            "timestamp": timestamps,
            "from_account": rows["chequing_account"].values,
            "to_account": payee_ids,
            "amount": np.round(amount, 2),
            "currency": "USD",
            "channel": "ach",
            "cross_border": cross_border,
            "instrument_type": None,
        }
    )


def _gen_big_oneoff(rng, cm, config, start, months) -> pd.DataFrame:
    counts = (rng.random(len(cm)) < 0.03).astype(int)
    rows = _expand(cm, counts)
    if len(rows) == 0:
        return _empty_tx_frame()
    timestamps = random_datetimes_for_months(rng, start, rows["month"].values, months)
    amount = rows["activity"].values * rng.uniform(3.0, 10.0, size=len(rows))
    direction_in = rng.random(len(rows)) < 0.5
    ext_ids, cross_border = external_accounts(rng, config, len(rows))
    channel = rng.choice(["wire", "e_transfer"], size=len(rows))
    from_account = np.where(direction_in, ext_ids, rows["chequing_account"].values)
    to_account = np.where(direction_in, rows["chequing_account"].values, ext_ids)
    return pd.DataFrame(
        {
            "customer_id": rows["customer_id"].values,
            "timestamp": timestamps,
            "from_account": from_account,
            "to_account": to_account,
            "amount": np.round(amount, 2),
            "currency": "USD",
            "channel": channel,
            "cross_border": cross_border,
            "instrument_type": None,
        }
    )


def _sample_excluding_self(rng: np.random.Generator, own_idx: np.ndarray, candidate_idx: np.ndarray) -> np.ndarray:
    """For each i, pick a random element of candidate_idx that isn't own_idx[i]."""
    m = len(candidate_idx)
    pick = rng.integers(0, m, size=len(own_idx))
    chosen = candidate_idx[pick]
    collision = chosen == own_idx
    if collision.any():
        alt_pick = (pick[collision] + 1) % m
        chosen[collision] = candidate_idx[alt_pick]
    return chosen


def _gen_peer_transfers(rng, cm, peer_roster, start, months) -> pd.DataFrame:
    """Legitimate same-bank customer-to-customer transfers: family/friends
    e-transfers, roommates splitting bills, a customer paying a local
    small business. Without this, same-bank graph degree is only ever
    nonzero for the fan-in/fan-out typology, which makes that signal
    unrealistically clean.
    """
    counts = rng.poisson(lam=PEER_TRANSFER_LAM, size=len(cm))
    rows = _expand(cm, counts)
    if len(rows) == 0:
        return _empty_tx_frame()
    timestamps = random_datetimes_for_months(rng, start, rows["month"].values, months)
    amount = rng.lognormal(mean=np.log(75.0), sigma=0.7, size=len(rows))

    cust_idx = rows["cust_idx"].values
    slot = rng.integers(0, peer_roster.shape[1], size=len(rows))
    peer_account = peer_roster[cust_idx, slot]

    return pd.DataFrame(
        {
            "customer_id": rows["customer_id"].values,
            "timestamp": timestamps,
            "from_account": rows["chequing_account"].values,
            "to_account": peer_account,
            "amount": np.round(amount, 2),
            "currency": "USD",
            "channel": "e_transfer",
            "cross_border": False,
            "instrument_type": None,
        }
    )


def _gen_hnw_rebalancing(rng, cm, invest_map, config, start, months) -> pd.DataFrame:
    """Legitimate look-alike: HNW customers moving funds between their own
    investment accounts with no following outbound cross-border wire. Not
    labeled anywhere - this is deliberately indistinguishable noise.
    """
    mask = cm["is_wealth_lookalike"].values
    counts = np.where(mask, rng.poisson(lam=1.0, size=len(cm)), 0)
    rows = _expand(cm, counts)
    if len(rows) == 0:
        return _empty_tx_frame()
    timestamps = random_datetimes_for_months(rng, start, rows["month"].values, months)
    from_acc, to_acc = [], []
    for cust in rows["customer_id"].values:
        a, b = rng.choice(np.array(invest_map[cust]), size=2, replace=False)
        from_acc.append(a)
        to_acc.append(b)
    amount = rng.uniform(5000.0, 80000.0, size=len(rows))
    return pd.DataFrame(
        {
            "customer_id": rows["customer_id"].values,
            "timestamp": timestamps,
            "from_account": from_acc,
            "to_account": to_acc,
            "amount": np.round(amount, 2),
            "currency": "USD",
            "channel": "internal_journal",
            "cross_border": False,
            "instrument_type": None,
        }
    )


def _gen_investment_trades(rng, accounts, config, start, months) -> pd.DataFrame:
    """Normal buy/sell activity for every investment account, so the
    wealth-layering typology's trade activity isn't trivially separable
    from ordinary portfolio activity.
    """
    invest_accounts = accounts[accounts["account_type"] == "investment"][["account_id", "customer_id"]]
    if invest_accounts.empty:
        return _empty_tx_frame()
    n_accts = len(invest_accounts)
    am = pd.DataFrame(
        {
            "account_id": np.repeat(invest_accounts["account_id"].values, months),
            "customer_id": np.repeat(invest_accounts["customer_id"].values, months),
            "month": np.tile(np.arange(1, months + 1), n_accts),
        }
    )
    lam = config["investment_activity"]["trades_per_month_mean"]
    counts = rng.poisson(lam=lam, size=len(am))
    rows = _expand(am, counts)
    if len(rows) == 0:
        return _empty_tx_frame()

    buy_time = random_datetimes_for_months(rng, start, rows["month"].values, months)
    sim_end = pd.Timestamp(add_months(start, months) - dt.timedelta(days=1)) + pd.Timedelta(hours=23, minutes=59)
    hold_days = rng.integers(1, 30, size=len(rows))
    sell_time = np.minimum(pd.to_datetime(buy_time) + pd.to_timedelta(hold_days, unit="D"), sim_end)

    amount = rng.lognormal(mean=np.log(8000.0), sigma=0.8, size=len(rows))
    sell_amount = amount * rng.uniform(0.9, 1.1, size=len(rows))
    instrument = rng.choice(["equity", "fixed_income"], size=len(rows), p=[0.6, 0.4])

    buys = pd.DataFrame(
        {
            "customer_id": rows["customer_id"].values,
            "timestamp": buy_time,
            "from_account": rows["account_id"].values,
            "to_account": MARKET,
            "amount": np.round(amount, 2),
            "currency": "USD",
            "channel": "security_buy",
            "cross_border": False,
            "instrument_type": instrument,
        }
    )
    sells = pd.DataFrame(
        {
            "customer_id": rows["customer_id"].values,
            "timestamp": sell_time,
            "from_account": MARKET,
            "to_account": rows["account_id"].values,
            "amount": np.round(sell_amount, 2),
            "currency": "USD",
            "channel": "security_sell",
            "cross_border": False,
            "instrument_type": instrument,
        }
    )
    return pd.concat([buys, sells], ignore_index=True)


def generate_normal_transactions(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    config: dict,
    start: dt.date,
    months: int,
) -> pd.DataFrame:
    chequing_map, invest_map = account_maps(accounts)

    n = len(customers)
    cust_ids = customers["customer_id"].values
    segments = customers["segment_true"].values
    activity = customers["expected_monthly_activity"].values
    employer_ids = np.array([f"EXT-HOME-EMP{idx:06d}" for idx in range(n)])

    ll_cfg = config["legit_lookalikes"]
    has_2plus_invest = np.array([len(invest_map.get(c, [])) >= 2 for c in cust_ids])
    is_structuring_lookalike = (segments == "small_business") & (rng.random(n) < ll_cfg["structuring_lookalike_rate"])
    is_wealth_lookalike = (
        (segments == "hnw_investment") & (rng.random(n) < ll_cfg["wealth_lookalike_rate"]) & has_2plus_invest
    )

    cm = pd.DataFrame(
        {
            "customer_id": np.repeat(cust_ids, months),
            "cust_idx": np.repeat(np.arange(n), months),
            "segment_true": np.repeat(segments, months),
            "activity": np.repeat(activity, months),
            "employer_account": np.repeat(employer_ids, months),
            "is_structuring_lookalike": np.repeat(is_structuring_lookalike, months),
            "is_wealth_lookalike": np.repeat(is_wealth_lookalike, months),
            "month": np.tile(np.arange(1, months + 1), n),
        }
    )
    cm["chequing_account"] = cm["customer_id"].map(chequing_map)

    # Persistent per-customer rosters of recurring counterparties (merchants,
    # billers) so distinct-counterparty features reflect normal repeat
    # behaviour instead of a fresh id on every transaction.
    merchant_ids_flat, merchant_cb_flat = external_accounts(
        rng, config, n * MERCHANT_ROSTER_SIZE, domestic_share_override=0.95
    )
    merchant_roster = merchant_ids_flat.reshape(n, MERCHANT_ROSTER_SIZE)
    merchant_roster_cb = merchant_cb_flat.reshape(n, MERCHANT_ROSTER_SIZE)

    ach_ids_flat, ach_cb_flat = external_accounts(rng, config, n * ACH_ROSTER_SIZE, domestic_share_override=0.98)
    ach_roster = ach_ids_flat.reshape(n, ACH_ROSTER_SIZE)
    ach_roster_cb = ach_cb_flat.reshape(n, ACH_ROSTER_SIZE)

    # Persistent peer roster (family/friends/roommates/local small
    # business) for legitimate same-bank customer-to-customer transfers.
    chequing_by_idx = np.array([chequing_map[c] for c in cust_ids])
    own_idx = np.arange(n)
    small_business_idx_pool = own_idx[segments == "small_business"]
    peer_roster_idx = np.empty((n, PEER_ROSTER_SIZE), dtype=int)
    for slot in range(PEER_ROSTER_SIZE):
        use_business = rng.random(n) < PEER_SMALL_BUSINESS_BIAS
        any_pick = _sample_excluding_self(rng, own_idx, own_idx)
        biz_pick = _sample_excluding_self(rng, own_idx, small_business_idx_pool)
        peer_roster_idx[:, slot] = np.where(use_business, biz_pick, any_pick)
    peer_roster = chequing_by_idx[peer_roster_idx]

    is_retail_hnw = cm["segment_true"].isin(["retail", "hnw_investment"]).values
    income_counts = np.where(is_retail_hnw, 1, (rng.random(len(cm)) < 0.6).astype(int))

    fragments = [
        _gen_income(rng, cm, income_counts, start, months),
        _gen_cash_deposits(rng, cm, config, start, months),
        _gen_card_spend(rng, cm, config, start, months, merchant_roster, merchant_roster_cb),
        _gen_ach_bills(rng, cm, config, start, months, ach_roster, ach_roster_cb),
        _gen_big_oneoff(rng, cm, config, start, months),
        _gen_peer_transfers(rng, cm, peer_roster, start, months),
        _gen_hnw_rebalancing(rng, cm, invest_map, config, start, months),
        _gen_investment_trades(rng, accounts, config, start, months),
    ]
    tx = pd.concat(fragments, ignore_index=True)
    tx["_typology"] = None
    tx["_is_injected"] = False
    return tx


# ---------------------------------------------------------------------------
# Typology injection
# ---------------------------------------------------------------------------

def _build_structuring_instance(rng, cust, month, chequing_map, config, start):
    cfg = config["typologies"]["structuring"]
    threshold = config["reporting_threshold"]
    n_dep = int(rng.integers(cfg["deposits_min"], cfg["deposits_max"] + 1))
    window_days = int(rng.integers(cfg["window_days_min"], cfg["window_days_max"] + 1))
    month_start, month_end = month_bounds(start, month)
    span = max((month_end - month_start).days - window_days, 0)
    burst_start = month_start + dt.timedelta(days=int(rng.integers(0, span + 1)) if span > 0 else 0)

    day_offsets = rng.integers(0, window_days + 1, size=n_dep)
    pct = rng.uniform(cfg["pct_of_threshold_min"], cfg["pct_of_threshold_max"], size=n_dep)
    amounts = threshold * pct
    acct = chequing_map[cust]
    timestamps = [
        dt.datetime.combine(burst_start + dt.timedelta(days=int(d)), dt.time())
        + dt.timedelta(seconds=int(rng.integers(0, 86400)))
        for d in day_offsets
    ]

    frag = pd.DataFrame(
        {
            "customer_id": cust,
            "timestamp": timestamps,
            "from_account": CASH,
            "to_account": acct,
            "amount": np.round(amounts, 2),
            "currency": "USD",
            "channel": "cash_deposit",
            "cross_border": False,
            "instrument_type": None,
            "_typology": "structuring",
            "_is_injected": True,
        }
    )
    label = {
        "customer_id": cust,
        "typology": "structuring",
        "role": "primary",
        "start_date": min(t.date() for t in timestamps),
        "end_date": max(t.date() for t in timestamps),
    }
    return frag, label


def _build_rapid_movement_instance(rng, cust, month, chequing_map, config, start):
    cfg = config["typologies"]["rapid_movement"]
    acct = chequing_map[cust]
    month_start, month_end = month_bounds(start, month)
    day_span = (month_end - month_start).days
    inflow_time = dt.datetime.combine(
        month_start + dt.timedelta(days=int(rng.integers(0, day_span + 1))), dt.time()
    ) + dt.timedelta(seconds=int(rng.integers(0, 86400)))
    inflow_amount = float(rng.uniform(cfg["inflow_min"], cfg["inflow_max"]))
    ext_in, _ = external_accounts(rng, config, 1)

    window_hours = float(rng.uniform(cfg["window_hours_min"], cfg["window_hours_max"]))
    n_out = int(rng.integers(1, 4))
    out_pct_total = float(rng.uniform(cfg["outflow_pct_min"], cfg["outflow_pct_max"]))
    splits = rng.dirichlet(np.ones(n_out)) * out_pct_total * inflow_amount
    out_offsets_hours = np.sort(rng.uniform(1, window_hours, size=n_out))
    out_times = [inflow_time + dt.timedelta(hours=float(h)) for h in out_offsets_hours]
    out_ext_ids, out_cross_border = external_accounts(rng, config, n_out)

    rows = [
        {
            "customer_id": cust,
            "timestamp": inflow_time,
            "from_account": ext_in[0],
            "to_account": acct,
            "amount": round(inflow_amount, 2),
            "currency": "USD",
            "channel": str(rng.choice(["wire", "e_transfer"])),
            "cross_border": False,
            "instrument_type": None,
            "_typology": "rapid_movement",
            "_is_injected": True,
        }
    ]
    for amt, t, ext, cb in zip(splits, out_times, out_ext_ids, out_cross_border):
        rows.append(
            {
                "customer_id": cust,
                "timestamp": t,
                "from_account": acct,
                "to_account": ext,
                "amount": round(float(amt), 2),
                "currency": "USD",
                "channel": str(rng.choice(["wire", "e_transfer"])),
                "cross_border": bool(cb),
                "instrument_type": None,
                "_typology": "rapid_movement",
                "_is_injected": True,
            }
        )
    frag = pd.DataFrame(rows)
    label = {
        "customer_id": cust,
        "typology": "rapid_movement",
        "role": "primary",
        "start_date": inflow_time.date(),
        "end_date": max(out_times).date(),
    }
    return frag, label


def _build_fan_instance(rng, senders, hub, downstream, month, chequing_map, config, start):
    cfg = config["typologies"]["fan_in_fan_out"]
    month_start, month_end = month_bounds(start, month)
    window_days = int(rng.integers(cfg["window_days_min"], cfg["window_days_max"] + 1))
    span = max((month_end - month_start).days - window_days, 0)
    burst_start = month_start + dt.timedelta(days=int(rng.integers(0, span + 1)) if span > 0 else 0)

    hub_acct = chequing_map[hub]
    rows = []
    inflow_total = 0.0
    last_in_time = None
    for s in senders:
        amt = float(rng.uniform(1000, 15000))
        day = int(rng.integers(0, window_days + 1))
        t = dt.datetime.combine(burst_start + dt.timedelta(days=day), dt.time()) + dt.timedelta(
            seconds=int(rng.integers(0, 86400))
        )
        rows.append(
            {
                "customer_id": s,
                "timestamp": t,
                "from_account": chequing_map[s],
                "to_account": hub_acct,
                "amount": round(amt, 2),
                "currency": "USD",
                "channel": str(rng.choice(["e_transfer", "ach"])),
                "cross_border": False,
                "instrument_type": None,
                "_typology": "fan_in_fan_out",
                "_is_injected": True,
            }
        )
        inflow_total += amt
        last_in_time = t if last_in_time is None else max(last_in_time, t)

    forwarded_share = float(rng.uniform(cfg["forwarded_share_min"], cfg["forwarded_share_max"]))
    forward_total = inflow_total * forwarded_share
    splits = rng.dirichlet(np.ones(len(downstream))) * forward_total
    last_out_time = last_in_time
    for d, amt in zip(downstream, splits):
        delay_hours = float(rng.uniform(1, window_days * 24))
        t = last_in_time + dt.timedelta(hours=delay_hours)
        rows.append(
            {
                "customer_id": hub,
                "timestamp": t,
                "from_account": hub_acct,
                "to_account": chequing_map[d],
                "amount": round(float(amt), 2),
                "currency": "USD",
                "channel": str(rng.choice(["e_transfer", "ach"])),
                "cross_border": False,
                "instrument_type": None,
                "_typology": "fan_in_fan_out",
                "_is_injected": True,
            }
        )
        last_out_time = max(last_out_time, t)

    frag = pd.DataFrame(rows)
    labels = [
        {
            "customer_id": s,
            "typology": "fan_in_fan_out",
            "role": "sender",
            "start_date": burst_start,
            "end_date": last_out_time.date(),
        }
        for s in senders
    ]
    labels.append(
        {
            "customer_id": hub,
            "typology": "fan_in_fan_out",
            "role": "hub",
            "start_date": burst_start,
            "end_date": last_out_time.date(),
        }
    )
    labels.extend(
        {
            "customer_id": d,
            "typology": "fan_in_fan_out",
            "role": "downstream",
            "start_date": burst_start,
            "end_date": last_out_time.date(),
        }
        for d in downstream
    )
    return frag, labels


def _build_wealth_layering_instance(rng, cust, month, invest_map, config, start):
    cfg = config["typologies"]["wealth_layering"]
    accts = np.array(invest_map[cust])
    month_start, month_end = month_bounds(start, month)
    window_days = int(rng.integers(cfg["window_days_min"], cfg["window_days_max"] + 1))
    span = max((month_end - month_start).days - window_days, 0)
    burst_start = month_start + dt.timedelta(days=int(rng.integers(0, span + 1)) if span > 0 else 0)
    cursor_time = dt.datetime.combine(burst_start, dt.time())

    rows = []
    n_journal = int(rng.integers(cfg["journal_transfers_min"], cfg["journal_transfers_max"] + 1))
    for _ in range(n_journal):
        a, b = rng.choice(accts, size=2, replace=False)
        amt = float(rng.uniform(20000, 150000))
        t = cursor_time + dt.timedelta(days=float(rng.uniform(0, window_days * 0.6)))
        rows.append(
            {
                "customer_id": cust,
                "timestamp": t,
                "from_account": a,
                "to_account": b,
                "amount": round(amt, 2),
                "currency": "USD",
                "channel": "internal_journal",
                "cross_border": False,
                "instrument_type": None,
                "_typology": "wealth_layering",
                "_is_injected": True,
            }
        )

    n_trades = int(rng.integers(cfg["trade_round_trips_min"], cfg["trade_round_trips_max"] + 1))
    for _ in range(n_trades):
        acct = str(rng.choice(accts))
        amt = float(rng.uniform(10000, 100000))
        buy_t = cursor_time + dt.timedelta(days=float(rng.uniform(0, window_days * 0.7)))
        sell_t = buy_t + dt.timedelta(days=int(rng.integers(1, 5)))
        instrument = str(rng.choice(["equity", "fixed_income"]))
        rows.append(
            {
                "customer_id": cust,
                "timestamp": buy_t,
                "from_account": acct,
                "to_account": MARKET,
                "amount": round(amt, 2),
                "currency": "USD",
                "channel": "security_buy",
                "cross_border": False,
                "instrument_type": instrument,
                "_typology": "wealth_layering",
                "_is_injected": True,
            }
        )
        rows.append(
            {
                "customer_id": cust,
                "timestamp": sell_t,
                "from_account": MARKET,
                "to_account": acct,
                "amount": round(amt * float(rng.uniform(0.95, 1.05)), 2),
                "currency": "USD",
                "channel": "security_sell",
                "cross_border": False,
                "instrument_type": instrument,
                "_typology": "wealth_layering",
                "_is_injected": True,
            }
        )

    wire_amt = float(rng.uniform(cfg["outbound_wire_min"], cfg["outbound_wire_max"]))
    wire_acct = str(rng.choice(accts))
    wire_time = cursor_time + dt.timedelta(days=float(window_days * 0.9))
    ext_id = foreign_accounts(rng, config, 1, high_risk_bias=0.6)[0]
    rows.append(
        {
            "customer_id": cust,
            "timestamp": wire_time,
            "from_account": wire_acct,
            "to_account": ext_id,
            "amount": round(wire_amt, 2),
            "currency": "USD",
            "channel": "wire",
            "cross_border": True,
            "instrument_type": None,
            "_typology": "wealth_layering",
            "_is_injected": True,
        }
    )

    frag = pd.DataFrame(rows)
    label = {
        "customer_id": cust,
        "typology": "wealth_layering",
        "role": "primary",
        "start_date": burst_start,
        "end_date": wire_time.date(),
    }
    return frag, label


def inject_typologies(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    config: dict,
    start: dt.date,
    months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chequing_map, invest_map = account_maps(accounts)
    n_total = len(customers)
    used: set[str] = set()
    tx_fragments = []
    labels_rows = []

    # --- structuring -------------------------------------------------
    cfg = config["typologies"]["structuring"]
    n_instances = max(1, round(cfg["rate"] * n_total))
    eligible = customers.loc[
        customers["segment_true"].isin(["retail", "small_business"]) & ~customers["customer_id"].isin(used),
        "customer_id",
    ].values
    chosen = rng.choice(eligible, size=min(n_instances, len(eligible)), replace=False)
    used.update(chosen)
    for cust, m in zip(chosen, assign_stratified_months(rng, len(chosen), months)):
        frag, lab = _build_structuring_instance(rng, cust, m, chequing_map, config, start)
        tx_fragments.append(frag)
        labels_rows.append(lab)

    # --- rapid movement ------------------------------------------------
    cfg = config["typologies"]["rapid_movement"]
    n_instances = max(1, round(cfg["rate"] * n_total))
    eligible = customers.loc[~customers["customer_id"].isin(used), "customer_id"].values
    chosen = rng.choice(eligible, size=min(n_instances, len(eligible)), replace=False)
    used.update(chosen)
    for cust, m in zip(chosen, assign_stratified_months(rng, len(chosen), months)):
        frag, lab = _build_rapid_movement_instance(rng, cust, m, chequing_map, config, start)
        tx_fragments.append(frag)
        labels_rows.append(lab)

    # --- fan-in / fan-out mule network -----------------------------
    # Networks are sized up front (not greedily grown until a customer
    # budget is exhausted) so we reliably get >= 3 networks whenever the
    # pool allows it - greedily consuming toward a customer target could
    # land all of the budget in a single oversized network, leaving no
    # instances to spread across the validation/test period bands.
    cfg = config["typologies"]["fan_in_fan_out"]
    avg_network_size = (cfg["senders_min"] + cfg["senders_max"]) / 2 + 1 + (
        cfg["downstream_min"] + cfg["downstream_max"]
    ) / 2
    target_customers = round(cfg["rate"] * n_total)
    n_networks_target = max(3, round(target_customers / avg_network_size))
    pool = [c for c in customers["customer_id"].values if c not in used]
    rng.shuffle(pool)
    networks = []
    cursor = 0
    for _ in range(n_networks_target):
        n_send = int(rng.integers(cfg["senders_min"], cfg["senders_max"] + 1))
        n_down = int(rng.integers(cfg["downstream_min"], cfg["downstream_max"] + 1))
        need = n_send + 1 + n_down
        if cursor + need > len(pool):
            break
        group = pool[cursor : cursor + need]
        cursor += need
        networks.append((group[:n_send], group[n_send], group[n_send + 1 :]))
    for senders, hub, downstream in networks:
        used.update(senders)
        used.add(hub)
        used.update(downstream)
    for (senders, hub, downstream), m in zip(networks, assign_stratified_months(rng, len(networks), months)):
        frag, labs = _build_fan_instance(rng, senders, hub, downstream, m, chequing_map, config, start)
        tx_fragments.append(frag)
        labels_rows.extend(labs)

    # --- wealth layering --------------------------------------------
    cfg = config["typologies"]["wealth_layering"]
    n_instances = max(1, round(cfg["rate"] * n_total))
    hnw_pool = customers.loc[
        (customers["segment_true"] == "hnw_investment") & ~customers["customer_id"].isin(used), "customer_id"
    ].values
    hnw_eligible = np.array([c for c in hnw_pool if len(invest_map.get(c, [])) >= 2])
    chosen = rng.choice(hnw_eligible, size=min(n_instances, len(hnw_eligible)), replace=False)
    used.update(chosen)
    for cust, m in zip(chosen, assign_stratified_months(rng, len(chosen), months)):
        frag, lab = _build_wealth_layering_instance(rng, cust, m, invest_map, config, start)
        tx_fragments.append(frag)
        labels_rows.append(lab)

    tx = pd.concat(tx_fragments, ignore_index=True) if tx_fragments else _empty_tx_frame()
    if "_typology" not in tx.columns:
        tx["_typology"] = None
        tx["_is_injected"] = False

    flat_labels = []
    for item in labels_rows:
        if isinstance(item, list):
            flat_labels.extend(item)
        else:
            flat_labels.append(item)
    labels = pd.DataFrame(flat_labels, columns=["customer_id", "typology", "role", "start_date", "end_date"])
    return tx, labels


# ---------------------------------------------------------------------------
# Orchestration, output, data card
# ---------------------------------------------------------------------------

def generate_dataset(config: dict, n_customers: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config["seed"])
    start, _end, months = sim_window(config)

    customers = generate_customers(rng, n_customers, start, config)
    accounts = generate_accounts(rng, customers)
    relationships = generate_relationships(rng, customers, accounts, config)

    normal_tx = generate_normal_transactions(rng, customers, accounts, config, start, months)
    injected_tx, labels = inject_typologies(rng, customers, accounts, config, start, months)

    all_tx = pd.concat([normal_tx, injected_tx], ignore_index=True)
    all_tx["timestamp"] = pd.to_datetime(all_tx["timestamp"])
    all_tx = all_tx.sort_values("timestamp").reset_index(drop=True)
    all_tx["transaction_id"] = [f"TXN{i:08d}" for i in range(len(all_tx))]

    transaction_labels = all_tx[["transaction_id", "_is_injected", "_typology"]].rename(
        columns={"_is_injected": "is_injected", "_typology": "typology"}
    )
    transactions = all_tx[
        [
            "transaction_id",
            "timestamp",
            "from_account",
            "to_account",
            "amount",
            "currency",
            "channel",
            "cross_border",
            "instrument_type",
        ]
    ].copy()

    return {
        "customers": customers,
        "accounts": accounts,
        "relationships": relationships,
        "transactions": transactions,
        "transaction_labels": transaction_labels,
        "labels": labels,
    }


def write_outputs(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["customers", "accounts", "relationships", "transactions", "transaction_labels", "labels"]:
        data[name].to_parquet(out_dir / f"{name}.parquet", index=False)


def write_data_card(data: dict[str, pd.DataFrame], config: dict, out_path: Path, n_customers: int) -> None:
    labels = data["labels"]
    tx = data["transactions"]
    tx_labels = data["transaction_labels"]
    by_typology = labels.groupby("typology")["customer_id"].nunique() if not labels.empty else pd.Series(dtype=int)
    n_labeled = labels["customer_id"].nunique() if not labels.empty else 0
    n_injected_tx = int(tx_labels["is_injected"].sum())

    lines = [
        "# Data Card: AML Scenario Tuning Lab synthetic dataset",
        "",
        "**All data on this page is synthetic.** No real customers, accounts, or transactions were used.",
        "",
        f"- Customers: {n_customers:,}",
        f"- Transactions: {len(tx):,}",
        f"- Simulation window: {config['sim_start']} + {config['months']} months",
        f"- Seed: {config['seed']}",
        f"- Reporting threshold: ${config['reporting_threshold']:,}",
        "",
        "## Injected typologies (customer-level)",
        "",
        "| Typology | Customers labeled |",
        "|---|---|",
    ]
    for typology, count in by_typology.items():
        lines.append(f"| {typology} | {count} |")
    lines += [
        "",
        f"- Total labeled customers: {n_labeled} ({n_labeled / n_customers:.2%} of all customers)",
        f"- Injected transactions: {n_injected_tx:,} ({n_injected_tx / max(len(tx), 1):.3%} of all transactions)",
        "",
        "## Legitimate look-alikes (deliberately NOT labeled)",
        "",
        (
            f"- ~{config['legit_lookalikes']['structuring_lookalike_rate']:.0%} of small-business customers make "
            "some cash deposits clustered near the reporting threshold as part of normal revenue activity (no "
            "tight day-window burst, no label)."
        ),
        (
            f"- ~{config['legit_lookalikes']['wealth_lookalike_rate']:.0%} of HNW customers with 2+ investment "
            "accounts make legitimate internal journal transfers between their own accounts (no outbound "
            "cross-border wire, no label)."
        ),
        "",
        "## Known simplifications",
        "",
        "- Single currency (USD); no FX modeling.",
        "- Country codes are fictional (e.g. HOME/ALD/DRK); any resemblance to real countries is unintended.",
        "- No branch-level modeling for structuring: all deposits are attributed to the account, not a branch.",
        (
            f"- Each customer has a persistent roster of {MERCHANT_ROSTER_SIZE} recurring merchants and "
            f"{ACH_ROSTER_SIZE} recurring billers; card/ACH transactions draw from that roster "
            f"({1 - CARD_NEW_COUNTERPARTY_PROB:.0%} / {1 - ACH_NEW_COUNTERPARTY_PROB:.0%} of the time "
            "respectively), with an occasional new counterparty, so distinct-counterparty-per-period features "
            "are meaningful rather than inflated by a fresh id on every transaction."
        ),
        (
            f"- Each customer also has a persistent roster of {PEER_ROSTER_SIZE} peers (family/friends/"
            "roommates, with a bias toward small-business peers representing local purchases) for ordinary "
            "customer-to-customer e-transfers, so same-bank graph degree is nonzero for typical unlabeled "
            "customers too, not just the fan-in/fan-out typology."
        ),
        "- Account balances are not modeled; only transaction flows.",
        (
            "- Investment trades do not model actual security positions or market prices; buy/sell amounts are "
            "independent random draws, not linked to a shared price series."
        ),
        (
            "- Each injected customer carries at most one typology, and typologies are injected independently "
            "of each other. Real laundering is messier and less separable than this."
        ),
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic AML transaction-monitoring data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--n-customers", type=int, default=None)
    parser.add_argument("--small", action="store_true", help="Use the small customer count for fast runs/tests.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "data")
    parser.add_argument("--data-card", type=Path, default=REPO_ROOT / "artifacts" / "data_card.md")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.n_customers is not None:
        n_customers = args.n_customers
    elif args.small:
        n_customers = config["n_customers_small"]
    else:
        n_customers = config["n_customers_default"]

    data = generate_dataset(config, n_customers)
    write_outputs(data, args.out_dir)
    write_data_card(data, config, args.data_card, n_customers)

    print(f"Generated {n_customers:,} customers, {len(data['transactions']):,} transactions -> {args.out_dir}")
    print(f"Data card written to {args.data_card}")


if __name__ == "__main__":
    main()
