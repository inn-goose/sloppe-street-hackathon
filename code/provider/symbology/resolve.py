"""symbology — one company, every name it goes by.

A *virtual* provider, like `calendar`: it fetches nothing and banks nothing. It exists so that no
other module has to know that Home Depot is `HD` to the competition, `HD` to Yahoo, `HD` to
stockanalysis, CIK `0000354950` to EDGAR and `US4370761029` by ISIN — or that Hays is `LSE:HAS`,
`HAS.L`, `HAS` on a London quote page, `GB0004161021`, and **has no CIK at all**.

## Why this is not bookkeeping

Three identifier traps in this universe have already produced wrong data in this run or would
have:

⛔ **`HAS` is Hasbro on a US exchange.** Asking Yahoo for `HAS` returns a toy company with a full,
plausible payload. The only safe address is `HAS.L`, and the guard is `long_name`, verified on
capture — measured: `Hays plc`, currency **`GBp`**.

⛔ **stockanalysis labels Home Depot's fiscal 2026 as "FY2027".** A vendor's fiscal-year label is
its own convention, not the filer's. Every join in this store keys on `period_end`, and this
module is where that rule is written down rather than rediscovered.

⛔ **Hays has no CIK and never will** — LSE/FCA filer, not an SEC registrant. Any code that
assumes a CIK exists silently drops 25 % of the competition score, so `cik` is `None` here
deliberately and loudly.

⚠️ **`GBp` is one hundredth of `GBP`.** Hays quotes in pence and the competition submits its EPS
in pence. The unit rides on the row so nothing has to remember it.
"""

from __future__ import annotations

from dataclasses import dataclass

from code.lib import config, store


@dataclass(frozen=True)
class Identity:
    ticker: str                 # our key, and the competition's
    short: str                  # bare symbol
    company: str
    isin: str
    cik: int | None             # None is a fact, not a gap
    yahoo: str
    stockanalysis: str
    sa_venue: str               # "" for US listings, "lon" for the London quote route
    corpus_dir: str
    currency: str               # the reporting currency
    quote_currency: str         # what the market quotes in — NOT always the same
    exchange: str
    fiscal_year_end: str


IDENTITIES = (
    Identity("HD", "HD", "Home Depot", "US4370761029", 354950, "HD", "HD", "",
             "home-depot", "USD", "USD", "NYSE", "late January / early February"),
    Identity("ADI", "ADI", "Analog Devices", "US0326541051", 6281, "ADI", "ADI", "",
             "analog-devices", "USD", "USD", "NASDAQ", "Saturday nearest 31 October"),
    Identity("LSE:HAS", "HAS", "Hays plc", "GB0004161021", None, "HAS.L", "HAS", "lon",
             "hays", "GBP", "GBp", "LSE", "30 June"),
    Identity("DE", "DE", "Deere & Company", "US2441991054", 315189, "DE", "DE", "",
             "deere", "USD", "USD", "NYSE", "late October"),
)

BY_TICKER = {i.ticker: i for i in IDENTITIES}
BY_YAHOO = {i.yahoo: i for i in IDENTITIES}
BY_CIK = {i.cik: i for i in IDENTITIES if i.cik}


def resolve(name: str) -> Identity | None:
    """Any spelling → the identity, or None. Never guesses."""
    key = name.strip()
    if key in BY_TICKER:
        return BY_TICKER[key]
    if key in BY_YAHOO:
        return BY_YAHOO[key]
    folded = key.casefold()
    for identity in IDENTITIES:
        if folded in {identity.short.casefold(), identity.company.casefold(),
                      identity.isin.casefold(), identity.corpus_dir}:
            return identity
    if key.isdigit() and int(key) in BY_CIK:
        return BY_CIK[int(key)]
    return None


def verify() -> list[dict]:
    """Check every claim here against what the providers actually banked.

    ⚠️ A symbology table is the one artefact in a store that is *authored* rather than derived, so
    it is the one most likely to drift. This grades it against the captures.
    """
    checks = []
    try:
        stats = {r["symbol"]: r for r in store.read(config.EXTRACTED / "yh_key_stats.parquet")}
    except FileNotFoundError:
        stats = {}
    for identity in IDENTITIES:
        row = stats.get(identity.yahoo) or {}
        name_ok = identity.company.split()[0].casefold() in (row.get("long_name") or "").casefold()
        ccy_ok = (row.get("currency") or "") == identity.quote_currency
        checks.append({
            "ticker": identity.ticker, "yahoo": identity.yahoo,
            "yahoo_long_name": row.get("long_name") or "",
            "yahoo_currency": row.get("currency") or "",
            "expected_quote_currency": identity.quote_currency,
            "name_matches": name_ok if row else None,
            "currency_matches": ccy_ok if row else None,
            "cik": identity.cik, "isin": identity.isin,
        })
    return checks


def main() -> int:
    rows = [{"ticker": i.ticker, "short": i.short, "company": i.company, "isin": i.isin,
             "cik": i.cik, "yahoo": i.yahoo, "stockanalysis": i.stockanalysis,
             "sa_venue": i.sa_venue, "corpus_dir": i.corpus_dir, "currency": i.currency,
             "quote_currency": i.quote_currency, "exchange": i.exchange,
             "fiscal_year_end": i.fiscal_year_end} for i in IDENTITIES]
    store.write(config.EXTRACTED / "symbology.parquet", rows)
    checks = verify()
    store.write(config.EXTRACTED / "symbology_checks.parquet", checks)

    print(f"extracted/symbology.parquet {len(rows)} identities")
    for i in IDENTITIES:
        print(f"  {i.ticker:<9}{i.company:<18}{i.isin}  "
              f"CIK {i.cik if i.cik else '— (no SEC registration)':<24}"
              f"yahoo {i.yahoo:<7}quotes {i.quote_currency}")
    bad = [c for c in checks if c["name_matches"] is False or c["currency_matches"] is False]
    print(f"  verification against banked captures: "
          f"{sum(1 for c in checks if c['name_matches'])} names and "
          f"{sum(1 for c in checks if c['currency_matches'])} currencies confirmed"
          + (f", {len(bad)} MISMATCHED" if bad else ""))
    for c in bad:
        print(f"    MISMATCH {c['ticker']}: yahoo says {c['yahoo_long_name']!r} "
              f"in {c['yahoo_currency']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
