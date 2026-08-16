"""extracted/sec_facts — every XBRL fact, with the filing that first carried it.

`companyfacts` is nested `taxonomy → concept → units → [observations]`, and each observation
states `start`, `end`, `val`, `fy`, `fp`, `form`, **`filed`** and `accn`. Unpivoting it gives the
only genuinely **point-in-time** panel in this store.

## The three columns that make this lane different from every vendor

* **`filed`** — when the fact became knowable. A vendor serves today's view of 2019; this serves
  the number as first reported, alongside every later restatement of it, as separate rows.
* **`frame`** — SEC's own canonical period label, present only on the observation it considers
  the representative one for that period. Its absence is how a restatement is told from an
  original without diffing.
* **`form`** — 10-Q vs 10-K vs 8-K. A quarterly value taken off a 10-K's comparative column is
  not the same fact as the one the 10-Q filed at the time.

⚠️ **A duplicate `(concept, start, end)` is the norm, not an error.** Each period is re-reported
in later filings as a comparative, so the panel carries several rows per period with different
`filed` dates. Taking the *earliest* gives as-first-reported; taking the *latest* gives the
restated series. Both are legitimate and they are different questions — the column is kept so the
consumer chooses rather than inheriting whichever happened to sort last.

⛔ **Quarterly durations must be filtered on their own length.** `us-gaap:Revenues` carries
3-month, 6-month, 9-month and 12-month spans under one concept, distinguished only by
`end - start`. Summing across them double-counts the year.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date

from code.lib import config, rawstore, store

#: Concepts worth unpivoting. The full set is 461–680 per filer and most of it is balance-sheet
#: detail the twelve targets never touch; this is the income-statement and share-count spine plus
#: the tags each specific target needs.
CONCEPTS = {
    # top line — filers tag revenue under several concepts across eras
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    # profitability
    "GrossProfit", "CostOfRevenue", "CostOfGoodsAndServicesSold",
    "OperatingIncomeLoss", "NetIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
    "ExtraordinaryItemsNoncontrollingInterest",
    "IncomeTaxExpenseBenefit", "EffectiveIncomeTaxRateContinuingOperations",
    # per share and the denominator behind it
    "EarningsPerShareDiluted", "EarningsPerShareBasic",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding",
    # capital return — the EPS denominator's driver
    "PaymentsForRepurchaseOfCommonStock", "CommonStockDividendsPerShareDeclared",
    # quality / one-offs
    "ShareBasedCompensation", "RestructuringCharges", "DepreciationDepletionAndAmortization",
    "ResearchAndDevelopmentExpense", "SellingGeneralAndAdministrativeExpense",
    # working capital, for the accrual legs
    "InventoryNet", "AccountsReceivableNetCurrent", "AccountsPayableCurrent",
    "NetCashProvidedByUsedInOperatingActivities", "PaymentsToAcquirePropertyPlantAndEquipment",
}


def _days(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    try:
        a = date(*(int(p) for p in start.split("-")))
        b = date(*(int(p) for p in end.split("-")))
    except (ValueError, TypeError):
        return None
    return (b - a).days


def build() -> list[dict]:
    rows: list[dict] = []
    for cap, body in rawstore.iter_captures("sec"):
        if cap["product"] != "company_facts":
            continue
        ticker = cap["for_ticker"]
        for taxonomy, concepts in (body.get("facts") or {}).items():
            for concept, node in concepts.items():
                # ⚡ EVERY concept, not a shortlist. A first version kept 46 hand-picked tags and
                # dropped 400+ per filer unread — including the segment, lease, pension and
                # revenue-disaggregation families. The filer decided what to tag; a reader that
                # second-guesses that is choosing its own coverage gap.
                label = node.get("label") or concept
                for unit, observations in (node.get("units") or {}).items():
                    for obs in observations:
                        value = obs.get("val")
                        if not isinstance(value, (int, float)):
                            continue
                        span = _days(obs.get("start"), obs.get("end"))
                        rows.append({
                            "ticker": ticker, "cik": cap["cik"],
                            "taxonomy": taxonomy, "concept": concept, "label": label,
                            "unit": unit,
                            "start": obs.get("start"), "end": obs.get("end"),
                            "span_days": span,
                            # the span a filer means by "a quarter" is 84-98 days; a year is 350-380
                            "duration": ("instant" if span is None else
                                         "quarter" if 80 <= span <= 100 else
                                         "half" if 170 <= span <= 195 else
                                         "nine_months" if 260 <= span <= 285 else
                                         "year" if 350 <= span <= 380 else "other"),
                            "value": float(value),
                            "fiscal_year": obs.get("fy"), "fiscal_period": obs.get("fp"),
                            "form": obs.get("form"), "filed": obs.get("filed"),
                            "accession": obs.get("accn"), "frame": obs.get("frame"),
                            "is_core": concept in CONCEPTS,
                            "captured_at": cap.get("fetched_at") or "",
                        })
    return rows


def main() -> int:
    rows = build()
    store.write(config.EXTRACTED / "sec_facts.parquet", rows)

    # as-first-reported: the earliest filing that carried each (concept, period)
    seen: dict[tuple, dict] = {}
    for r in rows:
        key = (r["ticker"], r["concept"], r["unit"], r["start"], r["end"])
        prev = seen.get(key)
        if prev is None or (r["filed"] or "9999") < (prev["filed"] or "9999"):
            seen[key] = r
    store.write(config.EXTRACTED / "sec_facts_first_reported.parquet", list(seen.values()))

    print(f"extracted/sec_facts.parquet                {len(rows):,} facts  "
          f"{len({r['concept'] for r in rows})} concepts")
    print(f"extracted/sec_facts_first_reported.parquet {len(seen):,} as-first-reported "
          f"({len(rows) - len(seen):,} later restatements of the same period)")
    durations = Counter(r["duration"] for r in rows)
    print(f"  durations: {dict(durations)}")
    for t in ("HD", "ADI", "DE"):
        mine = [r for r in rows if r["ticker"] == t]
        q = [r for r in mine if r["duration"] == "quarter"]
        filed = sorted({r["filed"] for r in mine if r["filed"]})
        print(f"  {t:<5}{len(mine):>7,} facts  {len(q):>6,} quarterly  "
              f"filed {filed[0] if filed else '-'}..{filed[-1] if filed else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
