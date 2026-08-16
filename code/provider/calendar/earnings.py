"""calendar — when each fiscal period ENDED and when it was REPORTED.

A *virtual* provider: it fetches nothing and banks nothing. It answers
one question — "which fiscal period is this document about, and what dates bound it" — from lanes
other providers already built, so nothing downstream has to learn which upstream happens to carry
a date.

## Why this is the load-bearing lane

Every metric in the panel is keyed on `(ticker, fiscal_year, fiscal_quarter)`. Get that key wrong
by one quarter and a seasonality model trains on the wrong season, a YoY bridge divides by the
wrong base, and the error is invisible — the numbers are all real, they are just filed under the
wrong period. So the key is never inferred from a publication date; it is read from three
independent statements the corpus makes about itself, and disagreements are recorded rather than
resolved silently:

1. **The filename's fiscal tag** (`q2-8k`, `fy-10k`, `h2-8k`) — the corpus's own label.
2. **The frontmatter `period`** (`Q2 2025`) — the corpus's other own label.
3. **The period-end DATE in the document's own table headers** (`August 3, 2025`) — the filer's.

⚠️ **These disagree, and the third one wins.** Measured below: the frontmatter period drifts on
transcripts (an AGM held in May 2026 is labelled `Q2 2027`), because it records the period the
*conversation* was about rather than the period the *numbers* are for. A date printed in a
statement header cannot drift.

## The 52/53-week trap

⛔ HD, ADI and DE all run 52/53-week fiscal years, so a quarter end moves by a day or two every
year and by a whole week when a 53rd week is inserted. **A calendar-month calendar is wrong for
three of the four names**, which is why the target period end is projected as *prior-year end +
364 days* (52 weeks, preserving the weekday) rather than as "same date next year", and the
projection is checked against the realised gaps this lane measures.

Hays is the exception: a plain 30 June year end, and its target is the FY rather than a quarter.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

from code.lib import config, store

_QUARTER_TAG = {"q1": 1, "q2": 2, "q3": 3, "q4": 4}

#: ⛔ Hays states its period ends in WORDS and its tables key on YEARS ("Year ended 30 June … |
#: 2025 | 2024"), so the date-token route that works for the three US filers finds nothing at all
#: — measured, it bound **0 of 239** Hays documents until this existed. The company runs a plain
#: 30-June year with no 52/53-week drift, so the tag alone fixes the period end exactly:
#: the most recent occurrence of the tag's month/day at or before the publication date.
_HAS_TAG_MONTH_DAY = {
    "h2": (6, 30), "q4": (6, 30),      # full year / Q4 both end the fiscal year
    "h1": (12, 31), "q2": (12, 31),    # interim / Q2 both end the first half
    "q1": (9, 30),
    "q3": (3, 31),
}


def _has_period_end(tag: str, report_date: date | None) -> str | None:
    """Hays' period end from its own fiscal calendar, walked back from publication."""
    if not report_date or tag not in _HAS_TAG_MONTH_DAY:
        return None
    month, day = _HAS_TAG_MONTH_DAY[tag]
    for year in (report_date.year, report_date.year - 1):
        candidate = date(year, month, day)
        if candidate <= report_date:
            return candidate.isoformat()
    return None


def _d(text: str | None) -> date | None:
    if not text:
        return None
    try:
        y, m, dd = (int(p) for p in str(text)[:10].split("-"))
        return date(y, m, dd)
    except (ValueError, TypeError):
        return None


def build() -> list[dict]:
    lanes = {r["doc_id"]: r for r in store.read(config.EXTRACTED / "document_lanes.parquet")}
    cols = store.read(config.EXTRACTED / "table_columns.parquet")

    # the dated period ends each document's own tables state
    per_doc: dict[str, list[dict]] = {}
    for c in cols:
        if c["period_end"]:
            per_doc.setdefault(c["doc_id"], []).append(c)

    rows = []
    for doc_id, lane in lanes.items():
        dated = per_doc.get(doc_id, [])
        report_date = _d(lane["published_at"])
        ends = sorted({c["period_end"] for c in dated})
        # ⚠️ The CURRENT period end is the latest date the document states that is not in the
        # future of its own publication. A guidance table can name a period end that has not
        # happened yet, and taking a plain max would key the document to a period it is
        # forecasting rather than reporting.
        current = None
        for candidate in reversed(ends):
            cd = _d(candidate)
            if cd and report_date and cd <= report_date:
                current = candidate
                break
        prior = None
        if current:
            cd = _d(current)
            for candidate in reversed([e for e in ends if e < current]):
                pd = _d(candidate)
                # the prior-year comparable sits ~a year back, not the previous quarter
                if pd and cd and 300 <= (cd - pd).days <= 430:
                    prior = candidate
                    break

        # the smallest span any dated column carries — 3 on a quarterly table, 12 on an annual one
        spans = [c["span_months"] for c in dated if c["span_months"]]
        span = min(spans) if spans else 0

        tag = (lane.get("filename_fiscal_tag") or "").lower()
        period_source = "table_date" if current else ""
        if not current and lane["ticker"] == "LSE:HAS":
            current = _has_period_end(tag, report_date)
            period_source = "has_fiscal_rule" if current else ""
            if current:
                prior_d = _d(current)
                prior = date(prior_d.year - 1, prior_d.month, prior_d.day).isoformat()
        rows.append({
            "period_source": period_source,
            "doc_id": doc_id,
            "ticker": lane["ticker"],
            "lane_family": lane["lane_family"],
            "report_date": lane["published_at"],
            "filename_tag": tag,
            "frontmatter_period": lane["period_label"],
            "tag_quarter": _QUARTER_TAG.get(tag, 0),
            "period_end": current,
            "prior_period_end": prior,
            "n_dated_columns": len(dated),
            "min_span_months": span,
            "reporting_lag_days": ((_d(lane["published_at"]) - _d(current)).days
                                   if current and lane["published_at"] else None),
        })
    return rows


def fiscal_calendar(rows: list[dict]) -> list[dict]:
    """Per (ticker, period_end): the fiscal label, and the realised gap to the prior year.

    Built only from `earnings_release` and `periodic_report` — a conference transcript states no
    statement dates, and an AGM's frontmatter period is about the meeting, not the accounts.
    """
    out = []
    by_ticker: dict[str, list[dict]] = {}
    for r in rows:
        if r["lane_family"] not in ("earnings_release", "periodic_report"):
            continue
        if not r["period_end"]:
            continue
        if not r["tag_quarter"] and r["filename_tag"] not in ("fy", "h1", "h2"):
            continue
        by_ticker.setdefault(r["ticker"], []).append(r)

    for ticker, group in by_ticker.items():
        seen: dict[str, dict] = {}
        for r in sorted(group, key=lambda x: (x["period_end"], x["report_date"])):
            key = r["period_end"]
            # several documents report the same period (8-K then 10-Q); keep the earliest report
            if key not in seen:
                seen[key] = {"ticker": ticker, "period_end": key,
                             "first_report_date": r["report_date"],
                             "tags": Counter(), "spans": Counter(), "n_docs": 0}
            seen[key]["tags"][r["filename_tag"]] += 1
            seen[key]["spans"][r["min_span_months"]] += 1
            seen[key]["n_docs"] += 1

        ordered = sorted(seen.values(), key=lambda x: x["period_end"])
        for entry in ordered:
            tag = entry["tags"].most_common(1)[0][0]
            prior_end = None
            this = _d(entry["period_end"])
            for other in ordered:
                od = _d(other["period_end"])
                if od and this and 300 <= (this - od).days <= 430:
                    prior_end = other["period_end"]
            out.append({
                "ticker": ticker,
                "period_end": entry["period_end"],
                "fiscal_tag": tag,
                "quarter": _QUARTER_TAG.get(tag, 0),
                "first_report_date": entry["first_report_date"],
                "reporting_lag_days": ((_d(entry["first_report_date"]) - this).days
                                       if this else None),
                "prior_year_period_end": prior_end,
                "year_gap_days": ((this - _d(prior_end)).days if prior_end and this else None),
                "n_docs": entry["n_docs"],
            })
    return out


def project_target_period_end(calendar: list[dict]) -> list[dict]:
    """The end date of each company's TARGET period, projected from its own history.

    ⛔ **52 weeks, not 12 months.** For HD/ADI/DE the fiscal quarter is a 13-week block, so the
    projection is `prior-year end + 364 days`, which preserves the weekday. A 53-week year inserts
    an extra week in Q4 and shifts everything after it, so the realised `year_gap_days` this lane
    measures is what says whether 364 or 371 applies — it is read, not assumed.
    """
    out = []
    for company in config.load_companies():
        rows = [r for r in calendar if r["ticker"] == company.ticker and r["quarter"]]
        target_q = company.fiscal_quarter
        if company.ticker == "LSE:HAS" or not target_q:
            # a plain 30 June year end, stated in every RNS
            out.append({"ticker": company.ticker, "target_period": company.period,
                        "projected_period_end": f"{company.fiscal_year}-06-30",
                        "basis": "stated fiscal year end (30 June)", "gap_days_used": None,
                        "anchor_period_end": None})
            continue
        same_q = sorted((r for r in rows if r["quarter"] == target_q),
                        key=lambda r: r["period_end"])
        if not same_q:
            continue
        anchor = same_q[-1]
        # ⛔ **The MOST RECENT realised gap, not the modal one.** A 53-week fiscal year inserts an
        # extra week and every quarter after it shifts by 7 days. Measured: Deere's Q2 FY2026
        # ended 2026-05-03 against 2025-04-27 — a **371-day** gap — so FY2026 already carries the
        # 53rd week, while the modal gap across its history is 364. Projecting Q3 on the modal gap
        # put the period end a week early (2026-07-26 against the true 2026-08-02), which would
        # mis-date every macro and FX window aligned to it. The latest completed quarter of the
        # target year is the only observation that knows whether the extra week has landed.
        latest = max((r for r in rows if r["year_gap_days"]),
                     key=lambda r: r["period_end"], default=None)
        modal_gaps = [r["year_gap_days"] for r in same_q if r["year_gap_days"]]
        if latest:
            gap, basis = latest["year_gap_days"], f"gap realised at {latest['period_end']}"
        elif modal_gaps:
            gap, basis = Counter(modal_gaps).most_common(1)[0][0], "modal realised year gap"
        else:
            gap, basis = 364, "52-week default (no realised gap observed)"
        projected = _d(anchor["period_end"]) + timedelta(days=gap)
        out.append({
            "ticker": company.ticker, "target_period": company.period,
            "projected_period_end": projected.isoformat(),
            "basis": f"anchor {anchor['period_end']} + {gap}d ({basis})",
            "gap_days_used": gap, "anchor_period_end": anchor["period_end"],
        })
    return out


def main() -> int:
    rows = build()
    store.write(config.EXTRACTED / "document_calendar.parquet", rows)
    cal = fiscal_calendar(rows)
    store.write(config.EXTRACTED / "fiscal_calendar.parquet", cal)
    targets = project_target_period_end(cal)
    store.write(config.EXTRACTED / "target_periods.parquet", targets)

    dated = sum(1 for r in rows if r["period_end"])
    print(f"extracted/document_calendar.parquet {len(rows):,} documents, "
          f"{dated:,} carry a statement period-end date")
    print(f"extracted/fiscal_calendar.parquet   {len(cal):,} fiscal periods")
    for ticker in ("HD", "ADI", "LSE:HAS", "DE"):
        mine = [c for c in cal if c["ticker"] == ticker]
        lags = [c["reporting_lag_days"] for c in mine if c["reporting_lag_days"] is not None]
        gaps = [c["year_gap_days"] for c in mine if c["year_gap_days"]]
        lag = sorted(lags)[len(lags) // 2] if lags else None
        print(f"  {ticker:<8}{len(mine):>3} periods  "
              f"{mine[0]['period_end'] if mine else '-'}..{mine[-1]['period_end'] if mine else '-'}"
              f"  median reporting lag {lag}d  year gaps {sorted(set(gaps))[:6]}")
    print("\nextracted/target_periods.parquet")
    for t in targets:
        print(f"  {t['ticker']:<8}{t['target_period']:<12}ends {t['projected_period_end']}"
              f"   [{t['basis']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
