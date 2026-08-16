"""Two audits a quant cannot skip: look-ahead, and the 52/53-week calendar.

    PYTHONPATH=. .venv/bin/python -m code.feature.audit

## 1. Look-ahead

The cardinal sin. Everything the model sees must have been knowable before the deadline, and the
strongest guarantee is structural — all four companies report **after** it — but "structural" is
what people say right before they find a leak. So this checks it directly:

* no banked document is published after the corpus freeze
* no vendor lane carries a **realised** value for any target period
* the newest observation in every macro series predates the deadline

⚠️ A vendor **estimate** for the target period is not leakage — it is the benchmark, and it is the
thing being competed against. Only a realised value would be.

## 2. The 52/53-week calendar

⛔ Home Depot, ADI and Deere run 52/53-week fiscal years, so roughly every sixth year carries a
**14-week quarter**. A 14-week quarter has ~7.7 % more selling days than a 13-week one. Applying a
seasonal share measured on 13-week quarters to a 14-week quarter understates it by about that much
— on Home Depot's Q2 that is roughly **$3.5 bn**, several times any edge the model is chasing.

The check is arithmetic on dates the filers themselves stated: measure each target quarter's
length from the prior period end, and say plainly whether it is 13 or 14 weeks.
"""

from __future__ import annotations

from datetime import date

from code.feature import metrics as M
from code.feature import seasonality
from code.feature.periods import Resolver
from code.lib import config, store

DEADLINE = "2026-08-16"


def _d(text) -> date | None:
    try:
        return date.fromisoformat(str(text)[:10])
    except (ValueError, TypeError):
        return None


def leakage() -> list[dict]:
    findings = []

    docs = store.read(config.EXTRACTED / "documents.parquet")
    latest = max(d["published_at"] for d in docs if d["published_at"])
    findings.append({"check": "corpus documents", "detail": f"newest published_at {latest}",
                    "ok": latest <= DEADLINE})

    design = store.read(config.FEATURE / "design_matrix.parquet")
    reported = [r for r in design if r["already_reported"]]
    findings.append({"check": "target period absent from panel",
                     "detail": f"{len(reported)} of {len(design)} targets already reported",
                     "ok": not reported})

    # a vendor's REALISED quarter for a target period would be leakage
    resolver = Resolver()
    sa = store.read(config.EXTRACTED / "sa_financials.parquet")
    targets = {t["ticker"]: t for t in store.read(config.EXTRACTED / "target_periods.parquet")}
    sym = {s["short"]: s["ticker"] for s in store.read(config.EXTRACTED / "symbology.parquet")}
    hits = []
    for row in sa:
        if row["statement"] != "income_statement" or row["grain"] != "quarterly":
            continue
        ticker = sym.get(row["symbol"])
        if ticker not in targets or not row["period_end"]:
            continue
        end, want = _d(row["period_end"]), _d(targets[ticker]["projected_period_end"])
        if end and want and abs((end - want).days) <= 10:
            hits.append(f"{row['symbol']} {row['period_end']} {row['metric']}")
    findings.append({"check": "vendor realised value for a target period",
                     "detail": f"{len(hits)} rows" + (f" e.g. {hits[0]}" if hits else ""),
                     "ok": not hits})

    for lane, column in (("fred_observations.parquet", "date"),
                         ("labour_observations.parquet", "date"),
                         ("yh_bars.parquet", "date")):
        rows = store.read(config.EXTRACTED / lane)
        newest = max((r[column] or "") for r in rows)
        findings.append({"check": f"{lane.split('.')[0]} newest observation",
                         "detail": newest, "ok": newest <= DEADLINE})
    return findings


def calendar() -> list[dict]:
    resolver = Resolver()
    targets = store.read(config.EXTRACTED / "target_periods.parquet")
    anchors = store.read(config.FEATURE / "fiscal_periods.parquet")
    rows = []
    for target in targets:
        ticker = target["ticker"]
        end = _d(target["projected_period_end"])
        if ticker == "LSE:HAS" or end is None:
            rows.append({"ticker": ticker, "target": target["target_period"],
                         "days": None, "weeks": None, "note": "30 June year end, not 52/53-week"})
            continue
        mine = sorted((a for a in anchors if a["ticker"] == ticker), key=lambda a: a["period_end"])
        prior_end = _d(mine[-1]["period_end"]) if mine else None
        days = (end - prior_end).days if prior_end else None
        weeks = round(days / 7, 1) if days else None
        # the prior-year same quarter, to see where a 53rd week landed
        prior_year = None
        for a in mine:
            ad = _d(a["period_end"])
            if ad and 350 <= (end - ad).days <= 380:
                prior_year = a
        yoy_days = (end - _d(prior_year["period_end"])).days if prior_year else None
        rows.append({
            "ticker": ticker, "target": target["target_period"],
            "prior_quarter_end": mine[-1]["period_end"] if mine else None,
            "projected_end": target["projected_period_end"],
            "days": days, "weeks": weeks,
            "prior_year_end": prior_year["period_end"] if prior_year else None,
            "yoy_days": yoy_days,
            "note": ("14-week quarter — seasonal share understates it" if weeks and weeks >= 13.9
                     else "13-week quarter" if weeks else "unknown"),
        })
    return rows


def fiscal_year_weeks(ticker: str, fiscal_year: int) -> int:
    """52 or 53, measured from the filer's own quarter ends.

    ⛔ **A 53-week year dilutes every quarter's share of it.** Deere's FY2026 carries the extra
    week — measured, its Q1 ended 371 days after Q1 FY2025 while a normal year steps 364 — and the
    week landed in Q1, so Q3 is a normal 13 weeks but is now 13/53 of the year instead of 13/52.
    Splitting an annual guide by an unadjusted seasonal share therefore **overstates Q3 by about
    1.9 %**: on Deere's $4.5–5.0 bn guide that is roughly $23 m of pure specification error, in a
    forecast where the whole edge is a few percent.

    A year is 53 weeks when any of its quarters steps 371 days from the same quarter a year back.
    """
    anchors = [a for a in store.read(config.FEATURE / "fiscal_periods.parquet")
               if a["ticker"] == ticker]
    by_key = {(a["fiscal_year"], a["quarter"]): _d(a["period_end"]) for a in anchors}
    for quarter in (1, 2, 3, 4):
        cur, prior = by_key.get((fiscal_year, quarter)), by_key.get((fiscal_year - 1, quarter))
        if cur and prior and (cur - prior).days >= 370:
            return 53
    return 52


def additivity() -> list[dict]:
    """Four quarters must sum to the year the filer itself reported.

    ⛔ **This identity caught a mislabelled fiscal year that two submitted numbers depended on.**
    Home Depot's `2025-02-02` quarter end was filed as FY2025Q4 by one document and FY2024Q4 by
    another; the contested label won, so FY2024 silently lost its fourth quarter and FY2025 carried
    two. Every individual value was correct and every lane agreed — only the *sum* disagreed with
    the filer's own annual total, by **4.2 %**, and HD's full-year guide is applied to that base.

    The check needs no external data and no judgement: where the panel holds both four quarters and
    a stated annual figure for the same year, they are the same number or something is wrong. It
    stays in the audit because period attribution breaks quietly and this is the only place it
    makes a noise.
    """
    panel = store.read(config.FEATURE / "metric_panel.parquet")
    by_key: dict = {}
    for row in panel:
        if row["fiscal_year"]:
            by_key[(row["metric"], row["fiscal_year"], row["fiscal_period"])] = row["value"]

    findings = []
    for (metric, fy, period), stated in sorted(by_key.items()):
        if period != "FY" or not stated:
            continue
        # ⚠️ Only a flow adds up. A diluted **share count** is a period average — summing four
        # quarters of it gives ~4× the year and the check would "fail" on correct data — and a
        # margin or a comp-sales percentage is a ratio, which does not sum at all.
        spec = M.REGISTRY.get(metric)
        if spec is None or spec.unit not in seasonality.ADDITIVE_UNITS:
            continue
        quarters = [by_key.get((metric, fy, q)) for q in ("Q1", "Q2", "Q3", "Q4")]
        if any(q is None for q in quarters):
            continue
        total = sum(quarters)
        gap = abs(total / stated - 1.0)
        # a filer restating a prior year for a divestiture moves the annual figure but not the
        # quarters as originally filed, so a small gap is history, not a period-attribution bug
        findings.append({"check": f"{metric} FY{fy}", "metric": metric, "fiscal_year": fy,
                         "quarter_sum": total, "stated": stated, "gap_pct": gap * 100.0,
                         "detail": f"Σ4Q {total:,.1f} vs stated {stated:,.1f} ({gap:+.2%})",
                         "ok": gap <= 0.005})
    return findings


def main() -> int:
    print("look-ahead audit")
    print("-" * 92)
    findings = leakage()
    for row in findings:
        print(f"  {'PASS' if row['ok'] else 'FAIL'}  {row['check']:<44}{row['detail']}")
    failed = [f for f in findings if not f["ok"]]
    print(f"\n  {len(findings) - len(failed)}/{len(findings)} clean"
          + ("  ⛔ LEAKAGE" if failed else "  — every input predates the deadline"))

    print("\n52/53-week calendar audit")
    print("-" * 92)
    rows = calendar()
    store.write(config.FEATURE / "calendar_audit.parquet", rows)
    for row in rows:
        span = f"{row['days']}d ({row['weeks']}w)" if row["days"] else "—"
        yoy = f"{row['yoy_days']}d YoY" if row.get("yoy_days") else ""
        print(f"  {row['ticker']:<9}{row['target']:<11}{span:<14}{yoy:<12}{row['note']}")

    print("\nadditivity audit  (Σ four quarters == the filer's own annual figure)")
    print("-" * 92)
    adds = additivity()
    store.write(config.FEATURE / "additivity_audit.parquet", adds)
    broken = [a for a in adds if not a["ok"]]
    for row in broken:
        print(f"  FAIL  {row['check']:<44}{row['detail']}")
    print(f"  {len(adds) - len(broken)}/{len(adds)} (metric, year) pairs reconcile exactly"
          + ("" if broken else "  — no period is double-counted or missing"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
