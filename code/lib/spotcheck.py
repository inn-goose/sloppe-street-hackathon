"""Can the store return the RIGHT number? Ground truths read by hand from the source documents.

    PYTHONPATH=. .venv/bin/python -m code.lib.spotcheck

Row counts prove a lane is populated; they do not prove it is correct. Each case below is a value
read directly out of a banked filing before any extractor existed, and the check asks whether the
store now returns it. A lane that reports 200,000 facts and cannot produce `HD Q2 FY2025 net sales
= 45,277` is not ready, however large it is.

⚠️ These are the **prior-year comparables for the twelve targets** specifically. Every forecast is
a YoY bridge off one of them, so an error here propagates straight into a submitted number.
"""

from __future__ import annotations

import re

from code.lib import config, store

#: (ticker, description, expected, tolerance, lane, matcher)
#: `matcher` gets the lane's rows and returns candidate values.
CASES = (
    ("HD", "Q2 FY2025 net sales ($m)", 45277.0, 1.0, "statement_facts",
     dict(label=r"^net sales$", period_end="2025-08-03", unit_kind="currency")),
    ("HD", "Q2 FY2025 comparable sales (%)", 1.0, 0.05, "statement_facts",
     dict(label=r"^comparable sales", period_end="2025-08-03", unit_kind="percent")),
    ("HD", "Q2 FY2025 adjusted diluted EPS ($)", 4.68, 0.005, "prose_facts",
     dict(metric=r"adjusted diluted earnings per share", published_at="2025-08-19")),
    ("HD", "Q1 FY2026 adjusted diluted EPS ($)", 3.43, 0.005, "prose_facts",
     dict(metric=r"adjusted diluted earnings per share", published_at="2026-05-19")),
    ("ADI", "Q2 FY2026 revenue ($m)", 3623.0, 1.0, "statement_facts",
     dict(label=r"^revenue$", period_end="2026-05-02", unit_kind="currency")),
    ("ADI", "Q2 FY2026 adjusted gross margin (%)", 73.0, 0.05, "statement_facts",
     dict(label=r"adjusted gross margin", period_end="2026-05-02", unit_kind="percent")),
    ("ADI", "Q2 FY2026 adjusted diluted EPS ($)", 3.09, 0.005, "statement_facts",
     dict(label=r"adjusted diluted earnings per share", period_end="2026-05-02")),
    # ⚠️ These six assert the PERIOD as well as the value. An earlier version left the period
    # unconstrained and called them passes — the value existed *somewhere* in the lane. That is
    # not the same claim: measured before the header fix, Deere's segment operating profit was
    # bound a full year early and its net sales carried a six-month span on a three-month figure,
    # and this test reported PASS on both. A right number on a wrong date is worse than a miss,
    # because nothing downstream can see it.
    ("DE", "Q2 FY2026 net sales and revenues ($m)", 13369.0, 1.0, "statement_facts",
     dict(label=r"net sales and revenues", period_year="2026", span_months="3",
          unit_kind="currency")),
    ("DE", "Q2 FY2026 diluted EPS ($)", 6.55, 0.005, "statement_facts",
     dict(label=r"diluted eps|fully diluted eps|diluted earnings per share",
          period_year="2026", span_months="3")),
    ("DE", "Q2 FY2026 P&PA operating profit ($m)", 706.0, 1.0, "statement_facts",
     dict(label=r"^operating profit$", period_year="2026", span_months="3",
          unit_kind="currency")),
    ("DE", "Q2 FY2026 six-month net sales ($m)", 22981.0, 1.0, "statement_facts",
     dict(label=r"net sales and revenues", period_year="2026", span_months="6",
          unit_kind="currency")),
    ("LSE:HAS", "FY2025 net fees (£m)", 972.4, 0.1, "statement_facts",
     dict(label=r"^net fees", period_year="2025", unit_kind="currency")),
    ("LSE:HAS", "FY2025 pre-exceptional operating profit (£m)", 45.6, 0.1, "statement_facts",
     dict(label=r"^operating profit", period_year="2025", unit_kind="currency")),
    ("LSE:HAS", "FY2025 pre-exceptional basic EPS (p)", 1.31, 0.005, "statement_facts",
     dict(label=r"basic earnings per share|^eps$", period_year="2025")),
    ("ADI", "Q2 FY2026 revenue, prior-year column ($m)", 2640.0, 1.0, "statement_facts",
     dict(label=r"^revenue$", period_end="2025-05-03", unit_kind="currency")),
)


def _candidates(rows, ticker, spec):
    out = []
    for row in rows:
        if row.get("ticker") != ticker:
            continue
        ok = True
        for field, pattern in spec.items():
            if pattern is None:
                continue
            value = str(row.get(field) or "")
            if field in ("label", "metric"):
                ok = ok and bool(re.search(pattern, value, re.IGNORECASE))
            else:
                ok = ok and value == pattern
            if not ok:
                break
        if ok:
            out.append(row)
    return out


def main() -> int:
    lanes = {name: store.read(config.EXTRACTED / f"{name}.parquet")
             for name in ("statement_facts", "prose_facts")}

    print(f"{'ground truth':<52}{'expected':>11}{'found':>12}{'n':>5}  verdict")
    print("-" * 100)
    passed = failed = 0
    misses = []
    for ticker, label, expected, tol, lane, spec in CASES:
        rows = _candidates(lanes[lane], ticker, spec)
        values = [r.get("value") for r in rows if r.get("value") is not None]
        hit = next((v for v in values if abs(v - expected) <= tol), None)
        # a scaled variant is still a hit — the store may carry millions where the doc wrote units
        if hit is None:
            hit = next((v for v in values
                        if any(abs(v / f - expected) <= tol for f in (1e6, 1e3, 100.0))), None)
        verdict = "PASS" if hit is not None else "MISS"
        if hit is None:
            failed += 1
            misses.append((ticker, label, lane, len(rows),
                           sorted({round(v, 3) for v in values})[:6]))
        else:
            passed += 1
        shown = f"{hit:,.3f}" if hit is not None else "—"
        print(f"{ticker + ' · ' + label:<52}{expected:>11,.2f}{shown:>12}{len(rows):>5}  {verdict}")
    print("-" * 100)
    print(f"{passed}/{passed + failed} ground truths reproduced from the store")
    if misses:
        print("\nmisses, with what the lane actually holds:")
        for ticker, label, lane, n, sample in misses:
            print(f"  {ticker:<9}{label:<46}{lane}  {n} candidate rows  values={sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
