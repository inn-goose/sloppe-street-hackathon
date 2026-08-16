"""Do independent lanes agree about the same number? The strongest test available, and free.

    PYTHONPATH=. .venv/bin/python -m code.lib.reconcile

For the three US filers the same quarterly figure is stated by up to five sources that share no
code path: the corpus's own tables, the corpus's prose, stockanalysis, SEC XBRL, and Yahoo's
statement history. Where two of them disagree on `(ticker, period_end, metric)`, one is wrong —
and *which* pairs disagree localises the fault far better than any single-lane check.

⛔ **The units are the whole difficulty, and getting them wrong is the failure this exists to
catch.** The corpus writes `3,623` under a "in millions" header; stockanalysis and SEC write
`3623465000`. Comparing them requires the corpus's declared scale to be applied — which is
exactly the `scale_known` caveat — so a disagreement here is either a real extraction error or a
scale that was never declared. Both are worth knowing and the report separates them.

⚠️ **Agreement is not proof of correctness** — two vendors can copy one upstream. But
*disagreement is proof of error*, and that asymmetry is what makes this worth running.
"""

from __future__ import annotations

import re
from collections import defaultdict

from code.lib import config, store

#: canonical metric -> how each lane spells it
METRICS = {
    "revenue": {
        "corpus": r"^(?:net sales|revenue|total revenue|net sales and revenues"
                  r"|total net sales and revenues)$",
        "sa": r"^revenue$",
        "sec": r"^(?:Revenues|RevenueFromContractWithCustomerExcludingAssessedTax)$",
        "yahoo": r"^totalRevenue$",
        "kind": "currency",
    },
    "net_income": {
        "corpus": r"^(?:net earnings|net income|net income attributable to deere & company)$",
        "sa": r"^netinc$",
        "sec": r"^NetIncomeLoss$",
        "yahoo": r"^netIncome$",
        "kind": "currency",
    },
    "diluted_eps": {
        "corpus": r"^(?:diluted earnings per share|fully diluted eps|diluted eps"
                  r"|diluted earnings per share \(gaap\))$",
        "sa": r"^epsdil$",
        "sec": r"^EarningsPerShareDiluted$",
        "yahoo": r"^$",
        "kind": "per_share",
    },
}

TICKERS = {"HD": "HD", "ADI": "ADI", "DE": "DE"}

#: Section headings that mark a row as a piece of the total rather than the total.
_DISAGGREGATED = re.compile(
    r"outside\s+the\s+u\.?s|united\s+states|canada|europe|asia|latin\s+america|geograph"
    r"|segment|agriculture|turf|construction|forestry|financial\s+services|precision"
    r"|by\s+(?:region|country|product|market)", re.IGNORECASE)
#: relative tolerance for a money figure, absolute for a per-share one
REL_TOL = 0.005
EPS_TOL = 0.011


def _corpus_series(facts, ticker, pattern, kind):
    """(period_end, value in BASE units). Only quarterly, scale-known, width-matching rows."""
    rx = re.compile(pattern, re.IGNORECASE)
    out = defaultdict(list)
    for r in facts:
        if r["ticker"] != ticker or not r["period_end"]:
            continue
        if r["span_months"] != 3 or not r["row_width_matches"]:
            continue
        # ⛔ consolidated only — a segment row carries the same label as the group total
        if not r.get("is_consolidated", True):
            continue
        # ⛔ …and so does a geographic or segment DISAGGREGATION inside one table. Deere states
        # `Net sales` twice in its 10-Q: once consolidated and once under
        # `Equipment operations outside the U.S. and Canada`.
        if _DISAGGREGATED.search(r.get("section") or ""):
            continue
        if not rx.match((r["label"] or "").strip()):
            continue
        if kind == "currency":
            if not r["scale_known"] or r["unit_kind"] != "currency":
                continue
            out[r["period_end"]].append(r["value_scaled"])
        else:
            if r["unit_kind"] != "per_share":
                continue
            out[r["period_end"]].append(r["value"])
    # a period repeats across documents; take the modal value rather than the first
    return {p: max(set(v), key=v.count) for p, v in out.items()}


def _sa_series(rows, symbol, pattern):
    rx = re.compile(pattern, re.IGNORECASE)
    return {r["period_end"]: r["value"] for r in rows
            if r["symbol"] == symbol and r["statement"] == "income_statement"
            and r["grain"] == "quarterly" and r["period_end"] and rx.match(r["metric"])}


def _sec_series(rows, ticker, pattern):
    rx = re.compile(pattern)
    out = {}
    for r in rows:
        if r["ticker"] != ticker or r["duration"] != "quarter" or not r["end"]:
            continue
        if not rx.match(r["concept"] or ""):
            continue
        # as-first-reported: the earliest filing that carried this period
        prev = out.get(r["end"])
        if prev is None or (r["filed"] or "9999") < prev[1]:
            out[r["end"]] = (r["value"], r["filed"] or "9999")
    return {k: v[0] for k, v in out.items()}


def _yahoo_series(rows, symbol, pattern):
    if pattern == r"^$":
        return {}
    rx = re.compile(pattern)
    return {r["period_end"]: r["value"] for r in rows
            if r["symbol"] == symbol and r["grain"] == "quarterly"
            and r["period_end"] and rx.match(r["metric"])}


def _agree(a: float, b: float, kind: str) -> bool:
    if kind == "per_share":
        return abs(a - b) <= EPS_TOL
    scale = max(abs(a), abs(b))
    return scale == 0 or abs(a - b) / scale <= REL_TOL


def main() -> int:
    facts = store.read(config.EXTRACTED / "statement_facts.parquet")
    sa = store.read(config.EXTRACTED / "sa_financials.parquet")
    sec = store.read(config.EXTRACTED / "sec_facts.parquet")
    yh = store.read(config.EXTRACTED / "yh_statements.parquet")

    print(f"{'metric':<14}{'ticker':<7}{'pair':<20}{'shared':>8}{'agree':>7}{'rate':>8}  worst")
    print("-" * 108)
    totals = defaultdict(lambda: [0, 0])
    problems = []

    for metric, spec in METRICS.items():
        for ticker, symbol in TICKERS.items():
            series = {
                "corpus": _corpus_series(facts, ticker, spec["corpus"], spec["kind"]),
                "sa": _sa_series(sa, symbol, spec["sa"]),
                "sec": _sec_series(sec, ticker, spec["sec"]),
                "yahoo": _yahoo_series(yh, symbol, spec["yahoo"]),
            }
            names = [n for n in series if series[n]]
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    shared = sorted(set(series[left]) & set(series[right]))
                    if not shared:
                        continue
                    ok = 0
                    worst = (0.0, None)
                    for period in shared:
                        a, b = series[left][period], series[right][period]
                        if _agree(a, b, spec["kind"]):
                            ok += 1
                        else:
                            scale = max(abs(a), abs(b)) or 1.0
                            gap = abs(a - b) / scale
                            if gap > worst[0]:
                                worst = (gap, (period, a, b))
                    totals[metric][0] += ok
                    totals[metric][1] += len(shared)
                    rate = ok / len(shared)
                    flag = "" if rate == 1.0 else (
                        f"{worst[1][0]} {left}={worst[1][1]:,.4g} {right}={worst[1][2]:,.4g}"
                        if worst[1] else "")
                    if rate < 1.0:
                        problems.append((metric, ticker, f"{left}~{right}", rate, flag))
                    print(f"{metric:<14}{ticker:<7}{left + '~' + right:<20}{len(shared):>8}"
                          f"{ok:>7}{rate:>8.0%}  {flag[:44]}")
    print("-" * 108)
    for metric, (ok, n) in totals.items():
        print(f"{metric:<14}{ok:>6,}/{n:<6,} agreeing observations  ({ok / max(n, 1):.1%})")
    grand_ok = sum(v[0] for v in totals.values())
    grand_n = sum(v[1] for v in totals.values())
    print(f"\n{grand_ok:,}/{grand_n:,} cross-lane comparisons agree "
          f"({grand_ok / max(grand_n, 1):.1%})")
    if problems:
        print(f"\n{len(problems)} lane pairs disagree somewhere — each is an extraction error in "
              f"one of the two:")
        for metric, ticker, pair, rate, flag in sorted(problems, key=lambda p: p[3])[:12]:
            print(f"  {metric:<14}{ticker:<6}{pair:<18}{rate:>6.0%}  {flag[:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
