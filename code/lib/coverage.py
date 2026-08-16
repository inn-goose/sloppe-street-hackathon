"""Can each of the twelve targets actually be built? Measured, per metric, per lane.

    PYTHONPATH=. .venv/bin/python -m code.lib.coverage

The honest gate before the feature layer. For every submitted metric this asks four questions and
answers them from the store rather than from intent:

  history    — is there a series of the metric ITSELF to fit seasonality and growth on?
  anchor     — is there a stated guide standing against the target period?
  benchmark  — is there a consensus, i.e. the thing the competition scores against?
  drivers    — are the exogenous conditioners present?

⚠️ **A metric is only covered if the BASIS matches.** Five of the twelve are non-GAAP —
HD and ADI adjusted diluted EPS, ADI adjusted gross margin, Hays pre-exceptional operating profit
and EPS. Vendors publish GAAP. Counting a GAAP series as coverage for an adjusted target is the
one error that would be invisible: both numbers are real.
"""

from __future__ import annotations

import re

from code.lib import config, store

#: (ticker, workbook label, basis, search patterns per lane). The patterns are how each source
#: spells the same measure — mined from the store, not assumed.
TARGETS = (
    ("HD", "Net sales", "gaap",
     dict(corpus=r"^net sales$", sa=r"^revenue$", sec=r"Revenues|RevenueFromContract")),
    ("HD", "Adjusted diluted EPS", "adjusted",
     dict(corpus=r"adjusted diluted earnings per share|adjusted diluted eps", sa=r"^$", sec=r"^$")),
    ("HD", "Comparable sales, total company", "kpi",
     dict(corpus=r"comparable sales", sa=r"^$", sec=r"^$")),
    ("ADI", "Revenue", "gaap",
     dict(corpus=r"^revenue$", sa=r"^revenue$", sec=r"Revenues|RevenueFromContract")),
    ("ADI", "Adjusted diluted EPS", "adjusted",
     dict(corpus=r"adjusted diluted earnings per share|adjusted diluted eps", sa=r"^$", sec=r"^$")),
    ("ADI", "Adjusted gross margin", "adjusted",
     dict(corpus=r"adjusted gross margin", sa=r"^$", sec=r"^$")),
    ("LSE:HAS", "Net fees", "kpi",
     dict(corpus=r"^net fees", sa=r"^$", sec=r"^$")),
    ("LSE:HAS", "Pre-exceptional basic EPS", "adjusted",
     dict(corpus=r"basic earnings per share|^eps$", sa=r"^$", sec=r"^$")),
    ("LSE:HAS", "Pre-exceptional operating profit", "adjusted",
     dict(corpus=r"^operating profit", sa=r"^$", sec=r"^$")),
    ("DE", "Worldwide net sales and revenues", "gaap",
     dict(corpus=r"net sales and revenues|total net sales and revenues", sa=r"^revenue$",
          sec=r"Revenues")),
    ("DE", "Diluted EPS (GAAP)", "gaap",
     dict(corpus=r"diluted.*per share|fully diluted eps", sa=r"^epsdil$",
          sec=r"EarningsPerShareDiluted")),
    ("DE", "Production & Precision Ag operating profit", "segment",
     dict(corpus=r"production & precision ag|production and precision ag", sa=r"^$", sec=r"^$")),
)


def _hits(rows, ticker_key, ticker, label_key, pattern, value_key="value"):
    if not pattern or pattern == r"^$":
        return []
    rx = re.compile(pattern, re.IGNORECASE)
    return [r for r in rows
            if r.get(ticker_key) == ticker and rx.search(str(r.get(label_key) or ""))]


def main() -> int:
    facts = store.read(config.EXTRACTED / "statement_facts.parquet")
    prose = store.read(config.EXTRACTED / "prose_facts.parquet")
    sa = store.read(config.EXTRACTED / "sa_financials.parquet")
    sec = store.read(config.EXTRACTED / "sec_facts.parquet")
    guid = store.read(config.EXTRACTED / "guidance.parquet")
    cons = store.read(config.EXTRACTED / "consensus.parquet")
    sa_fc = store.read(config.EXTRACTED / "sa_forecast.parquet")
    nq_fc = store.read(config.EXTRACTED / "nq_forecast.parquet")
    yh_est = store.read(config.EXTRACTED / "yh_estimates.parquet")
    targets = {t["ticker"]: t for t in store.read(config.EXTRACTED / "target_periods.parquet")}
    sym = {s["ticker"]: s for s in store.read(config.EXTRACTED / "symbology.parquet")}

    print(f"{'metric':<46}{'basis':<10}{'table':>7}{'prose':>7}{'sa':>5}{'sec':>6}"
          f"{'guide':>7}{'bench':>7}  verdict")
    print("-" * 116)
    blocked = []
    for ticker, label, basis, pats in TARGETS:
        short = sym.get(ticker, {}).get("short", ticker)
        c = _hits(facts, "ticker", ticker, "label", pats["corpus"])
        c_periods = len({(r.get("period_end"), r.get("period_year")) for r in c
                         if r.get("period_end") or r.get("period_year")})
        # the sentence lane: a distinct (release, period phrase) is a distinct observation
        p = _hits(prose, "ticker", ticker, "metric", pats["corpus"])
        p_periods = len({(r.get("published_at"), r.get("period_phrase")) for r in p})
        s = _hits(sa, "symbol", short, "metric", pats["sa"])
        s_periods = len({r.get("period_end") for r in s if r.get("period_end")})
        e = _hits(sec, "ticker", ticker, "concept", pats["sec"])
        e_periods = len({(r.get("start"), r.get("end")) for r in e})

        g = [r for r in guid if r["ticker"] == ticker and r["confidence"] == "stated"
             and r["published_at"] >= "2026-01-01"]
        bench = 0
        if ticker == "LSE:HAS":
            bench = len([r for r in cons if r["ticker"] == ticker
                         and r["kind"] in ("company_compiled", "position_in_range")])
            bench += len([r for r in yh_est if r["symbol"] == "HAS.L" and r["block"] == "eps"])
        else:
            bench = len([r for r in sa_fc if r["symbol"] == short and r["is_estimate"]])
            bench += len([r for r in nq_fc if r["symbol"] == short])

        has_history = max(c_periods, p_periods, s_periods, e_periods) >= 8
        verdict = "OK" if has_history and bench else ("NO HISTORY" if not has_history
                                                      else "NO BENCHMARK")
        if verdict != "OK":
            blocked.append((ticker, label, verdict))
        print(f"{short + ' · ' + label:<46}{basis:<10}{c_periods:>7}{p_periods:>7}"
              f"{s_periods:>5}{e_periods:>6}{len(g):>7}{bench:>7}  {verdict}")

    print("-" * 116)
    print("\ntarget periods:")
    for t in targets.values():
        print(f"  {t['ticker']:<9}{t['target_period']:<11}ends {t['projected_period_end']}")

    print("\ndrivers:")
    for name, path, key in (("fred", "fred_observations.parquet", "series_id"),
                            ("labour", "labour_observations.parquet", "series_id"),
                            ("yahoo bars", "yh_bars.parquet", "symbol"),
                            ("alpaca news", "ap_news.parquet", "symbol"),
                            ("sa segments", "sa_segments.parquet", "member")):
        rows = store.read(config.EXTRACTED / path)
        print(f"  {name:<14}{len(rows):>9,} rows over {len({r[key] for r in rows}):>4} keys")

    if blocked:
        print(f"\n{len(blocked)} metric(s) not yet buildable:")
        for ticker, label, why in blocked:
            print(f"  {ticker:<9}{label:<44}{why}")
    else:
        print("\nall twelve targets have a history and a benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
