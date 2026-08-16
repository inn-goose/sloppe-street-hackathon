"""extracted/sa_financials + sa_forecast + sa_segments — the vendor's typed view, long format.

The payload is column-oriented: `financialData` is a dict of metric → array, and `datekey`,
`fiscalYear`, `fiscalQuarter` are parallel arrays of the same length that key every other one.
Unpivoting it to one row per (listing, statement, period, metric) is the whole job.

## What this lane is FOR, and what it is not

It is the **second witness**. Every figure the document reader pulls out of a markdown table has an
independent counterpart here, so a disagreement is a bug report rather than a silent error — and
`feature/reconciliation.py` grades exactly that.

⛔ **It is not a substitute for the corpus, and the reason is the metric set.** Three of the twelve
targets are non-GAAP — HD's and ADI's *adjusted* diluted EPS, ADI's *adjusted* gross margin, Hays'
*pre-exceptional* operating profit and EPS. This vendor publishes `grossMargin` and `epsdil` on a
**GAAP** basis only. Reading a GAAP margin into an adjusted target is exactly the basis error the
panel's basis check exists to prevent, and it would be invisible: both numbers are real.

⚠️ **`"[PRO]"` marks a paywalled PERIOD, not a cell.** The period's dates are stated and every
figure in it reads `[PRO]`. Those periods are counted and refused rather than stored as nulls —
a row of nulls would claim the source said nothing, which is the opposite of what it says.

⚠️ **The quarterly page carries 20 periods, the annual one ~10 years.** Five years of quarters is
enough for a same-quarter seasonality estimate at n=5 and no more; that bound is real and the
model states it rather than fitting through it.
"""

from __future__ import annotations

from collections import Counter

from code.lib import config, rawstore, store
from code.vendor import devalue


def _captures() -> list[dict]:
    """Banked captures as the shape the builders below expect."""
    out = []
    for row, payload in rawstore.iter_captures("stockanalysis"):
        out.append({**row, "body": payload,
                    "captured_at": row.get("fetched_at") or "",
                    "venue": row.get("venue") or "us",
                    "period": row.get("period") or ""})
    return out

_KEYS = ("datekey", "fiscalYear", "fiscalQuarter")

#: Products whose payload is the column-oriented `financialData` block.
_STATEMENTS = {"income_statement", "balance_sheet", "cash_flow_statement", "ratios", "statistics"}


def _panel(doc: dict) -> tuple[dict, int]:
    fd = doc.get("financialData")
    if not isinstance(fd, dict):
        return {}, 0
    n = max((len(v) for v in fd.values() if isinstance(v, list)), default=0)
    return fd, n


def _titles(doc: dict) -> dict[str, str]:
    """The source's own display name per metric id — `map` is its schema, so it is read not guessed."""
    out = {}
    for entry in doc.get("map") or []:
        if isinstance(entry, dict) and entry.get("id"):
            out[entry["id"]] = entry.get("title") or entry["id"]
    return out


def build_financials(captures: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, refusals = [], []
    for cap in captures:
        if cap["product"] not in _STATEMENTS:
            continue
        doc = devalue.document(cap["body"])
        fd, n = _panel(doc)
        if not n:
            refusals.append({**{k: cap[k] for k in ("symbol", "for_ticker", "product", "period")},
                             "reason": "no financialData block"})
            continue
        titles = _titles(doc)
        dates = fd.get("datekey") or [None] * n
        years = fd.get("fiscalYear") or [None] * n
        quarters = fd.get("fiscalQuarter") or [None] * n
        currency = ((doc.get("info") or {}).get("quote") or {}).get("currency") or ""

        for i in range(n):
            # a wholly paywalled period states its dates and nothing else — count it, do not store
            values = {m: v[i] for m, v in fd.items()
                      if m not in _KEYS and isinstance(v, list) and i < len(v)}
            if values and all(v == devalue.PRO for v in values.values()):
                refusals.append({"symbol": cap["symbol"], "for_ticker": cap["for_ticker"],
                                 "product": cap["product"], "period": cap["period"],
                                 "reason": f"paywalled period {dates[i]}"})
                continue
            for metric, value in values.items():
                num = devalue.num(value)
                if num is None:
                    continue
                rows.append({
                    "symbol": cap["symbol"], "for_ticker": cap["for_ticker"],
                    "venue": cap["venue"], "statement": cap["product"],
                    "grain": cap["period"] or "current",
                    "period_end": dates[i], "fiscal_year": years[i],
                    "fiscal_quarter": quarters[i],
                    "metric": metric, "title": titles.get(metric, metric),
                    "value": num, "currency": currency,
                    "captured_at": cap["captured_at"],
                })
    return rows, refusals


#: Consensus arrays that are not themselves a forecast series.
_FC_KEYS = {"dates", "fiscalYear", "fiscalQuarter", "lastDate"}


def build_forecast(captures: list[dict]) -> list[dict]:
    """The sell-side consensus — **the benchmark the competition scores against**.

    The payload is `estimates.table.{annual,quarterly}`: parallel arrays keyed by `dates`, running
    from realised periods on the left into estimated ones on the right, with `lastDate` marking the
    index of the **last realised** period. Everything past it is a forecast, and that is read from
    the document rather than inferred from today's date.

    ⛔ **KEY ON `period_end`, NEVER ON THE VENDOR'S FISCAL LABEL.** Measured: for Home Depot the
    vendor labels the year ending Feb-2027 as **FY2027** while Home Depot itself calls that same
    year **fiscal 2026** — an off-by-one that would put every HD forecast on the wrong year, with
    real numbers throughout and nothing to flag it. The competition asks for HD `FY2026Q2`, which
    is the quarter ending 2026-08-02. The date is the only identifier both sides agree on.

    ⛔ **The basis breaks between the realised and estimated columns of the SAME array.** Measured
    on ADI `grossMargin`: the realised cells are GAAP (Q2-FY26 reads 67.33 against ADI's own
    reported GAAP 67.3 %) while the estimated cells read 72.30, which is an *adjusted* margin —
    ADI's own reported adjusted GM was 73.0 %. So one column carries two bases. `is_estimate` is
    stored on every row precisely so a consumer cannot difference across that break by accident;
    doing so would manufacture a ~5-point "margin expansion" that nobody forecast.

    ⚠️ **`[PRO]` marks a paywalled period**, and those are the far horizons (FY2027+). Refused, not
    nulled.

    ⚠️ It is a **current-state** surface: nothing says when the panel was computed, so
    `captured_at` is the only honest clock and this lane cannot be backtested point-in-time.
    """
    rows = []
    for cap in captures:
        if cap["product"] != "forecast":
            continue
        doc = devalue.document(cap["body"])
        info = doc.get("info") or {}
        currency = (info.get("quote") or {}).get("currency") or ""
        table = ((doc.get("estimates") or {}).get("table")) or {}
        for grain, block in table.items():
            if not isinstance(block, dict):
                continue
            dates = block.get("dates") or []
            years = block.get("fiscalYear") or [None] * len(dates)
            quarters = block.get("fiscalQuarter") or [None] * len(dates)
            last_actual = block.get("lastDate")
            last_actual = last_actual if isinstance(last_actual, int) else -1
            for i, period_end in enumerate(dates):
                for metric, series in block.items():
                    if metric in _FC_KEYS or not isinstance(series, list) or i >= len(series):
                        continue
                    num = devalue.num(series[i])
                    if num is None:
                        continue
                    rows.append({
                        "symbol": cap["symbol"], "for_ticker": cap["for_ticker"],
                        "grain": grain, "period_end": period_end,
                        "vendor_fiscal_year": years[i] if i < len(years) else None,
                        "vendor_fiscal_quarter": quarters[i] if i < len(quarters) else None,
                        "metric": metric, "value": num,
                        "is_estimate": i > last_actual,
                        "n_analysts": devalue.num((block.get("analysts") or [None] * len(dates))[i])
                        if i < len(block.get("analysts") or []) else None,
                        "currency": currency, "captured_at": cap["captured_at"],
                    })
    return rows


def build_segments(captures: list[dict]) -> list[dict]:
    """Revenue split by the segments and geographies the filer itself reports.

    The segment page nests its payload one level deeper than the statements —
    `data.data` — and the member names are the filer's own strings, so no mapping is applied.
    """
    rows = []
    for cap in captures:
        if cap["product"] not in ("revenue_by_segment", "revenue_by_geography"):
            continue
        doc = devalue.document(cap["body"])
        block = (doc.get("data") or {})
        panel = block.get("data")
        if not isinstance(panel, dict):
            continue
        # ⚠️ Row-oriented, unlike every other page on this source: `{grain: [{name, values:
        # [{x: date, y: value, growth}]}]}`. Assuming the column-oriented shape here returned
        # zero rows silently, which is why the grain is read from the payload rather than
        # inherited from the product.
        for grain, members in panel.items():
            if not isinstance(members, list):
                continue
            for member in members:
                if not isinstance(member, dict):
                    continue
                for point in member.get("values") or []:
                    if not isinstance(point, dict):
                        continue
                    value = devalue.num(point.get("y"))
                    if value is None:
                        continue
                    rows.append({
                        "symbol": cap["symbol"], "for_ticker": cap["for_ticker"],
                        "axis": "segment" if cap["product"].endswith("segment") else "geography",
                        "grain": grain,
                        "period_end": str(point.get("x") or "")[:10] or None,
                        "member": member.get("name") or "",
                        "value": value,
                        "yoy_change": devalue.num(point.get("change")),
                        "yoy_growth_pct": devalue.num(point.get("growth")),
                        "value_type": member.get("valueType") or block.get("valueType") or "",
                        "captured_at": cap["captured_at"],
                    })
    return rows


def main() -> int:
    captures = _captures()
    fin, refused = build_financials(captures)
    fc = build_forecast(captures)
    seg = build_segments(captures)

    store.write(config.EXTRACTED / "sa_financials.parquet", fin)
    store.write(config.EXTRACTED / "sa_forecast.parquet", fc)
    store.write(config.EXTRACTED / "sa_segments.parquet", seg)
    store.write(config.EXTRACTED / "sa_refusals.parquet", refused)

    quarters = {(r["symbol"], r["period_end"]) for r in fin
                if r["statement"] == "income_statement" and r["grain"] == "quarterly"}
    print(f"extracted/sa_financials.parquet {len(fin):,} facts  "
          f"{len({r['metric'] for r in fin})} distinct metrics  "
          f"{len(quarters):,} (symbol, quarter) income-statement periods")
    est = [r for r in fc if r["is_estimate"]]
    print(f"extracted/sa_forecast.parquet   {len(fc):,} rows "
          f"({len(est):,} forward estimates)  grains={dict(Counter(r['grain'] for r in fc))}")
    print(f"extracted/sa_segments.parquet   {len(seg):,} segment/geography rows")
    print(f"extracted/sa_refusals.parquet   {len(refused):,} refusals "
          f"({sum(1 for r in refused if 'paywalled' in r['reason'])} paywalled periods)")
    for t in ("HD", "ADI", "DE", "HAS"):
        mine = sorted({r["period_end"] for r in fin
                       if r["symbol"] == t and r["statement"] == "income_statement"
                       and r["grain"] == "quarterly" and r["period_end"]})
        f_mine = [r for r in fc if r["symbol"] == t]
        print(f"  {t:<5}{len(mine):>3} quarters {mine[0] if mine else '-'}..{mine[-1] if mine else '-'}"
              f"   forecast rows {len(f_mine):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
