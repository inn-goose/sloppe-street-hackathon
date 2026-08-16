"""extracted/yh_* — everything the banked Yahoo captures carry.

Eleven lanes out of two products. A first version read **6 of the 33 modules** `quoteSummary`
returns and left 27 on disk unread; the whole point of asking for every module in one request is
that none of them costs anything extra, so none of them is left.

⛔ **`meta.currency` rides on every bar row.** Measured: `HAS.L` quotes in **`GBp`** — pence —
while `RAND.AS` quotes EUR and the futures quote `USX`. A GBp series differenced against a GBP one
is out by 100×, and because a *return* is scale-free the error hides completely until a level is
used. The competition submits Hays EPS in pence, so this unit is live in the output too.

⛔ **`adjclose` is a RESTATEMENT, not an observation.** It moves with every later split and
dividend, so it is stored as what Yahoo says today and never treated as point-in-time. `close`
plus the dated action streams are the immutable facts.

⚠️ **`earningsTrend` is a current-state surface.** It states no as-of, so `captured_at` is the
only honest clock. It is the second, independent witness to the stockanalysis consensus — two
vendors disagreeing about the bar is information about how firm the bar is.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from code.lib import config, rawstore, store


def _iso(epoch) -> str | None:
    if not isinstance(epoch, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _num(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return _num(value.get("raw"))
    return None


def build() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {k: [] for k in (
        "bars", "actions", "estimates", "earnings_history", "calendar", "key_stats",
        "statements", "recommendations", "rating_actions", "insider", "ownership")}

    for row, body in rawstore.iter_captures("yahoo"):
        base = {"symbol": row["symbol"], "role": row.get("role") or "",
                "for_ticker": row.get("for_ticker") or "",
                "captured_at": row.get("fetched_at") or ""}

        if row["product"] == "bars":
            result = ((body.get("chart") or {}).get("result") or [{}])[0] or {}
            meta = result.get("meta") or {}
            currency = meta.get("currency") or ""
            stamps = result.get("timestamp") or []
            quote = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
            adj = (((result.get("indicators") or {}).get("adjclose") or [{}])[0] or {}) \
                .get("adjclose")
            for i, epoch in enumerate(stamps):
                close = _num((quote.get("close") or [None] * len(stamps))[i])
                if close is None:
                    continue
                out["bars"].append({
                    **base, "date": _iso(epoch), "currency": currency,
                    "open": _num((quote.get("open") or [None] * len(stamps))[i]),
                    "high": _num((quote.get("high") or [None] * len(stamps))[i]),
                    "low": _num((quote.get("low") or [None] * len(stamps))[i]),
                    "close": close,
                    "adjclose": _num(adj[i]) if isinstance(adj, list) and i < len(adj) else None,
                    "volume": _num((quote.get("volume") or [None] * len(stamps))[i]),
                })
            for kind in ("dividends", "splits"):
                for _k, event in ((result.get("events") or {}).get(kind) or {}).items():
                    if isinstance(event, dict):
                        out["actions"].append({
                            **base, "kind": kind[:-1], "date": _iso(event.get("date")),
                            "currency": currency, "amount": _num(event.get("amount")),
                            "numerator": _num(event.get("numerator")),
                            "denominator": _num(event.get("denominator")),
                            "ratio": event.get("splitRatio")})
            continue

        # ---- quoteSummary
        res = ((body.get("quoteSummary") or {}).get("result") or [{}])[0] or {}

        for trend in ((res.get("earningsTrend") or {}).get("trend") or []):
            if not isinstance(trend, dict):
                continue
            row_base = {**base, "period": trend.get("period"),
                        "period_end": trend.get("endDate"),
                        "growth": _num(trend.get("growth"))}
            for block, prefix in (("earningsEstimate", "eps"), ("revenueEstimate", "revenue")):
                for field, value in (trend.get(block) or {}).items():
                    v = _num(value)
                    if v is not None:
                        out["estimates"].append({**row_base, "block": prefix,
                                                 "metric": field, "value": v})
            for block in ("epsTrend", "epsRevisions"):
                for field, value in (trend.get(block) or {}).items():
                    v = _num(value)
                    if v is not None:
                        out["estimates"].append({**row_base, "block": block,
                                                 "metric": field, "value": v})

        for q in ((res.get("earningsHistory") or {}).get("history") or []):
            if isinstance(q, dict):
                out["earnings_history"].append({
                    **base, "period": q.get("period"), "quarter_end": q.get("quarter"),
                    "eps_actual": _num(q.get("epsActual")),
                    "eps_estimate": _num(q.get("epsEstimate")),
                    "eps_difference": _num(q.get("epsDifference")),
                    "surprise_pct": _num(q.get("surprisePercent"))})

        events = res.get("calendarEvents") or {}
        earnings = events.get("earnings") or {}
        dates = earnings.get("earningsDate") or []
        out["calendar"].append({
            **base, "next_earnings_date": _iso(dates[0]) if dates else None,
            "next_earnings_date_end": _iso(dates[-1]) if len(dates) > 1 else None,
            "is_estimate": bool(earnings.get("isEarningsDateEstimate")),
            "earnings_average": _num(earnings.get("earningsAverage")),
            "earnings_low": _num(earnings.get("earningsLow")),
            "earnings_high": _num(earnings.get("earningsHigh")),
            "revenue_average": _num(earnings.get("revenueAverage")),
            "ex_dividend_date": _iso(events.get("exDividendDate")),
            "dividend_date": _iso(events.get("dividendDate"))})

        key = res.get("defaultKeyStatistics") or {}
        fin = res.get("financialData") or {}
        price = res.get("price") or {}
        qt = res.get("quoteType") or {}
        profile = res.get("summaryProfile") or res.get("assetProfile") or {}
        summary = res.get("summaryDetail") or {}
        out["key_stats"].append({
            **base,
            "currency": price.get("currency") or fin.get("financialCurrency") or "",
            "financial_currency": fin.get("financialCurrency") or "",
            # ⚠️ `price.longName` is null on some listings even when the module is present
            # (measured: HD and ADI). `quoteType` carries the name, and the symbology check
            # depends on it — `HAS` alone would return Hasbro.
            "long_name": (price.get("longName") or qt.get("longName")
                          or price.get("shortName") or qt.get("shortName") or ""),
            "exchange": price.get("exchangeName") or qt.get("exchange") or "",
            "sector": profile.get("sector") or "", "industry": profile.get("industry") or "",
            "employees": _num(profile.get("fullTimeEmployees")),
            "shares_outstanding": _num(key.get("sharesOutstanding")),
            "float_shares": _num(key.get("floatShares")),
            "shares_short": _num(key.get("sharesShort")),
            "short_ratio": _num(key.get("shortRatio")),
            "held_by_insiders": _num(key.get("heldPercentInsiders")),
            "held_by_institutions": _num(key.get("heldPercentInstitutions")),
            "forward_eps": _num(key.get("forwardEps")),
            "trailing_eps": _num(key.get("trailingEps")),
            "book_value": _num(key.get("bookValue")),
            "profit_margins": _num(key.get("profitMargins")),
            "enterprise_value": _num(key.get("enterpriseValue")),
            "market_cap": _num(price.get("marketCap")) or _num(summary.get("marketCap")),
            "dividend_yield": _num(summary.get("dividendYield")),
            "payout_ratio": _num(summary.get("payoutRatio")),
            "beta": _num(summary.get("beta")),
            "target_mean_price": _num(fin.get("targetMeanPrice")),
            "target_high_price": _num(fin.get("targetHighPrice")),
            "target_low_price": _num(fin.get("targetLowPrice")),
            "number_of_analyst_opinions": _num(fin.get("numberOfAnalystOpinions")),
            "recommendation_mean": _num(fin.get("recommendationMean")),
            "revenue_growth": _num(fin.get("revenueGrowth")),
            "earnings_growth": _num(fin.get("earningsGrowth")),
            "gross_margins": _num(fin.get("grossMargins")),
            "operating_margins": _num(fin.get("operatingMargins")),
            "ebitda_margins": _num(fin.get("ebitdaMargins")),
            "total_revenue": _num(fin.get("totalRevenue")),
            "total_cash": _num(fin.get("totalCash")),
            "total_debt": _num(fin.get("totalDebt")),
            "free_cashflow": _num(fin.get("freeCashflow")),
            "operating_cashflow": _num(fin.get("operatingCashflow"))})

        # ---- the statement histories: 27 modules were previously banked and never read
        for module, grain in (("incomeStatementHistory", "annual"),
                              ("incomeStatementHistoryQuarterly", "quarterly"),
                              ("balanceSheetHistory", "annual"),
                              ("balanceSheetHistoryQuarterly", "quarterly"),
                              ("cashflowStatementHistory", "annual"),
                              ("cashflowStatementHistoryQuarterly", "quarterly")):
            node = res.get(module) or {}
            statements = (node.get("incomeStatementHistory") or node.get("balanceSheetStatements")
                          or node.get("cashflowStatements") or [])
            for statement in statements:
                if not isinstance(statement, dict):
                    continue
                end = _iso(statement.get("endDate")) or statement.get("endDate")
                for field, value in statement.items():
                    v = _num(value)
                    if v is not None and field not in ("endDate", "maxAge"):
                        out["statements"].append({
                            **base, "module": module, "grain": grain, "period_end": end,
                            "metric": field, "value": v})

        for r in ((res.get("recommendationTrend") or {}).get("trend") or []):
            if isinstance(r, dict):
                out["recommendations"].append({
                    **base, "period": r.get("period"),
                    "strong_buy": _num(r.get("strongBuy")), "buy": _num(r.get("buy")),
                    "hold": _num(r.get("hold")), "sell": _num(r.get("sell")),
                    "strong_sell": _num(r.get("strongSell"))})

        for a in ((res.get("upgradeDowngradeHistory") or {}).get("history") or []):
            if isinstance(a, dict):
                out["rating_actions"].append({
                    **base, "date": _iso(a.get("epochGradeDate")),
                    "firm": a.get("firm"), "to_grade": a.get("toGrade"),
                    "from_grade": a.get("fromGrade"), "action": a.get("action")})

        for t in ((res.get("insiderTransactions") or {}).get("transactions") or []):
            if isinstance(t, dict):
                out["insider"].append({
                    **base, "date": _iso(t.get("startDate")),
                    "filer": t.get("filerName"), "relation": t.get("filerRelation"),
                    "transaction": t.get("transactionText"),
                    "shares": _num(t.get("shares")), "value": _num(t.get("value")),
                    "ownership": t.get("ownership")})

        for h in ((res.get("institutionOwnership") or {}).get("ownershipList") or []):
            if isinstance(h, dict):
                out["ownership"].append({
                    **base, "report_date": _iso(h.get("reportDate")),
                    "organization": h.get("organization"),
                    "pct_held": _num(h.get("pctHeld")), "position": _num(h.get("position")),
                    "value": _num(h.get("value")), "pct_change": _num(h.get("pctChange"))})

    return out


def main() -> int:
    lanes = build()
    for name, rows in lanes.items():
        store.write(config.EXTRACTED / f"yh_{name}.parquet", rows)
    ccy = Counter(r["currency"] for r in lanes["bars"])
    print(f"extracted/yh_bars.parquet             {len(lanes['bars']):,} bars over "
          f"{len({r['symbol'] for r in lanes['bars']})} symbols  {dict(ccy.most_common(5))}")
    for name in ("actions", "estimates", "earnings_history", "calendar", "key_stats",
                 "statements", "recommendations", "rating_actions", "insider", "ownership"):
        print(f"extracted/yh_{name + '.parquet':<24} {len(lanes[name]):>8,} rows")
    for t in ("HD", "ADI", "DE", "HAS.L"):
        b = [r for r in lanes["bars"] if r["symbol"] == t]
        s = next((r for r in lanes["key_stats"] if r["symbol"] == t), None)
        c = next((r for r in lanes["calendar"] if r["symbol"] == t), None)
        if b:
            print(f"  {t:<7}{len(b):>6,} bars {b[0]['date']}..{b[-1]['date']}  "
                  f"{b[0]['currency']:<4} shares {(s or {}).get('shares_outstanding') or 0:,.0f}"
                  f"  next report {(c or {}).get('next_earnings_date')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
