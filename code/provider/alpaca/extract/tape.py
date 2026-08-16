"""extracted/ap_bars + ap_news — the SIP daily tape and the Benzinga news tape.

⚠️ **A raw article count measures TAGGING, not attention.** A market-wrap piece tags dozens of
symbols, so `n_symbols` rides on every article and the usable count is the single-symbol subset or
an inverse-tag weight. That choice stays with the consumer; this lane only records the tag list's
size and the symbols themselves.

⛔ **No headline text is read for meaning.** The headline is stored verbatim so a count, a diff or
a template match remains possible, but nothing here scores it — a sentiment reading would be an
inference, and inferences do not belong in a faithful view.
"""

from __future__ import annotations

from collections import Counter

from code.lib import config, rawstore, store


def build() -> tuple[list[dict], list[dict]]:
    bars, news = [], []
    for meta, body in rawstore.iter_captures("alpaca"):
        symbol = meta["symbol"]
        base = {"symbol": symbol, "captured_at": meta.get("fetched_at") or ""}
        if meta["product"] == "bars_1d":
            for row in (body.get("bars") or {}).get(symbol) or []:
                if not isinstance(row, dict):
                    continue
                bars.append({**base, "date": str(row.get("t") or "")[:10],
                             "open": row.get("o"), "high": row.get("h"), "low": row.get("l"),
                             "close": row.get("c"), "volume": row.get("v"),
                             "vwap": row.get("vw"), "trade_count": row.get("n")})
        elif meta["product"] == "news":
            for article in body.get("news") or []:
                if not isinstance(article, dict):
                    continue
                symbols = article.get("symbols") or []
                news.append({**base, "article_id": article.get("id"),
                             "created_at": str(article.get("created_at") or "")[:19],
                             "updated_at": str(article.get("updated_at") or "")[:19],
                             "date": str(article.get("created_at") or "")[:10],
                             "source": article.get("source"), "author": article.get("author"),
                             "headline": (article.get("headline") or "")[:300],
                             "summary": (article.get("summary") or "")[:400],
                             "n_symbols": len(symbols),
                             "symbols": ",".join(symbols)[:200],
                             "is_single_symbol": len(symbols) == 1})
    return bars, news


def main() -> int:
    bars, news = build()
    store.write(config.EXTRACTED / "ap_bars.parquet", bars)
    store.write(config.EXTRACTED / "ap_news.parquet", news)
    single = sum(1 for a in news if a["is_single_symbol"])
    dates = sorted(a["date"] for a in news if a["date"])
    print(f"extracted/ap_bars.parquet {len(bars):,} bars over "
          f"{len({b['symbol'] for b in bars})} symbols")
    print(f"extracted/ap_news.parquet {len(news):,} articles "
          f"({single:,} single-symbol, {single / max(len(news), 1):.0%})  "
          f"{dates[0] if dates else '-'}..{dates[-1] if dates else '-'}")
    per = Counter(a["symbol"] for a in news)
    print(f"  articles per symbol: {dict(per.most_common(5))}")
    tags = Counter(a["n_symbols"] for a in news)
    print(f"  tag-count distribution (1,2,3,…): "
          f"{[tags.get(i, 0) for i in range(1, 7)]}  max tags on one article: "
          f"{max((a['n_symbols'] for a in news), default=0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
