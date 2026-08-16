"""raw/stockanalysis — quarterly fundamentals, segment revenue and the analyst forecast.

    data/raw/stockanalysis/stock/snapshot/week/<as-of>/<product>/<TICKER>[/<period>].json.gz

The single highest-value external lane for this task, for what it carries:

* `financials/income-statement?p=quarterly` — the reported quarterly series, already reconciled,
  so every figure the corpus reader produces has an independent second witness.
* `metrics/revenue-by-segment` — segment revenue as the filer reports it.
* `forecast` — the **sell-side consensus**, which is the benchmark the competition scores against
  and which the frozen corpus does not contain for HD/ADI/DE at all.

## The document grammar, and the two states that are not failures

Every page has a SvelteKit `__data.json` twin. The envelope is `{"nodes":[…]}` and its state is
read **by position from the end**: the last node is the page, anything before it is the entity it
hangs off. A node before the last typed `error` means the source does not carry the ticker; the
last node typed `error` means the ticker exists but this page does not. Neither is an exception —
a ticker with no forecast page is information about coverage.

⚠️ **Hays is addressed differently.** US listings are `/stocks/<TICKER>/`; a London line is
`/quote/lon/<TICKER>/`. Guessing wrong returns a valid not-found envelope that reads as
"no coverage".

⛔ Banked as the JSON the site sent, wrapped in a stamp. The flattened `devalue` graph is
hydrated in `extracted/`, never here.
"""

from __future__ import annotations

import json
import time

from code.lib import config, rawstore

PROVIDER = "stockanalysis"
_STOCK = "https://stockanalysis.com/stocks/{ticker}/{page}__data.json"
_QUOTE = "https://stockanalysis.com/quote/{venue}/{ticker}/{page}__data.json"
_MIN_INTERVAL = 0.7
_TIMEOUT = 45

OK, NO_PAGE, NO_TICKER = "ok", "no_page", "no_ticker"

PRODUCTS: tuple[tuple[str, str, tuple[str | None, ...]], ...] = (
    ("forecast", "forecast", (None,)),
    ("statistics", "statistics", (None,)),
    ("income_statement", "financials/income-statement", ("quarterly", "annual")),
    ("balance_sheet", "financials/balance-sheet", ("quarterly", "annual")),
    ("cash_flow_statement", "financials/cash-flow-statement", ("quarterly", "annual")),
    ("ratios", "financials/ratios", ("quarterly", "annual")),
    ("revenue_by_segment", "metrics/revenue-by-segment", (None,)),
    ("revenue_by_geography", "metrics/revenue-by-geography", (None,)),
    ("dividend", "dividend", (None,)),
    ("overview", "", (None,)),
)

SYMBOLS: tuple[tuple[str, str, str | None], ...] = (
    ("HD", "HD", None), ("ADI", "ADI", None), ("DE", "DE", None), ("LSE:HAS", "HAS", "lon"),
    ("HD", "LOW", None), ("HD", "TSCO", None),
    ("ADI", "TXN", None), ("ADI", "MCHP", None), ("ADI", "NXPI", None),
    ("DE", "AGCO", None), ("DE", "CNH", None), ("DE", "CAT", None),
    ("LSE:HAS", "PAGE", "lon"), ("LSE:HAS", "RWA", "lon"), ("LSE:HAS", "STEM", "lon"),
)


def _session():
    import curl_cffi.requests as cr

    return cr.Session(impersonate="chrome")


def _state(text: str) -> str:
    try:
        nodes = json.loads(text).get("nodes")
    except (ValueError, AttributeError):
        return NO_TICKER
    if not isinstance(nodes, list) or not nodes:
        return NO_TICKER
    if any((n or {}).get("type") == "error" for n in nodes[:-1]):
        return NO_TICKER
    return NO_PAGE if (nodes[-1] or {}).get("type") == "error" else OK


def _url(ticker: str, venue: str | None, page: str) -> str:
    tail = f"{page}/" if page else ""
    return (_QUOTE.format(venue=venue, ticker=ticker, page=tail) if venue
            else _STOCK.format(ticker=ticker, page=tail))


def fetch() -> dict:
    config.ensure_dirs()
    session = _session()
    as_of = rawstore.now()
    ledger = rawstore.Ledger(PROVIDER)
    banked = refused = disk = 0
    last = 0.0
    dead: set[str] = set()

    print(f"stockanalysis: {len(SYMBOLS)} symbols x {len(PRODUCTS)} products -> "
          f"raw/stockanalysis/stock/snapshot/week/{rawstore.segment_of(as_of)}")
    for our, ticker, venue in SYMBOLS:
        if ticker in dead:
            continue
        for product, page, periods in PRODUCTS:
            for period in periods:
                wait = _MIN_INTERVAL - (time.monotonic() - last)
                if wait > 0:
                    time.sleep(wait)
                last = time.monotonic()
                url = _url(ticker, venue, page)
                try:
                    resp = session.get(url, params={"p": period} if period else None,
                                       timeout=_TIMEOUT)
                    text, status = resp.text, resp.status_code
                    state = _state(text) if status == 200 else "http_error"
                    served = resp.headers.get("Date")
                except Exception as exc:  # noqa: BLE001
                    text, status, state, served = "", 0, "transport_error", None
                    state = f"transport_error:{type(exc).__name__}"

                if state != OK:
                    refused += 1
                    ledger.add(provider=PROVIDER, product=product, period=period or "",
                               symbol=ticker, venue=venue or "us", for_ticker=our,
                               state=state, status=status,
                               fetched_at=as_of.isoformat(timespec="seconds"))
                    if state == NO_TICKER:
                        dead.add(ticker)
                    print(f"  -- {ticker:<6}{product:<22}{period or '':<10}{state}")
                    if ticker in dead:
                        break
                    continue

                directory = rawstore.capture_dir(PROVIDER, "snapshot", "week", as_of=as_of,
                                                 domain="stock")
                parts = [product, ticker] + ([period] if period else [])
                leaf = rawstore.leaf_name(*parts, ext="json", compression="gz")
                body = rawstore.envelope(text, fetched_at=as_of, served_at=served,
                                         request={"url": url, "p": period})
                written = rawstore.bank(directory / leaf, body)
                disk += written
                banked += 1
                ledger.add(provider=PROVIDER, product=product, period=period or "",
                           symbol=ticker, venue=venue or "us", for_ticker=our, state="ok",
                           path=str((directory / leaf).relative_to(config.RAW)),
                           source_bytes=len(text), disk_bytes=written,
                           fetched_at=as_of.isoformat(timespec="seconds"))
                print(f"  ok {ticker:<6}{product:<22}{period or '':<10}{len(text) // 1024:>4} KiB")
            if ticker in dead:
                break

    ledger.write()
    return {"banked": banked, "refused": refused, "disk": disk, "dead": sorted(dead)}


def main() -> int:
    stats = fetch()
    print(f"\nraw/stockanalysis  {stats['banked']} captures  {stats['disk'] / 1e6:.1f} MB on disk "
          f"({stats['refused']} refused)")
    if stats["dead"]:
        print(f"  not carried by the source: {', '.join(stats['dead'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
