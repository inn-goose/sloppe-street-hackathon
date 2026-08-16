"""raw/alpaca — daily bars and the Benzinga news tape, back to 2015. Verbatim JSON pages.

    data/raw/alpaca/stock/bars/day/<as-of>/<SYM>/page-0000.json.gz
    data/raw/alpaca/stock/news/event/<as-of>/<SYM>/page-0000.json.gz

## What this adds that nothing else in the store has

A **dated news tape**: Benzinga from 2015 with a per-article `symbols[]` list and a publication
timestamp. That is the attention leg (#125) at a real cadence — and an article *count* is a count,
so it survives the no-NLP boundary intact, unlike a sentiment score.

⛔ **US only.** Alpaca carries no LSE line, so Hays is absent by construction — the same 3-of-4
shape as SEC. Asking for `HAS` would return **Hasbro**, so it is not in the symbol list.

⚠️ **A raw article count measures TAGGING, not attention.** A market-wrap piece tags dozens of
symbols, so the usable count is the single-symbol subset or an inverse-tag weight. The tag list is
banked per article precisely so that choice stays with the consumer.

⛔ **The feed decides the backfill, and the default was costing 76 % of the history.** Measured on
this credential, same request, same 2015 start:

    feed=sip    2,669 bars, first 2016-01-04     <- what the key is entitled to
    feed=iex    1,518 bars, first 2020-07-27     <- what `iex` returns
    feed=boats    173 bars, first 2024-09-24
    feed=otc      403

`next_page_token` was `null` in every case, so the short answer looked complete — a truncated
backfill that reports itself as exhausted is exactly the failure a page-count check cannot see.
**2016-01-04 is Alpaca's own data floor**, not a window we chose, so that is the deepest this
provider goes; Yahoo carries 2014→ and remains the long series.

## Credentials

`ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY`, read by environment-variable **name** only. Never
read into output, never written to a file, never committed. Unset means refuse.
"""

from __future__ import annotations

import json
import time

from code.lib import config, rawstore

PROVIDER = "alpaca"
_NEWS = "https://data.alpaca.markets/v1beta1/news"
_BARS = "https://data.alpaca.markets/v2/stocks/bars"

_MIN_INTERVAL = 0.35
_TIMEOUT = 60
#: Measured against this endpoint: `limit=100` answers 400; 50 is the working ceiling.
_NEWS_PAGE = 50
_BARS_PAGE = 10000
_MAX_PAGES = 400

SYMBOLS = ("HD", "ADI", "DE", "LOW", "TXN", "MCHP", "AGCO", "CAT", "CNH")
#: Ask for more than the source can hold and let it answer with its own floor — a request that
#: stops at the horizon we assume would never reveal that the floor is elsewhere.
START = "2010-01-01T00:00:00Z"
#: Entitlement is measured, not assumed. See the module docstring for the four-feed comparison.
FEED = config.setting("alpaca", "feed", default="sip")


def _session():
    import curl_cffi.requests as cr

    key = config.secret("ALPACA_API_KEY_ID")
    sec = config.secret("ALPACA_API_SECRET_KEY")
    if not key or not sec:
        raise SystemExit(
            "alpaca: ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set. This fetcher will not "
            "proceed without them and will not substitute any other credential.")
    s = cr.Session(impersonate="chrome")
    s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec,
                      "Accept": "application/json"})
    return s


def _paged(s, url: str, params: dict, page_key: str, last: list[float]):
    token = None
    for page in range(_MAX_PAGES):
        wait = _MIN_INTERVAL - (time.monotonic() - last[0])
        if wait > 0:
            time.sleep(wait)
        last[0] = time.monotonic()
        query = dict(params)
        if token:
            query["page_token"] = token
        resp = s.get(url, params=query, timeout=_TIMEOUT)
        if resp.status_code != 200:
            yield page, resp, 0
            return
        try:
            body = json.loads(resp.text)
        except ValueError:
            yield page, resp, 0
            return
        payload = body.get(page_key)
        count = (len(payload) if isinstance(payload, list)
                 else sum(len(v) for v in (payload or {}).values()))
        yield page, resp, count
        token = body.get("next_page_token")
        if not token:
            return


def fetch() -> dict:
    config.ensure_dirs()
    s = _session()
    as_of = rawstore.now()
    ledger = rawstore.Ledger(PROVIDER)
    last = [0.0]
    banked = refused = disk = items = 0

    print(f"alpaca: {len(SYMBOLS)} US symbols from {START[:10]} -> "
          f"raw/alpaca/stock/<bars|news>/…/{rawstore.segment_of(as_of)}  "
          f"(Hays absent — no LSE line on this venue)")

    for symbol in SYMBOLS:
        for product, url, params, key, data_type, granularity in (
            ("bars_1d", _BARS,
             {"symbols": symbol, "timeframe": "1Day", "start": START, "limit": _BARS_PAGE,
              "adjustment": "raw", "feed": FEED}, "bars", "bars", "day"),
            ("news", _NEWS,
             {"symbols": symbol, "start": START, "limit": _NEWS_PAGE, "sort": "asc",
              "include_content": "false"}, "news", "news", "event"),
        ):
            pages = total = 0
            directory = rawstore.capture_dir(PROVIDER, data_type, granularity, as_of=as_of,
                                             domain="stock")
            for page, resp, count in _paged(s, url, params, key, last):
                if resp.status_code != 200:
                    refused += 1
                    ledger.add(provider=PROVIDER, product=product, symbol=symbol, page=page,
                               state="http_error", status=resp.status_code,
                               note=resp.text[:120],
                               fetched_at=as_of.isoformat(timespec="seconds"))
                    break
                leaf = rawstore.leaf_name(symbol, f"page-{page:04d}", ext="json",
                                          compression="gz")
                written = rawstore.bank(directory / leaf,
                                        rawstore.envelope(resp.text, fetched_at=as_of,
                                                          served_at=resp.headers.get("Date"),
                                                          request={"url": url,
                                                                   "symbol": symbol,
                                                                   "page": page}))
                disk += written
                banked += 1
                pages += 1
                total += count
                items += count
                ledger.add(provider=PROVIDER, product=product, symbol=symbol, page=page,
                           state="ok", n_items=count,
                           path=str((directory / leaf).relative_to(config.RAW)),
                           source_bytes=len(resp.text), disk_bytes=written,
                           fetched_at=as_of.isoformat(timespec="seconds"))
            print(f"  {'ok ' if total else '-- '}{symbol:<6}{product:<9}{total:>7,} items "
                  f"over {pages} page(s)")

    ledger.write()
    return {"banked": banked, "refused": refused, "disk": disk, "items": items}


def main() -> int:
    stats = fetch()
    print(f"\nraw/alpaca  {stats['banked']} pages  {stats['items']:,} items  "
          f"{stats['disk'] / 1e6:.1f} MB on disk  ({stats['refused']} refused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
