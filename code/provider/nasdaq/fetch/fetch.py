"""raw/nasdaq — the analyst REVISION panel. Keyless, verbatim JSON.

    data/raw/nasdaq/stock/snapshot/day/<as-of>/<product>/<SYM>.json.gz

## The one leg nothing else in this store carries

stockanalysis and Yahoo both give a consensus *level*. Neither gives its **drift**: how many
analysts raised and how many lowered over the last week and month, which is the §C
revision-diffusion family (#30–#33) and the best-documented pre-print predictor of the surprise's
sign. `api.nasdaq.com/api/analyst/<SYM>/earnings-forecast` publishes exactly that, so **one
capture already carries direction** with no snapshot history to accumulate first.

⚠️ **US only.** Hays is not on this surface, so the revision leg covers three of four names and
Hays' stays unsourced. Asking for `HAS` would return **Hasbro** with a complete payload.

⛔ **Forward-only and current-state.** No dated route: the panel is as-of the capture and nothing
in the body says otherwise, so this can never be backtested. A today's-bar input, like the
consensus itself.

⚠️ **Absence lives inside a 200**: `{"data": null, "status": {"rCode": 400 …}}`. A null `data` is
refused rather than banked.
"""

from __future__ import annotations

import json
import time

from code.lib import config, rawstore

PROVIDER = "nasdaq"
_BASE = "https://api.nasdaq.com/api"
_MIN_INTERVAL = 0.6
_TIMEOUT = 45

PRODUCTS = (
    ("earnings_forecast", "/analyst/{sym}/earnings-forecast"),
    ("price_target", "/analyst/{sym}/price-target"),
    ("earnings_surprise", "/company/{sym}/earnings-surprise"),
    ("earnings_date", "/company/{sym}/earnings-date"),
    ("eps_forecast", "/analyst/{sym}/eps-forecast"),
    ("revenue_eps", "/company/{sym}/revenue-eps"),
    ("financials_quarterly", "/company/{sym}/financials?frequency=2"),
)

SYMBOLS = (("HD", "HD"), ("ADI", "ADI"), ("DE", "DE"),
           ("HD", "LOW"), ("ADI", "TXN"), ("ADI", "MCHP"),
           ("DE", "AGCO"), ("DE", "CAT"), ("DE", "CNH"))


def _session():
    import curl_cffi.requests as cr

    s = cr.Session(impersonate="chrome")
    s.headers.update({"Accept": "application/json", "Referer": "https://www.nasdaq.com/"})
    return s


def fetch() -> dict:
    config.ensure_dirs()
    s = _session()
    as_of = rawstore.now()
    ledger = rawstore.Ledger(PROVIDER)
    banked = refused = disk = 0
    last = 0.0

    print(f"nasdaq: {len(SYMBOLS)} symbols x {len(PRODUCTS)} products -> "
          f"raw/nasdaq/stock/snapshot/day/{rawstore.segment_of(as_of)}")
    for our, sym in SYMBOLS:
        for product, path in PRODUCTS:
            wait = _MIN_INTERVAL - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            url = _BASE + path.format(sym=sym)
            try:
                resp = s.get(url, timeout=_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                refused += 1
                ledger.add(provider=PROVIDER, product=product, symbol=sym, for_ticker=our,
                           state=f"transport_error:{type(exc).__name__}",
                           fetched_at=as_of.isoformat(timespec="seconds"))
                print(f"  -- {sym:<6}{product:<22}{type(exc).__name__}")
                continue

            body = None
            if resp.status_code == 200:
                try:
                    body = json.loads(resp.text)
                except ValueError:
                    body = None
            payload = (body or {}).get("data")
            if not payload:
                note = ((body or {}).get("status") or {}).get("bCodeMessage") or "empty data"
                refused += 1
                ledger.add(provider=PROVIDER, product=product, symbol=sym, for_ticker=our,
                           state="empty", note=str(note)[:80], status=resp.status_code,
                           fetched_at=as_of.isoformat(timespec="seconds"))
                print(f"  -- {sym:<6}{product:<22}{str(note)[:44]}")
                continue

            directory = rawstore.capture_dir(PROVIDER, "snapshot", "day", as_of=as_of,
                                             domain="stock")
            leaf = rawstore.leaf_name(product, sym, ext="json", compression="gz")
            written = rawstore.bank(directory / leaf,
                                    rawstore.envelope(resp.text, fetched_at=as_of,
                                                      served_at=resp.headers.get("Date"),
                                                      request={"url": url}))
            disk += written
            banked += 1
            ledger.add(provider=PROVIDER, product=product, symbol=sym, for_ticker=our,
                       state="ok", path=str((directory / leaf).relative_to(config.RAW)),
                       source_bytes=len(resp.text), disk_bytes=written,
                       fetched_at=as_of.isoformat(timespec="seconds"))
            print(f"  ok {sym:<6}{product:<22}{len(resp.text) // 1024:>4} KiB")

    ledger.write()
    return {"banked": banked, "refused": refused, "disk": disk}


def main() -> int:
    stats = fetch()
    print(f"\nraw/nasdaq  {stats['banked']} captures  {stats['disk'] / 1e6:.2f} MB on disk  "
          f"({stats['refused']} refused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
