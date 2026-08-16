"""raw/yahoo — chart bars and the whole quoteSummary, banked as verbatim JSON.

    data/raw/yahoo/stock/bars/day/<as-of>/<SYM>.json.gz
    data/raw/yahoo/stock/snapshot/day/<as-of>/<SYM>.json.gz

⛔ **The library is the transport, never the parser.** `query2.finance.yahoo.com` discriminates on
the TLS handshake — plain `requests` answers 429 on every path — and `quoteSummary` needs a
cookie+crumb pair from another host. `yfinance.data.YfData` owns both. What lands on disk is the
bytes Yahoo sent, wrapped in a stamp envelope and gzipped; parsing at ingest would make the store
a record of this version of yfinance's opinion.

⛔ **A miss is not always a status code.** `quoteSummary` answers 404 with a null result for a
ticker Yahoo does not carry; the chart answers 200 with an error member; and a weekend-only window
answers 200 with `result[0]` present and **no `timestamp` key at all** — 1 kB of nothing. Each
reader recognises the shape it asked for, and anything else is refused rather than banked, because
a wrong capture in an immutable store satisfies every existence check forever.

⚠️ **The as-of is the fetch instant, not a data date.** Nothing in a `quoteSummary` body says when
the analyst panel was computed. The chart carries dated bars, but the *capture* is still a
snapshot of what Yahoo currently says those bars were — `adjclose` is restated on every later
split and dividend.

## Why this provider

`quoteSummary` carries `earningsTrend` — the sell-side consensus for the quarter being forecast —
plus `earningsHistory`, `calendarEvents` and the quarterly statements, in one request. It is also
the **only** external surface here that reaches all four names: Hays is `HAS.L`.
"""

from __future__ import annotations

import json
import time

from code.lib import config, rawstore

PROVIDER = "yahoo"

_QUOTE_SUMMARY = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
_OPTION_CHAIN = "https://query2.finance.yahoo.com/v7/finance/options/{ticker}"
_CHART = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"

_MIN_INTERVAL = 0.55
_TIMEOUT = 30

QUOTE_SUMMARY_MODULES = (
    "assetProfile", "balanceSheetHistory", "balanceSheetHistoryQuarterly",
    "calendarEvents", "cashflowStatementHistory", "cashflowStatementHistoryQuarterly",
    "defaultKeyStatistics", "earnings", "earningsHistory", "earningsTrend",
    "esgScores", "financialData", "fundOwnership", "fundPerformance", "fundProfile",
    "incomeStatementHistory", "incomeStatementHistoryQuarterly", "indexTrend",
    "industryTrend", "insiderHolders", "insiderTransactions", "institutionOwnership",
    "majorDirectHolders", "majorHoldersBreakdown", "netSharePurchaseActivity", "price",
    "quoteType", "recommendationTrend", "secFilings", "sectorTrend", "summaryDetail",
    "summaryProfile", "topHoldings", "upgradeDowngradeHistory",
)

#: ⛔ Hays is `HAS.L`. A bare `HAS` is **Hasbro** and answers with a full, plausible payload —
#: verified on capture against `longName`, which reads `Hays plc`, and currency, which reads `GBp`.
TARGETS = {"HD": "HD", "ADI": "ADI", "DE": "DE", "LSE:HAS": "HAS.L"}

#: ⚠️ A peer ticker is authored and a wrong one answers plausibly. Hays' set is the deepest on
#: purpose: it has no SEC, no Alpaca and no Nasdaq coverage, so the staffing cycle is the only
#: cross-sectional read available for a quarter of the competition score.
PEERS = {
    "HD": ["LOW", "SHW", "FND", "BLDR", "TSCO", "WSM"],
    "ADI": ["TXN", "MCHP", "NXPI", "STM", "ON", "SWKS", "MPWR", "IFX.DE"],
    "DE": ["AGCO", "CNH", "CAT", "TITN", "LNN", "KUBTY", "TRMB"],
    "LSE:HAS": ["PAGE.L", "STEM.L", "RWA.L", "RAND.AS", "ADEN.SW", "MAN", "RHI", "KFY"],
}

#: Hays reports **actual** and **LFL** growth side by side; the wedge between them is translation,
#: so a GBP cross per region is a direct input to its net-fee bridge. Deere states `+3.0 %`
#: currency translation in its own FY26 outlook — these price that statement.
FX = ["EURGBP=X", "AUDGBP=X", "GBPUSD=X", "EURUSD=X", "JPY=X", "CAD=X", "AUDUSD=X",
      "PLN=X", "CHF=X", "BRL=X", "MXN=X", "INR=X"]

MACRO = ["ZC=F", "ZS=F", "ZW=F", "LE=F", "HG=F", "CL=F", "^TNX", "^FVX", "^GSPC", "^SOX",
         "XLI", "XLK", "XLY", "XHB", "ITB"]

#: ⚡ **`period1=0` asks for everything, and that is deliberate.** A backfill bounded by the
#: horizon we assume can never reveal that the source holds more — the mistake this fetcher's
#: sibling made on Alpaca, where `feed=iex` returned 2020→ with a null page token and looked
#: complete against a 2015 request. Yahoo honours `period1` genuinely, so asking from zero makes
#: the source state its own floor. The competition corpus is dense from 2015; anything deeper is
#: free seasonality history.
HORIZON_START_EPOCH = 0


class Transport:
    """One yfinance `YfData` per process, doing the cookie+crumb handshake once."""

    def __init__(self) -> None:
        import os
        import tempfile

        import yfinance
        from yfinance import data as yfdata

        try:
            yfinance.set_tz_cache_location(
                os.path.join(tempfile.gettempdir(), "yfinance-cache", str(os.getpid())))
        except Exception as exc:  # noqa: BLE001 — degrading to no cache costs one handshake
            print(f"  note: yfinance cache unavailable ({exc.__class__.__name__})")
        self._data = yfdata.YfData()
        self._last = 0.0

    def get(self, url: str, params: dict):
        wait = _MIN_INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        return self._data.get(url, params=params, timeout=_TIMEOUT)


def universe() -> list[tuple[str, str, str]]:
    rows = [("target", ours, sym) for ours, sym in TARGETS.items()]
    rows += [("peer", ours, sym) for ours, peers in PEERS.items() for sym in peers]
    rows += [("fx", "", sym) for sym in FX]
    rows += [("macro", "", sym) for sym in MACRO]
    return rows


def fetch(products=("bars", "snapshot")) -> dict:
    config.ensure_dirs()
    transport = Transport()
    as_of = rawstore.now()
    ledger = rawstore.Ledger(PROVIDER)
    now_epoch = int(time.time())
    start_epoch = HORIZON_START_EPOCH
    banked = refused = disk = 0

    plan = universe()
    print(f"yahoo: {len(plan)} symbols x {len(products)} products -> "
          f"raw/yahoo/stock/<product>/day/{rawstore.segment_of(as_of)}")

    for role, ours, sym in plan:
        for product in products:
            if product == "bars":
                url = _CHART.format(ticker=sym)
                params = {"period1": start_epoch, "period2": now_epoch, "interval": "1d",
                          "events": "div,split", "includeAdjustedClose": "true"}
            elif product == "snapshot":
                url = _QUOTE_SUMMARY.format(ticker=sym)
                params = {"modules": ",".join(QUOTE_SUMMARY_MODULES),
                          "corsDomain": "finance.yahoo.com", "formatted": "false"}
            else:
                url = _OPTION_CHAIN.format(ticker=sym)
                params = {}

            try:
                resp = transport.get(url, params)
            except Exception as exc:  # noqa: BLE001
                refused += 1
                ledger.add(provider=PROVIDER, product=product, symbol=sym, role=role,
                           for_ticker=ours, state="transport_error",
                           note=type(exc).__name__,
                           fetched_at=as_of.isoformat(timespec="seconds"))
                print(f"  -- {product:<9}{sym:<10}{type(exc).__name__}")
                continue

            state, note, n = _judge(product, resp)
            if state != "ok":
                refused += 1
                ledger.add(provider=PROVIDER, product=product, symbol=sym, role=role,
                           for_ticker=ours, state=state, note=note, status=resp.status_code,
                           fetched_at=as_of.isoformat(timespec="seconds"))
                print(f"  -- {product:<9}{sym:<10}{note}")
                continue

            directory = rawstore.capture_dir(PROVIDER, product, "day", as_of=as_of,
                                             domain="stock")
            leaf = rawstore.leaf_name(sym, ext="json", compression="gz")
            body = rawstore.envelope(resp.text, fetched_at=as_of,
                                     served_at=resp.headers.get("Date"),
                                     request={"url": url, "params": {
                                         k: v for k, v in params.items() if k != "modules"}})
            written = rawstore.bank(directory / leaf, body)
            disk += written
            banked += 1
            ledger.add(provider=PROVIDER, product=product, symbol=sym, role=role,
                       for_ticker=ours, state="ok", note=note, n_items=n,
                       path=str((directory / leaf).relative_to(config.RAW)),
                       source_bytes=len(resp.text), disk_bytes=written,
                       fetched_at=as_of.isoformat(timespec="seconds"))
            print(f"  ok {product:<9}{sym:<10}{note}")

    ledger.write()
    return {"banked": banked, "refused": refused, "disk": disk}


def _judge(product: str, resp) -> tuple[str, str, int]:
    """(state, note, item count) — each surface spells absence differently."""
    try:
        body = json.loads(resp.text)
    except ValueError:
        return "not_json", f"HTTP {resp.status_code}, not JSON", 0
    if product == "bars":
        chart = body.get("chart") or {}
        result = (chart.get("result") or [{}])
        bars = len((result[0] or {}).get("timestamp") or []) if result else 0
        if bars:
            return "ok", f"{bars} bars", bars
        err = (chart.get("error") or {}).get("description") or f"no timestamps ({resp.status_code})"
        return "empty", str(err), 0
    if product == "snapshot":
        summary = body.get("quoteSummary") or {}
        result = summary.get("result")
        if result:
            mods = sorted(set().union(*(set(r) for r in result if isinstance(r, dict))))
            return "ok", f"{len(mods)} modules", len(mods)
        err = (summary.get("error") or {}).get("description") or f"empty ({resp.status_code})"
        return "empty", str(err), 0
    chain = body.get("optionChain") or {}
    result = chain.get("result") or []
    if result:
        exps = len((result[0] or {}).get("expirationDates") or [])
        return "ok", f"{exps} expirations", exps
    return "empty", "no chain listed", 0


def main() -> int:
    stats = fetch()
    print(f"\nraw/yahoo  {stats['banked']} captures  {stats['disk'] / 1e6:.1f} MB on disk  "
          f"({stats['refused']} refused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
