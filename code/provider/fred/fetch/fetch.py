"""raw/fred — macro series as verbatim CSV, keyless.

    data/raw/fred/series/<granularity>/<as-of>/<SERIES_ID>.csv.gz

⛔ **Banked bare, not enveloped.** The source's format is CSV, and splicing a JSON stamp into it
would stop it being the source's bytes. The stamp lives in the ledger instead.

⚡ **The keyless route is the point.** `api.stlouisfed.org` wants an API key;
`fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>` serves the same observations with none.

## Why macro, when the label is a reported fundamental

Three of the four issuers state their own demand driver out loud and none states its *level*.
Hays earns 32 % of group net fees in Germany; Home Depot guides comps against housing turnover;
Deere guides Large Ag down 15–20 % on crop economics. Each of these is published monthly through
the quarter the company reports once.

⚠️ **A discontinued series does not error — it freezes.** Measured here: the OECD German and UK
vacancy series stop in 2024 and 2023, and `DEUURHARMMDSMEI` ignores `cosd` entirely and answers
1991→2012. `is_stale` is recorded in the ledger so a consumer can refuse them; the live
replacements come from the `labour` provider.

⚠️ **Vintages are not point-in-time.** `fredgraph.csv` serves the *current* vintage, so a revised
series is revised all the way back. Correct for a forecast made today, mildly clairvoyant for a
backtest — stated rather than hidden.
"""

from __future__ import annotations

import csv
import io
import time

from code.lib import config, rawstore

PROVIDER = "fred"
_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_MIN_INTERVAL = 0.4
_TIMEOUT = 45

#: A series whose newest observation is older than this is discontinued for our purposes: the
#: quarter being forecast ended 2026-08 and a feature built on a 2024 level is a constant.
STALE_AFTER = "2025-06-01"

SERIES: tuple[tuple[str, str, str, str], ...] = (
    # (series id, which issuer it conditions, cadence, what it is)
    ("LMJVTTUVDEM647S", "LSE:HAS", "month", "Germany job vacancies (OECD — discontinued)"),
    ("LMJVTTUVGBM647S", "LSE:HAS", "month", "UK job vacancies (OECD — discontinued)"),
    ("LRHUTTTTDEM156S", "LSE:HAS", "month", "Germany unemployment rate"),
    ("LRHUTTTTGBM156S", "LSE:HAS", "month", "UK unemployment rate"),
    ("LRHUTTTTAUM156S", "LSE:HAS", "month", "Australia unemployment rate"),
    ("CLVMNACSCAB1GQDE", "LSE:HAS", "quarter", "Germany real GDP"),
    ("DEUPROINDMISMEI", "LSE:HAS", "month", "Germany industrial production (discontinued)"),
    # ⚡ **The closest thing to a nowcast of Home Depot's own line.** Census retail sales for
    # NAICS 444 — building materials, garden equipment and supplies dealers — is Home Depot's
    # actual category, published MONTHLY through a quarter the company reports once. Comparable
    # sales is the target; this is the category's own monthly trade. Nothing else in the store is
    # this close to a submitted metric.
    ("MRTSSM444USS", "HD", "month", "US retail sales: building materials & garden (NAICS 444, SA)"),
    ("MRTSSM444USN", "HD", "month", "US retail sales: building materials & garden (NSA)"),
    ("RSBMGESD", "HD", "month", "US retail sales: building materials & garden dealers"),
    ("HOUST", "HD", "month", "US housing starts"),
    ("HSN1F", "HD", "month", "US new one-family houses sold"),
    ("MSACSR", "HD", "month", "US monthly supply of new houses"),
    ("TLRESCONS", "HD", "month", "US total residential construction spending"),
    ("PCEDG", "HD", "month", "US personal consumption, durable goods"),
    ("PERMIT", "HD", "month", "US building permits"),
    ("MORTGAGE30US", "HD", "week", "US 30-year fixed mortgage rate"),
    ("EXHOSLUSM495S", "HD", "month", "US existing home sales"),
    ("CSUSHPINSA", "HD", "month", "Case-Shiller US national home price index"),
    ("RSXFS", "HD", "month", "US retail sales ex food services"),
    ("UMCSENT", "HD", "month", "US consumer sentiment"),
    ("TTLCONS", "HD", "month", "US total construction spending"),
    ("WPU081", "HD", "month", "PPI lumber and wood products"),
    # ⚡ Deere guides Large Ag on crop economics, so the ag cycle is its demand curve. Farm income
    # and crop receipts are the cash the customer buys equipment out of; machinery new orders are
    # the order book the segment ships against.
    ("WPU01", "DE", "month", "PPI farm products"),
    ("A33SNO", "DE", "month", "US new orders: machinery"),
    ("AMDMNO", "DE", "month", "US new orders: durable goods"),
    ("IPG333S", "DE", "month", "US industrial production: machinery"),
    ("CORNPRICE", "DE", "month", "Global price of corn"),
    ("SOYBNPRICE", "DE", "month", "Global price of soybeans"),
    ("WHEATPRICE", "DE", "month", "Global price of wheat"),
    ("PPIACO", "DE", "month", "PPI all commodities"),
    ("IPMAN", "DE", "month", "US industrial production, manufacturing"),
    ("A038RC1Q027SBEA", "DE", "quarter", "US farm proprietors income"),
    ("USSTHPI", "DE", "quarter", "US house price index"),
    ("IPG3344S", "ADI", "month", "US industrial production, semiconductors"),
    ("AMTMNO", "ADI", "month", "US manufacturers new orders, total"),
    ("A34SNO", "ADI", "month", "US new orders, computers and electronic products"),
    ("BUSINV", "ADI", "month", "US total business inventories"),
    ("IPG3341S", "ADI", "month", "US industrial production: computer & electronic products"),
    ("A31SNO", "ADI", "month", "US new orders: computers & electronic products, detail"),
    ("CES3133400001", "ADI", "month", "US employment: semiconductor & electronic components"),
    # Hays' book is the labour market; UK vacancies are covered live by the `labour` provider, but
    # a wage series conditions the fee per placement rather than the volume.
    ("LES1252881600Q", "LSE:HAS", "quarter", "US median usual weekly real earnings"),
    ("DGS10", "", "day", "US 10-year Treasury yield"),
    ("DTWEXBGS", "", "day", "Trade-weighted US dollar, broad"),
    ("DEXUSEU", "", "day", "USD per EUR"),
    ("DEXUSUK", "", "day", "USD per GBP"),
    ("DEXJPUS", "", "day", "JPY per USD"),
    ("VIXCLS", "", "day", "CBOE VIX"),
)

#: ⚡ Earlier than any series here begins, on purpose — the same rule the bars fetchers learned:
#: a window bounded by the horizon we assume can never reveal that the source holds more. FRED
#: answers with the series' own span, and `first_observation` in the ledger records what that was.
#: Deeper macro history is free and it is what a seasonal or cyclical regression is fitted on.
START = "1900-01-01"


def _session():
    import curl_cffi.requests as cr

    return cr.Session(impersonate="chrome")


def fetch() -> dict:
    config.ensure_dirs()
    session = _session()
    as_of = rawstore.now()
    ledger = rawstore.Ledger(PROVIDER)
    banked = refused = disk = 0
    last = 0.0

    print(f"fred: {len(SERIES)} series from {START} -> raw/fred/series/<cadence>/"
          f"{rawstore.segment_of(as_of)}")
    for series_id, for_ticker, cadence, description in SERIES:
        wait = _MIN_INTERVAL - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        last = time.monotonic()

        resp = session.get(_CSV, params={"id": series_id, "cosd": START}, timeout=_TIMEOUT)
        state, note, first, newest, n = _judge(resp)
        if state != "ok":
            refused += 1
            ledger.add(provider=PROVIDER, product="series", series_id=series_id,
                       for_ticker=for_ticker, description=description, state=state, note=note,
                       fetched_at=as_of.isoformat(timespec="seconds"))
            print(f"  -- {series_id:<18}{description[:44]:<46}{note}")
            continue

        directory = rawstore.capture_dir(PROVIDER, "series", cadence, as_of=as_of)
        leaf = rawstore.leaf_name(series_id, ext="csv", compression="gz")
        written = rawstore.bank(directory / leaf, resp.text)  # bare CSV — the source's own format
        disk += written
        banked += 1
        stale = newest < STALE_AFTER
        ledger.add(provider=PROVIDER, product="series", series_id=series_id,
                   for_ticker=for_ticker, description=description, cadence=cadence,
                   state="ok", n_observations=n, first_observation=first,
                   last_observation=newest, is_stale=stale, window_honoured=first >= START,
                   path=str((directory / leaf).relative_to(config.RAW)),
                   source_bytes=len(resp.text), disk_bytes=written,
                   fetched_at=as_of.isoformat(timespec="seconds"))
        mark = "!!" if stale else "ok"
        print(f"  {mark} {series_id:<18}{description[:44]:<46}{note}")

    ledger.write()
    return {"banked": banked, "refused": refused, "disk": disk,
            "stale": sum(1 for r in ledger.rows if r.get("is_stale"))}


def _judge(resp) -> tuple[str, str, str, str, int]:
    if resp.status_code != 200:
        return "http_error", f"HTTP {resp.status_code}", "", "", 0
    text = resp.text
    # ⛔ FRED answers 200 with an HTML error page for an unknown id; a CSV opens with its header.
    if not text.lstrip().lower().startswith(("observation_date", "date", '"observation_date')):
        return "not_csv", f"not a CSV ({text.lstrip()[:32]!r})", "", "", 0
    rows = list(csv.reader(io.StringIO(text)))
    body = [r for r in rows[1:] if len(r) >= 2 and r[1].strip() not in ("", ".")]
    if not body:
        return "empty", "no observations in window", "", "", 0
    first, newest = body[0][0], body[-1][0]
    note = f"{len(body):>6} obs {first}..{newest}" + ("  STALE" if newest < STALE_AFTER else "")
    return "ok", note, first, newest, len(body)


def main() -> int:
    stats = fetch()
    print(f"\nraw/fred  {stats['banked']} series  {stats['disk'] / 1e6:.2f} MB on disk  "
          f"({stats['refused']} refused, {stats['stale']} stale and flagged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
