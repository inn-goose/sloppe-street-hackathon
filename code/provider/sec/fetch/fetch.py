"""raw/sec — EDGAR XBRL company facts and the filing index. Keyless, verbatim JSON.

    data/raw/sec/company/facts/none/<as-of>/CIK<0000000000>.json.gz
    data/raw/sec/company/index/none/<as-of>/CIK<0000000000>.json.gz

## Why this provider matters more than any other external one

Every vendor lane here publishes a *restated* history: stockanalysis and Yahoo serve today's view
of what 2019 looked like. `companyfacts` serves each fact **with the accession and the `filed`
date that first carried it**, so a fact can be read as of the day it became knowable. That is the
difference between a model fitted on what was known and one fitted on what we know now — and for
a task judged against a frozen benchmark, the second flatters itself.

⛔ **Hays has no CIK and never will.** It is an LSE filer under the FCA, not an SEC registrant, so
this provider covers three of the four names by construction. The corpus is the only source for
Hays, which is why the corpus reader is not optional.

⚠️ **EDGAR requires a declaring `User-Agent`** and publishes 10 req/s. The value is read from the
`SEC_EDGAR_USER_AGENT` environment variable **by name only** — never read into this process's
output, never written to a file, never committed. Unset means refuse, not invent.
"""

from __future__ import annotations

import json
import time

from code.lib import config, rawstore

PROVIDER = "sec"
_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_MIN_INTERVAL = 0.25
_TIMEOUT = 120

#: Evidenced from the corpus's own `source_url` values, not remembered: 15 HD documents, 7 ADI
#: and 25 DE carry an `/edgar/data/<CIK>/` path.
CIKS = {"HD": 354950, "ADI": 6281, "DE": 315189}


def session(user_agent: str):
    import curl_cffi.requests as cr

    s = cr.Session(impersonate="chrome")
    s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    return s


def require_user_agent() -> str:
    ua = config.secret("SEC_EDGAR_USER_AGENT")
    if not ua:
        raise SystemExit(
            "sec: SEC_EDGAR_USER_AGENT is not set. EDGAR requires a declaring User-Agent and this "
            "fetcher will not invent one, nor substitute any other credential. Export it and "
            "rerun; the value is never printed, stored or committed.")
    return ua


def get(s, url: str, host: str, last: list[float]):
    wait = _MIN_INTERVAL - (time.monotonic() - last[0])
    if wait > 0:
        time.sleep(wait)
    last[0] = time.monotonic()
    s.headers["Host"] = host
    return s.get(url, timeout=_TIMEOUT)


def fetch() -> dict:
    config.ensure_dirs()
    s = session(require_user_agent())
    as_of = rawstore.now()
    ledger = rawstore.Ledger(PROVIDER)
    last = [0.0]
    banked = refused = disk = 0

    print(f"sec: {len(CIKS)} filers (Hays has no CIK — LSE/FCA, not an SEC registrant) -> "
          f"raw/sec/company/<facts|index>/none/{rawstore.segment_of(as_of)}")
    for ticker, cik in CIKS.items():
        for product, data_type, url in (
            ("company_facts", "facts", _FACTS.format(cik=cik)),
            ("submissions", "index", _SUBMISSIONS.format(cik=cik)),
        ):
            resp = get(s, url, "data.sec.gov", last)
            if resp.status_code != 200:
                refused += 1
                ledger.add(provider=PROVIDER, product=product, for_ticker=ticker, cik=cik,
                           state="http_error", status=resp.status_code,
                           fetched_at=as_of.isoformat(timespec="seconds"))
                print(f"  -- {ticker:<5}{product:<16}HTTP {resp.status_code}")
                continue
            try:
                body = json.loads(resp.text)
            except ValueError:
                refused += 1
                ledger.add(provider=PROVIDER, product=product, for_ticker=ticker, cik=cik,
                           state="not_json", fetched_at=as_of.isoformat(timespec="seconds"))
                continue
            # ⛔ a 200 that is not the shape we asked for is refused, never banked
            if product == "company_facts" and "facts" not in body:
                refused += 1
                ledger.add(provider=PROVIDER, product=product, for_ticker=ticker, cik=cik,
                           state="no_facts_member",
                           fetched_at=as_of.isoformat(timespec="seconds"))
                print(f"  -- {ticker:<5}{product:<16}no facts member")
                continue

            if product == "company_facts":
                note = ", ".join(f"{k}:{len(v)}" for k, v in (body.get("facts") or {}).items())
            else:
                recent = ((body.get("filings") or {}).get("recent") or {}).get("form") or []
                note = f"{len(recent)} recent filings, {body.get('name', '')}"

            directory = rawstore.capture_dir(PROVIDER, data_type, "none", as_of=as_of,
                                             domain="company")
            leaf = rawstore.leaf_name(f"CIK{cik:010d}", ext="json", compression="gz")
            enveloped = rawstore.envelope(resp.text, fetched_at=as_of,
                                          served_at=resp.headers.get("Date"),
                                          request={"url": url})
            written = rawstore.bank(directory / leaf, enveloped)
            disk += written
            banked += 1
            ledger.add(provider=PROVIDER, product=product, for_ticker=ticker, cik=cik,
                       state="ok", note=note,
                       path=str((directory / leaf).relative_to(config.RAW)),
                       source_bytes=len(resp.text), disk_bytes=written,
                       fetched_at=as_of.isoformat(timespec="seconds"))
            print(f"  ok {ticker:<5}{product:<16}{len(resp.text) / 1e6:>6.1f} MB  {note[:64]}")

    ledger.write()
    return {"banked": banked, "refused": refused, "disk": disk}


def main() -> int:
    stats = fetch()
    print(f"\nraw/sec  {stats['banked']} captures  {stats['disk'] / 1e6:.1f} MB on disk  "
          f"({stats['refused']} refused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
