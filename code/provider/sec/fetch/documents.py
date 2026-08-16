"""raw/sec documents — the ticker↔CIK map, peer filing indexes, and peer earnings-release bodies.

    data/raw/sec/company/index/none/<as-of>/company_tickers.json.gz
    data/raw/sec/company/index/none/<as-of>/CIK<0000000000>.json.gz
    data/raw/sec/documents/event/<filing date>/CIK<0000000000>/<accession>/<doc>.htm.gz

## Why fetch peer documents when the corpus has none

The competition corpus is four companies with **no cross-section at all**, so §R (peer
read-through) is dead in it. But the read-through is real and dated: Texas Instruments and
Microchip both guided their September quarter **before ADI reports its own**, Lowe's guides
against Home Depot's, AGCO and CNH against Deere's. A peer's stated forward range for an
overlapping period is the best exogenous check on a target's own guide — and it exists only
inside an 8-K's `EX-99.x` exhibit, which no vendor lane here carries.

⛔ **The four target companies' own releases are NOT fetched.** The frozen corpus already holds
them, and re-fetching from a live source would quietly replace a competition-issued frozen
document with a current one — which makes a run unreproducible after the fact.

⛔ **Exhibit names are not prefixed consistently, and a prefix match silently finds nothing.**
Measured on a first sweep: 10 of 16 peers returned zero exhibits — including Texas Instruments,
which files an earnings 8-K every quarter — because filers name the same document
`tm2521234d1_ex99-1.htm`, `q22026exhibit991.htm`, `a8-kq226xex991.htm`. The stable signal is that
an earnings release is an **EX-99** exhibit, so the match is on `99` appearing anywhere in the
name, gated by extension and a size floor rather than by position.

⚠️ HTML is banked **bare** — splicing a JSON stamp into markup would stop it being the source's
bytes. The stamp lives in the ledger.
"""

from __future__ import annotations

import json
import re

from code.lib import config, rawstore
from code.provider.sec.fetch.fetch import get, require_user_agent, session

PROVIDER = "sec"
_TICKERS = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{doc}"
_FILING_INDEX = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/index.json"

PEERS = {
    "HD": ["LOW", "TSCO", "SHW", "BLDR"],
    "ADI": ["TXN", "MCHP", "NXPI", "ON", "SWKS", "MPWR"],
    "DE": ["AGCO", "CNH", "CAT", "TITN", "LNN", "TRMB"],
}

#: ⚡ Deep enough for a peer's own guidance HISTORY, not just its latest quarter. A single recent
#: release says what a peer guided; six years of them says how that peer's guide has related to
#: what it then reported — which is the conservatism prior (#214) applied cross-sectionally, and
#: the only version of it available for names outside the frozen corpus.
RECENT_FILINGS = 400
MAX_8K_OPENED = 48
MAX_EXHIBITS_PER_PEER = 28
RESULT_FORMS = {"8-K", "8-K/A"}
EXHIBIT_MARK = re.compile(r"(?:^|[^0-9])99", re.IGNORECASE)
EXHIBIT_EXT = (".htm", ".html", ".txt")
#: A cover page or stub is a few hundred bytes; a results release is tens of thousands.
MIN_EXHIBIT_BYTES = 6000


def fetch() -> dict:
    config.ensure_dirs()
    s = session(require_user_agent())
    as_of = rawstore.now()
    ledger = rawstore.Ledger(f"{PROVIDER}_documents")
    last = [0.0]
    banked = refused = disk = exhibits_total = 0

    index_dir = rawstore.capture_dir(PROVIDER, "index", "none", as_of=as_of, domain="company")

    # ---- 1. the ticker → CIK map
    resp = get(s, _TICKERS, "www.sec.gov", last)
    tickers: dict[str, int] = {}
    if resp.status_code == 200:
        try:
            payload = json.loads(resp.text)
        except ValueError:
            payload = {}
        for entry in (payload.values() if isinstance(payload, dict) else []):
            if isinstance(entry, dict) and entry.get("ticker"):
                tickers[str(entry["ticker"]).upper()] = int(entry["cik_str"])
        leaf = rawstore.leaf_name("company_tickers", ext="json", compression="gz")
        written = rawstore.bank(index_dir / leaf,
                                rawstore.envelope(resp.text, fetched_at=as_of,
                                                  request={"url": _TICKERS}))
        disk += written
        banked += 1
        ledger.add(provider=PROVIDER, product="company_tickers", state="ok",
                   n_items=len(tickers),
                   path=str((index_dir / leaf).relative_to(config.RAW)),
                   source_bytes=len(resp.text), disk_bytes=written,
                   fetched_at=as_of.isoformat(timespec="seconds"))
        print(f"  ok company_tickers      {len(tickers):,} tickers mapped to CIKs")
    else:
        refused += 1
        print(f"  -- company_tickers      HTTP {resp.status_code}")

    wanted = [(t, p) for t, peers in PEERS.items() for p in peers]
    print(f"sec documents: {len(wanted)} peers "
          f"(targets excluded — the frozen corpus is authoritative for those)")

    for target, peer in wanted:
        cik = tickers.get(peer)
        if not cik:
            refused += 1
            ledger.add(provider=PROVIDER, product="submissions", symbol=peer,
                       for_ticker=target, state="no_cik",
                       fetched_at=as_of.isoformat(timespec="seconds"))
            print(f"  -- {peer:<6}no CIK in the SEC map")
            continue

        resp = get(s, _SUBMISSIONS.format(cik=cik), "data.sec.gov", last)
        if resp.status_code != 200:
            refused += 1
            continue
        leaf = rawstore.leaf_name(f"CIK{cik:010d}", ext="json", compression="gz")
        written = rawstore.bank(index_dir / leaf,
                                rawstore.envelope(resp.text, fetched_at=as_of,
                                                  request={"url": "submissions", "cik": cik}))
        disk += written
        banked += 1
        ledger.add(provider=PROVIDER, product="submissions", symbol=peer, for_ticker=target,
                   cik=cik, state="ok",
                   path=str((index_dir / leaf).relative_to(config.RAW)),
                   source_bytes=len(resp.text), disk_bytes=written,
                   fetched_at=as_of.isoformat(timespec="seconds"))

        recent = ((json.loads(resp.text).get("filings") or {}).get("recent") or {})
        forms = recent.get("form") or []
        accns = recent.get("accessionNumber") or []
        dates = recent.get("filingDate") or []
        primary = recent.get("primaryDocument") or []

        banked_here = opened = 0
        for i in range(min(len(forms), RECENT_FILINGS)):
            if forms[i] not in RESULT_FORMS or banked_here >= MAX_EXHIBITS_PER_PEER:
                continue
            if opened >= MAX_8K_OPENED:
                break
            opened += 1
            accn = (accns[i] or "").replace("-", "")
            idx = get(s, _FILING_INDEX.format(cik=cik, accn=accn), "www.sec.gov", last)
            if idx.status_code != 200:
                continue
            try:
                items = ((json.loads(idx.text).get("directory") or {}).get("item") or [])
            except ValueError:
                continue
            names = [it["name"] for it in items
                     if isinstance(it, dict) and isinstance(it.get("name"), str)
                     and EXHIBIT_MARK.search(it["name"])
                     and it["name"].lower().endswith(EXHIBIT_EXT)
                     and it["name"].lower() != str(primary[i] or "").lower()]
            for name in names[:2]:
                body = get(s, _ARCHIVE.format(cik=cik, accn=accn, doc=name), "www.sec.gov", last)
                if body.status_code != 200 or len(body.text) < MIN_EXHIBIT_BYTES:
                    continue
                ext = name.rsplit(".", 1)[-1]
                doc_dir = rawstore.ingest_dir(PROVIDER, "documents", "event", dates[i])
                doc_leaf = rawstore.leaf_name(f"CIK{cik:010d}", accn, name.rsplit(".", 1)[0],
                                              ext=ext, compression="gz")
                # bare bytes — HTML is the source's own format
                written = rawstore.bank(doc_dir / doc_leaf, body.text)
                disk += written
                banked += 1
                banked_here += 1
                exhibits_total += 1
                ledger.add(provider=PROVIDER, product="earnings_exhibit", symbol=peer,
                           for_ticker=target, cik=cik, accession=accns[i], document=name,
                           form=forms[i], filing_date=dates[i], state="ok",
                           path=str((doc_dir / doc_leaf).relative_to(config.RAW)),
                           source_bytes=len(body.text), disk_bytes=written,
                           fetched_at=as_of.isoformat(timespec="seconds"))
                if banked_here >= MAX_EXHIBITS_PER_PEER:
                    break
        print(f"  ok {peer:<6}CIK {cik:<9}{banked_here:>2} earnings exhibits  (informs {target})")

    ledger.write()
    return {"banked": banked, "refused": refused, "disk": disk, "exhibits": exhibits_total}


def main() -> int:
    stats = fetch()
    print(f"\nraw/sec documents  {stats['banked']} captures "
          f"({stats['exhibits']} peer earnings exhibits)  {stats['disk'] / 1e6:.1f} MB on disk  "
          f"({stats['refused']} refused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
