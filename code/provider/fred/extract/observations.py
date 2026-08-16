"""extracted/fred_observations — the macro panel, long format.

⚠️ **`.` is FRED's missing marker, not a zero.** Refused rather than filled — a macro series with
a hole is a hole, and interpolating here would launder a guess into the store's contract.

⚠️ **`is_stale` rides on every row**, carried from the fetcher's own measurement. Three of the 31
series are discontinued (the OECD German and UK vacancy series stop in 2024 and 2023), and a dead
series does not error — it freezes a feature at its last value. Anything that ranks or differences
a macro level must filter on this; the live replacements are in the `labour` provider.
"""

from __future__ import annotations

import csv
import io

from code.lib import config, rawstore, store


def build() -> list[dict]:
    rows = []
    for meta, text in rawstore.iter_captures("fred", parse=False):
        reader = csv.reader(io.StringIO(text))
        if not next(reader, None):
            continue
        for record in reader:
            if len(record) < 2:
                continue
            date, raw = record[0].strip(), record[1].strip()
            if raw in ("", "."):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            rows.append({
                "series_id": meta["series_id"], "for_ticker": meta.get("for_ticker") or "",
                "description": meta.get("description") or "", "cadence": meta.get("cadence") or "",
                "date": date, "value": value,
                "is_stale": bool(meta.get("is_stale")),
                "last_observation": meta.get("last_observation") or "",
                "captured_at": meta.get("fetched_at") or "",
            })
    return rows


def main() -> int:
    rows = build()
    store.write(config.EXTRACTED / "fred_observations.parquet", rows)
    live = {r["series_id"] for r in rows if not r["is_stale"]}
    stale = {r["series_id"] for r in rows if r["is_stale"]}
    print(f"extracted/fred_observations.parquet {len(rows):,} observations  "
          f"{len(live)} live series, {len(stale)} stale and flagged")
    for ticker in ("HD", "ADI", "DE", "LSE:HAS", ""):
        mine = {r["series_id"] for r in rows if r["for_ticker"] == ticker}
        print(f"  {ticker or '(shared regime)':<16}{len(mine):>2} series")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
