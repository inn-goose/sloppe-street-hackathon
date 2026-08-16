"""extracted/labour_observations — UK, German and euro-area labour series, long format.

⛔ **Three sources, three completely different JSON shapes**, and none of them is a flat list:

* **ONS** nests observations by frequency — `{"months": [{"date": "2026 JUN", "value": "728"}]}` —
  and writes the date as a human string, not ISO.
* **Eurostat** is JSON-stat: a **sparse dictionary keyed by a flat integer index** into the
  cartesian product of its dimensions, so a value's period has to be recovered by dividing that
  index down through the dimension sizes. There is no date on the observation at all.
* **ECB** is SDMX-JSON: `dataSets[0].series["0:0:0:…"].observations` keyed by the position of the
  period in `structure.dimensions.observation[0].values`.

Getting any of the three wrong yields plausible numbers on the wrong dates, which is the failure
mode nothing downstream can see. Each reader below decodes its source's own index rather than
assuming an order.
"""

from __future__ import annotations

import re
from collections import Counter

from code.lib import config, rawstore, store

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
_ONS_MONTH = re.compile(r"^(\d{4})\s+([A-Z]{3})$")
_ONS_QUARTER = re.compile(r"^(\d{4})\s*Q(\d)$", re.IGNORECASE)


def _ons_date(raw: str) -> str | None:
    raw = (raw or "").strip().upper()
    m = _ONS_MONTH.match(raw)
    if m and m.group(2) in _MONTHS:
        return f"{m.group(1)}-{_MONTHS[m.group(2)]:02d}-01"
    m = _ONS_QUARTER.match(raw)
    if m:
        return f"{m.group(1)}-{(int(m.group(2)) - 1) * 3 + 1:02d}-01"
    if re.fullmatch(r"\d{4}", raw):
        return f"{raw}-01-01"
    return None


def _period_to_date(period: str) -> str | None:
    """`2026-Q2` / `2026M06` / `2026` → an ISO day."""
    period = (period or "").strip().upper()
    m = re.fullmatch(r"(\d{4})-?Q([1-4])", period)
    if m:
        return f"{m.group(1)}-{(int(m.group(2)) - 1) * 3 + 1:02d}-01"
    m = re.fullmatch(r"(\d{4})-?M?(\d{2})", period)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{m.group(1)}-{int(m.group(2)):02d}-01"
    if re.fullmatch(r"\d{4}", period):
        return f"{period}-01-01"
    return None


def _from_ons(meta, body) -> list[dict]:
    rows = []
    for frequency in ("months", "quarters", "years"):
        for point in body.get(frequency) or []:
            if not isinstance(point, dict):
                continue
            date = _ons_date(point.get("date") or "")
            try:
                value = float(str(point.get("value")).replace(",", ""))
            except (TypeError, ValueError):
                continue
            if date:
                rows.append({"date": date, "value": value, "frequency": frequency[:-1]})
    return rows


def _from_eurostat(meta, body) -> list[dict]:
    """JSON-stat: a sparse `value` dict keyed by a flat index over the dimension product."""
    dim = body.get("dimension") or {}
    order = body.get("id") or list(dim)
    sizes = body.get("size") or [len((dim.get(d, {}).get("category") or {}).get("index") or {})
                                 for d in order]
    if "time" not in order:
        return []
    time_axis = order.index("time")
    time_index = ((dim.get("time") or {}).get("category") or {}).get("index") or {}
    periods = [p for p, _i in sorted(time_index.items(), key=lambda kv: kv[1])]

    rows = []
    for flat, value in (body.get("value") or {}).items():
        try:
            idx = int(flat)
        except (TypeError, ValueError):
            continue
        # decode the flat index into per-dimension positions, right to left
        position = []
        for size in reversed(sizes):
            position.append(idx % size)
            idx //= size
        position.reverse()
        slot = position[time_axis]
        if slot >= len(periods) or not isinstance(value, (int, float)):
            continue
        date = _period_to_date(periods[slot])
        if date:
            rows.append({"date": date, "value": float(value),
                         "frequency": "quarter" if "Q" in periods[slot].upper() else "month"})
    return rows


def _from_ecb(meta, body) -> list[dict]:
    """SDMX-JSON: observations keyed by the period's POSITION, resolved through the structure."""
    structure = body.get("structure") or {}
    obs_dims = ((structure.get("dimensions") or {}).get("observation") or [{}])
    periods = [v.get("id") for v in (obs_dims[0].get("values") or [])]
    rows = []
    for series in ((body.get("dataSets") or [{}])[0].get("series") or {}).values():
        for slot, values in (series.get("observations") or {}).items():
            try:
                index = int(slot)
            except (TypeError, ValueError):
                continue
            if index >= len(periods) or not values:
                continue
            value = values[0]
            date = _period_to_date(periods[index] or "")
            if date and isinstance(value, (int, float)):
                rows.append({"date": date, "value": float(value), "frequency": "month"})
    return rows


_READERS = {"ons": _from_ons, "eurostat": _from_eurostat, "ecb": _from_ecb}


def build() -> list[dict]:
    out = []
    for meta, body in rawstore.iter_captures("labour"):
        reader = _READERS.get(meta.get("source") or "")
        if reader is None:
            continue
        for point in reader(meta, body):
            out.append({
                "source": meta["source"], "series_id": meta["series_id"],
                "description": meta.get("description") or "",
                "for_ticker": meta.get("for_ticker") or "",
                "cadence": meta.get("cadence") or "",
                "captured_at": meta.get("fetched_at") or "", **point})
    return out


def main() -> int:
    rows = build()
    store.write(config.EXTRACTED / "labour_observations.parquet", rows)
    print(f"extracted/labour_observations.parquet {len(rows):,} observations  "
          f"{dict(Counter(r['source'] for r in rows))}")
    for series in sorted({r["series_id"] for r in rows}):
        mine = sorted((r for r in rows if r["series_id"] == series), key=lambda r: r["date"])
        print(f"  {mine[0]['source']:<9}{series[:46]:<48}{len(mine):>5} obs  "
              f"{mine[0]['date']}..{mine[-1]['date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
