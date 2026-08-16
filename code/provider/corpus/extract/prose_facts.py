"""extracted/prose_facts — values these filers state in a SENTENCE, not in a table.

## Why this lane is not optional

The coverage audit found exactly one of the twelve targets unbuildable: **Home Depot's adjusted
diluted EPS**, with 3 usable periods. HD publishes it in a non-GAAP reconciliation table only from
fiscal 2024, but it has stated it **in the headline paragraph of every release for a decade**:

    "Adjusted diluted earnings per share for the first quarter of fiscal 2026 were $3.43,
     compared with adjusted diluted earnings per share of $3.56 in the same period of fiscal 2025."

One sentence, two observations — the quarter and its prior-year comparable. Nothing else in the
store carries that series.

## This is a template match, not NLP

The boundary is on the **method**, not the corpus: a dictionary, a classifier or an
LLM is out; a template match, a count and a diff are in — the same trust class as any numeric
table. Every pattern below is a fixed frame with the value in a capture group. Nothing is scored,
classified or inferred; a sentence either fits the frame or it is not read.

## The frames, verbatim from the releases

| Frame | Filer | Text |
|---|---|---|
| `metric_for_period_was` | HD | `Adjusted diluted earnings per share for the first quarter of fiscal 2026 were $3.43` |
| `reported_of_for` | HD/DE | `reported sales of $41.8 billion for the first quarter of fiscal 2026` |
| `metric_moved_pct` | HD | `Comparable sales for the first quarter of fiscal 2026 increased 0.6%` |
| `moved_pct_to_value` | DE | `Worldwide net sales and revenues increased 5 percent, to $13.369 billion` |
| `or_per_share` | HD/DE | `, or $3.30 per diluted share` |
| `compared_with` | HD/DE | `compared with adjusted diluted EPS of $3.56 in the same period of fiscal 2025` |

⚡ **The comparative half is banked as its own row.** `compared with … in the same period of
fiscal 2025` states the prior-year value in the same sentence, so one release yields two dated
observations and the series doubles for free — and the overlap between consecutive releases is a
free consistency check.
"""

from __future__ import annotations

import re
from collections import Counter

from code.lib import config, store
from code.provider.corpus.extract import _corpus
from code.vendor.sentence_grammar import Sentences, evidence

_NUM = r"\d{1,4}(?:,\d{3})*(?:\.\d+)?"
_SCALE = r"(?:\s*(?:billion|million|thousand|bn|%|percent|percentage\s+points?|bps|cents|pence|p\b))?"
_MONEY = rf"(?P<cur>[$£€])?\s?(?P<num>-?{_NUM})(?P<scale>{_SCALE})"
_PERIOD = (r"(?:the\s+)?(?:first|second|third|fourth)\s+quarter(?:\s+ended\s+[^,]{4,24})?"
           r"(?:\s+of\s+)?(?:fiscal\s+)?(?:year\s+)?(?:\d{4})?"
           r"|(?:the\s+)?(?:fiscal|full)[\s-]?year(?:\s+\d{4})?"
           r"|(?:the\s+)?(?:six|nine|twelve)\s+months(?:\s+ended\s+[^,]{4,24})?"
           r"|(?:the\s+)?(?:year|quarter)\s+ended\s+[^,]{4,24}"
           r"|(?:the\s+)?(?:half|first\s+half|second\s+half)(?:\s+of\s+\w+\s*\d{0,4})?")

#: `Adjusted diluted earnings per share for the first quarter of fiscal 2026 were $3.43`
_FOR_PERIOD_WAS = re.compile(
    rf"(?P<metric>[A-Z][\w\s,&'()./-]{{2,64}}?)\s+for\s+(?P<period>{_PERIOD})\s+"
    rf"(?:were|was|is|are|totall?ed|came\s+in\s+at)\s+{_MONEY}")

#: `reported sales of $41.8 billion for the first quarter of fiscal 2026`
_REPORTED_OF_FOR = re.compile(
    rf"reported\s+(?P<metric>[\w\s,&'()./-]{{2,52}}?)\s+of\s+{_MONEY}\s+for\s+"
    rf"(?P<period>{_PERIOD})", re.IGNORECASE)

#: `Comparable sales for the first quarter of fiscal 2026 increased 0.6%`
_MOVED_PCT = re.compile(
    rf"(?P<metric>[A-Z][\w\s,&'()./-]{{2,64}}?)\s+for\s+(?P<period>{_PERIOD})\s+"
    rf"(?P<dir>increased|decreased|rose|fell|grew|declined)\s+(?:by\s+)?"
    rf"(?P<num>-?{_NUM})\s*(?P<scale>%|percent)")

#: `Worldwide net sales and revenues increased 5 percent, to $13.369 billion, for the second …`
_MOVED_TO = re.compile(
    rf"(?P<metric>[A-Z][\w\s,&'()./-]{{2,64}}?)\s+(?P<dir>increased|decreased|rose|fell|grew|"
    rf"declined)\s+(?P<pct>-?{_NUM})\s*(?:%|percent)\s*,?\s*to\s+{_MONEY}"
    rf"(?:\s*,?\s*for\s+(?P<period>{_PERIOD}))?")

#: `, or $3.30 per diluted share`
_PER_SHARE = re.compile(
    rf",?\s*or\s+{_MONEY}\s+per\s+(?P<kind>diluted|basic|common)?\s*share", re.IGNORECASE)

#: `compared with adjusted diluted earnings per share of $3.56 in the same period of fiscal 2025`
_COMPARED = re.compile(
    rf"compared\s+with\s+(?P<metric>[\w\s,&'()./-]{{2,64}}?)\s+of\s+{_MONEY}"
    rf"(?:\s+(?:in|for)\s+(?P<period>[^.;]{{4,60}}?))?(?=[.,;])", re.IGNORECASE)

_SCALE_FACTOR = {"billion": 1e9, "bn": 1e9, "million": 1e6, "thousand": 1e3}


def _value(num: str, scale: str, cur: str | None):
    """(value as stated, value in base units, unit kind). Scale is applied, never guessed."""
    try:
        raw = float(num.replace(",", ""))
    except (TypeError, ValueError):
        return None, None, ""
    token = (scale or "").strip().lower()
    if token in ("%", "percent", "percentage point", "percentage points"):
        return raw, raw, "percent"
    if token == "bps":
        return raw, raw / 100.0, "percent"
    if token in ("cents",):
        return raw, raw / 100.0, "per_share"
    if token in ("pence", "p"):
        return raw, raw, "pence"
    factor = _SCALE_FACTOR.get(token, 1.0)
    kind = "currency" if cur else ("currency" if factor > 1 else "number")
    return raw, raw * factor, kind


def build() -> list[dict]:
    lanes = {r["doc_id"]: r for r in store.read(config.EXTRACTED / "document_lanes.parquet")}
    rows: list[dict] = []

    def emit(doc, sents, m, frame, metric, period, num, scale, cur, direction=""):
        stated, base, kind = _value(num, scale, cur)
        if stated is None:
            return
        metric = " ".join((metric or "").split()).strip(" ,-")
        if not metric or len(metric) < 3:
            return
        span = sents.at(m.start())
        lane = lanes.get(doc.doc_id, {})
        rows.append({
            "doc_id": doc.doc_id, "ticker": doc.ticker, "published_at": doc.published_at,
            "lane_family": lane.get("lane_family", ""),
            "document_period": doc.period_label,
            "frame": frame, "metric": metric[:80],
            "period_phrase": " ".join((period or "").split())[:80],
            "direction": direction,
            "value": stated, "value_base": base, "unit_kind": kind,
            "currency": cur or "", "scale": (scale or "").strip().lower(),
            "is_comparative": frame in ("compared_with",),
            "evidence": evidence(doc_body, span, m.start(), m.end())[:300],
        })

    for doc in _corpus.iter_documents():
        doc_body = re.sub(r"\s+", " ", doc.body)
        sents = Sentences(doc_body)

        for m in _FOR_PERIOD_WAS.finditer(doc_body):
            emit(doc, sents, m, "metric_for_period_was", m.group("metric"), m.group("period"),
                 m.group("num"), m.group("scale"), m.group("cur"))
        for m in _REPORTED_OF_FOR.finditer(doc_body):
            emit(doc, sents, m, "reported_of_for", m.group("metric"), m.group("period"),
                 m.group("num"), m.group("scale"), m.group("cur"))
        for m in _MOVED_PCT.finditer(doc_body):
            direction = m.group("dir").lower()
            num = m.group("num")
            if direction in ("decreased", "fell", "declined"):
                num = f"-{num}"
            emit(doc, sents, m, "metric_moved_pct", m.group("metric"), m.group("period"),
                 num, m.group("scale"), None, direction)
        for m in _MOVED_TO.finditer(doc_body):
            emit(doc, sents, m, "moved_pct_to_value", m.group("metric"), m.group("period") or "",
                 m.group("num"), m.group("scale"), m.group("cur"), m.group("dir").lower())
        for m in _PER_SHARE.finditer(doc_body):
            kind = (m.group("kind") or "diluted").lower()
            emit(doc, sents, m, "or_per_share", f"{kind} earnings per share", "",
                 m.group("num"), m.group("scale"), m.group("cur"))
        for m in _COMPARED.finditer(doc_body):
            emit(doc, sents, m, "compared_with", m.group("metric"), m.group("period") or "",
                 m.group("num"), m.group("scale"), m.group("cur"))
    return rows


def main() -> int:
    rows = build()
    store.write(config.EXTRACTED / "prose_facts.parquet", rows)
    print(f"extracted/prose_facts.parquet {len(rows):,} stated values  "
          f"{dict(Counter(r['frame'] for r in rows))}")
    for ticker in ("HD", "ADI", "LSE:HAS", "DE"):
        mine = [r for r in rows if r["ticker"] == ticker]
        print(f"  {ticker:<9}{len(mine):>6,} values  "
              f"{len({r['metric'].lower() for r in mine}):>4} distinct metrics")
    print("\n  HD adjusted diluted EPS — the metric the coverage audit blocked on:")
    hits = [r for r in rows if r["ticker"] == "HD"
            and re.search(r"adjusted diluted", r["metric"], re.I)]
    for r in sorted(hits, key=lambda r: r["published_at"])[-10:]:
        print(f"    {r['published_at']}  {r['value']:>7.2f}  {r['frame']:<22}"
              f"[{r['period_phrase'][:38]}]")
    print(f"    -> {len(hits)} observations over "
          f"{len({r['published_at'] for r in hits})} releases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
