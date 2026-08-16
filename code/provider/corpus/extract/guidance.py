"""extracted/guidance — every forward range the four companies stated, in their own words.

Runs the vendored SEC guidance grammar (`code/vendor/guidance_grammar.py`, a **declared
pre-existing component**) over every banked document. That grammar was mined from a
2,600-document stratified sample of SEC earnings releases across ~3,000 filers, so it knows the
frames a filer actually writes a range in and the noise classes that look like one.

## Why this lane decides the forecast

Every one of the twelve targets has a stated guide standing against it, and the guide is the
single strongest anchor available:

* ADI guided Q3 FY26 revenue and adjusted EPS with an explicit ± band.
* Home Depot guided FY26 sales growth, comparable sales and adjusted EPS growth.
* Deere guided FY26 net income and per-segment sales.
* Hays guided FY26 pre-exceptional operating profit against its own compiled consensus.

⚠️ **A guide is not a forecast of the same shape as the target.** HD guides the *fiscal year* in
growth terms while the target is a *quarter* in dollars; Deere guides net income while the target
is EPS and a segment's operating profit. Converting one to the other is the feature layer's job —
this lane only records, faithfully, what was said.

⚠️ **`period` is the filer's own phrase, unresolved.** "full year", "the third quarter",
"fiscal 2026" — resolving that to a fiscal key is a decision and lives one layer up.

⛔ **Run over every lane, including transcripts.** Management states ranges out loud that never
reach the release, and the grammar's own vetoes (prior-guide quotation, comparison base, reported
past tense, phone numbers, ASC citations) are what keep the noise out — not a lane filter.
"""

from __future__ import annotations

from collections import Counter

from code.lib import config, store
from code.provider.corpus.extract import _corpus, _guidance_forms
from code.vendor import guidance_grammar

#: ⚠️ The vendored grammar returned proxy-statement mechanics as HD's and ADI's most recent
#: "guidance" — equity-grant vesting bands, PSU payout ranges, CEO salary bands. They are ranges
#: stated in forward-looking language, so no *frame* can exclude them; the veto is on the subject.
_COMP_NOISE = _guidance_forms._NOT_GUIDANCE


def _hits(body: str):
    """Vendored frames first, then the supplementary ones, with spans deduped by value.

    The two grammars overlap deliberately: where both see a range they agree, and a duplicate is
    collapsed on `(metric, period, low, high)`. What the supplementary set adds is the forms the
    vendored sample never contained — `+/-`, a worded `flat` bound, and a worded direction.
    """
    seen: dict[tuple, object] = {}
    for hit in guidance_grammar.extract(body):
        if _COMP_NOISE.search(hit.metric) or _COMP_NOISE.search(hit.evidence):
            continue
        seen[(hit.metric.lower(), hit.period.lower(), hit.low, hit.high)] = hit
    for hit in _guidance_forms.extract(body):
        key = (hit.metric.lower(), hit.period.lower(), hit.low, hit.high)
        if key not in seen:
            seen[key] = hit
    return list(seen.values())


def build() -> list[dict]:
    lanes = {r["doc_id"]: r for r in store.read(config.EXTRACTED / "document_lanes.parquet")}
    rows: list[dict] = []
    for doc in _corpus.iter_documents():
        lane = lanes.get(doc.doc_id, {})
        for hit in _hits(doc.body):
            mid = (hit.low + hit.high) / 2.0
            denom = abs(hit.low) if abs(hit.low) > 1e-9 else abs(hit.high)
            rows.append({
                "doc_id": doc.doc_id, "ticker": doc.ticker,
                "published_at": doc.published_at,
                "lane_family": lane.get("lane_family", ""),
                "document_period": doc.period_label,
                "metric": hit.metric,
                "period": hit.period,
                "low": hit.low, "high": hit.high,
                "midpoint": mid,
                "width": hit.high - hit.low,
                "relative_width": (hit.high - hit.low) / denom if denom else None,
                "is_point": hit.low == hit.high,
                "unit": hit.unit, "scale": hit.scale, "scale_source": hit.scale_source,
                "frame": hit.frame, "confidence": hit.confidence,
                "evidence": hit.evidence[:400],
            })
    return rows


def main() -> int:
    rows = build()
    store.write(config.EXTRACTED / "guidance.parquet", rows)
    stated = [r for r in rows if r["confidence"] == "stated"]
    print(f"extracted/guidance.parquet {len(rows):,} stated forward ranges "
          f"({len(stated):,} on the `stated` tier, {len(rows) - len(stated):,} `weak`)")
    print(f"  frames: {dict(Counter(r['frame'] for r in rows))}")
    print(f"  lanes : {dict(Counter(r['lane_family'] for r in rows).most_common(6))}")
    for ticker in ("HD", "ADI", "LSE:HAS", "DE"):
        mine = [r for r in rows if r["ticker"] == ticker]
        s = [r for r in mine if r["confidence"] == "stated"]
        recent = [r for r in s if r["published_at"] >= "2026-01-01"]
        print(f"  {ticker:<9}{len(mine):>5} ranges  {len(s):>5} stated  "
              f"{len(recent):>4} stated in 2026  "
              f"{len({r['metric'].lower() for r in s}):>4} distinct metrics")
    top = Counter(r["metric"].lower() for r in stated).most_common(12)
    print("  most-guided metrics: " + ", ".join(f"{m}({n})" for m, n in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
