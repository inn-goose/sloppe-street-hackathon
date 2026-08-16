"""feature/guides — every stated guide, bound to a canonical period and a registry metric.

    PYTHONPATH=. .venv/bin/python -m code.feature.guides

The guidance lane holds 2,064 stated ranges keyed by the filer's own words: *"fiscal 2026"*,
*"the third quarter"*, *"the year ending 30 June"*. Nothing joins to that. This module resolves each
one onto `(metric, fiscal_year, fiscal_period)` and normalises its bounds onto the panel's scale, so
a guide and the outcome it was about become the same key.

⚠️ **A guide is data, not a parameter.** ADI's `49.0 %` operating-margin guide and Deere's
`$4.5–5.0 bn` were previously written into the model as literals, which quietly meant those bridges
existed only at the target period and could never be scored. Read from the lane instead, the same
guides exist for dozens of past periods — ADI has guided its adjusted operating margin 25 times, HD
its total sales growth 21 times — and every rule built on one becomes measurable out of sample.

## Binding a guide to a metric

`GUIDE_MAP` matches the filer's phrasing to a registry metric and records the guide's **shape**,
which decides how it may ever be compared:

* `level` — `revenue of $3.9 bn ± $100 m`, compared against the realised level
* `growth` — `total sales growth of 2.5 % to 4.5 %`, compared against realised year-on-year growth
* `margin` — `gross margin of approximately 33.1 %`, compared against the realised margin

⛔ Comparing a growth guide against a level is the tidiest way to produce a confident, meaningless
number, so the shape travels with the row and unclassifiable guides are dropped rather than guessed.
"""

from __future__ import annotations

import re

from collections import Counter, defaultdict

from code.feature import metrics as M
from code.feature.periods import Key, Resolver
from code.lib import config, store

#: (ticker, pattern against the filer's own wording, registry metric, shape).
#: Mined from `extracted/guidance` — these are phrasings the filers actually used.
GUIDE_MAP: tuple[tuple[str, str, str, str], ...] = (
    ("ADI", r"forecasting revenue|^revenue$", "adi_revenue", "level"),
    ("ADI", r"adjusted eps|adjusted diluted eps", "adi_adj_diluted_eps", "level"),
    ("ADI", r"planning for reported eps|^reported eps$|^eps$", "adi_diluted_eps", "level"),
    ("ADI", r"adjusted operating margin", "adi_adj_operating_margin", "margin"),
    # ⛔ ADI guides GAAP **and** adjusted operating margin separately, and they run ~16 pp apart.
    # A bare `operating margin` is the GAAP one; binding it to the adjusted metric compares two
    # different measures and reports the basis gap as a guidance beat.
    ("ADI", r"adjusted gross margin", "adi_adj_gross_margin", "margin"),
    ("HD", r"total sales growth|^sales growth$", "hd_net_sales", "growth"),
    ("HD", r"comparable sales growth|^comp sales$|^comparable sales$",
     "hd_comparable_sales", "level"),
    # ⚠️ HD states one guide for two measures — "Diluted earnings-per-share **and** Adjusted
    # diluted earnings-per-share **to both increase** approximately flat to 4.0 %" — and the metric
    # capture lands on `per share to both increase`. Matching only the tidy phrasings left one of
    # the twelve targets with no guide at all while the guide sat in the lane.
    ("HD", r"diluted earnings-per-share|adjusted diluted earnings-per-share"
           r"|earnings-per-share to grow|per share to both increase",
     "hd_adj_diluted_eps", "growth"),
    ("HD", r"^gross margin$", "hd_gross_margin", "margin"),
    ("HD", r"^operating margin$", "hd_operating_margin", "margin"),
    ("DE", r"net income|earnings", "de_net_income", "level"),
    # ⚡ Deere guides its biggest segment's sales by name — "precision ag net sales ... down 5 % for
    # fiscal year 2026" — which is the direct driver of that segment's profit, a submitted target.
    ("DE", r"precision ag net sales", "de_ppa_net_sales", "growth"),
    # ⛔ **`segment's operating margin` is deliberately NOT bound.** Deere states three segment
    # margins in the same release (Production & Precision Ag, Small Ag & Turf, Construction &
    # Forestry) and the period field comes back empty on all of them, so nothing in the row says
    # which segment it belongs to. Binding it would attach another segment's margin to P&PA — the
    # same class of error as reading a GAAP margin into an adjusted target, and just as invisible.
    ("LSE:HAS", r"operating profit", "has_pre_exc_operating_profit", "level"),
)

#: A range wider than this relative to its own midpoint is not a point forecast of anything.
MAX_RELATIVE_WIDTH = 2.0
#: Guidance states bounds in the filer's chosen scale; the panel is in millions.
_SCALE = {"billion": 1e3, "million": 1.0, "thousand": 1e-3, "": 1.0}


def _to_millions(value: float, unit: str, scale: str) -> float:
    if unit == "%":
        return value
    return value * _SCALE.get((scale or "").lower(), 1.0)


def _period_by_document(raw: list[dict], resolver: Resolver) -> dict[str, tuple[int, str]]:
    """What period each release's forward-looking guides are about, from the ones that resolved.

    ⚡ **A guide that names no period can be settled by its siblings, but never by its document.**
    ADI's Q2 FY2026 release guides revenue *and* adjusted EPS *and* operating margin for the same
    upcoming quarter; only some of those sentences repeat the words "third quarter". Resolving the
    explicit ones and lending that period to the rest recovers real guides — ADI's Q3 adjusted-EPS
    range is a submitted target and was otherwise unjoinable.

    ⛔ This is not the same as falling back to the document's own period, which was tried and was
    wrong: guidance in a Q2 release is about Q3 or the year, not about the quarter being reported.
    Assigning it to the reporting period paired guides with the wrong outcomes and made ADI look as
    if it beat its operating-margin guide 96 % of the time by 16 pp. Only a *sibling guide's*
    resolved period is evidence, and only where the siblings agree.
    """
    votes: dict[str, Counter] = defaultdict(Counter)
    for guide in raw:
        if guide["confidence"] != "stated" or not guide["period"]:
            continue
        key = resolver.by_phrase(guide["ticker"], guide["period"], guide["doc_id"])
        if key.ok:
            votes[guide["doc_id"]][(key.fiscal_year, key.fiscal_period)] += 1
    out = {}
    for doc_id, counter in votes.items():
        # ⚠️ **Unanimity, not a majority.** ADI's Q2 FY2026 8-K names the period exactly once — on
        # revenue, "third quarter of fiscal 2026" — and leaves it implicit on the adjusted-EPS,
        # reported-EPS and operating-margin sentences beside it. Requiring two agreeing siblings
        # rejected that document and cost a submitted target its only guide. One unambiguous
        # sibling is the cleanest evidence there is; what must be refused is *disagreement*, so a
        # release guiding both a quarter and the year lends its period to neither.
        if len(counter) == 1:
            out[doc_id] = next(iter(counter))
    return out


def build() -> tuple[list[dict], dict[str, int]]:
    resolver = Resolver()
    raw = store.read(config.EXTRACTED / "guidance.parquet")
    sibling_period = _period_by_document(raw, resolver)

    rows: list[dict] = []
    refused: dict[str, int] = {"unresolved_period": 0, "too_wide": 0, "no_metric_match": 0}
    for guide in raw:
        if guide["confidence"] != "stated":
            continue
        matched = False
        for ticker, pattern, metric_name, shape in GUIDE_MAP:
            if guide["ticker"] != ticker:
                continue
            if not re.search(pattern, guide["metric"] or "", re.IGNORECASE):
                continue
            matched = True
            metric = M.REGISTRY.get(metric_name)
            if metric is None:
                continue
            # ⛔ **A guide whose period cannot be read from its own words is refused, never
            # assigned to the document's period.** Guidance is about the future: an undated "we're
            # forecasting revenue of $3.9 bn" in a Q2 release is about Q3 or the year, never about
            # the quarter the release is reporting. Defaulting to the document's own period looked
            # like it rescued data — ADI's guide count went from 33 to 52 — but it paired guides
            # with the wrong outcomes, and the damage was legible: ADI appeared to beat its
            # operating-margin guide **96 %** of the time by a median **16 pp**, which is not a
            # filer being conservative, it is a guide compared against the wrong quarter.
            key = resolver.by_phrase(ticker, guide["period"] or "", guide["doc_id"])
            source = "phrase"
            if not key.ok and (borrowed := sibling_period.get(guide["doc_id"])):
                key = Key(borrowed[0], borrowed[1], "sibling")
                source = "sibling"
            if not key.ok:
                refused["unresolved_period"] += 1
                continue

            low = _to_millions(guide["low"], guide["unit"], guide["scale"])
            high = _to_millions(guide["high"], guide["unit"], guide["scale"])
            if low > high:
                low, high = high, low
            midpoint = (low + high) / 2.0
            if midpoint == 0 or abs(high - low) / max(abs(midpoint), 1e-9) > MAX_RELATIVE_WIDTH:
                refused["too_wide"] += 1
                continue
            rows.append({
                "ticker": ticker, "metric": metric_name, "shape": shape,
                "fiscal_year": key.fiscal_year, "fiscal_period": key.fiscal_period,
                "label": key.label(),
                "low": low, "high": high, "midpoint": midpoint,
                "width": high - low,
                "relative_width": abs(high - low) / abs(midpoint),
                "is_point": bool(guide["is_point"]),
                "unit": metric.unit, "guide_unit": guide["unit"], "scale": guide["scale"],
                "frame": guide["frame"], "published_at": guide["published_at"],
                "doc_id": guide["doc_id"],
                "wording": guide["metric"],
            })
        if not matched:
            refused["no_metric_match"] += 1

    rows.sort(key=lambda r: (r["metric"], r["fiscal_year"] or 0, r["fiscal_period"],
                             r["published_at"]))
    return rows, refused


def main() -> int:
    rows, refused = build()
    store.write(config.FEATURE / "guides.parquet", rows)
    keys = {(r["metric"], r["fiscal_year"], r["fiscal_period"]) for r in rows}
    print(f"feature/guides.parquet {len(rows):,} stated guides bound to {len(keys):,} "
          f"(metric, period) keys")
    print(f"  refused: " + ", ".join(f"{k} {v:,}" for k, v in refused.items()))

    print(f"\n{'metric':<32}{'shape':<8}{'guides':>8}{'periods':>9}{'first':>10}{'last':>10}")
    print("-" * 80)
    seen: dict[tuple, list[dict]] = {}
    for row in rows:
        seen.setdefault((row["metric"], row["shape"]), []).append(row)
    for (metric, shape), group in sorted(seen.items()):
        periods = sorted({r["label"] for r in group})
        print(f"{metric:<32}{shape:<8}{len(group):>8}{len(periods):>9}"
              f"{periods[0]:>10}{periods[-1]:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
