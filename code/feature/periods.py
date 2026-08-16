"""feature/fiscal_periods — the canonical period key everything else joins on.

    (ticker, fiscal_year, fiscal_period)     fiscal_period ∈ Q1 Q2 Q3 Q4 H1 H2 M9 FY TTM

## Why this is the foundation, and why it is a FEATURE not an extraction

Six lanes describe a period six different ways and none joins to another:

| lane | how it says "Deere's second quarter of fiscal 2026" |
|---|---|
| `statement_facts` | `period_end = 2026-05-03`, `span_months = 3` |
| `statement_facts` (measure tables) | no column at all — the period is the document's |
| `prose_facts` | `"the second quarter of fiscal 2026"` |
| `guidance` | `"fiscal 2026"` · `"third quarter of"` · `""` |
| `sa_financials` | `period_end = 2026-05-03`, `fiscalYear = 2026` |
| `nq_forecast` | `"Jul 2026"` |

Reconciling them is a decision — which label wins, how far a date may be from an anchor, what a
12-month span ending in Q2 means — so it belongs here rather than in a faithful view.

## The three decisions this module makes

⛔ **1. The FILER'S label is canonical, and every vendor is translated into it.** The competition
asks for `HD FY2026Q2`, which is Home Depot's own name for that quarter; **stockanalysis calls the
same year FY2027**. Measured, and it is not a quirk — a vendor's fiscal-year convention is its
own. So no vendor label is ever read as a key: a vendor row is keyed by its `period_end` **date**,
which both sides agree on, and resolved through the filer's map.

⛔ **2. The map is LEARNED, not ruled.** Home Depot labels a fiscal year by the calendar year it
*starts*; ADI, Deere and Hays by the year it *ends*. A hardcoded rule gets one of them wrong.
Instead every earnings release and periodic report states its period twice — once in the corpus's
filename tag (`q2`) and once in its frontmatter (`Q2 2025`) — and those two independent labels
**agree on 237 of 238 documents (99.6 %)**, which is what makes the learned map trustworthy. The
document's own statement tables then supply the period-end date, giving
`(period_end) → (fiscal_year, quarter)` directly.

⛔ **3. The SPAN is part of the key, not a footnote.** A figure with `period_end = 2026-05-03` is
Deere's Q2 if its span is 3 months and its **first half** if the span is 6 — the same date, two
different facts, and the release states both side by side. Collapsing them mixes quarterly and
year-to-date figures in one panel, which is the single most damaging thing that can happen to a
quarterly forecast.

⚠️ Every resolution carries a `grade`, and a consumer may refuse a tier:
`anchored` (exact date match) · `near` (within a few days) · `offset` (a whole number of fiscal
years from an anchor) · `phrase` (parsed from words) · `document` (inherited) · `unresolved`.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from code.lib import config, store

_Q_TAG = {"q1": 1, "q2": 2, "q3": 3, "q4": 4}
_FRONT = re.compile(r"^Q([1-4])\s+(\d{4})$", re.IGNORECASE)
_FRONT_FY = re.compile(r"^(?:FY|fiscal(?:\s+year)?)\s*(\d{4})$", re.IGNORECASE)

#: A date this far from an anchor is the same period — 52/53-week years drift a few days.
NEAR_DAYS = 6
#: …and this far from an anchor shifted by whole fiscal years is the same quarter, N years back.
OFFSET_DAYS = 9
MAX_YEARS_BACK = 14

ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4}


@dataclass(frozen=True)
class Key:
    fiscal_year: int | None
    fiscal_period: str          # Q1..Q4 | H1 | H2 | M9 | FY | TTM
    grade: str

    @property
    def ok(self) -> bool:
        return self.fiscal_year is not None and self.fiscal_period != ""

    def label(self) -> str:
        return f"FY{self.fiscal_year}{self.fiscal_period}" if self.ok else ""


UNRESOLVED = Key(None, "", "unresolved")


def _d(text) -> date | None:
    if not text:
        return None
    try:
        y, m, dd = (int(p) for p in str(text)[:10].split("-"))
        return date(y, m, dd)
    except (ValueError, TypeError):
        return None


def _period_of(quarter: int, span: int) -> str:
    """Position plus length. The same date is a quarter or a half depending on the span."""
    if not quarter:
        return ""
    if span in (0, 3):
        return f"Q{quarter}"
    if span == 6:
        return "H1" if quarter == 2 else ("H2" if quarter == 4 else "M6")
    if span == 9:
        return "M9" if quarter == 3 else "M9"
    if span == 12:
        # ⚠️ twelve months ending on Q4 is the fiscal year; ending anywhere else it is a
        # trailing-twelve-month figure, which is a different thing and must not be filed as FY.
        return "FY" if quarter == 4 else "TTM"
    return f"Q{quarter}"


class Resolver:
    """Learned anchors plus the rules for reaching a period they do not cover."""

    def __init__(self) -> None:
        self.anchors: dict[str, dict[date, tuple[int, int]]] = defaultdict(dict)
        self.votes: dict[tuple[str, date], Counter] = defaultdict(Counter)
        self.doc_context: dict[str, tuple[int | None, int | None]] = {}
        self._learn()

    # ---------------------------------------------------------------- learning

    def _learn(self) -> None:
        cal = {r["doc_id"]: r for r in store.read(config.EXTRACTED / "document_calendar.parquet")}
        lanes = {r["doc_id"]: r for r in store.read(config.EXTRACTED / "document_lanes.parquet")}

        for doc_id, row in cal.items():
            lane = lanes.get(doc_id, {})
            ticker = lane.get("ticker") or ""
            front = (row.get("frontmatter_period") or "").strip()
            tag = _Q_TAG.get((row.get("filename_tag") or "").lower())
            m = _FRONT.match(front)
            fy = int(m.group(2)) if m else None
            fq = int(m.group(1)) if m else None
            if fy is None:
                fym = _FRONT_FY.match(front)
                if fym:
                    fy, fq = int(fym.group(1)), 4
            # the document's own period, used to resolve bare phrases inside it
            quarter = tag or fq
            self.doc_context[doc_id] = (fy, quarter)

            # ⛔ anchors come only from lanes that state accounts; a transcript's frontmatter
            # period describes the conversation, not the accounts (an AGM held in May 2026 is
            # labelled `Q2 2027`), and one such row would poison every date near it.
            if lane.get("lane_family") not in ("earnings_release", "periodic_report"):
                continue
            period_end = _d(row.get("period_end"))
            if period_end is None or fy is None or quarter is None:
                continue
            self.votes[(ticker, period_end)][(fy, quarter)] += 1

        for (ticker, period_end), counter in self.votes.items():
            (fy, quarter), _n = counter.most_common(1)[0]
            self.anchors[ticker][period_end] = (fy, quarter)
        self.pruned = self._prune()
        self.repaired = self._repair_sequence()

    #: A fiscal quarter is ~91 days. Two anchors closer than this cannot both be quarter ends.
    MIN_QUARTER_DAYS = 70

    def _prune(self) -> list[tuple[str, str, str]]:
        """Drop anchors the quarter-spacing invariant refuses. Returns what was dropped and why.

        ⛔ **A single stray date becomes a permanent mis-dating.** Measured: ADI acquired an
        anchor at **2023-06-30 labelled FY2023Q3** on one vote, sitting 62 days after the real Q2
        (2023-04-29) and 29 days before the real Q3 (2023-07-29) — a calendar-quarter date read
        off a table in a document whose own accounts end elsewhere. Every fact within six days of
        it would have resolved to the wrong period, and nothing downstream could see it.

        The rule needs no judgement: fiscal quarters are ~91 days apart, so where two anchors sit
        closer than 70 days the one with fewer votes is not a period end. Ties are left alone
        rather than guessed at.
        """
        dropped = []
        for ticker, table in self.anchors.items():
            ordered = sorted(table)
            for left, right in zip(ordered, ordered[1:]):
                if (right - left).days >= self.MIN_QUARTER_DAYS:
                    continue
                lv = sum(self.votes[(ticker, left)].values())
                rv = sum(self.votes[(ticker, right)].values())
                if lv == rv:
                    continue
                loser = left if lv < rv else right
                keeper = right if lv < rv else left
                if loser in table:
                    del table[loser]
                    dropped.append((ticker, loser.isoformat(),
                                    f"{(right - left).days}d from {keeper.isoformat()} "
                                    f"({min(lv, rv)} vote vs {max(lv, rv)})"))
        return dropped

    def _repair_sequence(self) -> list[tuple[str, str, str, str]]:
        """Fiscal quarters advance one at a time. Repair any anchor that says otherwise.

        ⛔ **Two anchors a year apart both read `FY2025Q4` for Home Depot.** Its fiscal 2024 ran
        Feb-2024 → Feb-2025, so `2025-02-02` is FY2024Q4 — but one document labelled it 2025 and
        won a split vote. The damage was silent and compounding: FY2024 lost its fourth quarter so
        no annual total could be derived for it, FY2025 acquired **two** fourth quarters, and the
        FY2025 quarter-sum came to **166,189** against a stated **159,514**. Both numbers are real;
        one is the wrong year. HD's full-year guide is applied to that base, so the error would
        have flowed straight into two submitted figures.

        The invariant needs no external data: ordered by period end, `(fiscal_year, quarter)` must
        increase by exactly one quarter each step. Where it does not, the anchor with the weaker
        support is corrected to what the sequence requires — a unanimous neighbour is never
        overruled by a contested one.
        """
        repaired = []
        for ticker, table in self.anchors.items():
            ordered = sorted(table)
            for i in range(1, len(ordered)):
                prev_end, cur_end = ordered[i - 1], ordered[i]
                pfy, pq = table[prev_end]
                cfy, cq = table[cur_end]
                want = (pfy + 1, 1) if pq == 4 else (pfy, pq + 1)
                if (cfy, cq) == want:
                    continue
                prev_votes = self.votes[(ticker, prev_end)]
                cur_votes = self.votes[(ticker, cur_end)]
                prev_firm = len(prev_votes) == 1
                cur_firm = len(cur_votes) == 1
                # only overrule the contested one; if both are firm this is a real gap, not an error
                if cur_firm and not prev_firm:
                    back = (cfy - 1, 4) if cq == 1 else (cfy, cq - 1)
                    table[prev_end] = back
                    repaired.append((ticker, prev_end.isoformat(),
                                     f"FY{pfy}Q{pq}", f"FY{back[0]}Q{back[1]}"))
                elif prev_firm and not cur_firm:
                    table[cur_end] = want
                    repaired.append((ticker, cur_end.isoformat(),
                                     f"FY{cfy}Q{cq}", f"FY{want[0]}Q{want[1]}"))
        return repaired

    # ---------------------------------------------------------------- resolving

    def by_date(self, ticker: str, period_end, span_months: int = 3) -> Key:
        target = _d(period_end)
        if target is None:
            return UNRESOLVED
        table = self.anchors.get(ticker) or {}
        if not table:
            return UNRESOLVED

        hit = table.get(target)
        if hit:
            return Key(hit[0], _period_of(hit[1], span_months), "anchored")

        best = min(table, key=lambda a: abs((a - target).days), default=None)
        if best is not None and abs((best - target).days) <= NEAR_DAYS:
            fy, quarter = table[best]
            return Key(fy, _period_of(quarter, span_months), "near")

        # ⚡ a whole number of fiscal years from an anchor: a prior-year comparative column is the
        # same quarter one year back, and 52/53-week years make the gap 364 or 371 days, never 365
        for years in range(1, MAX_YEARS_BACK + 1):
            for step in (364, 365, 371):
                for direction in (1, -1):
                    shifted = target + timedelta(days=direction * years * step)
                    near = min(table, key=lambda a: abs((a - shifted).days), default=None)
                    if near is not None and abs((near - shifted).days) <= OFFSET_DAYS:
                        fy, quarter = table[near]
                        return Key(fy - direction * years,
                                   _period_of(quarter, span_months), "offset")
        return UNRESOLVED

    def by_year(self, ticker: str, year, span_months: int, quarter_hint: int = 0) -> Key:
        """A column headed only with a YEAR — `| … | 2025 | 2024 |`.

        ⛔ **Hays' entire annual disclosure is keyed this way** and it has no date anywhere in the
        header, so the date resolver cannot see it and the phrase resolver rejects a bare `2025`
        for having no period word. The span supplies what the header omits: a 12-month column
        headed `2025` in an annual statement is FY2025.

        ⚠️ A bare year with a *quarterly* span is genuinely ambiguous — the year does not say
        which quarter — so it is refused rather than guessed.
        """
        try:
            fy = int(str(year)[:4])
        except (TypeError, ValueError):
            return UNRESOLVED
        if not 1990 <= fy <= 2100:
            return UNRESOLVED
        if quarter_hint:
            # the table names its quarter in words above the year columns
            return Key(fy, _period_of(quarter_hint, span_months or 3), "year_column")
        if span_months == 12:
            return Key(fy, "FY", "year_column")
        if span_months == 6:
            return Key(fy, "H1", "year_column")
        return UNRESOLVED

    def by_phrase(self, ticker: str, phrase: str, doc_id: str | None = None) -> Key:
        """`"the second quarter of fiscal 2025"` → FY2025 Q2. A bare quarter takes the document's
        fiscal year; a bare "full year" takes the document's year too."""
        text = " ".join((phrase or "").lower().split())
        if not text:
            return UNRESOLVED
        ctx_year, _ctx_q = self.doc_context.get(doc_id or "", (None, None))

        year = None
        ym = re.search(r"(?:fiscal(?:\s+year)?|fy)\s*(\d{4})", text) or re.search(r"\b(20\d{2})\b",
                                                                                 text)
        if ym:
            year = int(ym.group(1))

        quarter = None
        qm = re.search(r"\b(first|second|third|fourth)\s+quarter\b", text)
        if qm:
            quarter = ORDINAL[qm.group(1)]
        else:
            qm = re.search(r"\bq([1-4])\b", text)
            if qm:
                quarter = int(qm.group(1))

        # ⛔ **`"fiscal 2026"` is a full-year phrase and matching only `"fiscal year"` missed it.**
        # Home Depot's guidance is entirely annual and it writes the period exactly that way, so
        # every HD guide resolved to nothing and the conservatism signal was silent for the one
        # company whose forecast depends on splitting an annual guide.
        if quarter is None and re.search(
                r"\b(?:full[\s-]?year|fiscal\s+year|twelve\s+months|annual"
                r"|fiscal\s*(?:19|20)\d{2}|fy\s*(?:19|20)?\d{2})\b", text):
            period = "FY"
        elif re.search(r"\b(?:six\s+months|first\s+half|half[\s-]?year|h1)\b", text):
            period = "H1"
        elif re.search(r"\b(?:second\s+half|h2)\b", text):
            period = "H2"
        elif re.search(r"\bnine\s+months\b", text):
            period = "M9"
        elif quarter:
            period = f"Q{quarter}"
        else:
            return UNRESOLVED

        if year is None:
            year = ctx_year
            if year is None:
                return UNRESOLVED
            return Key(year, period, "document")
        return Key(year, period, "phrase")


def build() -> tuple[list[dict], dict]:
    resolver = Resolver()
    rows = []
    for ticker, table in resolver.anchors.items():
        for period_end, (fy, quarter) in sorted(table.items()):
            votes = resolver.votes[(ticker, period_end)]
            rows.append({
                "ticker": ticker, "period_end": period_end.isoformat(),
                "fiscal_year": fy, "quarter": quarter,
                "fiscal_period": f"Q{quarter}",
                "label": f"FY{fy}Q{quarter}",
                "n_votes": sum(votes.values()),
                "unanimous": len(votes) == 1,
            })
    rows.sort(key=lambda r: (r["ticker"], r["period_end"]))

    # a free invariant: within a ticker, consecutive anchored quarters are ~91 days apart
    gaps = defaultdict(list)
    for ticker in {r["ticker"] for r in rows}:
        mine = [r for r in rows if r["ticker"] == ticker]
        for a, b in zip(mine, mine[1:]):
            gaps[ticker].append((_d(b["period_end"]) - _d(a["period_end"])).days)
    stats = {"anchors": len(rows),
             "not_unanimous": sum(1 for r in rows if not r["unanimous"]),
             "pruned": resolver.pruned,
             "gaps": {t: Counter(g).most_common(4) for t, g in gaps.items()}}
    return rows, stats


def fiscal_weeks(anchors: list[dict]) -> list[dict]:
    """52 or 53, per (ticker, fiscal year), measured from the filer's own quarter ends.

    ⛔ **A 53-week year dilutes every quarter's share of it.** Home Depot, ADI and Deere all run
    52/53-week calendars, so roughly every sixth year carries a 14-week quarter — ~7.7 % more
    selling days. Deere's FY2026 has it: its Q1 ended **371 days** after Q1 FY2025 where a normal
    year steps 364. The extra week landed in Q1, so Q3 is a normal 13 weeks but is now 13/53 of the
    year rather than 13/52, and splitting an annual guide by an unadjusted seasonal share overstates
    Q3 by about 1.9 %.

    ⛔ **Measure the year end to the year end, and nothing else.** Testing *any* quarter's
    year-on-year step reported Deere's FY2025 **and** FY2026 as 53 weeks, which cannot both be true.
    The step from one Q1 to the next spans the four quarters *ending* at that Q1 — a window that
    straddles two fiscal years — so a long step gets attributed to whichever year it happens to be
    read from. Deere's extra week sits in FY2025 (Q4 ended 2025-11-02, **371 days** after FY2024's),
    which makes FY2026 an ordinary 52 weeks; the loose test claimed the opposite and would have
    shrunk Deere's guided Q3 by ~1.9 % for a week that is not there.

    A fiscal year is its own Q4-to-Q4 span. A year whose Q4 has not been reported yet cannot be
    measured, and says so rather than guessing.
    """
    ends = {(a["ticker"], a["fiscal_year"], a["quarter"]): _d(a["period_end"]) for a in anchors}
    rows = []
    for ticker, fiscal_year in sorted({(a["ticker"], a["fiscal_year"]) for a in anchors}):
        this_end = ends.get((ticker, fiscal_year, 4))
        last_end = ends.get((ticker, fiscal_year - 1, 4))
        span = (this_end - last_end).days if this_end and last_end else None
        rows.append({"ticker": ticker, "fiscal_year": fiscal_year,
                     "weeks": 53 if span is not None and span >= 370 else 52,
                     "year_span_days": span,
                     "measured": span is not None})
    return rows


def main() -> int:
    rows, stats = build()
    store.write(config.FEATURE / "fiscal_periods.parquet", rows)
    weeks = fiscal_weeks(rows)
    store.write(config.FEATURE / "fiscal_weeks.parquet", weeks)
    long_years = [w for w in weeks if w["weeks"] == 53]
    print(f"feature/fiscal_weeks.parquet {len(weeks)} (ticker, year) calendars, "
          f"{len(long_years)} of 53 weeks: "
          + ", ".join(f"{w['ticker']} FY{w['fiscal_year']}" for w in long_years[-6:]))
    print(f"feature/fiscal_periods.parquet {stats['anchors']:,} learned anchors "
          f"({stats['not_unanimous']} with a split vote)")
    if stats["pruned"]:
        print(f"  {len(stats['pruned'])} refused by the quarter-spacing invariant:")
        for ticker, when, why in stats["pruned"]:
            print(f"    {ticker:<9}{when}  {why}")
    for ticker in ("HD", "ADI", "LSE:HAS", "DE"):
        mine = [r for r in rows if r["ticker"] == ticker]
        if not mine:
            continue
        print(f"  {ticker:<9}{len(mine):>3} anchors  {mine[0]['label']} ({mine[0]['period_end']})"
              f" .. {mine[-1]['label']} ({mine[-1]['period_end']})")
        print(f"           quarter-to-quarter gaps: {stats['gaps'][ticker]}")

    # resolve the four target periods as a sanity check
    resolver = Resolver()
    targets = store.read(config.EXTRACTED / "target_periods.parquet")
    print("\n  the four target periods, resolved from their projected end dates:")
    for t in targets:
        span = 12 if t["ticker"] == "LSE:HAS" else 3
        key = resolver.by_date(t["ticker"], t["projected_period_end"], span)
        print(f"    {t['ticker']:<9}{t['target_period']:<11}{t['projected_period_end']}  "
              f"-> {key.label() or '(unresolved)':<12}[{key.grade}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
