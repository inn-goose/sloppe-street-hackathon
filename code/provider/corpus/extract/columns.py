"""extracted/table_columns + table_spec — what each numeric column of each table actually is.

This is the join key the whole store turns on. `table_cells` says "the third number on the row
labelled *Net sales*"; this lane says "the third number is the % change against the prior-year
quarter, and the table is written in millions".

## How the binding is decided

1. **The column-header row is the table's own**, found by scoring every candidate row in the
   table's head and taking the most *specific* one. ⚠️ This is the correction that mattered most:
   the 2015-era HD release writes a **span row** above the date row —
   `| | Three Months Ended | Three Months Ended | % | Fiscal Year Ended | … |` — and it is a
   perfectly good header row by every structural test, so a first-match scan binds to it and every
   column loses its date. A date token is worth more than a year, a year more than a span; the
   highest score wins and the rows above it become the span context.
2. **Alignment is LEFT, and it is graded.** ⚠️ That same 2015 release writes seven header tokens
   over six numeric columns — `Increase (Decrease)` appears a third time because the converter
   split a trailing `%` into its own cell. Aligning from the right shifts every column by one.
3. **The binding is then CHECKED against arithmetic the table states itself** (see
   `statements.py`): where a (current, prior, %change) triple is bound, `current/prior − 1` must
   reproduce the stated change. Judgement-free, needs no external data, and fails loudly when the
   columns are shifted.

⛔ **This extractor re-parses the banked document rather than reading a summary of it.** A first
version passed the head of each table through a truncated string blob and **half the corpus's
tables came back unbound** — the blob cut mid-row on any wide statement. A lossy intermediate
between two extractors is a silent recall hole, so there isn't one.

## Two traps this lane exists to hold

⚠️ **A bare year IS a number.** `| | 2026 | | 2025 | | % Change |` parses as three numeric tokens
under any honest number grammar, so a header row cannot be found by "the row with no numbers". It
is found by what its tokens *are*.

⚠️ **The scale is not on the number.** `13,369` is thirteen billion or thirteen thousand depending
on a parenthetical six lines above the table. A mis-scaled pair is internally consistent, so no
invariant can catch it — the declaration has to be read, and where it is absent the table is
marked `scale=""` rather than assumed.
"""

from __future__ import annotations

import re
from collections import Counter

from code.lib import config, store
from code.lib.text import iter_tables, parse_number
from code.provider.corpus.extract import _corpus

_MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december"
           "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
# "August 3,2025" — the converter drops the space after the comma, so it is optional.
_DATE = re.compile(rf"\b({_MONTHS})\.?\s+(\d{{1,2}})\s*,?\s*((?:19|20)\d{{2}})\b", re.IGNORECASE)
_YEAR_ONLY = re.compile(r"^\(?(?:FY\s*)?((?:19|20)\d{2})\)?\*?$", re.IGNORECASE)
_CHANGE = re.compile(
    r"(?:%|\bchange\b|\bincrease\b|\bdecrease\b|\bgrowth\b|\bvariance\b|\bvs\.?\b|\bbps\b"
    r"|\bbasis\s*points?\b|\bfav(?:ourable)?\b|\bunfav\w*\b|\bdiff\w*\b|\bmovement\b)",
    re.IGNORECASE)
_LFL = re.compile(r"\b(?:lfl|like[\s-]for[\s-]like|organic|constant\s+currency|underlying)\b",
                  re.IGNORECASE)
_GUIDE = re.compile(r"\b(?:guidance|outlook|forecast|estimate|target|plan|budget)\b", re.IGNORECASE)
#: ⚡ A column naming the BASIS a figure is measured on, not a period and not a change. Hays heads
#: its net-fee table `| | Actual | LFL |`; "Actual" fell through to `other`, which made it look
#: like a stub, which made the whole header unusable — so the binder fell back to the span row and
#: the two measures became indistinguishable. Actual carries FX and disposals; LFL strips them.
_BASIS = re.compile(r"^\s*(?:actual|reported|as\s+reported|statutory|headline|pro\s*forma"
                    r"|adjusted|total)\s*$", re.IGNORECASE)
_PERIODIC = re.compile(
    r"\b(?:three|six|nine|twelve|3|6|9|12)\s+months?\s+ended"
    r"|\b(?:first|second|third|fourth)\s+quarter"
    r"|\bquarter\s+ended|\byear\s+(?:to\s+date|ended|ending)|\bfiscal\s+year\s+ended"
    r"|\bhalf[\s-]?year|\bsix\s+months|\bfull\s+year|\bytd\b|\bq[1-4]\b"
    r"|\bperiod\s+ended|\bweeks?\s+ended|\bmonths?\s+ended", re.IGNORECASE)

_SCALE_WORDS = re.compile(
    r"\b(?:in\s+)?(?P<scale>millions?|billions?|thousands?)\b"
    r"|£s?\s*(?P<gbp>million|billion|thousand)\b"
    r"|\$\s*in\s+(?P<usd>millions?|billions?)", re.IGNORECASE)
_PER_SHARE_STUB = re.compile(r"per[\s-]share|per\s+diluted|except\s+per", re.IGNORECASE)

_SPAN_MONTHS = (
    (re.compile(r"\btwelve\s+months|\bfiscal\s+year\s+ended|\byear\s+ended|\bfull[\s-]year"
                r"|\byear\s+to\s+date\b|\b12\s+months", re.IGNORECASE), 12),
    (re.compile(r"\bnine\s+months|\b9\s+months|\bthird\s+quarter\s+and\s+nine", re.IGNORECASE), 9),
    (re.compile(r"\bsix\s+months|\b6\s+months|\bhalf[\s-]?year|\bfirst\s+half"
                r"|\bsecond\s+half", re.IGNORECASE), 6),
    (re.compile(r"\bthree\s+months|\b3\s+months|\bquarter", re.IGNORECASE), 3),
)

#: How far into a table to look for its column header. ⚠️ Measured rather than assumed: raising
#: this from 6 to 12 moved unbound tables by a rounding error, which says the unbound mass is not
#: a scan-depth problem — those tables genuinely carry no period dimension (fair-value levels,
#: maturity buckets, reconciliation footnotes) and refusing them is correct.
MAX_HEADER_SCAN = 12

_QUARTER_WORD = re.compile(r"\b(first|second|third|fourth)\s+quarter\b", re.IGNORECASE)
_QUARTER_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4}
#: How specific each token kind is. A date pins a column to a day; a span only says "a quarter".
_SPECIFICITY = {"period_date": 12, "period_year": 6, "change": 2, "lfl_change": 2,
                "guidance": 2, "basis": 2, "period_span": 1, "other": 0, "empty": 0}


def _classify(token: str) -> str:
    t = token.strip()
    if not t:
        return "empty"
    if _GUIDE.search(t):
        return "guidance"
    if _LFL.search(t):
        return "lfl_change"
    if _BASIS.match(t):
        return "basis"
    if _DATE.search(t):
        return "period_date"
    if _YEAR_ONLY.match(t):
        return "period_year"
    if _CHANGE.search(t):
        return "change"
    if _PERIODIC.search(t):
        return "period_span"
    return "other"


_MONTH_NUM = {n: i + 1 for i, n in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _period_end(token: str) -> str | None:
    m = _DATE.search(token)
    if not m:
        return None
    key = m.group(1)[:3].lower()
    if key not in _MONTH_NUM:
        return None
    return f"{int(m.group(3)):04d}-{_MONTH_NUM[key]:02d}-{int(m.group(2)):02d}"


def _period_year(token: str) -> int | None:
    m = _YEAR_ONLY.match(token.strip())
    if m:
        return int(m.group(1))
    m = _DATE.search(token)
    return int(m.group(3)) if m else None


def _span_months(text: str) -> int | None:
    for rx, months in _SPAN_MONTHS:
        if rx.search(text):
            return months
    return None


def _scale(*texts: str) -> tuple[str, str]:
    """(scale word, declaration verbatim). The nearest declaration wins."""
    for text in texts:
        if not text:
            continue
        last = None
        for last in _SCALE_WORDS.finditer(text):
            pass
        if last:
            word = (last.group("scale") or last.group("gbp") or last.group("usd") or "")
            return word.lower().rstrip("s"), " ".join(last.group(0).split())
    return "", ""


_SCALE_ONLY = re.compile(r"^[^A-Za-z]*(?:\$|£|€)?\s*(?:in\s+)?(?:millions?|billions?|thousands?)"
                         r"[\s,\w]*$", re.IGNORECASE)


_HEADING = re.compile(r"^\s{0,3}#{1,4}\s+(.{2,80}?)\s*$", re.MULTILINE)


def _scope_label(rows, header_idx: int, pre_context: str = "") -> str:
    """WHICH entity or segment this table is about, read off the caption above its header.

    ⛔ **Without this, a segment figure is indistinguishable from a consolidated one.** Deere's
    Q2-FY2026 release states `Operating profit` three times — 706 for Production & Precision
    Agriculture, 719 for Small Ag & Turf, 561 for Construction & Forestry — in three tables whose
    only difference is the caption above the header. One of the twelve submitted numbers **is**
    that P&PA line, and cross-lane reconciliation traced Deere's revenue and net-income
    disagreements against SEC XBRL to exactly this: a segment row read as the consolidated total
    (2017-04-30 corpus 2.97 bn against SEC 8.29 bn).

    The caption is the first cell above the header that names something rather than a period, a
    change or a units declaration.
    """
    for row in rows[:header_idx]:
        for cell in row.cells:
            text = cell.strip()
            if len(text) < 3 or _SCALE_ONLY.match(text):
                continue
            if _classify(text) != "other":
                continue
            return text[:80]
    # ⛔ **The scope can be a MARKDOWN HEADING above the table, not a cell inside it.** Hays'
    # preliminary report repeats one table verbatim per division — `| Year ended 30 June | 2025 |
    # 2024 | Reported | LFL |` with rows `Net fees` and `Pre-exceptional operating profit` — and
    # names the division only in a `## Germany` heading above it. Measured, that made Germany's
    # 308.9 and 52.1 indistinguishable from the group's 972.4 and 45.6, and both are two of the
    # twelve targets.
    headings = _HEADING.findall(pre_context or "")
    if headings:
        return headings[-1].strip()[:80]
    return ""


def _span_by_position(rows, header_idx: int, header_cells: list[str],
                      stub_offset: int) -> list[int]:
    """The span in months governing EACH column, resolved POSITIONALLY.

    ⛔ **A spanning header is positional and collapsing it destroys the quarter.** Deere writes

        | Deere & Company |  | Second Quarter |  | Year to Date |  |
        | $ in millions   |  | 2026 | 2025 | % Change | 2026 | 2025 | % Change |
        | Net sales …     |  | 13,369 | 12,763 | 5% | 22,981 | 21,272 | 8% |

    — the first three value columns are a **3-month** period and the next three are **6-month**.
    Taking one span for the whole table (the first version did) makes 13,369 and 22,981
    indistinguishable, so a quarterly panel silently ingests year-to-date figures. Measured before
    this fix: every Deere `net sales and revenues` fact carried the same span.

    Both rows live in the same cell grid, so each header token is governed by the nearest span
    token at or to the left of it — the standard resolution, and it needs no guessing.
    """
    spans: list[tuple[int, int]] = []
    for row in rows[:header_idx]:
        for index, cell in enumerate(row.cells):
            months = _span_months(cell)
            if months:
                spans.append((index, months))
    if not spans:
        # ⛔ **The stub itself can carry the span, and Hays' does.** Its accounts head the table
        # `| Year ended 30 June (In £s million) | 2025 | 2024 | …` with **no row above it**, so a
        # span search that only looks upward finds nothing and every Hays figure is filed with an
        # unknown length. That is what left the net-fee series with a single period.
        stub_span = _span_months(header_cells[0] if header_cells else "")
        if stub_span:
            positions = [i for i, c in enumerate(header_cells) if c.strip()][stub_offset:]
            return [stub_span] * len(positions)
        return []
    spans.sort()
    # cell indices of the header's own value columns, after the stub
    header_positions = [i for i, c in enumerate(header_cells) if c.strip()][stub_offset:]
    out = []
    for position in header_positions:
        governing = 0
        for index, months in spans:
            if index <= position:
                governing = months
            else:
                break
        out.append(governing or spans[0][1])
    return out


def _header_candidate(cells: list[str]) -> tuple[list[str], bool, int] | None:
    """(column tokens WITH the stub already removed, had-a-stub, specificity score).

    ⛔ **The returned list is already stub-stripped.** A first version returned the stripped list
    *and* a `start` offset, and the caller applied the offset again — stripping the first real
    column a second time. Measured on Deere's segment table
    `| $ in millions | 2026 | 2025 | % Change |`: the bound columns became `[2025, % Change]`, so
    the **current** year's figure was filed under the **prior** year and the table graded `short`
    for a width it actually matched. That single off-by-one mis-dated Deere's segment operating
    profit — one of the twelve submitted numbers — by a full year, with entirely plausible values
    throughout. Returning a boolean instead of an offset makes the mistake unrepresentable.
    """
    non_empty = [c.strip() for c in cells if c.strip()]
    if len(non_empty) < 2:
        return None
    kinds = [_classify(c) for c in non_empty]
    had_stub = False
    # ⛔ **A stub can read as a PERIOD and still be a stub.** Hays heads its statements
    # `| Year ended 30 June (In £s million) | 2025 | 2024 | Actual growth | LFL growth |` — the
    # first cell matches "year ended" and so classifies as `period_span`, not as `other`. Treated
    # as a value column it shifts everything one place left, and the **FY2024** figure is filed
    # as **FY2025**: measured, Hays' pre-exceptional basic EPS returned 4.03p (the prior year)
    # where the accounts say 1.31p. It also swallowed the `(In £s million)` scale declaration.
    # A leading `period_span` is descriptive; a value column is a date, a year or a change.
    if kinds[0] in ("other", "empty", "period_span"):
        had_stub, kinds, non_empty = True, kinds[1:], non_empty[1:]
    if len(kinds) < 2 or any(k in ("other", "empty") for k in kinds):
        return None
    if not any(k.startswith("period") for k in kinds):
        # ⚡ **A MEASURE header, not a period one — and refusing it lost Hays' whole disclosure.**
        # Its quarterly updates head the net-fee table `| | Actual | LFL |`: the columns are two
        # ways of measuring the same growth, and the period is the DOCUMENT's ("Quarterly Update
        # for the Three Months Ended 30 June 2026"). Requiring a period column rejected all four
        # FY2026 updates outright — the exact series the Hays forecast is built from, for the name
        # that is 25 % of the score.
        #
        # Accepted only when every token is a measure and at least two are, so an ordinary data
        # row cannot masquerade as a header. The period is then taken from the document, and the
        # rows carry `period_from_document` so a consumer knows which clock it came off.
        # ⛔ **A data row of percentages looks exactly like a measure header** — `| Germany |
        # (8)% | (8)% |` classifies as `change, change` just as `| | Actual | LFL |` does. Without
        # this guard the binder picked the first data row as the header, and the column token
        # became `(8)%` instead of `Actual`. The distinction is real and load-bearing: Hays'
        # **actual** growth carries FX and the six-country disposal while **LFL** strips both, and
        # the net-fee bridge needs actual. A header names its columns; it does not hold values.
        if (len(kinds) >= 2
                and all(k in ("change", "lfl_change", "guidance", "basis") for k in kinds)
                and all(parse_number(t) is None for t in non_empty)):
            return non_empty, had_stub, sum(_SPECIFICITY[k] for k in kinds)
        return None
    return non_empty, had_stub, sum(_SPECIFICITY[k] for k in kinds)


def build() -> tuple[list[dict], list[dict]]:
    lanes = {r["doc_id"]: r for r in store.read(config.EXTRACTED / "document_lanes.parquet")}

    columns: list[dict] = []
    specs: list[dict] = []

    for doc in _corpus.iter_documents():
        body = doc.body
        lines = body.splitlines()
        lane = lanes.get(doc.doc_id, {})
        for table in iter_tables(body):
            widths = Counter(len(r.numbers) for r in table.rows if r.numbers)
            if not widths:
                continue
            modal = widths.most_common(1)[0][0]

            best = None  # (score, row_index, tokens, had_stub)
            for i, row in enumerate(table.rows[:MAX_HEADER_SCAN]):
                cand = _header_candidate(row.cells)
                if cand is None:
                    continue
                tokens, had_stub, score = cand
                # a row that matches the numeric width exactly is worth more than one that does not
                score += 8 if len(tokens) == modal else 0
                if best is None or score >= best[0]:
                    best = (score, i, tokens, had_stub)

            # ⚠️ 24 lines, not 6. Hays' divisional tables sit well below their `## Germany`
            # heading — with a 6-line window the heading was out of reach and Germany's net fees
            # stayed indistinguishable from the group's. The nearest heading ABOVE a table is what
            # scopes it, so the window has to be wide enough to contain one.
            pre = "\n".join(lines[max(0, table.start_line - 24):table.start_line - 1])
            stub = ""
            if best and best[3]:  # had_stub
                stub = [c.strip() for c in table.rows[best[1]].cells if c.strip()][0]
            scale, scale_src = _scale(stub, pre, table.header_text)
            per_share_stub = bool(_PER_SHARE_STUB.search(f"{stub} {pre}"))

            column_spans: list[int] = []
            if best is None:
                grade, bound, span_text, header_idx = "unbound", [], "", -1
            else:
                _score, header_idx, tokens, had_stub = best
                span_text = " ".join(" ".join(r.cells) for r in table.rows[:header_idx])
                column_spans = _span_by_position(table.rows, header_idx,
                                                 table.rows[header_idx].cells,
                                                 1 if had_stub else 0)
                if len(tokens) == modal:
                    grade, bound = "exact", tokens
                elif len(tokens) > modal:
                    grade, bound = "left_truncated", tokens[:modal]
                else:
                    grade, bound = "short", tokens

            scope = _scope_label(table.rows, header_idx, pre) if header_idx >= 0 else ""
            # a caption naming the filer itself is the consolidated statement, not a segment
            consolidated = (not scope
                            or scope.casefold() in doc.company.casefold()
                            or doc.company.casefold().startswith(scope.casefold()[:12]))
            specs.append({
                "scope": scope, "is_consolidated": consolidated,
                "doc_id": doc.doc_id, "table_idx": table.index, "ticker": doc.ticker,
                "published_at": doc.published_at, "lane_family": lane.get("lane_family", ""),
                "document_type": doc.document_type,
                "modal_width": modal, "n_rows": len(table.rows),
                "n_header_tokens": len(bound), "align_grade": grade,
                "header_row_idx": header_idx, "stub": stub[:160],
                "scale": scale, "scale_source": scale_src[:120],
                "per_share_stub": per_share_stub,
                "span_text": span_text[:200], "span_months": _span_months(span_text) or 0,
            })

            # ⚡ A year column in a quarterly table needs the QUARTER, and the table states it in
            # words: Deere heads its segment tables `Production & Precision Agriculture | Second
            # Quarter` over columns headed `2026 | 2025 | % Change`. Without this the year alone
            # is unresolvable and the segment's operating profit — a submitted number — never
            # enters the panel.
            qm = _QUARTER_WORD.search(f"{span_text} {scope} {pre}")
            quarter_hint = _QUARTER_ORDINAL.get(qm.group(1).lower()) if qm else 0
            table_span = _span_months(span_text) or 0
            period_from_document = bool(bound) and not any(
                _classify(t).startswith("period") for t in bound)
            for ordinal, token in enumerate(bound):
                kind = _classify(token)
                # precedence: the column's own words, then the span row governing ITS position,
                # then the table's. Only the middle one distinguishes a quarter from a YTD.
                own_span = (_span_months(token)
                            or (column_spans[ordinal] if ordinal < len(column_spans) else 0))
                columns.append({
                    "doc_id": doc.doc_id, "table_idx": table.index, "ticker": doc.ticker,
                    "published_at": doc.published_at, "ordinal": ordinal,
                    "token": token[:120], "kind": kind,
                    "period_end": _period_end(token),
                    "period_year": _period_year(token),
                    "span_months": own_span or table_span,
                    "quarter_hint": quarter_hint or 0,
                    "period_from_document": period_from_document,
                    "align_grade": grade, "scale": scale,
                    "per_share_stub": per_share_stub,
                    "lane_family": lane.get("lane_family", ""),
                })
    return columns, specs


def main() -> int:
    cols, specs = build()
    store.write(config.EXTRACTED / "table_columns.parquet", cols)
    store.write(config.EXTRACTED / "table_spec.parquet", specs)
    grades = Counter(s["align_grade"] for s in specs)
    kinds = Counter(c["kind"] for c in cols)
    scaled = sum(1 for s in specs if s["scale"])
    dated = sum(1 for c in cols if c["period_end"])
    print(f"extracted/table_spec.parquet    {len(specs):,} tables  {dict(grades)}")
    print(f"                                scale declared on {scaled:,} "
          f"({scaled / max(len(specs), 1):.0%})")
    print(f"extracted/table_columns.parquet {len(cols):,} bound columns  {dict(kinds)}")
    print(f"                                {dated:,} carry an explicit period-end date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
