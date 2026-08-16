"""Markdown → typed tokens. The bottom of the provider layer.

Everything here is a template match, a split or a count over bytes the corpus
already contains. Nothing infers meaning: a cell either parses as a number under
a stated grammar or it is refused, and the refusal is visible to the caller.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterator

# Zero-width and non-breaking junk the PDF→markdown conversion leaves behind.
_JUNK = dict.fromkeys(map(ord, "​‌‍‎‏﻿\xa0⁠"), " ")
_DASHES = {ord(c): "-" for c in "‐‑‒–—―−"}
_CURRENCY = "$£€"

# A cell is numeric only if the WHOLE cell is a number once markers are stripped.
# This is what keeps "Interest and other (income) expense:" from yielding a value.
_NUMBER = re.compile(r"^\(?\s*[-+]?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\)?$")
_BARE_NUMBER = re.compile(r"^\(?\s*[-+]?\s*\d+(?:\.\d+)?\s*\)?$")
#: ⚠️ A footnote marker survives the `{}` scrub as `{(1)}` — Hays writes `Net fees {(1)}`, which
#: is a different label from `Net fees` to any exact matcher and split the series in two.
_TRAILING_FOOTNOTE = re.compile(r"\s*[{\[]?\(\d{1,2}\)[}\]]?\s*$")
_SEPARATOR = re.compile(r"^:?-{2,}:?$")


def scrub(text: str) -> str:
    """Normalise the invisible characters without touching anything meaningful."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_JUNK).translate(_DASHES)
    text = text.replace("{}", " ").replace("&amp;", "&")
    return text


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Read the corpus's JSON-valued YAML header. Same grammar as starter/search.py."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    meta: dict[str, object] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        raw = raw.strip()
        try:
            meta[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            meta[key.strip()] = raw
    return meta, text[end + 5 :]


@dataclass(frozen=True)
class Num:
    """One numeric token, with the unit markers the document itself carried."""

    value: float
    percent: bool = False
    currency: str | None = None
    pence: bool = False
    raw: str = ""


def parse_number(cell: str) -> Num | None:
    """Parse a whole cell as a number, or refuse it.

    Handles the four conventions the corpus actually uses: a leading currency
    symbol, accounting parentheses for negatives, a trailing percent sign, and
    Hays' pence suffix (`1.31p`). `-`, `—` and `N/A` are refusals, not zeros —
    a dash in a financial table means "not applicable", and reading it as 0
    would manufacture a data point.
    """
    raw = scrub(cell).strip()
    if not raw:
        return None

    body = _TRAILING_FOOTNOTE.sub("", raw).strip()
    if body.upper() in {"N/A", "NA", "NM", "-", "--", "*"}:
        return None

    percent = False
    pence = False
    currency = None

    if body.endswith("%"):
        percent = True
        body = body[:-1].strip()
    if body.lower().endswith("bps"):
        percent = True  # basis points are converted by the caller, never here
        body = body[:-3].strip()
    if body.endswith(("p", "P")) and _BARE_NUMBER.match(body[:-1].strip() or "x"):
        pence = True
        body = body[:-1].strip()
    # `"" in "$£€"` is True, so the emptiness guard is load-bearing: a lone `%` cell strips to
    # nothing and would otherwise index off the end.
    if body[:1] and body[:1] in _CURRENCY:
        currency = body[0]
        body = body[1:].strip()
    if body and body.endswith(tuple(_CURRENCY)):
        currency = body[-1]
        body = body[:-1].strip()
    if not body:
        return None

    if not _NUMBER.match(body):
        return None

    negative = body.startswith("(") and body.endswith(")")
    digits = body.strip("()").replace(",", "").replace("+", "").strip()
    if not digits or digits in {"-", "."}:
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    if negative:
        value = -value
    return Num(value=value, percent=percent, currency=currency, pence=pence, raw=raw)


@dataclass
class Row:
    cells: list[str]
    line: int
    label: str = ""
    numbers: list[Num] = field(default_factory=list)


@dataclass
class Table:
    index: int
    start_line: int
    rows: list[Row]

    @property
    def header_text(self) -> str:
        head = self.rows[:3]
        return " ".join(" ".join(c for c in r.cells if c.strip()) for r in head)

    def labelled(self, *needles: str, exact: bool = False) -> list[Row]:
        wanted = [n.casefold() for n in needles]
        out = []
        for row in self.rows:
            label = row.label.casefold()
            if not label:
                continue
            hit = any(label == w for w in wanted) if exact else any(w in label for w in wanted)
            if hit:
                out.append(row)
        return out


def _split_cells(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [scrub(c).strip() for c in inner.split("|")]


def _row_label(cells: list[str]) -> str:
    for cell in cells:
        text = cell.strip()
        if not text or text in _CURRENCY or text == "%":
            continue
        if parse_number(text) is not None:
            return ""
        return _TRAILING_FOOTNOTE.sub("", text).strip().rstrip(":").strip()
    return ""


def _row_numbers(cells: list[str]) -> list[Num]:
    """Ordered numeric tokens, with a `%` in a neighbouring cell folded in.

    The financial tables split `4.9 %` across two cells and `$ 45,277` across
    two more, so a token's unit lives beside it rather than on it.
    """
    nums: list[Num] = []
    pending_currency: str | None = None
    for index, cell in enumerate(cells):
        text = cell.strip()
        if not text:
            continue
        if text in _CURRENCY:
            pending_currency = text
            continue
        if text == "%":
            if nums:
                last = nums[-1]
                nums[-1] = Num(last.value, True, last.currency, last.pence, last.raw)
            continue
        num = parse_number(text)
        if num is None:
            pending_currency = None
            continue
        if pending_currency and num.currency is None:
            num = Num(num.value, num.percent, pending_currency, num.pence, num.raw)
        pending_currency = None
        # a `%` immediately to the right, skipping blanks
        for nxt in cells[index + 1 :]:
            stripped = nxt.strip()
            if not stripped:
                continue
            if stripped == "%":
                num = Num(num.value, True, num.currency, num.pence, num.raw)
            break
        nums.append(num)
    return nums


def iter_tables(body: str) -> Iterator[Table]:
    """Yield every pipe table in the document, separator rows dropped."""
    lines = body.splitlines()
    index = 0
    position = 0
    while position < len(lines):
        if not lines[position].lstrip().startswith("|"):
            position += 1
            continue
        start = position
        block: list[Row] = []
        while position < len(lines) and lines[position].lstrip().startswith("|"):
            cells = _split_cells(lines[position])
            position += 1
            if cells and all(not c or _SEPARATOR.match(c) for c in cells):
                continue
            row = Row(cells=cells, line=position)
            row.label = _row_label(cells)
            row.numbers = _row_numbers(cells)
            block.append(row)
        if len(block) >= 2:
            yield Table(index=index, start_line=start + 1, rows=block)
            index += 1


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z$£€•])")


def sentences(body: str) -> list[str]:
    """Flatten the body to sentences for the stated-value templates."""
    flat = re.sub(r"\s+", " ", scrub(body))
    return [s.strip() for s in _SENTENCE.split(flat) if s.strip()]


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def pct_change(current: float, prior: float) -> float | None:
    if prior in (0, None) or current is None:
        return None
    return (current / prior - 1.0) * 100.0
