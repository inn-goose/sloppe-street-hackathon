"""VENDORED PRE-EXISTING COMPONENT — see vendor/README.md.

The sentence substrate every value-binding rule in a pre-existing SEC extractor shares:
an abbreviation-aware sentence index, a *local* prose-vs-deck test, and evidence quoting.

Carried over unmodified in behaviour from that extractor's stated-date grammar module. It reads any
SEC or RNS document and knows nothing about the four companies in this challenge.
"""

from __future__ import annotations

import bisect
import re

_ABBR = {
    "inc", "corp", "co", "ltd", "llc", "llp", "lp", "plc", "na", "sa", "ag", "nv", "bv",
    "us", "uk", "st", "mr", "mrs", "ms", "dr", "jr", "sr", "no", "nos", "vs", "etc",
    "approx", "prof", "gen", "sen", "rep", "ave", "blvd", "rd", "dept", "est", "fig", "al",
    "am", "pm", "eg", "ie", "cf", "ca", "dba", "cir", "sec", "reg", "art", "para", "ch",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
}
# The closing quote is load-bearing: without it `day." Earnings Call The Company will host …`
# is one sentence, and a value binds to the wrong subject.
_SPLIT_CAND = re.compile(r"\.[\"']?\s+(?=[\"'(]?[A-Z0-9])")
_LAST_TOKEN = re.compile(r"([A-Za-z][A-Za-z.&]*)$")


def _is_abbrev(text: str, pos: int) -> bool:
    m = _LAST_TOKEN.search(text, max(0, pos - 40), pos)
    if not m:
        return False
    tok = m.group(1).replace(".", "").lower()
    return len(tok) <= 1 or tok in _ABBR


def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of *text*'s sentences — flat prose, abbreviation-aware."""
    out, prev = [], 0
    for m in _SPLIT_CAND.finditer(text):
        if _is_abbrev(text, m.start()):
            continue
        out.append((prev, m.start() + 1))
        prev = m.end()
    out.append((prev, len(text)))
    return [(a, b) for a, b in out if b > a]


class Sentences:
    """An abbreviation-aware sentence index over one document, built once."""

    __slots__ = ("spans", "_starts")

    def __init__(self, text: str):
        self.spans = sentences(text)
        self._starts = [a for a, _b in self.spans]

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, i):
        return self.spans[i]

    def __iter__(self):
        return iter(self.spans)

    def index_at(self, pos: int) -> int:
        return max(0, bisect.bisect_right(self._starts, pos) - 1)

    def at(self, pos: int) -> tuple[int, int]:
        return self.spans[self.index_at(pos)] if self.spans else (0, 0)


_DECK_DIGITS = 0.085
_DECK_SENTENCE = 700
_DECK_MIN_LEN = 2000


def is_prose(text: str) -> bool:
    """Whether *text* reads as prose (a release / RNS body) rather than a slide deck."""
    if len(text) < _DECK_MIN_LEN:
        return True
    if sum(c.isdigit() for c in text) / len(text) > _DECK_DIGITS:
        return False
    stops = text.count(". ")
    return stops > 0 and len(text) / stops <= _DECK_SENTENCE


def is_prose_at(text: str, start: int, end: int, reach: int = 1300) -> bool:
    """Prose-vs-deck judged **around the mention**, not over the document.

    An earnings release is prose at the top and financial tables at the bottom, so a
    document-wide digit ratio calls the whole thing a deck and silently disables the
    proximity tiers on the very sentence that announces the number.
    """
    return is_prose(text[max(0, start - reach):end + reach])


EVIDENCE = 400


def evidence(text: str, sent: tuple[int, int], start: int, end: int) -> str:
    """The sentence, centred on the mention when it is too long to show whole."""
    a, b = sent
    if b - a <= EVIDENCE:
        return re.sub(r"\s+", " ", text[a:b])
    pad = (EVIDENCE - (end - start)) // 2
    lo, hi = max(a, start - pad), min(b, end + pad)
    return re.sub(r"\s+", " ", text[lo:hi])


_evidence = evidence
