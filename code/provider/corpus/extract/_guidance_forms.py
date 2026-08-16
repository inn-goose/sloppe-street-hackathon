"""Guidance forms the vendored SEC grammar does not carry — built during the event.

The vendored grammar (`code/vendor/guidance_grammar.py`) was mined from a broad sample of SEC
earnings releases and it is good at what it saw. Measured against **these four filers**, it caught
Deere's FY26 net-income guide and its segment margins, and it missed Home Depot and Analog Devices
outright. The reason is form, not quality: each of these issuers writes its guide in a shape that
sample did not contain.

Every pattern below was mined from the banked releases, verbatim:

| Form | Filer | Text as written |
|---|---|---|
| `plus_minus` | **ADI** | `revenue of $3.9 billion, +/- $100 million` · `EPS to be $2.60, +/-$0.15` · `approximately 39.0%, +/-150 bps` |
| `flat_to` | **HD** | `Comparable sales growth of approximately flat to 2.0%` · `to grow approximately flat to 4.0%` |
| `approx_range` | **HD** | `Total sales growth of approximately 2.5% to 4.5%` |
| `approx_point` | **HD** | `Gross margin of approximately 33.1%` |
| `directional` | **DE** | `Down 15 to 20%` · `Down 5 to 10%` · `Up ~5%` · `Down ~15%` |

⛔ **`+/-` is ADI's ONLY guidance form**, and no range grammar can see it: there is one number and
a tolerance, not two bounds. Every quarterly guide ADI has ever given — the exact anchor for two
of the twelve targets — was invisible until this existed.

⛔ **`flat` is a bound.** Home Depot guides comparable sales `flat to 2.0%`, so the low bound is a
word. A numeric-only pattern reads `2.0%` as a point and loses the range; worse, it silently drops
the guided-flat case, which is precisely the state HD is in.

⛔ **`Down 15 to 20%` is NEGATIVE and the sign is a word.** Read unsigned it becomes +15 to +20 —
an inversion, not a rounding error, and it would flip the sign of Deere's largest segment.

⚠️ **A proxy statement is not guidance.** The vendored grammar returned equity-grant vesting bands
(`0.00–200.00 %`), PSU payout ranges and CEO salary bands as HD's and ADI's most recent "guidance".
They are ranges in forward-looking language, so no frame can exclude them — the veto has to be on
the *subject*.
"""

from __future__ import annotations

import re

from code.vendor.guidance_grammar import Guidance, _metric, _metric_source, _period
from code.vendor.sentence_grammar import Sentences, evidence

_N = r"\d{1,4}(?:,\d{3})*(?:\.\d+)?"
_SCALE = r"(?:\s*(?:%|percent|billion|million|thousand|bn|mm|bps|basis\s+points))?"
_APPROX = r"(?:(?:approximately|about|roughly|around|circa|~)\s*)?"
_CUR = r"[$£€]?\s?"

#: `$3.9 billion, +/- $100 million` · `$2.60, +/-$0.15` · `39.0%, +/-150 bps`
_PLUS_MINUS = re.compile(
    rf"(?P<mid>{_CUR}{_N})(?P<midscale>{_SCALE})\s*,?\s*"
    rf"\(?\s*(?:\+/-|±|\+ or -|plus or minus)\s*"
    rf"(?P<tol>{_CUR}{_N})(?P<tolscale>{_SCALE})\s*\)?", re.IGNORECASE)

#: `approximately flat to 2.0%` — the low bound is a word.
_FLAT_TO = re.compile(
    rf"{_APPROX}(?P<lo>flat|unchanged)\s+to\s+{_APPROX}(?P<hi>-?{_N})(?P<hiscale>{_SCALE})",
    re.IGNORECASE)

#: `approximately 2.5% to 4.5%` — both bounds carry the unit.
_APPROX_RANGE = re.compile(
    rf"(?:approximately|about|roughly|around)\s+(?P<lo>-?{_CUR}{_N})(?P<loscale>{_SCALE})\s*"
    rf"(?:to|-|through)\s*{_APPROX}(?P<hi>-?{_CUR}{_N})(?P<hiscale>{_SCALE})", re.IGNORECASE)

#: `Gross margin of approximately 33.1%` · `net interest expense of approximately $2.3 billion`
_APPROX_POINT = re.compile(
    rf"\bof\s+(?:approximately|about|roughly|around)\s+(?P<v>-?{_CUR}{_N})(?P<vscale>{_SCALE})",
    re.IGNORECASE)

#: `Down 15 to 20%` · `Up ~5%` · `Down ~15%` · `Flat to up 5%`
_DIRECTIONAL = re.compile(
    rf"\b(?P<dir>down|up|flat)\s*(?:to\s+(?P<dir2>up|down)\s*)?~?\s*"
    rf"(?P<lo>{_N})?\s*(?:(?:to|-|–)\s*~?\s*(?P<hi>{_N}))?\s*(?P<scale>%|percent|bps)",
    re.IGNORECASE)

#: ⚠️ Subject-level veto. Every one of these produced a false "guidance" row on the first run.
_NOT_GUIDANCE = re.compile(
    r"\b(?:equity\s+grant|grant\s+date|RSU|PSU|restricted\s+stock|performance\s+share"
    r"|vest\w*|payout|award|salary|bonus|compensation|proposal|proxy|say[\s-]on[\s-]pay"
    r"|director|shareholder\s+vote|option\s+exercis\w*|severance|retirement|pension\s+plan"
    r"|LIBOR|SOFR|margin\s+payable|interest\s+rate\s+swap|covenant|credit\s+facility"
    r"|maturit\w+|ratings?\s+from\s+time|notional)\b", re.IGNORECASE)

_FORWARD = re.compile(
    r"\b(?:guidance|guides?|guiding|outlook|expects?|expected|expecting|anticipat\w+"
    r"|forecast\w*|targets?|targeting|sees|project\w+|estimat\w+|plan(?:ning|s)?\s+for"
    r"|we\s+are\s+forecasting|reaffirms?|raises?|lowers?)\b", re.IGNORECASE)

LEFT_REACH = 240
RIGHT_REACH = 200

_SCALE_RANK = {"": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9}
_SCALE_WORD = {"bn": "billion", "mm": "million", "percent": "%", "basis points": "bps"}


def _norm_scale(raw: str) -> str:
    tok = (raw or "").strip().lower()
    return _SCALE_WORD.get(tok, tok)


def _number(raw: str) -> float | None:
    try:
        return float(re.sub(r"[^\d.\-]", "", raw))
    except ValueError:
        return None


def _plus_minus_bounds(mid: float, midscale: str, tol: float, tolscale: str):
    """(low, high, unit, scale) with the tolerance converted onto the midpoint's own scale.

    ⚠️ The two halves are routinely written at DIFFERENT magnitudes — `$3.9 billion, +/- $100
    million` — so the tolerance must be rescaled before it is applied. Subtracting 100 from 3.9
    would give a negative revenue guide.

    ⚠️ **bps against a percent is the same problem in another unit**: `39.0%, +/-150 bps` is
    ±1.5 percentage points, not ±150.
    """
    ms, ts = _norm_scale(midscale), _norm_scale(tolscale)
    if ms == "%" or ts in ("%", "bps"):
        tolerance = tol / 100.0 if ts == "bps" else tol
        return mid - tolerance, mid + tolerance, "%", ""
    factor = _SCALE_RANK.get(ts, 1.0) / _SCALE_RANK.get(ms, 1.0)
    tolerance = tol * factor
    return mid - tolerance, mid + tolerance, "$", ms


def _unit_of(*raws: str) -> tuple[str, str]:
    unit = scale = ""
    for raw in raws:
        tok = _norm_scale(raw)
        if tok == "%":
            unit = "%"
        elif tok in ("billion", "million", "thousand"):
            scale = scale or tok
        elif tok == "bps":
            unit = "%"
    if any("$" in (r or "") or "£" in (r or "") or "€" in (r or "") for r in raws):
        unit = unit or "$"
    return unit, scale


def _context(text: str, start: int, end: int, sents: Sentences):
    idx = sents.index_at(start)
    s0, s1 = sents[idx]
    left = text[max(s0, start - LEFT_REACH):start]
    near = text[max(s0, start - LEFT_REACH):min(s1, end + RIGHT_REACH)]
    return (s0, s1), left, near


def _emit(text: str, sents: Sentences, m, frame: str, low: float, high: float,
          unit: str, scale: str) -> Guidance | None:
    if low > high:
        low, high = high, low
    (s0, s1), left, near = _context(text, m.start(), m.end(), sents)
    if not _FORWARD.search(near):
        return None
    if _NOT_GUIDANCE.search(near):
        return None
    metric = _metric(_metric_source(left))
    if not metric or _NOT_GUIDANCE.search(metric):
        return None
    return Guidance(
        metric=metric,
        period=_period(text, max(s0, m.start() - LEFT_REACH), s0, m.start(), m.end(), s1),
        low=low, high=high, unit=unit, scale=scale, scale_source="bound" if scale else "",
        frame=frame, confidence="stated",
        evidence=evidence(text, (s0, s1), m.start(), m.end()))


def extract(text: str) -> list[Guidance]:
    """Every supplementary form, deduped against itself. Spans are claimed once."""
    sents = Sentences(text)
    out: list[Guidance] = []
    claimed: list[tuple[int, int]] = []

    def _free(m) -> bool:
        return not any(m.start() < e and s < m.end() for s, e in claimed)

    for m in _PLUS_MINUS.finditer(text):
        if not _free(m):
            continue
        mid, tol = _number(m.group("mid")), _number(m.group("tol"))
        if mid is None or tol is None or tol == 0:
            continue
        low, high, unit, scale = _plus_minus_bounds(mid, m.group("midscale"), tol,
                                                    m.group("tolscale"))
        hit = _emit(text, sents, m, "plus_minus", low, high, unit, scale)
        if hit:
            claimed.append((m.start(), m.end()))
            out.append(hit)

    for m in _FLAT_TO.finditer(text):
        if not _free(m):
            continue
        high = _number(m.group("hi"))
        if high is None:
            continue
        unit, scale = _unit_of(m.group("hiscale"))
        hit = _emit(text, sents, m, "flat_to", 0.0, high, unit or "%", scale)
        if hit:
            claimed.append((m.start(), m.end()))
            out.append(hit)

    for m in _APPROX_RANGE.finditer(text):
        if not _free(m):
            continue
        low, high = _number(m.group("lo")), _number(m.group("hi"))
        if low is None or high is None:
            continue
        unit, scale = _unit_of(m.group("loscale"), m.group("hiscale"), m.group(0))
        if not (unit or scale):
            continue
        hit = _emit(text, sents, m, "approx_range", low, high, unit, scale)
        if hit:
            claimed.append((m.start(), m.end()))
            out.append(hit)

    for m in _DIRECTIONAL.finditer(text):
        if not _free(m):
            continue
        low = _number(m.group("lo") or "0")
        high = _number(m.group("hi") or m.group("lo") or "0")
        if low is None or high is None:
            continue
        direction = (m.group("dir2") or m.group("dir") or "").lower()
        if direction == "down":
            low, high = -high, -low          # ⛔ sign is a word; unsigned would invert it
        elif direction == "flat" and not m.group("lo"):
            low = high = 0.0
        unit, scale = _unit_of(m.group("scale"))
        hit = _emit(text, sents, m, "directional", low, high, unit or "%", scale)
        if hit:
            claimed.append((m.start(), m.end()))
            out.append(hit)

    for m in _APPROX_POINT.finditer(text):
        if not _free(m):
            continue
        value = _number(m.group("v"))
        if value is None:
            continue
        unit, scale = _unit_of(m.group("vscale"), m.group(0))
        if not (unit or scale):
            continue
        hit = _emit(text, sents, m, "approx_point", value, value, unit, scale)
        if hit:
            claimed.append((m.start(), m.end()))
            out.append(hit)

    return out
