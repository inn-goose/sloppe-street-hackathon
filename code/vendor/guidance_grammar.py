"""VENDORED PRE-EXISTING COMPONENT — see vendor/README.md.

Guidance-range grammar: the frames a filer states a forward range in, and the binder.
Carried over unmodified in behaviour from a pre-existing SEC extractor's guidance-vocabulary
module; the only edit is the import path for the sentence substrate.

Every pattern here was **mined from the corpus, not guessed** (2,600-document stratified sample of
SEC `earnings_release`, 2004→2026). Three measurements shaped the module:

- The bare range shape (*two numbers joined by a separator*) hits **95.7 %** of releases at ~17
  hits/document, and is overwhelmingly phone numbers, financial-table column pairs, ASU/ASC
  citations, page references and maturity buckets. So a range alone is **not** the handle; the
  handle is a range inside one of the frames the corpus wrote — `in the range of` (957),
  `to be between` (194), `to range from` (87), `between … and …` (694).
- The metric is **not adjacent** to the frame. The corpus writes `<METRIC> is/are (now) expected to
  be in the range of <LO> to <HI>`, so a naive left-capture censuses the verb phrase instead of the
  noun. `_VERB_RUN` strips that clause first.
- Every row in the >100 %-relative-width tail was a **reported comparison** ("revenue increased …
  to $3,516 million") bound across two unrelated numbers. That is what `_REPORTED` and the width
  guard exist for.
- The magnitude is often **not on the number**. A guidance table declares it once in a parenthetical
  header and every cell inherits it. See `_SCALE_DECL`.

⚠️ **A guided decline is a range too.** The bounds are signed: without a sign on the low bound,
"comparable store sales in the range of -8% to -2%" parses as 8 → 2, inverts, and is refused —
which removes guided declines specifically and biases the surviving distribution optimistic.
"""
import re
from dataclasses import dataclass

from code.vendor.sentence_grammar import Sentences, evidence as _evidence, is_prose_at

# Signed: a guided decline states a negative bound, and dropping those biases the dataset.
_SIGN = r"(?:-\s?)?"
_MONEY = rf"{_SIGN}\$\s?\d{{1,4}}(?:,\d{{3}})*(?:\.\d+)?"
_PLAIN = rf"{_SIGN}\d{{1,4}}(?:,\d{{3}})*(?:\.\d+)?"
_NUM = rf"(?:{_MONEY}|{_PLAIN})"
# The scale is the filer's own word and is stored beside the number, never multiplied into it.
_SCALE = r"(?:\s*(?:%|percent|billion|million|thousand|bn|mm|[BM]\b|cents|bps|basis\s+points))?"
_SEP = r"(?:-|to|and)"
_APPROX = r"(?:(?:approximately|about|roughly|nearly|around|circa)\s+)?"

_FRAMES = (
    ("range_of", re.compile(
        rf"(?:in|to|within)\s+(?:the\s+|a\s+)?range\s+of\s+{_APPROX}(?P<lo>{_NUM})"
        rf"(?P<loscale>{_SCALE})\s*{_SEP}\s*{_APPROX}(?P<hi>{_NUM})(?P<hiscale>{_SCALE})",
        re.IGNORECASE)),
    ("range_from", re.compile(
        rf"rang\w*\s+(?:from|between)\s+{_APPROX}(?P<lo>{_NUM})(?P<loscale>{_SCALE})\s*{_SEP}\s*"
        rf"{_APPROX}(?P<hi>{_NUM})(?P<hiscale>{_SCALE})", re.IGNORECASE)),
    ("between", re.compile(
        rf"between\s+{_APPROX}(?P<lo>{_NUM})(?P<loscale>{_SCALE})\s+and\s+{_APPROX}"
        rf"(?P<hi>{_NUM})(?P<hiscale>{_SCALE})", re.IGNORECASE)),
    # No connector at all — `Revenue of $5.16 - $5.25 billion`. Recall upside, graded apart and
    # gated hard; see _keep_dash. The low bound may not START on a bare year, and that is a
    # RECALL guard: finditer is non-overlapping, so a refused match consumes the real range too.
    ("dash", re.compile(
        rf"(?<![\d.,])(?!(?:19|20)\d{{2}}\s*(?:-|to)\s)(?P<lo>{_NUM})(?P<loscale>{_SCALE})"
        rf"\s*(?:-|to)\s*(?P<hi>{_NUM})(?P<hiscale>{_SCALE})", re.IGNORECASE)),
)

# Measured noise classes, as forms. Checked in a tight window around the match only.
_STOP = re.compile(
    r"(?:\bdial(?:ing)?\b|toll[\s-]?free|passcode|\breplay\b|\(\d{3}\)\s*\d{3}[-.]\d{4}"
    r"|\b\d{3}[-./]\d{3}[-.]\d{4}\b|\+1[-\s]?\d{3}"
    r"|\b(?:ASU|ASC|FSP|SOP|FAS|IFRS|SFAS)\b\s*(?:No\.)?\s*\d"
    r"|pages?\s+\d+\s*(?:-|to|through)"
    r"|\d\s*(?:-|to)\s*\d+\s*years?\b"
    r"|\d[\d,]*\s+and\s+\d[\d,]*\s+(?:\w+\s+){0,3}shares?\b)", re.IGNORECASE)

# ONE word list feeds both the right-anchored strip and the left-anchored one.
_CLAUSE = (r"is|are|was|were|will|would|should|may|might|can|could|remains?|stays?|continues?"
           r"|continue|expects?|expected|expecting|anticipates?|anticipated|projects?|projected"
           r"|forecasts?|forecasted|estimates?|estimated|targets?|targeting|sees|seeing|guides?"
           r"|guided|now|still|again|also|currently|approximately|about|to|be|being|been|of|at"
           r"|in|on|for|the|a|an|our|its|their|we|it|they|company|management|between|from"
           r"|full[\s-]?year|rais\w+|lower\w+|narrow\w+|updat\w+|revis\w+|reiterat\w+|reaffirm\w+")
_VERB_RUN = re.compile(rf"(?:\b(?:{_CLAUSE})\b[\s,]*)+$", re.IGNORECASE)
_STATEMENT_NOUN = re.compile(r"[\s,]*\b(?:guidance|guide|range|outlook|forecast|target|estimates?"
                             r"|projections?|expectations?)\b\s*$", re.IGNORECASE)
_QUALIFIER = re.compile(
    r"(?:,\s*(?:excluding|including|net\s+of|before|after|assuming|reflecting|driven\s+by)"
    r"[^,;]{0,70},?"
    r"|\bas\s+a\s+percent(?:age)?\s+of\s+[^,;]{0,40}"
    r"|\bon\s+a\s+[^,;]{0,40}\bbasis"
    r"|,\s*prior\s+to\s+[^,;]{0,40},?"
    r"|\bin\s+the\s+aggregate)\s*$", re.IGNORECASE)
_METRIC = re.compile(r"([A-Za-z][\w&%/.'\-]*(?:\s+[A-Za-z][\w&%/.'\-]*){0,4})[\s,]*$")
_UNIT_PHRASE_LEAD = re.compile(r"^per\s+[\w.'-]+(?:\s+[\w.'-]+)?\s+and\s+", re.IGNORECASE)
# A flattened guidance table puts a neighbouring CELL in front of the metric noun. Every form here
# comes from a census of what actually leads a metric across the shipped dataset.
_TABLE_DEBRIS = re.compile(
    r"^(?:(?:guidance|guide|outlook|forecast)"
    r"|n/?a"
    r"|no\s+change"
    r"|(?:low|high)\s+(?:low|high|end)"
    r"|actual"
    r"|(?:non-?gaap|gaap)\s+(?:non-?gaap|gaap)"
    r"|'\d{2}"
    r"|(?:inc|corp|corporation|incorporated|ltd|limited|plc|llc|l\.l\.c|lp|n\.v|s\.a|ag)\.?"
    r")(?:\s+|$)", re.IGNORECASE)

_FORWARD = re.compile(r"\b(?:guidance|guides?|guiding|outlook|expects?|expected|expecting"
                      r"|anticipat\w+|forecast\w*|targets?|targeting|sees|project\w+|estimat\w+"
                      r"|full[\s-]?year\s+(?:19|20)\d{2})\b", re.IGNORECASE)
_PERIOD = re.compile(
    r"(?:(?:the\s+)?(?:quarter|year|period)\s+end(?:ing|ed)\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*"
    r"(?:19|20)\d{2}"
    r"|Q[1-4]\s*(?:FY)?\s*(?:19|20)?\d{0,4}"
    r"|(?:first|second|third|fourth)-quarter\s+(?:of\s+)?(?:fiscal\s+)?(?:19|20)\d{2}"
    r"|(?:first|second|third|fourth)\s+quarter"
    r"(?:\s+(?:of\s+)?(?:fiscal\s+)?(?:19|20)?\d{2,4})?|full[\s-]?year(?:\s+(?:19|20)\d{2})?"
    r"|fiscal(?:\s+year)?\s*(?:19|20)?\d{2,4}|FY\s*(?:19|20)?\d{2}"
    r"|(?:first|second)\s+half(?:\s+(?:of\s+)?(?:19|20)?\d{2,4})?"
    r"|(?:19|20)\d{2})", re.IGNORECASE)
# A range the filing quotes only to compare against is NOT this filing's guide.
_COMPARISON_BASE = re.compile(
    r"(?:over|from|versus|vs\.?|compared\s+(?:to|with)|than|against|relative\s+to|"
    r"vs\.|prior\s+(?:year|period)|year[\s-]ago)"
    r"\s+(?:the\s+|its\s+|our\s+)?(?:same\s+|prior\s+|comparable\s+)?$", re.IGNORECASE)
_BASE_LOOKBACK = 34
_GUIDANCE_SCOPE = re.compile(
    r"(?:(?:guidance|outlook|forecast|expectations?|estimates?)\s+for\s+(?:the\s+)?"
    r"|\bfor\s+(?:the\s+)?(?:full\s+)?)$", re.IGNORECASE)
BASE_FALLBACK_REACH = 240
_BASE_TAIL = 14

_PRIOR = re.compile(
    r"(?:(?:,|\s)\s*(?:down\s+|up\s+)?from"
    r"|\(\s*previously"
    r"|\b(?:previous|previously|prior|original|initial|earlier)\s+"
    r"(?:guidance|range|forecast|outlook|estimate|expectation|view)s?\s*(?:of|was|were|range\s+of)?"
    r"|\b(?:top|bottom|high|low|mid|middle|upper|lower)(?:[\s-]+end)?\s+of\s+(?:the\s+)?"
    r"(?:[\w'’]+\s+){0,4}(?:guidance|range)\s*(?:range\s+)?(?:of)?"
    r"|\b(?:previous|previously|prior|original|initial|earlier)\s+"
    r"(?:announced|provided|issued|reported|stated|revised)?\s*"
    r"(?:guidance|range|ranges|forecast|outlook|estimate)s?\b[^.;]{0,60}"
    r"|\bup\s+from\s+[^.;]{0,60}\b(?:guidance|range|ranges|estimate)s?\s+of"
    r"|\bfrom\s+a\s+range\s+of"
    r"|\b(?:exceed\w*|beat\w*|miss\w*|above|below|within|versus|vs\.?|"
    r"compared\s+(?:to|with)|fell\s+\w+|came\s+in\s+\w+)\s+"
    r"(?:the\s+)?(?:[\w'’]+\s+){0,4}(?:guidance|range|forecast|outlook|estimate)s?\s*"
    r"(?:of|range\s+of)?)\s*$", re.IGNORECASE)
_ABILITY = re.compile(r"\babilit(?:y|ies)\s+to\b", re.IGNORECASE)
# `ranging from` is a PARTICIPLE describing a set of terms, not a forecast.
_RANGING = re.compile(r"^ranging\b", re.IGNORECASE)
_LEADING_PERIOD = re.compile(
    r"^(?:(?:for|in)\s+)?(?:the\s+)?(?:Q[1-4]|(?:first|second|third|fourth)\s+quarter"
    r"|full[\s-]?year|fiscal(?:\s+year)?|FY)\s*(?:19|20)?\d{0,4}\s*(?:of\s+)?", re.IGNORECASE)
_TRAILING_PERIOD = re.compile(
    rf"[\s,]*(?:\b(?:for|in|during|of)\s+)?(?:the\s+)?"
    rf"(?:{_PERIOD.pattern}|fiscal\s+year|full[\s-]?year|year|quarter|period)\s*$", re.IGNORECASE)
_LEADING_FILLER = re.compile(
    rf"^(?:{_CLAUSE}|and|or|both|with|plus"
    r"|that|which|this|these|those"
    r"|year|quarter|half|month"
    r"|billion|million|thousand|cents|bps|[bmBM]"
    r"|announc\w+|report\w+|provid\w+|issu\w+|introduc\w+|initiat\w+|confirm\w+|maintain\w+)\s+",
    re.IGNORECASE)

_UNIT_SCALE = {"bn": "billion", "mm": "million", "b": "billion", "m": "million",
               "percent": "%", "basis points": "bps"}
_SCALE_INHERIT_RATIO = 100.0
# A guidance TABLE declares its magnitude ONCE in a parenthetical header. NO DIGIT may appear
# inside the parenthetical — that is the difference between a header and a prose aside.
_SCALE_DECL = re.compile(
    r"\([^)\d]{0,45}\b(?P<scale>millions?|billions?|thousands?)\b[^)\d]{0,70}\)", re.IGNORECASE)
SCALE_DECL_REACH = 400
_PER_SHARE_UNIT = re.compile(
    r"per\s+(?:diluted\s+|basic\s+|common\s+|adjusted\s+){0,2}(?:share|unit)\b|\bEPS\b",
    re.IGNORECASE)
_PER_ANY_TAIL = re.compile(r"^\W{0,3}per\s+[\w-]+", re.IGNORECASE)
_SCALE_TAIL_REACH = 40
_BARE_YEAR = re.compile(r"^(?:19|20)\d{2}$")
_MODAL = re.compile(r"\b(?:can|could|may|might)\s*$", re.IGNORECASE)
_DANGLING_PREP = re.compile(r"\b(?:of|at|to|by|from|with)$", re.IGNORECASE)

_REPORTED = re.compile(r"\b(?:increas\w+|decreas\w+|grew|grow\w*|declin\w+|ros\w*|rose|fell|fall\w*"
                       r"|improv\w+|totall?\w*|report\w+|compared|deliver\w+|generat\w+|record\w+"
                       r"|represent\w+|reflect\w+|shipp\w+|sold|paid|was|were)\s*$", re.IGNORECASE)
_PAST_COMPARISON = re.compile(
    r"\b(?:increased|decreased|declined|grew|rose|fell|improved|totall?ed|reported|delivered"
    r"|generated|achieved|posted|earned|exceeded|impacted|contributed|represented|reflected"
    r"|shipped|sold|paid|was|were|had)\b", re.IGNORECASE)
_NO_NOUN = frozenset(
    "gaap non-gaap diluted basic adjusted core total net prior previous up down which "
    "respectively other same first second third fourth "
    "the a an and or to of in on for with at by from as that this its our their it we "
    "million billion thousand cents bps b m increase increases decrease decreases "
    "compared versus vs basis level range midpoint mid-point point change based "
    "u.s. us underlying forma".split())
_BASIS_LEAD = re.compile(r"\b(?:reported|restated|adjusted|normalized|underlying|comparable"
                         r"|pro\s+forma)\b", re.IGNORECASE)
_RESPECTIVELY = re.compile(r"\brespectively\b", re.IGNORECASE)
_RESPECTIVELY_REACH = 220

LEFT_REACH = 220
RIGHT_REACH = 200
NEIGHBOUR_MAX = 600
MAX_DASH_WIDTH = 1.0


@dataclass(frozen=True)
class Guidance:
    metric: str
    period: str
    low: float
    high: float
    unit: str
    scale: str
    scale_source: str
    frame: str
    confidence: str
    evidence: str


def _number(raw: str) -> float | None:
    try:
        return float(raw.replace("$", "").replace(",", "").replace(" ", "").strip())
    except ValueError:
        return None


def _unit_and_scale(*raws: str) -> tuple[str, str]:
    unit = scale = ""
    for raw in raws:
        tok = (raw or "").strip().lower()
        tok = _UNIT_SCALE.get(tok, tok)
        if tok == "%":
            unit = "%"
        elif tok in ("billion", "million", "thousand", "cents", "bps"):
            scale = scale or tok
    if any("$" in (r or "") for r in raws):
        unit = "$"
    return unit, scale


def _qualifier_owns_the_range(raw_left: str) -> bool:
    """Decided once, on the untouched left context, because `_VERB_RUN` erases the evidence."""
    m = _QUALIFIER.search(raw_left.rstrip(" :,"))
    return bool(m) and bool(_DANGLING_PREP.search(m.group(0).rstrip(" ,")))


def _strip_qualifier(text: str) -> str:
    m = _QUALIFIER.search(text)
    if not m:
        return text
    if _DANGLING_PREP.search(m.group(0).rstrip(" ,")):
        return text
    return text[:m.start()]


def _quotes_a_prior_guide(left: str) -> bool:
    return bool(_PRIOR.search(left)) or bool(
        _PRIOR.search(_VERB_RUN.sub("", left.rstrip(" :,"))))


def _metric_source(left: str) -> str:
    """The metric window as captured, before any leading strip — what a tense veto must read."""
    strip_q = (lambda t: t) if _qualifier_owns_the_range(left) else _strip_qualifier
    trimmed, prev = left.rstrip(" :,"), None
    while prev != trimmed:
        prev = trimmed
        trimmed = strip_q(_STATEMENT_NOUN.sub("", _VERB_RUN.sub("", trimmed)))
        trimmed = _TRAILING_PERIOD.sub("", trimmed)
        trimmed = trimmed.rstrip(" :,")
    m = _METRIC.search(trimmed)
    return m.group(1).strip() if m else ""


def _metric(captured: str) -> str:
    if not captured:
        return ""
    metric, prev = _LEADING_PERIOD.sub("", captured), None
    while prev != metric:
        prev = metric
        metric = _UNIT_PHRASE_LEAD.sub("", metric)
        metric = _TABLE_DEBRIS.sub("", metric)
        metric = _LEADING_FILLER.sub("", _LEADING_PERIOD.sub("", metric)).strip()
    return metric


def _period(text: str, left_from: int, sent_start: int, start: int, end: int,
            sent_end: int) -> str:
    """Nearest on the left, else first on the right, both scoped to the sentence."""
    def _left(lo: int) -> str:
        skipped = []
        for i, m in enumerate(reversed(list(_PERIOD.finditer(text, lo, start)))):
            if _COMPARISON_BASE.search(text[max(lo, m.start() - _BASE_LOOKBACK):m.start()]) or \
                    any(m.end() + _BASE_TAIL >= s0 and m.start() <= e0 for s0, e0 in skipped):
                skipped.append((m.start() - _BASE_TAIL, m.end()))
                continue
            if i and not _GUIDANCE_SCOPE.search(text[max(lo, m.start() - 40):m.start()]):
                break
            if i and start - m.end() > BASE_FALLBACK_REACH:
                break
            return " ".join(m.group(0).split())
        return ""

    own = _left(sent_start)
    if own:
        return own
    base_end = -1
    for m in _PERIOD.finditer(text, end, min(sent_end, end + RIGHT_REACH)):
        if _COMPARISON_BASE.search(text[max(end, m.start() - _BASE_LOOKBACK):m.start()]) or \
                m.start() - base_end <= _BASE_TAIL:
            base_end = m.end()
            continue
        return " ".join(m.group(0).split())
    return _left(left_from)


def _same_unit_kind(*raws: str) -> bool:
    """Both bounds must measure the same KIND of thing — a free validator."""
    kinds = set()
    for raw in raws:
        tok = _UNIT_SCALE.get((raw or "").strip().lower(), (raw or "").strip().lower())
        if tok == "%":
            kinds.add("pct")
        elif tok == "bps":
            kinds.add("bps")
        elif tok == "cents":
            kinds.add("cents")
        elif tok in ("billion", "million", "thousand"):
            kinds.add("scale")
    return len(kinds) <= 1


def _one_scale_covers_both(low: float, high: float, loscale: str, hiscale: str) -> bool:
    lo_tok = _UNIT_SCALE.get((loscale or "").strip().lower(), (loscale or "").strip().lower())
    hi_tok = _UNIT_SCALE.get((hiscale or "").strip().lower(), (hiscale or "").strip().lower())
    if lo_tok and hi_tok:
        return lo_tok == hi_tok
    if bool(lo_tok) != bool(hi_tok):
        a, b = abs(low), abs(high)
        if min(a, b) == 0:
            return True
        return max(a, b) / min(a, b) < _SCALE_INHERIT_RATIO
    return True


def _table_scale(text: str, start: int, end: int, metric: str) -> tuple[str, str]:
    if _PER_SHARE_UNIT.search(metric) or _PER_ANY_TAIL.search(text[end:end + _SCALE_TAIL_REACH]):
        return "", ""
    found = ("", "")
    for m in _SCALE_DECL.finditer(text, max(0, start - SCALE_DECL_REACH), start):
        found = (m.group("scale").lower().rstrip("s"), " ".join(m.group(0).split()))
    return found


def _usable_metric(metric: str) -> bool:
    if not metric or _REPORTED.search(metric) or _PAST_COMPARISON.search(metric):
        return False
    words = [w for w in re.split(r"[\s,]+", metric.lower()) if w]
    return bool(words) and not set(words) <= _NO_NOUN


def _width(low: float, high: float) -> float:
    denom = abs(low) if abs(low) > 1e-9 else abs(high)
    return (high - low) / denom if denom else 0.0


def _keep_dash(low: float, high: float, near: str) -> bool:
    return bool(_FORWARD.search(near)) and _width(low, high) <= MAX_DASH_WIDTH


def extract(text: str) -> list[Guidance]:
    """The frames are a **ladder, not independent passes**: a weaker frame may not re-bind
    numbers a stronger one already consumed."""
    sents = Sentences(text)
    out: list[Guidance] = []
    claimed: list[tuple[int, int]] = []
    for frame, rx in _FRAMES:
        for m in rx.finditer(text):
            lo_at, hi_at = m.start("lo"), m.end("hi")
            if any(lo_at < c_end and c_start < hi_at for c_start, c_end in claimed):
                continue
            if _STOP.search(text[max(0, m.start() - 55):m.end() + 25]):
                continue
            low, high = _number(m.group("lo")), _number(m.group("hi"))
            if low is None or high is None or low > high:
                continue

            idx = sents.index_at(m.start())
            sent_start, sent_end = sents[idx]
            near_from = sent_start
            if idx and sents[idx - 1][1] - sents[idx - 1][0] <= NEIGHBOUR_MAX:
                near_from = sents[idx - 1][0]
            if not _same_unit_kind(m.group("loscale"), m.group("hiscale")):
                continue
            if not _one_scale_covers_both(low, high, m.group("loscale"), m.group("hiscale")):
                continue
            if any(_BARE_YEAR.match(m.group(g).strip()) and not (m.group(g + "scale") or "").strip()
                   for g in ("lo", "hi")):
                continue
            left = text[max(sent_start, m.start() - LEFT_REACH):m.start()]
            if _quotes_a_prior_guide(left) or _ABILITY.search(left) or _MODAL.search(left):
                continue
            unit, scale = _unit_and_scale(m.group("loscale"), m.group("hiscale"), m.group(0))
            if not (unit or scale):
                continue
            captured = _metric_source(left)
            metric = _metric(captured)
            if not _usable_metric(metric) or _PAST_COMPARISON.search(_BASIS_LEAD.sub("", captured)):
                continue
            if " and " in f" {metric.lower()} " and _RESPECTIVELY.search(
                    text[m.end():m.end() + _RESPECTIVELY_REACH]):
                continue
            near = text[max(near_from, m.start() - LEFT_REACH):
                        min(sent_end, m.end() + RIGHT_REACH)]
            if frame == "dash":
                if not is_prose_at(text, m.start(), m.end()):
                    continue
                if not _keep_dash(low, high, near):
                    continue
            if _RANGING.match(m.group(0)) and not _FORWARD.search(near):
                # The span is claimed even though the row is refused — a veto that only moves a
                # row down the ladder is not a veto.
                claimed.append((lo_at, hi_at))
                continue
            claimed.append((lo_at, hi_at))
            ev = _evidence(text, (sent_start, sent_end), m.start(), m.end())
            scale_source = "bound" if scale else ""
            if unit == "$" and not scale:
                scale, decl = _table_scale(text, m.start(), m.end(), metric)
                if scale:
                    scale_source, ev = "table_header", f"{decl} … {ev}"
            out.append(Guidance(
                metric=metric,
                period=_period(text, max(near_from, m.start() - LEFT_REACH),
                               max(sent_start, m.start() - LEFT_REACH), m.start(), m.end(),
                               sent_end),
                low=low, high=high, unit=unit, scale=scale, scale_source=scale_source, frame=frame,
                confidence="stated" if frame != "dash" else "weak",
                evidence=ev))
    return dedupe(out)


def dedupe(rows: list[Guidance]) -> list[Guidance]:
    """One row per distinct stated range. A stated frame wins over `dash`."""
    best: dict[tuple, Guidance] = {}
    for r in rows:
        key = (r.metric.lower(), r.period.lower(), r.low, r.high, r.unit, r.scale)
        prev = best.get(key)
        if prev is None or (prev.frame == "dash" and r.frame != "dash"):
            best[key] = r
    return list(best.values())
