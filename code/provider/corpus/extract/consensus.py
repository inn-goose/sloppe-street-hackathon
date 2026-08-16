"""extracted/consensus — the benchmark, stated by the company itself.

The competition scores every metric against Wall Street's own miss, and that benchmark is frozen
internally and never supplied. For three of the four names the only route to it is a vendor
(stockanalysis, Yahoo, Nasdaq). **Hays publishes it outright**, in almost every RNS it files:

    "As of 9 July 2026, company complied consensus for FY26 pre-exceptional operating profit
     is £43.5m, with a £37.0-46.0m range, based on 10 analysts."

That single sentence carries the point estimate, the dispersion, the contributor count **and its
own as-of date** — more than any vendor lane in this store gives for any company. And because
Hays has published it for years, this lane is a *time series* of the bar, so how consensus
migrates through a fiscal year is measurable rather than assumed.

## The second signal: where management says it will land in that range

    "we currently expect FY26 pre-exceptional operating profit will be at the top of the
     £37.0-46.0m consensus range"

Management placing itself **within** a stated consensus range is guidance of a kind no numeric
frame can express — the guidance grammar correctly refuses it, because the range being quoted is
somebody else's. Captured here as `position` (`top` / `bottom` / `in_line` / `above` / `below`),
with the range it refers to.

## Two traps mined from the bytes

⛔ **Hays writes both `compiled` and `complied`** — its own typo, in official RNS announcements,
across multiple years. A pattern matching only the correct spelling loses whole fiscal years.

⛔ **The analyst count is often a WORD** — "based on nine analysts", "based on ten analysts" —
and it is the dispersion weight. Reading only digits silently drops those observations.
"""

from __future__ import annotations

import re
from collections import Counter

from code.lib import config, store
from code.provider.corpus.extract import _corpus
from code.vendor.sentence_grammar import Sentences, evidence

_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
             "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
             "nineteen": 19, "twenty": 20}

#: ⛔ **Hays' RNS are PDF-converted and the converter injects spaces INSIDE tokens.** Verbatim
#: from the 10 July 2026 announcement — the single most important consensus row in the corpus:
#:
#:     "As of 9 July 2026 , c ompany complied consensus for FY26 pre -exceptional operating
#:      profit is £4 3.5 m, with a £37 .0-46 .0m range, based on 1 0 analysts ."
#:
#: `£4 3.5 m` is £43.5m, `1 0 analysts` is 10, `pre -exceptional` is one word. A number pattern
#: that assumes contiguous digits matches none of it and the row vanishes silently. Every numeric
#: group below therefore admits internal whitespace and `_num` strips it — which is safe only
#: because each group is anchored between literal words, never floating in free text.
_SP = r"[\s ]*"
_DIGITS = rf"\d(?:[\d,{_SP}]*\d)?(?:{_SP}\.{_SP}\d+)?"
_MONEY = rf"£{_SP}(?P<v>{_DIGITS}){_SP}(?P<scale>m\b|million|bn\b|billion)?"
_PERIOD_RX = re.compile(r"(FY\s?\d{2,4}|fiscal\s+(?:year\s+)?\d{4}|H[12]\s?\d{0,4}"
                        r"|full[\s-]?year)", re.IGNORECASE)

#: ⛔ `compl?ied` is not a typo in this pattern — it is a typo in the SOURCE, and a frequent one:
#: Hays writes both `compiled` and `complied` in official announcements across several years.
#:
#: The subject is captured whole between `consensus` and `is`, then the period is pulled out of it
#: separately — writing the period as its own optional group failed on
#: "consensus for FY26 pre-exceptional operating profit is", where the period leads the subject
#: rather than trailing it.
_COMPILED = re.compile(
    rf"(?:compiled|complied){_SP}consensus{_SP}"
    rf"(?:for{_SP})?(?P<subject>[A-Za-z][\w\s&/-]{{0,70}}?){_SP}is{_SP}{_MONEY}"
    rf"(?:{_SP},?{_SP}with{_SP}a{_SP}£{_SP}(?P<lo>{_DIGITS}){_SP}[-–]{_SP}"
    rf"(?P<hi>{_DIGITS}){_SP}(?:m\b|million))?"
    rf"[^.]{{0,50}}?based{_SP}on{_SP}(?P<n>[\d\s]{{1,5}}|[a-z]+){_SP}analysts",
    re.IGNORECASE)

#: "in line with market consensus expectations of c.£196 million"
_MARKET = re.compile(
    r"market\s+consensus\s+(?:expectations?\s+)?(?:of\s+)?(?:c\.?|circa|about|approximately)?\s*"
    + _MONEY, re.IGNORECASE)

#: "at the top of the £37.0-46.0m consensus range" — management's placement within somebody
#: else's range, which no numeric guidance frame can express.
_POSITION = re.compile(
    rf"(?P<pos>at{_SP}the{_SP}top|towards?{_SP}the{_SP}(?:upper|top)|above"
    rf"|at{_SP}the{_SP}bottom|around{_SP}the{_SP}bottom|towards?{_SP}the{_SP}lower|below"
    rf"|in{_SP}line{_SP}with|at{_SP}the{_SP}mid)"
    rf"[^.]{{0,60}}?(?:of{_SP})?(?:the{_SP})?"
    rf"(?:£{_SP}(?P<lo>{_DIGITS}){_SP}[-–]{_SP}(?P<hi>{_DIGITS}){_SP}(?:m\b|million){_SP})?"
    rf"(?:market{_SP}|current{_SP})?consensus", re.IGNORECASE)

_AS_OF = re.compile(r"as\s+of\s+(?P<d>\d{1,2}\s+[A-Z][a-z]+\s+\d{4})", re.IGNORECASE)

_POSITION_CLASS = (
    (re.compile(r"top|upper", re.I), "top"),
    (re.compile(r"above", re.I), "above"),
    (re.compile(r"bottom|lower", re.I), "bottom"),
    (re.compile(r"below", re.I), "below"),
    (re.compile(r"mid", re.I), "midpoint"),
    (re.compile(r"in\s+line", re.I), "in_line"),
)


def _num(raw: str | None) -> float | None:
    if not raw:
        return None
    cleaned = re.sub(r"[^\d.]", "", raw)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _count(raw: str) -> int | None:
    """The contributor count — the dispersion weight, and it is written three ways.

    ⚠️ Digits (`10`), a word (`ten`), and **digits with an injected space** (`1 0`), which is the
    PDF converter again. The space form is the FY26 row, so dropping it loses the count on the
    most recent and most important observation.
    """
    raw = (raw or "").strip().lower()
    despaced = re.sub(r"\s+", "", raw)
    if despaced.isdigit():
        return int(despaced)
    return _WORD_NUM.get(despaced)


def _scaled(value: float | None, scale: str | None) -> float | None:
    if value is None:
        return None
    tok = (scale or "m").lower()
    return value * 1000.0 if tok in ("bn", "billion") else value


def _classify(text: str) -> str:
    for rx, label in _POSITION_CLASS:
        if rx.search(text):
            return label
    return ""


def build() -> list[dict]:
    lanes = {r["doc_id"]: r for r in store.read(config.EXTRACTED / "document_lanes.parquet")}
    rows: list[dict] = []
    for doc in _corpus.iter_documents():
        body = re.sub(r"\s+", " ", doc.body)
        sents = Sentences(body)
        lane = lanes.get(doc.doc_id, {})
        base = {"doc_id": doc.doc_id, "ticker": doc.ticker,
                "published_at": doc.published_at,
                "lane_family": lane.get("lane_family", ""),
                "currency": "GBP" if doc.ticker == "LSE:HAS" else "USD"}

        for m in _COMPILED.finditer(body):
            span = sents.at(m.start())
            as_of = _AS_OF.search(body[max(0, m.start() - 140):m.start()])
            subject = " ".join((m.group("subject") or "").split())
            period_hit = _PERIOD_RX.search(subject)
            metric = _PERIOD_RX.sub("", subject).strip(" -,") if period_hit else subject
            rows.append({**base, "kind": "company_compiled",
                         "metric": metric[:60],
                         "period": period_hit.group(1).replace(" ", "") if period_hit else "",
                         "value": _scaled(_num(m.group("v")), m.group("scale")),
                         "low": _num(m.group("lo")), "high": _num(m.group("hi")),
                         "n_analysts": _count(m.group("n")),
                         "position": "", "as_of": as_of.group("d") if as_of else "",
                         "evidence": evidence(body, span, m.start(), m.end())[:320]})

        for m in _MARKET.finditer(body):
            span = sents.at(m.start())
            rows.append({**base, "kind": "market_consensus", "metric": "", "period": "",
                         "value": _scaled(_num(m.group("v")), m.group("scale")),
                         "low": None, "high": None, "n_analysts": None,
                         "position": _classify(body[max(0, m.start() - 80):m.end()]),
                         "as_of": "",
                         "evidence": evidence(body, span, m.start(), m.end())[:320]})

        for m in _POSITION.finditer(body):
            span = sents.at(m.start())
            low, high = _num(m.group("lo")), _num(m.group("hi"))
            rows.append({**base, "kind": "position_in_range", "metric": "", "period": "",
                         "value": None, "low": low, "high": high, "n_analysts": None,
                         "position": _classify(m.group("pos")), "as_of": "",
                         "evidence": evidence(body, span, m.start(), m.end())[:320]})
    return rows


def main() -> int:
    rows = build()
    store.write(config.EXTRACTED / "consensus.parquet", rows)
    kinds = Counter(r["kind"] for r in rows)
    print(f"extracted/consensus.parquet {len(rows):,} rows  {dict(kinds)}")
    for ticker in ("LSE:HAS", "HD", "ADI", "DE"):
        mine = [r for r in rows if r["ticker"] == ticker]
        print(f"  {ticker:<9}{len(mine):>4} rows")
    print("\n  Hays company-compiled consensus, by publication date:")
    seen = set()
    for r in sorted((r for r in rows if r["ticker"] == "LSE:HAS"
                     and r["kind"] == "company_compiled"),
                    key=lambda r: r["published_at"]):
        key = (r["period"], r["value"])
        if key in seen:
            continue
        seen.add(key)
        rng = f"  range {r['low']}–{r['high']}" if r["low"] else ""
        print(f"    {r['published_at']}  {r['period'] or '?':<8}£{r['value']:>7.1f}m  "
              f"{r['n_analysts'] or '?':>3} analysts{rng}   as-of {r['as_of'] or '-'}")
    print("\n  stated position within a consensus range:")
    for r in sorted((r for r in rows if r["kind"] == "position_in_range" and r["position"]),
                    key=lambda r: r["published_at"])[-6:]:
        rng = f"£{r['low']}–{r['high']}m" if r["low"] else "(range not restated)"
        print(f"    {r['published_at']} {r['ticker']:<9}{r['position']:<9}{rng}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
