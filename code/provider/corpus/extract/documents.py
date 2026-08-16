"""extracted/documents + document_lanes — one row per banked document.

Frontmatter as typed columns, counts over the body, and the document class the corpus encodes in
its own filename.

⚠️ **`document_type` is too coarse to dispatch on.** It says FILING / CALL_TRANSCRIPT / SLIDE,
which collapses an earnings release, a 10-Q, a proxy and an RNS holdings notice into one bucket —
four different sources with four different value grammars. The filename carries the real class
(`q2-8k`, `fy-10k`, `call-qna`, `slide`), so it is read from there.

⚠️ **`is_prose` gates every distance-based reader downstream.** 87 of 1,139 documents are
converted slide decks whose OCR is visibly damaged, and a deck has no sentence structure, so any
rule that binds a value to a subject by proximity binds across unrelated cells there.
"""

from __future__ import annotations

import re
from collections import Counter

from code.lib import config, store
from code.lib.text import iter_tables
from code.provider.corpus.extract import _corpus
from code.vendor.sentence_grammar import Sentences, is_prose

_HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_IMAGE = re.compile(r"<!--\s*image\s*-->|\*Image:", re.IGNORECASE)
_PAGE = re.compile(r"<!--\s*PAGE\s+\d+\s*-->", re.IGNORECASE)

_FILENAME = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})__(?P<sym>[a-z0-9]+)-(?P<mkt>[a-z]{2})-(?P<stamp>\d{8})-"
    r"(?P<lane>.+?)__(?P<docid>\d+)$")
_VARIANT = re.compile(r"-(\d+)$")
#: The fiscal tag is written in two positions — leading on a filing (`q2-8k`) and INFIXED on a
#: transcript (`call-q4-pres`). A leading-only strip left 29 transcripts in a bogus family.
_FISCAL_PREFIX = re.compile(r"^(q[1-4]|h[12]|fy)-")
_FISCAL_INFIX = re.compile(r"(?<=-)(q[1-4]|h[12]|fy)-")

LANE_FAMILY = {
    "8k": ("earnings_release", "Issuer results release / RNS statement"),
    "10q": ("periodic_report", "Quarterly report — statements and notes"),
    "10k": ("periodic_report", "Annual report — statements and notes"),
    "call-pres": ("call_presentation", "Prepared remarks of an earnings call"),
    "call-qna": ("call_qna", "Analyst Q&A of an earnings call"),
    "call": ("call_full", "Earnings call, undivided"),
    "call-conf-pres": ("conference_presentation", "Broker-conference prepared remarks"),
    "call-conf-qna": ("conference_qna", "Broker-conference Q&A"),
    "call-agm-pres": ("agm", "Annual general meeting address"),
    "call-agm-qna": ("agm", "Annual general meeting Q&A"),
    "slide": ("slide_deck", "Converted investor presentation (OCR)"),
    "filing": ("other_filing", "Proxy, RNS notice, or other issuer filing"),
}


def _family(lane: str) -> tuple[str, str, str]:
    variant = ""
    core = lane
    m = _VARIANT.search(lane)
    if m and not core.endswith(("10q", "10k")):
        variant, core = m.group(1), core[: m.start()]
    core = _FISCAL_INFIX.sub("", _FISCAL_PREFIX.sub("", core))
    if core in LANE_FAMILY:
        return LANE_FAMILY[core][0], core, variant
    for key in sorted(LANE_FAMILY, key=len, reverse=True):
        if core.startswith(key):
            return LANE_FAMILY[key][0], core, variant
    return "other_filing", core, variant


def build() -> tuple[list[dict], list[dict]]:
    docs, lanes = [], []
    for doc in _corpus.iter_documents():
        body = doc.body
        title = _HEADING.search(body)
        tables = list(iter_tables(body))
        digits = sum(c.isdigit() for c in body)
        prose = is_prose(body)

        stem = doc.doc_id.rsplit("/", 1)[-1]
        m = _FILENAME.match(stem)
        if m:
            family, lane, variant = _family(m.group("lane"))
            tag = _FISCAL_PREFIX.match(m.group("lane")) or _FISCAL_INFIX.search(m.group("lane"))
            fiscal_tag = tag.group(1) if tag else ""
            market = m.group("mkt")
        else:
            family, lane, variant, market, fiscal_tag = "other_filing", "", "", "", ""

        docs.append({
            "doc_id": doc.doc_id, "ticker": doc.ticker, "company": doc.company,
            "isin": doc.isin, "published_at": doc.published_at,
            "document_type": doc.document_type, "period_label": doc.period_label,
            "source_url": doc.source_url, "has_source_url": doc.source_url is not None,
            "corpus_frozen_at": doc.corpus_frozen_at, "sha256": doc.sha256,
            "title": title.group(1).strip() if title else "",
            "chars": len(body),
            "digit_ratio": round(digits / len(body), 5) if body else 0.0,
            "n_sentences": len(Sentences(body)),
            "n_tables": len(tables),
            "n_table_rows": sum(len(t.rows) for t in tables),
            "n_numeric_cells": sum(len(r.numbers) for t in tables for r in t.rows),
            "n_images": len(_IMAGE.findall(body)),
            "n_pages": len(_PAGE.findall(body)),
            "is_prose": prose,
        })
        lanes.append({
            "doc_id": doc.doc_id, "ticker": doc.ticker, "published_at": doc.published_at,
            "document_type": doc.document_type, "market": market, "lane": lane,
            "lane_family": family, "variant": variant, "filename_fiscal_tag": fiscal_tag,
            "period_label": doc.period_label,
            "title": title.group(1).strip() if title else "",
            "chars": len(body), "n_tables": len(tables),
            "n_numeric_cells": sum(len(r.numbers) for t in tables for r in t.rows),
            "is_prose": prose, "has_source_url": doc.source_url is not None,
        })
    return docs, lanes


def main() -> int:
    docs, lanes = build()
    store.write(config.EXTRACTED / "documents.parquet", docs)
    store.write(config.EXTRACTED / "document_lanes.parquet", lanes)
    decks = sum(1 for d in docs if not d["is_prose"])
    fams = Counter(r["lane_family"] for r in lanes)
    print(f"extracted/documents.parquet      {len(docs):,} documents, {decks} read as deck, "
          f"{sum(d['n_numeric_cells'] for d in docs):,} numeric cells")
    print(f"extracted/document_lanes.parquet {len(lanes):,} rows  "
          f"{dict(fams.most_common(6))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
