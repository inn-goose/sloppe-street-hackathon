"""The banked corpus, read back. Every corpus extractor walks this and nothing else.

⛔ **Reads `data/raw/`, never `starter/challenge/offline-data/`.** That is the whole point of
banking: an extract is a pure function of hashed bytes, and an edit to the supplied tree shows up
as a hash mismatch rather than as a silently different forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from code.lib import config, rawstore
from code.lib.text import parse_frontmatter


@dataclass
class Document:
    doc_id: str
    ticker: str
    company: str
    isin: str
    published_at: str
    document_type: str
    period_label: str
    source_url: str | None
    corpus_frozen_at: str
    lane_dir: str
    sha256: str
    text: str          # the whole file, header included
    body: str          # the document after the frontmatter
    meta: dict


def iter_documents() -> Iterator[Document]:
    for row in rawstore.iter_ledger("corpus"):
        if row.get("product") != "document":
            continue
        text = rawstore.read(config.RAW / row["path"])
        meta, body = parse_frontmatter(text)
        # the ledger key is the banked path minus its extension chain, which is stable and unique
        doc_id = row["source_path"][:-3] if row["source_path"].endswith(".md") \
            else row["source_path"]
        yield Document(
            doc_id=doc_id,
            ticker=row.get("ticker") or "",
            company=row.get("company") or "",
            isin=row.get("isin") or "",
            published_at=row.get("published_at") or "",
            document_type=row.get("document_type") or "",
            period_label=row.get("period_label") or "",
            source_url=row.get("source_url"),
            corpus_frozen_at=row.get("corpus_frozen_at") or "",
            lane_dir=row.get("lane_dir") or "",
            sha256=row.get("sha256") or "",
            text=text, body=body, meta=meta,
        )


def count() -> int:
    return sum(1 for r in rawstore.iter_ledger("corpus") if r.get("product") == "document")
