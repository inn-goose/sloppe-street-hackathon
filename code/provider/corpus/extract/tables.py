"""extracted/table_cells + extracted/table_headers — every number in every table, faithfully.

This is the widest lane in the store and the one every financial figure ultimately comes from.
One row per numeric token, carrying:

  * the row label the document itself wrote (`Net sales`, `Operating profit`),
  * the token's **ordinal position** among the numbers on that row, and
  * the unit markers the document put beside it (`$`, `%`, `p`, accounting parentheses).

⛔ **No column is bound to a period here.** A financial table writes `| Net sales | $ | 45,277 |
| | $ | 43,175 | | | 4.9 | % |` and which of those three numbers is "this quarter" depends on a
header two rows up. That binding is a decision, so it lives in `feature/`, and what is stored here
is only what the bytes say: token 0, token 1, token 2.

⚠️ Ordinal position is the key, not cell index. The converters emit ragged rows — empty spacer
cells, a `$` in its own cell, a `%` trailing the number — so a cell index is not stable across two
rows of the *same* table while the ordinal is.
"""

from __future__ import annotations

from code.lib import config, store
from code.lib.text import iter_tables
from code.provider.corpus.extract import _corpus

# A table with more numeric columns than this is a page-wide artefact of the PDF converter
# (a slide's whole body flattened into one row), never a statement. Kept, but flagged, because
# refusing it here would silently drop the Deere slide bridges.
WIDE_ROW = 24


def build() -> tuple[list[dict], list[dict]]:
    cells: list[dict] = []
    headers: list[dict] = []

    for doc in _corpus.iter_documents():
        body = doc.body
        lines = body.splitlines()
        for table in iter_tables(body):
            head = table.rows[0]
            # A statement table declares its magnitude ONCE, usually in the line above it or in
            # its own stub cell ("in millions, except per share data", "In £s million"). Banking
            # the preceding context is what lets the column binder read a scale it can never
            # recover from the number itself — a mis-scaled table is internally consistent, so no
            # arithmetic check can catch it.
            pre = "\n".join(lines[max(0, table.start_line - 24):table.start_line - 1])
            headers.append({
                "doc_id": doc.doc_id,
                "ticker": doc.ticker,
                "published_at": doc.published_at,
                "table_idx": table.index,
                "start_line": table.start_line,
                "n_rows": len(table.rows),
                "header_text": table.header_text[:600],
                "header_cells": " | ".join(c for c in head.cells if c.strip())[:600],
                "row_labels": " | ".join(r.label for r in table.rows if r.label)[:1200],
                "max_numbers": max((len(r.numbers) for r in table.rows), default=0),
                "pre_context": pre[-400:],
                "all_rows": "␟".join(
                    "␞".join(c for c in r.cells) for r in table.rows[:8])[:2000],
            })
            # ⛔ **A statement groups its rows with LABEL-ONLY rows, and without that grouping two
            # different measures share one label.** Deere's Q2-2017 10-Q states `Net sales` twice
            # in the same table — 8,287 m against the consolidated line and 2,968 m under a
            # sub-heading — which is what made corpus↔SEC revenue disagree on 22 % of Deere's
            # quarters. The nearest preceding row that has a label and no numbers is that row's
            # section, and it is read, never inferred.
            section = ""
            for row_idx, row in enumerate(table.rows):
                if not row.numbers:
                    if row.label and len(row.label) > 2:
                        section = row.label[:80]
                    continue
                for ordinal, num in enumerate(row.numbers):
                    cells.append({
                        "doc_id": doc.doc_id,
                        "ticker": doc.ticker,
                        "published_at": doc.published_at,
                        "document_type": doc.document_type,
                        "period_label": doc.period_label,
                        "table_idx": table.index,
                        "row_idx": row_idx,
                        "line": row.line,
                        "label": row.label,
                        "section": section,
                        "ordinal": ordinal,
                        "n_numbers": len(row.numbers),
                        "value": num.value,
                        "percent": num.percent,
                        "currency": num.currency,
                        "pence": num.pence,
                        "raw": num.raw[:40],
                        "wide_row": len(row.numbers) > WIDE_ROW,
                    })
    return cells, headers


def main() -> int:
    cells, headers = build()
    store.write(config.EXTRACTED / "table_cells.parquet", cells)
    store.write(config.EXTRACTED / "table_headers.parquet", headers)
    labelled = sum(1 for c in cells if c["label"])
    print(f"extracted/table_cells.parquet   {len(cells):,} numeric tokens "
          f"({labelled:,} on a labelled row, {labelled / max(len(cells), 1):.1%})")
    print(f"extracted/table_headers.parquet {len(headers):,} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
