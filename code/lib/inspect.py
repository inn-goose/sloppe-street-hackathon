"""Query the store from the shell. The verification loop every grammar in here was built against.

Not a convenience: a rule written from imagination gets a financial corpus wrong in both
directions, so every pattern in this package was mined by running these queries over the real
bytes first and refusing anything the corpus does not actually write.

    python -m providers.inspect labels   --ticker DE --lane earnings_release --top 30
    python -m providers.inspect headers  --ticker HD --grep "Months Ended" --limit 5
    python -m providers.inspect cells    --ticker ADI --label "Adjusted gross margin"
    python -m providers.inspect table    --doc <doc_id> --table 3
    python -m providers.inspect grep     --ticker LSE:HAS --pattern "consensus" --width 160
    python -m providers.inspect facts    --ticker HD --metric net_sales
    python -m providers.inspect tables   --list
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from code.lib import config, store
from code.lib.text import parse_frontmatter, iter_tables


def _lane_map() -> dict[str, str]:
    return {r["doc_id"]: r["lane_family"]
            for r in store.read(config.EXTRACTED / "document_lanes.parquet")}


def _filter(rows, args):
    lanes = _lane_map()
    out = []
    for r in rows:
        if args.ticker and r.get("ticker") != args.ticker:
            continue
        if args.lane and lanes.get(r.get("doc_id"), "") != args.lane:
            continue
        if args.since and str(r.get("published_at", "")) < args.since:
            continue
        out.append(r)
    return out


def cmd_labels(args) -> None:
    rows = _filter(store.read(config.EXTRACTED / "table_cells.parquet"), args)
    counts = Counter(r["label"] for r in rows if r["label"])
    print(f"{len(rows):,} tokens, {len(counts):,} distinct row labels")
    for label, n in counts.most_common(args.top):
        print(f"  {n:>6}  {label[:96]}")


def cmd_headers(args) -> None:
    rows = _filter(store.read(config.EXTRACTED / "table_headers.parquet"), args)
    rx = re.compile(args.grep, re.IGNORECASE) if args.grep else None
    shown = 0
    for r in rows:
        blob = f"{r['header_text']} {r['header_cells']}"
        if rx and not rx.search(blob):
            continue
        print(f"\n— {r['doc_id']}  table {r['table_idx']}  rows={r['n_rows']} "
              f"maxnum={r['max_numbers']}")
        print(f"  header_cells: {r['header_cells'][:220]}")
        print(f"  row_labels  : {r['row_labels'][:220]}")
        shown += 1
        if shown >= args.limit:
            break


def cmd_cells(args) -> None:
    rows = _filter(store.read(config.EXTRACTED / "table_cells.parquet"), args)
    rx = re.compile(args.label, re.IGNORECASE) if args.label else None
    seen = 0
    grouped = store.group_by(rows, "doc_id", "table_idx", "row_idx")
    for (doc_id, t_idx, r_idx), group in grouped.items():
        label = group[0]["label"]
        if rx and not rx.search(label or ""):
            continue
        vals = ", ".join(
            f"{g['value']:g}{'%' if g['percent'] else ''}" for g in sorted(group, key=lambda g: g["ordinal"]))
        print(f"{doc_id}  t{t_idx}r{r_idx}  {label[:44]:<44} [{vals}]")
        seen += 1
        if seen >= args.limit:
            break


def cmd_table(args) -> None:
    docs = {d["doc_id"]: d for d in store.read(config.RAW / "documents.parquet")}
    doc = docs[args.doc]
    _meta, body = parse_frontmatter(doc["text"])
    for table in iter_tables(body):
        if args.table is not None and table.index != args.table:
            continue
        print(f"\n=== {args.doc} table {table.index} @line {table.start_line} ===")
        for i, row in enumerate(table.rows):
            nums = ", ".join(f"{n.value:g}{'%' if n.percent else ''}" for n in row.numbers)
            print(f"  r{i:<3} {row.label[:46]:<46} | {nums}")
            if args.raw:
                print(f"        raw: {row.cells}")
        if args.table is not None:
            break


def cmd_grep(args) -> None:
    docs = _filter(store.read(config.RAW / "documents.parquet"), args)
    rx = re.compile(args.pattern, re.IGNORECASE)
    hits = 0
    for doc in docs:
        _meta, body = parse_frontmatter(doc["text"])
        flat = re.sub(r"\s+", " ", body)
        for m in rx.finditer(flat):
            lo = max(0, m.start() - args.width // 2)
            print(f"{doc['published_at']} {doc['doc_id'][:52]:<52} …{flat[lo:lo + args.width]}…")
            hits += 1
            if hits >= args.limit:
                return
    print(f"\n{hits} hits")


def cmd_facts(args) -> None:
    path = config.EXTRACTED / "statement_facts.parquet"
    rows = _filter(store.read(path), args)
    if args.metric:
        rows = [r for r in rows if args.metric in (r.get("metric") or "")]
    rows.sort(key=lambda r: (r.get("fiscal_year", 0), r.get("fiscal_period", ""), r["doc_id"]))
    for r in rows[: args.limit]:
        print(f"{r['ticker']:<8}{r.get('fiscal_label',''):<12}{r.get('metric',''):<34}"
              f"{r['value']:>14,.3f} {r.get('unit',''):<8}{r.get('check',''):<10}{r['doc_id'][:44]}")
    print(f"\n{len(rows)} rows")


def cmd_tables(args) -> None:
    for path in sorted(config.STORE.rglob("*.parquet")):
        print(f"  {path.relative_to(config.STORE)}: {store.describe(path).split(': ', 1)[1]}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["labels", "headers", "cells", "table", "grep", "facts",
                                       "tables"])
    p.add_argument("--ticker")
    p.add_argument("--lane")
    p.add_argument("--since")
    p.add_argument("--label")
    p.add_argument("--metric")
    p.add_argument("--grep")
    p.add_argument("--pattern")
    p.add_argument("--doc")
    p.add_argument("--table", type=int)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--width", type=int, default=180)
    p.add_argument("--raw", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    {"labels": cmd_labels, "headers": cmd_headers, "cells": cmd_cells, "table": cmd_table,
     "grep": cmd_grep, "facts": cmd_facts, "tables": cmd_tables}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
