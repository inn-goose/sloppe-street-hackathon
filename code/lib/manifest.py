"""What is actually in `data/raw/`, measured from the ledgers and the files on disk.

    PYTHONPATH=. .venv/bin/python -m code.lib.manifest
    PYTHONPATH=. .venv/bin/python -m code.lib.manifest --tree

A refusal count is part of the inventory rather than a footnote: a provider that banked 20 things
and refused 30 is telling you something about its coverage, and a manifest that counts only
successes hides exactly that.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from code.lib import config, rawstore

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TIME = re.compile(r"\d{2}-\d{2}-\d{2}-\d{6}")

PROVIDERS = ("corpus", "yahoo", "stockanalysis", "sec", "sec_documents", "fred", "labour",
             "nasdaq", "alpaca")


def _disk(provider: str) -> tuple[int, int]:
    root = config.RAW / provider.replace("_documents", "")
    if not root.exists():
        return 0, 0
    files = [p for p in root.rglob("*") if p.is_file() and p.name != "_ledger.jsonl"]
    return len(files), sum(p.stat().st_size for p in files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", action="store_true", help="also print the path grammar in use")
    args = parser.parse_args()

    print(f"{'provider':<15}{'banked':>8}{'refused':>9}{'source':>10}{'disk':>9}  products")
    print("-" * 112)
    total_banked = total_refused = total_src = total_disk = 0
    for provider in PROVIDERS:
        rows = list(rawstore.iter_ledger(provider))
        if not rows:
            print(f"{provider:<15}{'—':>8}{'—':>9}{'—':>10}{'—':>9}  NOT FETCHED")
            continue
        ok = [r for r in rows if r.get("state") == "ok" or r.get("product") == "document"]
        bad = [r for r in rows if r not in ok]
        src = sum(r.get("source_bytes") or 0 for r in ok)
        disk = sum(r.get("disk_bytes") or 0 for r in ok)
        products = Counter(r.get("product") or r.get("source") or "?" for r in ok)
        summary = ", ".join(f"{k}:{v}" for k, v in products.most_common(4))
        print(f"{provider:<15}{len(ok):>8,}{len(bad):>9,}{src / 1e6:>9.1f}M{disk / 1e6:>8.1f}M"
              f"  {summary[:58]}")
        total_banked += len(ok)
        total_refused += len(bad)
        total_src += src
        total_disk += disk
    print("-" * 112)
    print(f"{'TOTAL':<15}{total_banked:>8,}{total_refused:>9,}"
          f"{total_src / 1e6:>9.1f}M{total_disk / 1e6:>8.1f}M")

    files, size = 0, 0
    for provider in set(p.replace("_documents", "") for p in PROVIDERS):
        n, b = _disk(provider)
        files += n
        size += b
    print(f"\n  {files:,} capture files on disk, {size / 1e6:.1f} MB "
          f"(gzipped; gunzip reproduces the source bytes exactly)")

    if args.tree:
        print("\n  path grammar in use (dated segments collapsed):")
        seen: Counter = Counter()
        for path in config.RAW.rglob("*.gz"):
            parts = list(path.relative_to(config.RAW).parts)
            shape = []
            for part in parts[:-1]:
                if _DATE.fullmatch(part):
                    shape.append("<YYYY-MM-DD>")
                elif _TIME.fullmatch(part):
                    shape.append("<HH-MM-SS-ffffff>")
                elif len(shape) > 4:
                    shape.append("<key>")
                else:
                    shape.append(part)
            seen["/".join(shape) + "/<leaf>"] += 1
        for shape, n in sorted(seen.items()):
            print(f"    {n:>6,}  {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
