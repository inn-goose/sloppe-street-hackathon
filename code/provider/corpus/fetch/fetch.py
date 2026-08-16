"""raw/corpus — the frozen competition corpus, banked verbatim.

    data/raw/corpus/documents/event/<YYYY-MM-DD>/<company>/<lane>/<doc>.md.gz

The corpus ships as markdown files already, so "fetching" it is a copy — but banking it is not
ceremony. Everything downstream reads `data/raw/`, never the competition repo, so a run is a pure
function of hashed bytes and any edit to the supplied tree shows up as a hash mismatch rather than
as a silently different forecast.

⛔ **Banked as `.md`, not folded into a columnar file.** The document's own format is markdown;
re-encoding it at ingest would mean the bytes could only be read back through the writer.

⚠️ The as-of segment is the document's **publication date**, not the fetch instant: this source
publishes its own date, so it is banked through the ingest route rather than the capture one.
"""

from __future__ import annotations

import hashlib
import time

from code.lib import config, rawstore
from code.lib.text import parse_frontmatter

PROVIDER = "corpus"


def fetch() -> dict:
    config.ensure_dirs()
    fetched_at = rawstore.now()
    ledger = rawstore.Ledger(PROVIDER)
    banked = skipped = disk = 0

    for ticker, slug in config.CORPUS_DIRS.items():
        root = config.CORPUS_ROOT / slug
        for path in sorted(root.rglob("*.md")):
            if path.name in {"INDEX.md", "README.md"}:
                skipped += 1
                continue
            data = path.read_bytes()
            text = data.decode("utf-8")
            meta, _body = parse_frontmatter(text)
            published = str(meta.get("published_at") or "0000-00-00")
            lane = path.parent.name

            # the corpus publishes its own date, so the segment is that date
            directory = rawstore.ingest_dir(PROVIDER, "documents", "event", published)
            leaf = rawstore.leaf_name(slug, lane, path.stem, ext="md", compression="gz")
            written = rawstore.bank(directory / leaf, data)

            disk += written
            banked += 1
            ledger.add(
                provider=PROVIDER, product="document",
                path=str((directory / leaf).relative_to(config.RAW)),
                source_path=str(path.relative_to(config.CORPUS_ROOT)),
                ticker=str(meta.get("ticker") or ticker),
                company=str(meta.get("company") or ""),
                isin=str(meta.get("isin") or ""),
                published_at=published,
                document_type=str(meta.get("document_type") or ""),
                period_label=str(meta.get("period") or ""),
                source_url=meta.get("source_url") or None,
                corpus_frozen_at=str(meta.get("corpus_frozen_at") or ""),
                lane_dir=lane,
                sha256=hashlib.sha256(data).hexdigest(),
                source_bytes=len(data), disk_bytes=written,
                fetched_at=fetched_at.isoformat(timespec="seconds"),
            )

    ledger.write()
    return {"banked": banked, "skipped": skipped, "disk": disk}


def main() -> int:
    started = time.time()
    stats = fetch()
    print(f"raw/corpus  {stats['banked']:,} documents banked verbatim as .md.gz  "
          f"{stats['disk'] / 1e6:.1f} MB on disk  "
          f"({stats['skipped']} navigation files skipped)  ({time.time() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
