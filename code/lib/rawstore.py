"""`data/raw/` — verbatim bytes, on a fixed path grammar.

    <root>/raw/<provider>/[<domain>/]<data_type>/<granularity>/<as-of…>/<leaf>.<ext>[.gz]
    └────────── prefix ──┘ └──────────────── base ───────────────────┘  └─── suffix ───┘

⛔ **raw is the source's own format.** JSON stays JSON, CSV stays CSV, HTML stays HTML. Nothing is
re-encoded, re-serialised, or folded into a columnar file at ingest: a store that banks a JSON
body inside a parquet column has already transformed it, and the bytes can then only be read back
through the reader that wrote them. Parquet belongs to `extracted/` and `feature/`, which are our
own layouts and may be shaped however they read best.

⛔ **The leaf is the source's own id, verbatim** — `HD.json`, `AP2Y.json`, `DE-q2-8k.htm`. That is
what lets the store reconcile against the source by set difference rather than by re-derivation.

⚠️ **gzip is storage, not a transform.** `gunzip` reproduces the bytes exactly, and `mtime=0`
makes re-compressing unchanged bytes byte-identical — so a re-fetch that changed nothing is
detectable as a no-op instead of looking like new data.

## The as-of, and why it is the fetch instant for most of this store

Nothing on a current-state surface says when it is as of: `quoteSummary` answers with the analyst
panel standing *now*; a stockanalysis page answers with the statements standing *now*. The only
honest name for the directory is the moment we looked. A source that publishes its own dated
period — a FRED series, an SEC filing — is banked under **that** period instead, through
:func:`ingest_dir`.

## The envelope

A JSON source is banked wrapped: `{"fetched_at":…, "served_at":…, "request":…, "response": <body>}`
so a capture is **placeable without its path**. A delimited or markup source is banked bare —
splicing a stamp into CSV or HTML would stop it being the source's bytes.
"""

from __future__ import annotations

import gzip
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from code.lib import config

#: The vocabularies. A typo becomes a new tree otherwise, and a new tree is an invisible gap.
DATA_TYPES = {"snapshot", "bars", "series", "documents", "contracts", "news", "index", "facts"}
GRANULARITIES = {"tick", "hour", "day", "week", "month", "quarter", "year", "event", "none"}


def now() -> datetime:
    return datetime.now(timezone.utc)


def segment_of(as_of: datetime) -> str:
    """`<YYYY-MM-DD>/<HH-MM-SS-ffffff>` — the capture instant as two path segments."""
    return f"{as_of:%Y-%m-%d}/{as_of:%H-%M-%S-%f}"


def _check(data_type: str, granularity: str) -> None:
    if data_type not in DATA_TYPES:
        raise ValueError(f"unknown data_type {data_type!r}; expected one of {sorted(DATA_TYPES)}")
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"unknown granularity {granularity!r}; expected one of {sorted(GRANULARITIES)}")


def ingest_dir(provider: str, data_type: str, granularity: str, segment: str,
               domain: str | None = None) -> Path:
    """The base component for a source-published period (a series date, a filing date)."""
    _check(data_type, granularity)
    parts = [config.RAW, provider]
    if domain:
        parts.append(domain)
    parts += [data_type, granularity, segment]
    path = Path(*[str(p) for p in parts])
    return path


def capture_dir(provider: str, data_type: str, granularity: str, *, as_of: datetime,
                domain: str | None = None) -> Path:
    """The base component for a capture instant — the shape is fixed, callers cannot deepen it."""
    return ingest_dir(provider, data_type, granularity, segment_of(as_of), domain=domain)


def leaf_name(*parts: str, ext: str, compression: str | None = None) -> str:
    """The suffix — the source's own id, its format, its compression.

    ⛔ Refuses anything that would escape the capture directory.
    """
    segments: list[str] = []
    for part in parts:
        for piece in str(part).split("/"):
            if piece in ("", ".", ".."):
                if piece:
                    raise ValueError(
                        f"{part!r} is not a leaf segment — it would escape the capture directory")
                continue
            segments.append(piece)
    if not segments:
        raise ValueError("a leaf needs at least one segment — the source's own id")
    name = "/".join(segments) + f".{ext.lstrip('.')}"
    return f"{name}.{compression.lstrip('.')}" if compression else name


def envelope(body: str, *, fetched_at: datetime, served_at: str | None = None,
             request: dict | None = None) -> str:
    """A JSON body, stamped and still verbatim under `response`."""
    return json.dumps({
        "fetched_at": fetched_at.isoformat(timespec="seconds"),
        "served_at": served_at,
        "request": request or {},
        "response": json.loads(body),
    }, separators=(",", ":"))


def bank(path: Path, body: str | bytes, *, compress: bool = True) -> int:
    """Write one capture. Returns bytes on disk.

    ⚠️ `mtime=0` is load-bearing: gzip stamps the mtime into its header, so without it two
    compressions of identical bytes differ and every re-fetch looks like new data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = body.encode("utf-8") if isinstance(body, str) else body
    if compress:
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
            handle.write(data)
        payload = buffer.getvalue()
    else:
        payload = data
    path.write_bytes(payload)
    return len(payload)


def read(path: Path) -> str:
    """A banked capture back as text — gunzip if it is gzipped, otherwise as-is."""
    data = path.read_bytes()
    if path.suffix == ".gz" or data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("utf-8", "replace")


def read_response(path: Path):
    """The `response` member of a banked JSON envelope, or the whole body if it is bare."""
    body = json.loads(read(path))
    return body.get("response", body) if isinstance(body, dict) else body


class Ledger:
    """An append-only JSONL index of what a fetch banked.

    ⚠️ It is **metadata about** raw, not raw, so it is a plain text line per capture rather than a
    columnar file — greppable, diffable, and readable without this package.
    """

    def __init__(self, provider: str):
        self.path = config.RAW / provider / "_ledger.jsonl"
        self.rows: list[dict] = []

    def add(self, **row) -> None:
        self.rows.append(row)

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        return self.path

    def __len__(self) -> int:
        return len(self.rows)


def iter_ledger(provider: str):
    """Every ledger row for a provider, for the extractors to walk."""
    path = config.RAW / provider / "_ledger.jsonl"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_captures(provider: str, product: str | None = None, *, parse: bool = True):
    """`(ledger row, payload)` for every successfully banked capture.

    `parse=True` returns the JSON body's `response` member; `parse=False` returns the text, which
    is what a CSV or HTML capture needs. Rows that were refused carry no path and are skipped —
    a refusal is inventory, not input.
    """
    for row in iter_ledger(provider):
        if not row.get("path") or row.get("state") not in (None, "ok"):
            continue
        path = config.RAW / row["path"]
        if not path.exists():
            continue
        yield row, (read_response(path) if parse else read(path))
