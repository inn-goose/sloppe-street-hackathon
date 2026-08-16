"""The local parquet store.

Three roots, and the boundary between them is the whole point:

  raw/        verbatim bytes of the frozen corpus, banked with a content hash
              and an as-of stamp. Never rewritten, never interpreted.
  extracted/  a faithful typed view of raw. One row per fact the document
              actually states. No opinions, no fills, no derived columns.
  feature/    joins across extraction lanes. This is where decisions live.

Everything downstream reads parquet, never the corpus tree, so a rerun is a
pure function of the banked bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


def _column(values: list[Any]) -> pa.Array:
    kinds = {type(v) for v in values if v is not None}
    if not kinds:
        return pa.array(values, type=pa.string())
    if kinds <= {bool}:
        return pa.array(values, type=pa.bool_())
    if kinds <= {int, bool}:
        return pa.array([None if v is None else int(v) for v in values], type=pa.int64())
    if kinds <= {int, float, bool}:
        return pa.array([None if v is None else float(v) for v in values], type=pa.float64())
    return pa.array([None if v is None else str(v) for v in values], type=pa.string())


def to_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    if not rows:
        return pa.table({})
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return pa.table({name: _column([row.get(name) for row in rows]) for name in names})


def write(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table(list(rows)), path, compression="zstd")
    return path


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} has not been built yet")
    return pq.read_table(path).to_pylist()


def exists(path: Path) -> bool:
    return path.exists()


def describe(path: Path) -> str:
    meta = pq.read_metadata(path)
    size = path.stat().st_size
    return f"{path.name}: {meta.num_rows:,} rows x {meta.num_columns} cols, {size / 1024:.0f} KiB"


def group_by(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for row in rows:
        out.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    return out
