"""extracted/statement_facts — a number bound to a row label, a period and a unit.

The join `table_cells × table_columns` on `(doc_id, table_idx, ordinal)`. This is the lane the
whole forecast stands on, because it is the only one that carries the **non-GAAP** measures: five
of the twelve targets are adjusted or pre-exceptional, and no vendor publishes those.

## The free validator, and why it is the centrepiece

A financial table states its own arithmetic. Where a row carries a (current, prior, %change)
triple, `current / prior − 1` must reproduce the stated change. That check needs **no external
data and no judgement**, and it is the only thing that can catch a column binding that is shifted
by one — the failure mode where every number is real and every one is filed under the wrong
period.

Every fact is graded `verified` / `unchecked` / `failed` on it, and `failed` rows are kept rather
than dropped: a table whose arithmetic does not close is usually a *restated* comparative or a
sub-total the converter mangled, and hiding it would hide the evidence.

## Units, and the two ways they are stated

⛔ **Scale is declared once for the table and inherited by every cell** ("in millions"), so it is
applied from `table_spec.scale` — never guessed from a number's magnitude. A mis-scaled pair is
internally consistent, so no invariant can see it.

⛔ **Per-share rows do NOT inherit the table's scale.** `$4.58` in a table headed "in millions,
except per share data" is four dollars fifty-eight, not four million. The exception is stated in
the stub and is detected from the row label, because the *cell* decides, not the header.

⚠️ **A percent is never scaled.** `33.1` under a millions header is a margin.
"""

from __future__ import annotations

import re
from collections import Counter

from code.lib import config, store

_SCALE_FACTOR = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "": 1.0}

# Row labels whose values are per-share and therefore refuse the table's scale.
_PER_SHARE = re.compile(
    r"\bper\s+(?:diluted\s+|basic\s+|common\s+|adjusted\s+|ordinary\s+){0,2}share"
    r"|\bper\s+share\b|\bEPS\b|earnings\s+per|dividend[s]?\s+per|\bDPS\b", re.IGNORECASE)
# Row labels that are ratios even when the number carries no % marker.
_RATIO = re.compile(
    r"\bmargin\b|\brate\b|\bratio\b|\bpercent|\bconversion\b|\byield\b|\bgrowth\b"
    r"|\bcomparable\s+sales\b|\bcomps?\b|\bROE\b|\bROIC\b|\bROA\b", re.IGNORECASE)
# Row labels that are share/unit counts — a count is not money and must not take a currency.
_COUNT = re.compile(
    r"\bshares?\b|\bweighted\s+average\b|\bheadcount\b|\bemployees\b|\bstores\b|\boffices\b"
    r"|\bconsultants?\b|\btransactions\b", re.IGNORECASE)

#: Floor on the tolerance, in percentage points. The real tolerance is DERIVED per row — see
#: `_tolerance` — because a fixed one is wrong in both directions.
CHECK_TOL_PP = 0.05


def _decimals(raw: str, value: float) -> int:
    """How many decimals the document actually printed. `(25)` is 0, `4.9` is 1."""
    text = (raw or "").strip()
    if "." in text:
        tail = text.rsplit(".", 1)[-1]
        digits = "".join(c for c in tail if c.isdigit())
        if digits:
            return len(digits)
        return 0
    if text:
        return 0
    return 0 if abs(value - round(value)) < 1e-9 else 2


def _tolerance(cur: dict, prior: dict, change: dict) -> float:
    """The tolerance the DATA's own precision implies, in percentage points.

    ⛔ **A fixed tolerance was the residual failure class, and it was my bug, not the corpus's.**
    Measured on the survivors: Deere states `Construction and forestry` at 2,256 against 2,991 and
    prints the change as **−25**, where the exact ratio is −24.57 — a 0.43 pp gap that a 0.15 pp
    threshold calls a failure. The table is not wrong; it rounded, and so did its operands.

    Two independent sources of slack, added:

    * **the change's own rounding** — half a unit in the last place it printed, so `−25` carries
      ±0.5 pp and `4.9` carries ±0.05 pp;
    * **the operands' rounding, propagated** — `875` and `886` are whole millions, so the ratio
      inherits `0.5/|b| + 0.5·|a|/b²` before it is expressed as a percentage.

    That makes the check a statement about measurement precision rather than an arbitrary
    threshold, which is what lets a real disagreement stand out.
    """
    ulp = 0.5 * (10 ** -_decimals(change.get("raw", ""), change["value"]))
    a, b = abs(cur["value"]), abs(prior["value"])
    a_ulp = 0.5 * (10 ** -_decimals(cur.get("raw", ""), cur["value"]))
    b_ulp = 0.5 * (10 ** -_decimals(prior.get("raw", ""), prior["value"]))
    propagated = 100.0 * ((a_ulp / b) + (a * b_ulp / (b * b))) if b else 0.0
    return max(CHECK_TOL_PP, ulp + propagated)
#: A change against a tiny base is numerically meaningless — 0.1 → 0.2 is "+100 %" and rounding
#: alone moves it by tens of points. Those are graded `unchecked`, not `failed`.
CHECK_MIN_BASE = 1.0


def _unit_kind(label: str, percent: bool, currency: str | None, per_share_stub: bool) -> str:
    if percent:
        return "percent"
    if _PER_SHARE.search(label):
        return "per_share"
    if _RATIO.search(label):
        return "percent"
    if _COUNT.search(label):
        return "count"
    if currency:
        return "currency"
    return "currency" if per_share_stub is False else "currency"


def build() -> tuple[list[dict], dict]:
    cells = store.read(config.EXTRACTED / "table_cells.parquet")
    columns = store.read(config.EXTRACTED / "table_columns.parquet")
    specs = {(s["doc_id"], s["table_idx"]): s
             for s in store.read(config.EXTRACTED / "table_spec.parquet")}
    lanes = {r["doc_id"]: r for r in store.read(config.EXTRACTED / "document_lanes.parquet")}

    col_by_key = {(c["doc_id"], c["table_idx"], c["ordinal"]): c for c in columns}
    modal_width = {(s["doc_id"], s["table_idx"]): s["modal_width"] for s in specs.values()}

    facts: list[dict] = []
    for cell in cells:
        key = (cell["doc_id"], cell["table_idx"], cell["ordinal"])
        column = col_by_key.get(key)
        if column is None or not cell["label"]:
            continue
        spec = specs.get((cell["doc_id"], cell["table_idx"]), {})
        lane = lanes.get(cell["doc_id"], {})

        label = cell["label"]
        kind = _unit_kind(label, cell["percent"], cell["currency"], spec.get("per_share_stub", False))
        scale = spec.get("scale") or ""
        # ⛔ only a currency magnitude inherits the table's scale
        factor = _SCALE_FACTOR.get(scale, 1.0) if kind == "currency" else 1.0

        facts.append({
            "doc_id": cell["doc_id"], "ticker": cell["ticker"],
            "published_at": cell["published_at"],
            "lane_family": lane.get("lane_family", ""),
            "table_idx": cell["table_idx"], "row_idx": cell["row_idx"],
            "ordinal": cell["ordinal"],
            "label": label,
            # the nearest label-only row above this one — the statement's own grouping
            "section": cell.get("section") or "",
            # ⛔ WHICH entity the row belongs to. A row labelled `Operating profit` is Deere's
            # Production & Precision Ag segment or its Small Ag & Turf segment depending only on
            # the caption above its table, and one of the twelve targets is the former.
            "scope": spec.get("scope") or "",
            "is_consolidated": bool(spec.get("is_consolidated", True)),
            "column_token": column["token"], "column_kind": column["kind"],
            "period_end": column["period_end"], "period_year": column["period_year"],
            "span_months": column["span_months"],
            "quarter_hint": column.get("quarter_hint") or 0,
            # a measure-header table takes its period from the document, not from a column
            "period_from_document": bool(column.get("period_from_document")),
            "document_period": lane.get("period_label") or "",
            "align_grade": column["align_grade"],
            # ⛔ **A row with FEWER numbers than its table's columns is mis-bindable, and the
            # binding will still look plausible.** Measured on HD's selected-sales-data table:
            # `| Comparable sales | 1.0 % | (3.3) % | N/A | 0.4 % | (3.1) % | N/A |` yields four
            # numbers against six columns, because the two `N/A` change cells produce none. Bound
            # by ordinal, the six-month figure (0.4) inherits the quarter's date — a real number
            # filed under the wrong period, which no downstream check can see. Flagged here so a
            # consumer can refuse the tier; the prose lane covers the affected metric independently.
            "row_numbers": cell["n_numbers"],
            "table_width": modal_width.get((cell["doc_id"], cell["table_idx"]), 0),
            "row_width_matches": cell["n_numbers"] == modal_width.get(
                (cell["doc_id"], cell["table_idx"]), 0),
            # the token as printed — the validator needs the stated PRECISION, not just the value
            "raw": cell.get("raw") or "",
            "value": cell["value"],
            # ⚠️ **`value_scaled` is only meaningful where `scale_known`.** The scale is declared
            # on 52 % of tables; on the rest the factor is 1.0, so a figure written in millions is
            # carried as if it were units and `value_scaled == value`. Mixing the two across
            # tables would be a silent 10^6 error, and no invariant can see it because a
            # mis-scaled pair is internally consistent. Consumers must filter on `scale_known` or
            # use `value` with the table's own declaration.
            "value_scaled": cell["value"] * factor,
            "scale": scale, "scale_factor": factor,
            "scale_known": bool(scale) or kind != "currency",
            "unit_kind": kind, "currency": cell["currency"],
            "check": "unchecked", "check_error": None,
        })

    facts, checked = _validate(facts)
    return facts, {"facts": len(facts), "checks": dict(checked)}


_VS_YEARS = re.compile(r"((?:19|20)\d{2})\D{1,8}((?:19|20)\d{2})")
_BPS = re.compile(r"\bbps\b|\bbasis\s*points?\b", re.IGNORECASE)


def _validate(facts: list[dict]) -> tuple[list[dict], Counter]:
    """Recompute every stated % change from the columns it names.

    ⛔ **A change column's OPERANDS are named in its own token where the table is ambiguous.**
    Measured on HD's 2015 selected-financials table, the layout is
    `[2014, 2013, 2012, "2014 vs. 2013", "2013 vs. 2012"]` — three periods and two changes — and
    the naive "two columns to my left" rule pairs 2013 against 2012 for the first change and
    against a *change column* for the second. That single wrong assumption produced 3,825 failed
    triples against 911 closed ones, i.e. the validator was mostly measuring its own bug. Reading
    the operands off the token fixes it, and where the token is generic the positional fallback is
    only taken when the layout is unambiguously `[period, period, change]` blocks.

    ⛔ **A common-size table cannot close and must not be graded `failed`.** Where a table
    expresses every line as a percent of sales, its top line reads `100.0 | 100.0 | 100.0` while
    the change columns describe the underlying dollars. Detected at table level from that
    signature and marked `common_size`.

    ⚠️ **A basis-point change is a DIFFERENCE, not a ratio.** `33.1 % | 33.4 % | (30) bps` is
    `(33.1 − 33.4) × 100`. Checked on the right arithmetic rather than refused.
    """
    by_row: dict[tuple, list[dict]] = {}
    for f in facts:
        by_row.setdefault((f["doc_id"], f["table_idx"], f["row_idx"]), []).append(f)

    # table-level: a common-size table announces itself with an all-100.0 line
    common_size: set[tuple] = set()
    for (doc_id, table_idx, _row), group in by_row.items():
        periods = [f["value"] for f in group if f["column_kind"].startswith("period")]
        if len(periods) >= 2 and all(abs(v - 100.0) < 1e-9 for v in periods):
            common_size.add((doc_id, table_idx))

    checked: Counter = Counter()
    for (doc_id, table_idx, _row), group in by_row.items():
        group.sort(key=lambda f: f["ordinal"])
        by_ord = {f["ordinal"]: f for f in group}
        is_common = (doc_id, table_idx) in common_size

        for f in group:
            if f["column_kind"] not in ("change", "lfl_change"):
                continue
            if is_common:
                f["check"] = "common_size"
                checked["common_size"] += 1
                continue

            cur = prior = None
            # (1) the token names its operands
            named = _VS_YEARS.search(f["column_token"] or "")
            if named:
                want_cur, want_prior = named.group(1), named.group(2)
                for other in group:
                    token = other["column_token"] or ""
                    if other["column_kind"].startswith("period"):
                        if token.strip() == want_cur or str(other["period_year"]) == want_cur:
                            cur = other
                        elif token.strip() == want_prior or str(other["period_year"]) == want_prior:
                            prior = other
            # (2) positional, ONLY for an unambiguous [period, period, change] block
            if cur is None or prior is None:
                left2, left1 = by_ord.get(f["ordinal"] - 2), by_ord.get(f["ordinal"] - 1)
                left3 = by_ord.get(f["ordinal"] - 3)
                block_ok = (
                    left2 is not None and left1 is not None
                    and left2["column_kind"].startswith("period")
                    and left1["column_kind"].startswith("period")
                    and (left3 is None or left3["column_kind"] in ("change", "lfl_change"))
                )
                if block_ok:
                    cur, prior = left2, left1

            if cur is None or prior is None:
                f["check"] = "unbindable"
                checked["unbindable"] += 1
                continue
            if abs(prior["value"]) < CHECK_MIN_BASE or prior["value"] == 0:
                f["check"] = "base_too_small"
                checked["base_too_small"] += 1
                continue

            # a bps column is a difference in percentage points, not a ratio
            derived = _tolerance(cur, prior, f)
            if _BPS.search(f["column_token"] or "") or (
                    cur["unit_kind"] == "percent" and prior["unit_kind"] == "percent"
                    and abs(f["value"]) > 100):
                recomputed = (cur["value"] - prior["value"]) * 100.0
                tol = max(1.0, derived)
            elif cur["unit_kind"] == "percent" and prior["unit_kind"] == "percent":
                # a margin table's "change" is usually the pp difference
                diff = cur["value"] - prior["value"]
                ratio = (cur["value"] / prior["value"] - 1.0) * 100.0
                recomputed = diff if abs(diff - f["value"]) <= abs(ratio - f["value"]) else ratio
                tol = derived
            else:
                recomputed = (cur["value"] / prior["value"] - 1.0) * 100.0
                tol = derived

            error = abs(recomputed - f["value"])
            verdict = "verified" if error <= tol else "failed"
            for target in (cur, prior, f):
                target["check"] = verdict
                target["check_error"] = round(error, 4)
            checked[verdict] += 1
    return facts, checked


def main() -> int:
    facts, stats = build()
    store.write(config.EXTRACTED / "statement_facts.parquet", facts)
    grades = Counter(f["check"] for f in facts)
    verified = grades.get("verified", 0)
    failed = grades.get("failed", 0)
    rate = verified / max(verified + failed, 1)
    print(f"extracted/statement_facts.parquet {len(facts):,} facts "
          f"bound to a period column")
    print(f"  arithmetic invariant: {stats['checks'].get('verified', 0):,} triples closed, "
          f"{stats['checks'].get('failed', 0):,} did not  "
          f"({rate:.1%} of graded facts verified)")
    print(f"  fact grades: {dict(grades)}")
    kinds = Counter(f["unit_kind"] for f in facts)
    print(f"  unit kinds: {dict(kinds)}")
    matched = sum(1 for f in facts if f["row_width_matches"])
    print(f"  row width matches its table on {matched:,} of {len(facts):,} facts "
          f"({matched / max(len(facts), 1):.1%}) — the rest are mis-bindable and flagged")
    for t in ("HD", "ADI", "LSE:HAS", "DE"):
        mine = [f for f in facts if f["ticker"] == t]
        v = sum(1 for f in mine if f["check"] == "verified")
        print(f"  {t:<9}{len(mine):>7,} facts  {v:>6,} verified  "
              f"{len({f['label'] for f in mine}):>5,} distinct labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
