"""feature/metric_panel — one row per (ticker, metric, fiscal period), reconciled across lanes.

The join the whole model stands on: `periods.Resolver` supplies *when*, `metrics.REGISTRY`
supplies *what*, and this puts the two together over four independent lanes.

## What reconciliation means here

A period is usually stated by several lanes and several documents. Rather than take the first, the
panel keeps **every witness**, then reports the consensus value and the disagreement:

* `value` — the modal witness (ties broken by the median), in the registry's unit
* `n_witnesses` / `n_lanes` — how much support it has
* `spread` — the relative gap between the extreme witnesses
* `witnesses` — the lanes that spoke, so a disagreement is traceable

⚠️ **Agreement across lanes is not proof; disagreement IS proof of error.** A wide `spread` on a
key means one lane is wrong and the panel says so rather than averaging it away.

## Units, made uniform once

The workbook wants `USDm` / `GBPm` / `USD per share` / percentage points. The corpus carries
millions-or-units depending on a declaration, every vendor carries base units. Everything is
converted here, once, and the registry's `unit` is the contract from this point on.

⛔ **Only `scale_known` corpus facts are admitted for a currency metric.** Where a table never
declared its magnitude the value is carried as if it were units, and mixing that with a vendor's
base units is a silent 10⁶ error that no invariant can catch.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict

from code.feature import metrics as M
from code.feature.periods import Resolver
from code.lib import config, store


def _num(value):
    return None if value is None else float(value)


def _norm(metric: M.Metric, value: float, lane: str) -> float | None:
    """A lane's value in the registry's unit."""
    if value is None:
        return None
    if metric.unit == "currency_m":
        return value / M.TO_MILLIONS if lane != "corpus_millions" else value
    if metric.unit == "shares_m":
        return value / M.TO_MILLIONS if lane != "corpus_millions" else value
    return value


def _match(pattern: str, text: str) -> bool:
    return bool(pattern) and bool(re.match(pattern, (text or "").strip(), re.IGNORECASE))


def build() -> tuple[list[dict], dict]:
    resolver = Resolver()
    facts = store.read(config.EXTRACTED / "statement_facts.parquet")
    prose = store.read(config.EXTRACTED / "prose_facts.parquet")
    sa = store.read(config.EXTRACTED / "sa_financials.parquet")
    sec = store.read(config.EXTRACTED / "sec_facts.parquet")
    sym = {s["ticker"]: s for s in store.read(config.EXTRACTED / "symbology.parquet")}

    veto = {m.name: re.compile(m.section_veto, re.IGNORECASE) for m in M.REGISTRY.values()}
    scope_rx = {m.name: (re.compile(m.scope, re.IGNORECASE) if m.scope else None)
                for m in M.REGISTRY.values()}

    witnesses: dict[tuple, list[dict]] = defaultdict(list)
    refusals: Counter = Counter()

    # ── corpus statement tables ────────────────────────────────────────────────
    #: unit the registry declares -> unit kinds a fact may carry to satisfy it
    compatible = {"currency_m": {"currency"}, "shares_m": {"count", "currency"},
                  "per_share": {"per_share"}, "percent": {"percent"}, "ratio": {"percent"}}
    segment_rx = {t: re.compile(p, re.IGNORECASE) for t, p in M.SEGMENT_NAMES.items()}

    for row in facts:
        for metric in M.BY_TICKER.get(row["ticker"], ()):
            if not _match(metric.corpus, row["label"]):
                continue
            # ⛔ **The unit kind must match what the metric IS.** ADI states `Adjusted gross
            # margin` in dollars and `…percentage` in points; without this gate a dollar figure
            # entered a percent target and the panel returned 2,645,262 for 73.0.
            if row["unit_kind"] not in compatible.get(metric.unit, set()):
                refusals[f"{metric.name}:unit_kind_{row['unit_kind']}"] += 1
                continue
            # ⛔ **A change column is not a level.** Deere's P&PA operating profit resolved to
            # −0.00003 because the `−39 %` change column matched the same row label as the 706.
            if row.get("column_kind") in ("change", "lfl_change"):
                refusals[f"{metric.name}:change_column"] += 1
                continue
            # ⛔ **A consolidated metric refuses a segment's row.** See `SEGMENT_NAMES`.
            if metric.basis != "segment":
                seg = segment_rx.get(metric.ticker)
                context = f"{row.get('scope') or ''} {row.get('section') or ''}"
                if seg is not None and seg.search(context):
                    refusals[f"{metric.name}:segment_context"] += 1
                    continue
            if metric.unit in ("currency_m", "shares_m") and not row["scale_known"]:
                refusals[f"{metric.name}:scale_unknown"] += 1
                continue
            if not row["row_width_matches"]:
                refusals[f"{metric.name}:row_mis_bindable"] += 1
                continue
            if veto[metric.name].search(row.get("section") or ""):
                refusals[f"{metric.name}:disaggregated_section"] += 1
                continue
            # ⛔ A `period_span` column is a DESCRIPTION ("Three Months Ended"), not an identifier.
            # Measured on HD's Q1-FY2026 10-Q, a value of 4,002 bound to such a column and landed
            # in the same key as the true 41,765 — the two are a segment and the total.
            if row.get("column_kind") == "period_span":
                refusals[f"{metric.name}:span_column_not_a_period"] += 1
                continue
            # ⛔ ONE mechanism decides scope, not two. A segment metric must match its own segment;
            # a consolidated metric must not match ANY segment name. The earlier
            # `is_consolidated` flag — true only when a table's caption named the company — was a
            # second, cruder test that refused ADI's `Results Summary` table outright and with it
            # the entire adjusted-EPS series, because a perfectly consolidated statement carries a
            # caption that is not the company's name.
            rx = scope_rx[metric.name]
            if rx is not None and not rx.search(row.get("scope") or ""):
                refusals[f"{metric.name}:wrong_segment"] += 1
                continue

            if row["period_from_document"]:
                key = resolver.by_phrase(row["ticker"], row.get("document_period") or "",
                                         row["doc_id"])
            else:
                span = row["span_months"] or 3
                if row["period_end"]:
                    key = resolver.by_date(row["ticker"], row["period_end"], span)
                elif row["period_year"]:
                    key = resolver.by_year(row["ticker"], row["period_year"],
                                           row["span_months"] or 0,
                                           row.get("quarter_hint") or 0)
                else:
                    key = resolver.by_phrase(row["ticker"],
                                             row.get("document_period") or "", row["doc_id"])
            if not key.ok:
                refusals[f"{metric.name}:unresolved_period"] += 1
                continue
            value = (row["value_scaled"] if metric.unit in ("currency_m", "shares_m")
                     else row["value"])
            witnesses[(metric.name, key.fiscal_year, key.fiscal_period)].append({
                "lane": "corpus", "value": _norm(metric, value, "corpus"),
                "doc_id": row["doc_id"], "published_at": row["published_at"],
                "grade": key.grade, "check": row["check"],
            })

    # ── corpus prose ───────────────────────────────────────────────────────────
    for row in prose:
        for metric in M.BY_TICKER.get(row["ticker"], ()):
            if not _match(metric.prose, row["metric"]):
                continue
            key = resolver.by_phrase(row["ticker"], row["period_phrase"], row["doc_id"])
            if not key.ok:
                refusals[f"{metric.name}:unresolved_phrase"] += 1
                continue
            value = row["value_base"] if metric.unit in ("currency_m", "shares_m") else row["value"]
            witnesses[(metric.name, key.fiscal_year, key.fiscal_period)].append({
                "lane": "prose", "value": _norm(metric, value, "prose"),
                "doc_id": row["doc_id"], "published_at": row["published_at"],
                "grade": key.grade, "check": "",
            })

    # ── stockanalysis ──────────────────────────────────────────────────────────
    for row in sa:
        if row["statement"] != "income_statement" or row["grain"] != "quarterly":
            continue
        ticker = next((t for t, s in sym.items() if s["short"] == row["symbol"]), None)
        if ticker is None:
            continue
        for metric in M.BY_TICKER.get(ticker, ()):
            if not _match(metric.sa, row["metric"]):
                continue
            key = resolver.by_date(ticker, row["period_end"], 3)
            if not key.ok:
                refusals[f"{metric.name}:sa_unresolved"] += 1
                continue
            witnesses[(metric.name, key.fiscal_year, key.fiscal_period)].append({
                "lane": "sa", "value": _norm(metric, _num(row["value"]), "sa"),
                "doc_id": "", "published_at": row["captured_at"][:10],
                "grade": key.grade, "check": "",
            })

    # ── SEC XBRL, as first reported ────────────────────────────────────────────
    seen_first: dict[tuple, dict] = {}
    for row in sec:
        if row["duration"] not in ("quarter", "year"):
            continue
        for metric in M.BY_TICKER.get(row["ticker"], ()):
            if not _match(metric.sec, row["concept"]):
                continue
            ident = (metric.name, row["ticker"], row["start"], row["end"], row["unit"])
            prev = seen_first.get(ident)
            if prev is None or (row["filed"] or "9999") < (prev["filed"] or "9999"):
                seen_first[ident] = row
    for (name, ticker, _s, end, _u), row in seen_first.items():
        metric = M.REGISTRY[name]
        key = resolver.by_date(ticker, end, 3 if row["duration"] == "quarter" else 12)
        if not key.ok:
            refusals[f"{name}:sec_unresolved"] += 1
            continue
        witnesses[(name, key.fiscal_year, key.fiscal_period)].append({
            "lane": "sec", "value": _norm(metric, _num(row["value"]), "sec"),
            "doc_id": row["accession"] or "", "published_at": row["filed"],
            "grade": key.grade, "check": "",
        })

    # ── reconcile ──────────────────────────────────────────────────────────────
    rows = []
    for (name, fy, period), group in witnesses.items():
        values = [w["value"] for w in group if w["value"] is not None]
        if not values:
            continue
        # ⚡ **The modal witness wins, and its support is reported.** A statement repeats the same
        # figure across the release, the 10-Q and later comparatives, so the truth is the value
        # most documents state — measured on HD Q1-FY2026 net sales, 41,765 has seven witnesses
        # against a segment's 37,763 with two. A mean would land between them and be wrong by
        # construction; `consensus_share` is what says how firm the winner is.
        rounded = [round(v, 4) for v in values]
        counts = Counter(rounded)
        modal, count = counts.most_common(1)[0]
        value = modal if count > 1 else statistics.median(values)
        consensus_share = count / len(values)
        lo, hi = min(values), max(values)
        scale = max(abs(lo), abs(hi)) or 1.0
        metric = M.REGISTRY[name]
        lanes = sorted({w["lane"] for w in group})
        rows.append({
            "ticker": metric.ticker, "metric": name, "workbook_label": metric.workbook_label,
            "is_target": metric.is_target, "unit": metric.unit, "basis": metric.basis,
            "fiscal_year": fy, "fiscal_period": period, "label": f"FY{fy}{period}",
            "value": value, "n_witnesses": len(values), "n_lanes": len(lanes),
            "lanes": ",".join(lanes),
            "consensus_share": round(consensus_share, 4),
            "n_distinct": len(counts),
            "spread": round(abs(hi - lo) / scale, 5),
            "min": lo, "max": hi,
            "grade": Counter(w["grade"] for w in group).most_common(1)[0][0],
            "verified": any(w["check"] == "verified" for w in group),
            "latest_published_at": max((w["published_at"] or "") for w in group),
            # kept with their support so a losing candidate can be reinstated, by vote, when the
            # winner breaks an identity
            "_candidates": sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0])),
        })
    rows.extend(_derive_fiscal_years(rows))
    rows.sort(key=lambda r: (r["ticker"], r["metric"], r["fiscal_year"] or 0, r["fiscal_period"]))
    return rows, {"refusals": refusals}


def _derive_fiscal_years(rows: list[dict]) -> list[dict]:
    """Sum four quarters into a fiscal year where the filer does not state one.

    ⚡ **Home Depot and Deere guide the FULL YEAR**, so a guide cannot be compared to an outcome
    unless the outcome exists at annual grain. HD's 10-K states annual totals, but not on every
    metric and not in a form that always binds — measured, the growth guides had **zero**
    comparable outcomes before this, which silenced the conservatism signal for the company whose
    guidance is entirely annual.

    ⛔ **Only complete years, and only additive metrics.** Three quarters summed into a "year"
    understates it by a quarter and would read as a catastrophic guidance miss; a margin or a
    growth rate is not additive at all. Derived rows are marked `derived_fy` so nothing mistakes
    them for something the filer published.
    """
    from collections import defaultdict as _dd

    additive = {"currency_m", "per_share", "shares_m"}
    stated = {(r["metric"], r["fiscal_year"], r["fiscal_period"]) for r in rows}
    buckets: dict[tuple, dict[str, dict]] = _dd(dict)
    for row in rows:
        if row["fiscal_period"] in ("Q1", "Q2", "Q3", "Q4") and row["fiscal_year"]:
            buckets[(row["metric"], row["fiscal_year"])][row["fiscal_period"]] = row

    out = []
    for (name, fy), quarters in buckets.items():
        metric = M.REGISTRY.get(name)
        if metric is None or metric.unit not in additive:
            continue
        if len(quarters) != 4 or (name, fy, "FY") in stated:
            continue
        total = sum(q["value"] for q in quarters.values())
        out.append({
            "ticker": metric.ticker, "metric": name, "workbook_label": metric.workbook_label,
            "is_target": metric.is_target, "unit": metric.unit, "basis": metric.basis,
            "fiscal_year": fy, "fiscal_period": "FY", "label": f"FY{fy}FY",
            "value": total,
            "n_witnesses": sum(q["n_witnesses"] for q in quarters.values()),
            "n_lanes": max(q["n_lanes"] for q in quarters.values()),
            "lanes": "derived_fy",
            "consensus_share": min(q["consensus_share"] for q in quarters.values()),
            "n_distinct": 1,
            "spread": max(q["spread"] for q in quarters.values()),
            "min": total, "max": total,
            "grade": "derived_fy",
            "verified": all(q["verified"] for q in quarters.values()),
            "latest_published_at": max(q["latest_published_at"] for q in quarters.values()),
        })
    return out


#: `Q1+Q2 = H1`, `H1+Q3 = M9`, `M9+Q4 = FY`, `Q1+Q2+Q3+Q4 = FY`. Every filer reports on this
#: ladder; the identities are definitional, so they hold without any assumption about the business.
_LADDER: tuple[tuple[tuple[str, ...], str], ...] = (
    (("Q1", "Q2"), "H1"),
    (("H1", "Q3"), "M9"),
    (("M9", "Q4"), "FY"),
    (("Q1", "Q2", "Q3", "Q4"), "FY"),
    (("H1", "Q3", "Q4"), "FY"),
)
#: Only these periods sit on the ladder — TTM and the like are spans, not rungs.
_PERIODS = frozenset({"Q1", "Q2", "Q3", "Q4", "H1", "H2", "M9", "FY"} - {"H2"})
#: A cell this far from what the ladder implies is contradicting it, not rounding into it.
LADDER_TOL = 0.005


def _violations(values: dict[str, float]) -> list[tuple]:
    """Which ladder identities are observable in this group, and broken."""
    broken = []
    for parts, whole in _LADDER:
        if whole not in values or any(p not in values for p in parts):
            continue
        total = sum(values[p] for p in parts)
        if abs(total - values[whole]) > LADDER_TOL * max(abs(values[whole]), 1e-9):
            broken.append((parts, whole))
    return broken


def _solve_for(period: str, values: dict[str, float]) -> list[float]:
    """Every value the ladder implies for one period, given the others."""
    out = []
    for parts, whole in _LADDER:
        members = set(parts) | {whole}
        if period not in members:
            continue
        others = members - {period}
        if any(o not in values for o in others):
            continue
        out.append(values[whole] - sum(values[p] for p in parts if p != period)
                   if period in parts else sum(values[p] for p in parts))
    return out


def _plausible(name: str, period: str, was: float, now: float,
               history: dict[tuple, list[tuple[int, float]]], fy: int,
               all_positive: bool) -> str | None:
    """Would this repair produce a number the filer could plausibly have reported?

    ⛔ **A ladder solution is arithmetic, and arithmetic will happily return a value that cannot
    exist.** Deere's pre-2018 quarters are contaminated with year-to-date columns, and contaminated
    quarters can be *mutually* consistent while all being wrong — so the single-fault test blamed
    the one cell that was right. It "repaired" Q1 net sales to **−$19.6 bn** and rewrote Deere's
    FY2014 net income from its correct **$3,162 m** to **$5,675 m**.

    Two refusals, both measured against the panel's own history rather than any outside view:

    * a metric that has never once been negative does not begin now
    * a repair must not land **farther** from its neighbouring years than the value it replaces —
      if the "fix" makes the series less like itself, the diagnosis picked the wrong cell
    """
    if all_positive and now < 0:
        return "sign"
    neighbours = [v for y, v in history.get((name, period), []) if 0 < abs(y - fy) <= 2]
    if len(neighbours) >= 2:
        centre = statistics.median(neighbours)
        if abs(now - centre) > abs(was - centre):
            return "farther_from_neighbours"
    return None


def _reconcile_ladder(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Arbitrate contested cells with the period algebra — but only under a single-fault diagnosis.

    ⚡ **The panel's modal vote is a popularity contest, and popularity loses to arithmetic.**
    Deere's FY2019 Q1 net income had *sixteen* witnesses and **12 %** agreement: the vote returned
    **$122 m** where the year minus the other three quarters gives **$498 m**, which is what Deere
    reported. More witnesses would not have helped, because the witnesses disagree. What settles it
    is that the quarters must reach the year — and that matters beyond one cell, since quarterly
    shares are measured from these years and Deere's Q3 net income share feeds a submitted number.

    ⛔ **Blaming the least-agreed cell does not work, and the first attempt at it was wrong.** That
    rule assumes exactly one bad value per identity; Deere's pre-2018 years are corrupt in several
    cells at once, so each identity blamed a different term, the pass oscillated (`FY2017 H1` flipped
    between 995.4 and 1,006.2 on alternating iterations) and it "repaired" net sales to **−20,923**.
    Imputation on top of multiply-broken data manufactures numbers that never existed.

    So the test is a **single-fault diagnosis** instead: a cell is repaired only when removing it
    makes every remaining identity hold, *and* the identities that determine it all agree on one
    replacement. That is a deduction, not an attribution — and when several cells are bad no
    candidate passes, so the group is left broken on purpose for the audit to report and for
    seasonality to drop. Refusing to guess is the whole point; a year that cannot be reconciled is
    not a year to measure a seasonal share from.
    """
    from collections import defaultdict as _dd

    additive = {"currency_m", "per_share"}
    groups: dict[tuple, dict[str, dict]] = _dd(dict)
    for row in rows:
        if row["fiscal_year"]:
            groups[(row["metric"], row["fiscal_year"])][row["fiscal_period"]] = row

    # the pre-repair series, so a candidate fix is judged against untouched neighbours
    history: dict[tuple, list[tuple[int, float]]] = _dd(list)
    positive: dict[str, bool] = {}
    for row in rows:
        if row["fiscal_year"]:
            history[(row["metric"], row["fiscal_period"])].append(
                (row["fiscal_year"], row["value"]))
            positive[row["metric"]] = positive.get(row["metric"], True) and row["value"] >= 0

    repairs: list[dict] = []
    for (name, fy), cells in sorted(groups.items()):
        metric = M.REGISTRY.get(name)
        if metric is None or metric.unit not in additive:
            continue
        values = {p: c["value"] for p, c in cells.items() if p in _PERIODS}
        broken = _violations(values)

        # ⚠️ EPS is additive across quarters only up to the share count: a filer computes annual EPS
        # on the year's average shares, not by summing four quarterly figures. Measured, ADI's FY2021
        # sums 6.5% away from its stated annual EPS purely from the Maxim issuance, so overruling
        # there would corrupt a correct number. Per-share gaps may be filled, never overruled.
        if broken and metric.unit != "per_share":
            candidates = []
            for suspect in list(values):
                trial = {k: v for k, v in values.items() if k != suspect}
                if _violations(trial):
                    continue                       # removing it does not explain the rest
                solved = _solve_for(suspect, trial)
                if not solved:
                    continue
                if max(solved) - min(solved) > LADDER_TOL * max(abs(solved[0]), 1e-9):
                    continue                       # the identities disagree about the replacement
                candidates.append((suspect, solved[0]))
            # ⚡ **When arithmetic narrows to two suspects, lane agreement picks between them.**
            # Deere's FY2019 Q1 and Q2 both explain the break — removing either makes the ladder
            # hold — so the algebra alone must refuse. But Q1's sixteen witnesses agree only **12 %**
            # of the time while Q2's ten agree 30 %, and the ladder's answer for Q1 (498) is Deere's
            # reported figure. A cell its own sources cannot agree on is the fault; a clear margin is
            # required so this never decides a close call.
            if len(candidates) > 1:
                ranked = sorted(candidates, key=lambda c: (cells[c[0]]["consensus_share"],
                                                           -cells[c[0]]["n_witnesses"]))
                best, runner = cells[ranked[0][0]], cells[ranked[1][0]]
                if (best["consensus_share"] < 0.55
                        and runner["consensus_share"] - best["consensus_share"] >= 0.10):
                    candidates = [ranked[0]]
            if len(candidates) == 1:
                period, fixed = candidates[0]
                cell = cells[period]
                refusal = _plausible(name, period, cell["value"], fixed, history, fy,
                                     positive.get(name, False))
                if refusal:
                    repairs.append({
                        "metric": name, "fiscal_year": fy, "fiscal_period": period,
                        "action": "implausible", "identity": refusal,
                        "was": cell["value"], "now": fixed,
                        "consensus_share": cell["consensus_share"],
                        "n_witnesses": cell["n_witnesses"],
                    })
                    continue
                repairs.append({
                    "metric": name, "fiscal_year": fy, "fiscal_period": period,
                    "action": "overruled", "identity": "single-fault",
                    "was": cell["value"], "now": fixed,
                    "consensus_share": cell["consensus_share"],
                    "n_witnesses": cell["n_witnesses"],
                })
                cell["value"] = cell["min"] = cell["max"] = fixed
                cell["grade"] = "ladder_repaired"
                cell["lanes"] = f"{cell['lanes']}+ladder"
                values[period] = fixed
                broken = _violations(values)
            else:
                repairs.append({
                    "metric": name, "fiscal_year": fy, "fiscal_period": "",
                    "action": "unreconciled", "identity": f"{len(candidates)} candidates",
                    "was": None, "now": None, "consensus_share": 0.0, "n_witnesses": 0,
                })

        if broken:
            continue                               # never fill into a group that still contradicts
        for parts, whole in _LADDER:
            members = list(parts) + [whole]
            missing = [p for p in members if p not in values]
            if len(missing) != 1:
                continue
            gap = missing[0]
            value = _solve_for(gap, values)[0]
            donors = [cells[p] for p in members if p != gap]
            filled = {
                "ticker": metric.ticker, "metric": name,
                "workbook_label": metric.workbook_label, "is_target": metric.is_target,
                "unit": metric.unit, "basis": metric.basis,
                "fiscal_year": fy, "fiscal_period": gap, "label": f"FY{fy}{gap}",
                "value": value,
                "n_witnesses": min(d["n_witnesses"] for d in donors),
                "n_lanes": min(d["n_lanes"] for d in donors),
                "lanes": "ladder",
                "consensus_share": min(d["consensus_share"] for d in donors),
                "n_distinct": 1, "spread": 0.0, "min": value, "max": value,
                "grade": "ladder_derived",
                "verified": all(d["verified"] for d in donors),
                "latest_published_at": max(d["latest_published_at"] for d in donors),
            }
            cells[gap] = filled
            values[gap] = value
            rows.append(filled)
            repairs.append({
                "metric": name, "fiscal_year": fy, "fiscal_period": gap,
                "action": "derived", "identity": f"{'+'.join(parts)}={whole}",
                "was": None, "now": value,
                "consensus_share": filled["consensus_share"],
                "n_witnesses": filled["n_witnesses"],
            })
    return rows, repairs


#: A period and the periods it contains. `FY` contains everything; `M9` contains `H1` and Q1-Q3.
_CONTAINS: dict[str, tuple[str, ...]] = {
    "FY": ("M9", "H1", "Q1", "Q2", "Q3", "Q4"),
    "M9": ("H1", "Q1", "Q2", "Q3"),
    "H1": ("Q1", "Q2"),
    "H2": ("Q3", "Q4"),
}


def _repair_containment(rows: list[dict]) -> list[dict]:
    """A period cannot be smaller than a period inside it. Reinstate a candidate that obeys.

    ⛔ **This is the invariant that caught Hays, and the ladder could not.** Hays reports a first
    half and a full year but never a second half, so `H1 + H2 = FY` is never observable and the
    reconciliation had nothing to test. Meanwhile the modal vote returned a **FY2022 net fees of
    £534.2 m against a first half of £651.9 m** — a year smaller than its own half, which is not a
    close call, it is impossible.

    The damage ran straight into a submitted number. Hays' net fees are forecast from a stated
    growth chain whose base is the prior year, and three of these broken years sat in that history;
    the corrupted base made the stated-growth anchor look 25 % unreliable, so the model discarded
    it in favour of a momentum carry and forecast **£849 m** where Hays' own published quarterly
    growth implies **£904 m**.

    ⚡ **The right answer was already in the candidate set** — £1,189.4 m, the true figure, was
    present and outvoted 1-to-3 by a regional sub-total. So the repair does not invent anything: it
    takes the smallest banked candidate that satisfies containment, and if none does, it leaves the
    cell alone and says so.
    """
    by_key = {(r["metric"], r["fiscal_year"], r["fiscal_period"]): r for r in rows}
    repairs = []
    for row in rows:
        inner_names = _CONTAINS.get(row["fiscal_period"])
        metric = M.REGISTRY.get(row["metric"])
        # ⚠️ **Only a flow is contained by its parts.** A diluted share count is a period *average*,
        # so a year below its own busiest quarter is entirely normal — including `shares_m` here
        # raised 35 false violations across ADI, Deere and HD before this line existed.
        if not inner_names or metric is None or metric.unit != "currency_m":
            continue
        inner = [by_key[(row["metric"], row["fiscal_year"], p)] for p in inner_names
                 if (row["metric"], row["fiscal_year"], p) in by_key]
        # only meaningful where every part is positive; a loss-making quarter breaks the ordering
        parts = [c["value"] for c in inner if c["value"] is not None and c["value"] > 0]
        if not parts or row["value"] is None or row["value"] <= 0:
            continue
        biggest = max(parts)
        if row["value"] >= biggest:
            continue
        # ⚡ **Among the candidates that obey, take the best-supported — not the smallest.** The
        # failure mode here is a regional sub-total beating the group figure, so the winner is
        # always too *low*; picking the smallest legal candidate landed Hays' FY2022 on 651.9, which
        # is precisely its own first half. Candidates are pre-sorted by witness count, then by size.
        fixed = next((v for v, _n in row.get("_candidates") or [] if v >= biggest), None)
        repairs.append({
            "metric": row["metric"], "label": row["label"], "was": row["value"],
            "now": fixed, "min_allowed": biggest,
            "contained_by": ",".join(c["fiscal_period"] for c in inner if c["value"] == biggest),
            "action": "reinstated" if fixed is not None else "no_candidate_obeys",
            "consensus_share": row["consensus_share"], "n_witnesses": row["n_witnesses"],
        })
        if fixed is not None:
            row["value"] = row["min"] = row["max"] = fixed
            row["grade"] = "containment_repaired"
            row["lanes"] = f"{row['lanes']}+containment"
    return repairs


def main() -> int:
    rows, stats = build()
    containment = _repair_containment(rows)
    store.write(config.FEATURE / "containment_repairs.parquet", containment)
    rows, repairs = _reconcile_ladder(rows)
    for row in rows:
        row.pop("_candidates", None)
    store.write(config.FEATURE / "ladder_repairs.parquet", repairs)
    store.write(config.FEATURE / "metric_panel.parquet", rows)
    wide = [r for r in rows if r["spread"] > 0.01]
    print(f"feature/metric_panel.parquet {len(rows):,} (metric, period) keys  "
          f"{sum(r['n_witnesses'] for r in rows):,} witnesses  "
          f"{len(wide)} keys with >1% lane disagreement")
    print(f"\n{'target':<44}{'periods':>8}{'wit':>6}{'lanes':>7}{'latest':>12}  spread")
    print("-" * 96)
    for metric in M.TARGETS:
        mine = [r for r in rows if r["metric"] == metric.name]
        if not mine:
            print(f"{metric.ticker + ' · ' + metric.workbook_label:<44}{'—':>8}  NO DATA")
            continue
        latest = max(mine, key=lambda r: (r["fiscal_year"], r["fiscal_period"]))
        worst = max(r["spread"] for r in mine)
        print(f"{metric.ticker + ' · ' + metric.workbook_label:<44}{len(mine):>8}"
              f"{sum(r['n_witnesses'] for r in mine):>6}{len({l for r in mine for l in r['lanes'].split(',')}):>7}"
              f"{latest['label']:>12}  {worst:.1%}")
    if stats["refusals"]:
        print("\n  refusals (why a fact did not enter the panel):")
        for reason, n in stats["refusals"].most_common(8):
            print(f"    {n:>7,}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
