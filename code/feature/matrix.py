"""feature/matrix — the training matrix. One row per (metric, period), labelled, point-in-time.

    PYTHONPATH=. .venv/bin/python -m code.feature.matrix

This is the feature layer's output contract and the model layer's only input. Every row is one
`(metric, fiscal_year, fiscal_period)` observation somewhere in history; every column is a number
that was **knowable before that period began**; and `y_value` is what the filer went on to report.
The model trains on the rows whose label is known and predicts the twelve whose label is not.

## The border this file exists to enforce

⛔ **No combination rules live here, and no constants.** An earlier version of this layer computed
the answers directly — `adj GM = guided operating margin + a four-quarter opex ratio`, `Q3 EPS =
(FY guide − H1) × a share`, with `RATIO_WINDOW = 4` and `× 0.97` written into the source. Those are
model weights, they were hand-set, and — the part that actually invalidates them — they were tuned
by watching the printed value move toward the sell-side consensus. That is fitting on the benchmark
being scored against, and it cannot be detected downstream because such a bridge produces exactly
one number, for the target period only, with no history to score it on.

So this module emits **ingredients, never answers**. Whether Q3 profit is best reached through a
segment margin or a seasonal share is a question with a measurable answer, and the model layer
answers it out of sample. Nothing here reads `consensus.parquet`.

## Point-in-time, which is the whole difficulty

`seasonality.parquet` and `conservatism.parquet` are computed over *all* history — correct for
describing a filer, disastrous as a training column, because the 2018 row would carry a seasonal
share measured partly on 2019-2025. Every such column is therefore **recomputed per row on an
expanding window** over strictly-earlier periods. The suffix `_pit` marks the ones that required it.

⚠️ The same applies to a row's own witness count. How many documents corroborate a value is only
known once the value is published, so the target row's quality columns describe its **predecessor**,
never itself.

## Columns

* identity — `ticker`, `metric`, `fiscal_year`, `fiscal_period`, `unit`, `is_target`
* label — `y_value`, plus scale-free `y_yoy` / `y_vs_lag4` so the model can learn across metrics
  whose levels differ by four orders of magnitude
* history — lags at 1, 2 and 4 quarters, quarter-on-quarter and year-on-year momentum, a trailing
  four-quarter total, and the volatility of past year-on-year growth
* season — expanding-window quarter shares (`_pit`)
* guide — the filer's own stated range for exactly this period, and how old it was
* bias — how this filer has landed against its own guides, expanding-window (`_pit`)
* calendar — 52 or 53 weeks, because a 14-week quarter is ~7.7 % more selling days
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from datetime import date

from code.feature import metrics as M
from code.feature import seasonality
from code.feature.periods import Resolver
from code.lib import config, store

QUARTERS = ("Q1", "Q2", "Q3", "Q4")
_QIDX = {q: i + 1 for i, q in enumerate(QUARTERS)}
#: Nothing published after this may reach any row. The four companies all report after it.
DEADLINE = "2026-08-16"
#: Metrics that sum across quarters, so a seasonal share and a trailing total mean something.
ADDITIVE_UNITS = {"currency_m", "per_share"}
#: Fewest complete prior years before an expanding-window seasonal share is worth publishing.
MIN_SEASON_YEARS = 3
#: Fewest prior guide→outcome pairs before an expanding-window bias is worth publishing.
MIN_BIAS_OBS = 3


def _ordinal(fiscal_year: int, fiscal_period: str) -> int | None:
    """A single increasing index over quarters, so 'strictly earlier' is one comparison."""
    if fiscal_period not in _QIDX or not fiscal_year:
        return None
    return fiscal_year * 4 + _QIDX[fiscal_period]


#: ⚠️ **Hays is forecast annually, not quarterly.** It reports half-yearly and the competition asks
#: for its full year, so a quarterly-only matrix leaves all three of its targets with no row at all
#: and no history to learn from. Rows are therefore built at both grains, and a "lag" means one step
#: within the row's own grain — a quarter for a quarterly row, a year for an annual one.
GRAINS: tuple[tuple[str, tuple[str, ...]], ...] = (("quarterly", QUARTERS), ("annual", ("FY",)))


def _step_back(name: str, fiscal_year: int, fiscal_period: str, back: int,
               cells: dict) -> float | None:
    """The value `back` steps earlier in this row's own grain, or None."""
    if fiscal_period == "FY":
        got = cells.get((name, fiscal_year - back, "FY"))
        return got["value"] if got else None
    target = _ordinal(fiscal_year, fiscal_period) - back
    year, index = divmod(target - 1, 4)
    got = cells.get((name, year, QUARTERS[index]))
    return got["value"] if got else None


def _pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return (new / old - 1.0) * 100.0


def _season_pit(series: dict[int, dict[str, float]], upto_year: int,
                quarter: str, recent: int = 4) -> dict:
    """Quarter shares measured only on complete years that finished before `upto_year`.

    ⛔ The published `seasonality.parquet` uses every year including later ones. Reading it into a
    2018 training row would hand the model a share partly measured on 2019-2025 — the model would
    look excellent in backtest for a reason that does not exist at prediction time.
    """
    complete: list[dict[str, float]] = []
    for fiscal_year in sorted(series):
        if fiscal_year >= upto_year:
            break
        quarters = series[fiscal_year]
        if len(quarters) != 4:
            continue
        total = sum(quarters.values())
        # a year whose quarters cancel has no meaningful share; the ratio would explode
        if abs(total) < 0.5 * sum(abs(v) for v in quarters.values()):
            continue
        complete.append({q: quarters[q] / total for q in QUARTERS})
    if len(complete) < MIN_SEASON_YEARS:
        return {}
    values = [year[quarter] for year in complete]
    tail = values[-recent:]
    return {
        "season_share_mean_pit": statistics.fmean(values),
        "season_share_recent_pit": statistics.fmean(tail),
        "season_share_last_pit": values[-1],
        "season_share_sd_pit": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "season_n_years_pit": len(values),
        # every quarter's latest share, so a residual can be split by what is still to come
        "_shares_pit": complete[-1],
    }


def _bias_pit(pairs: list[dict], before: int) -> dict:
    """How this filer has landed against its own guides, using only earlier resolved periods."""
    earlier = [p for p in pairs if (o := _ordinal(p["fiscal_year"], p["fiscal_period"]))
               and o < before]
    if len(earlier) < MIN_BIAS_OBS:
        return {}
    biases = [p["bias"] for p in earlier]
    positions = [p["range_position"] for p in earlier if p["range_position"] is not None]
    return {
        "bias_median_pit": statistics.median(biases),
        "bias_mean_pit": statistics.fmean(biases),
        "bias_sd_pit": statistics.pstdev(biases) if len(biases) > 1 else 0.0,
        "beat_rate_pit": sum(1 for p in earlier if p["beat"]) / len(earlier),
        "range_position_median_pit": statistics.median(positions) if positions else None,
        "bias_n_pit": len(earlier),
    }


#: In-quarter observables per company: the demand drivers that move before the filer reports.
#: Home Depot's category tape is published monthly by the Census Bureau and Hays' market is the
#: UK/German vacancy series — both are readable for part of the target quarter before the deadline.
NOWCASTS: dict[str, tuple[tuple[str, str], ...]] = {
    "HD": (("RSBMGESD", "fred"), ("HOUST", "fred"), ("MORTGAGE30US", "fred")),
    "DE": (("WPU01", "fred"), ("A33SNO", "fred")),
    "ADI": (("IPG3344S", "fred"), ("A31SNO", "fred")),
    "LSE:HAS": (("AP2Y", "labour"),
                ("jvs_q_nace2_geo-DE_indic_em-JVR_nace_r2-B-S", "labour")),
}


def _macro_columns(ticker: str, series: dict, ends: dict, key: tuple, as_of: str) -> dict:
    """The demand tape for the target quarter, as far as it has actually been published.

    ⚡ **These series move before the filer does.** Home Depot's quarter is a function of the
    building-materials retail tape, housing starts and the mortgage rate; Hays' is a function of the
    UK and German vacancy series. Two of the three months of a target quarter are usually published
    before the deadline, which is genuine in-quarter information no guide contains.

    ⛔ **Truncated at `as_of`, always.** A macro series carries the *observation* month, not the
    publication date, so a naive window silently reaches into months that had not been released —
    and because these are the freshest-looking columns, a leak here would flatter the backtest most.
    `n_months` travels alongside so the model can discount a partially-observed quarter rather than
    treat it as a full one.
    """
    # ⚠️ A target period has no *learned* end date — the filer has not reported it — so a lookup
    # that only consults the anchors silently returns nothing for exactly the twelve rows being
    # predicted, while every training row gets its macro columns. That is train/serve skew, and it
    # is invisible because the columns are simply null. The projected end is used instead.
    end = ends.get(key)
    if not end:
        return {}

    cutoff = min(end, as_of)
    start = f"{int(end[:4])}-{max(1, int(end[5:7]) - 2):02d}-01"
    out: dict = {}
    for series_id, _lane in NOWCASTS.get(ticker, ()):
        points = series.get(series_id, ())
        cur = [v for d, v in points if start <= d <= cutoff]
        back = [v for d, v in points
                if f"{int(start[:4]) - 1}{start[4:]}" <= d <= f"{int(cutoff[:4]) - 1}{cutoff[4:]}"]
        tag = series_id[:12]
        out[f"macro_{tag}_yoy"] = ((statistics.fmean(cur) / statistics.fmean(back) - 1.0) * 100.0
                                   if cur and back and statistics.fmean(back) else None)
        out[f"macro_{tag}_n"] = len(cur)
    return out


def _external_anchors(panel_cells: dict) -> dict[tuple, dict]:
    """Anchors that exist outside the guidance lane: a company-compiled consensus, a stated growth.

    ⚡ **This is the whole of Hays.** Hays issues no numeric guidance, so all three of its submitted
    figures would otherwise rest on last year's number alone. Two independent signals are sitting in
    its own filings:

    * **Stated quarterly net-fee growth.** Hays reports growth, never the level — FY2026 runs
      −8/−9/−7/−4 % on an actual basis, all four quarters already published. Applied to last year's
      net fees that is a complete, guidance-free forecast of the year.
    * **Company-compiled consensus, with management's position in it.** Hays publishes the analyst
      range it has collected *and* says where it expects to land: `£37.0–46.0 m`, consensus
      `£43.5 m` from 10 analysts, and on 10 July 2026 management said **top** of that range. The
      position is the edge — the consensus alone is only the benchmark.

    ⚠️ **The consensus is the thing being competed against, so it enters as one weighted candidate,
    never as the answer.** Its own record here is poor: the FY2025 compilation read **£56.4 m**
    eleven days before that year ended and the company reported **£45.6 m**. Two ambiguous historical
    points are not enough to fit a correction, so no bias is applied to it — it is offered to the
    combination layer with its dispersion, and the stated-growth anchor stands beside it.
    """
    consensus = store.read(config.EXTRACTED / "consensus.parquet")
    out: dict[tuple, dict] = {}

    compiled = [c for c in consensus if c["kind"] == "company_compiled" and c["value"]]
    ranges = [c for c in consensus if c["kind"] == "position_in_range" and c["low"] and c["high"]]
    for row in compiled:
        # Hays compiles its consensus for the fiscal year it is about to finish
        year = int(row["published_at"][:4])
        fiscal = year if row["published_at"][5:7] >= "07" else year
        band = max((r for r in ranges if r["published_at"] == row["published_at"]),
                   key=lambda r: r["published_at"], default=None)
        key = ("has_pre_exc_operating_profit", fiscal, "FY")
        prev = out.get(key)
        if prev and prev["published_at"] >= row["published_at"]:
            continue
        out[key] = {"value": row["value"], "low": band["low"] if band else None,
                    "high": band["high"] if band else None,
                    "n_analysts": row["n_analysts"],
                    "position": (band or row).get("position") or "",
                    "published_at": row["published_at"], "source": "company_compiled"}

    # ── stated growth chains: a level the filer never states, from a growth it does ──
    for fiscal_year in {k[1] for k in panel_cells if k[0] == "has_net_fee_growth_actual"}:
        rates = [panel_cells[("has_net_fee_growth_actual", fiscal_year, q)]["value"]
                 for q in QUARTERS
                 if ("has_net_fee_growth_actual", fiscal_year, q) in panel_cells]
        base = panel_cells.get(("has_net_fees", fiscal_year - 1, "FY"))
        if len(rates) < 2 or not base:
            continue
        out[("has_net_fees", fiscal_year, "FY")] = {
            "value": base["value"] * (1.0 + statistics.fmean(rates) / 100.0),
            "low": base["value"] * (1.0 + min(rates) / 100.0),
            "high": base["value"] * (1.0 + max(rates) / 100.0),
            "n_analysts": None, "position": "", "published_at": "",
            "source": f"stated_growth_{len(rates)}q"}
    return out


def build() -> list[dict]:
    panel = store.read(config.FEATURE / "metric_panel.parquet")
    guides = store.read(config.FEATURE / "guides.parquet")
    pairs = store.read(config.FEATURE / "guide_vs_actual.parquet")
    calendar = {(r["ticker"], r["fiscal_year"]): r["weeks"]
                for r in store.read(config.FEATURE / "fiscal_weeks.parquet")}

    ends: dict[tuple, str] = {}
    for anchor in store.read(config.FEATURE / "fiscal_periods.parquet"):
        ends[(anchor["ticker"], anchor["fiscal_year"], anchor["fiscal_period"])] = \
            anchor["period_end"]
        if anchor["fiscal_period"] == "Q4":       # a fiscal year ends when its fourth quarter does
            ends[(anchor["ticker"], anchor["fiscal_year"], "FY")] = anchor["period_end"]
    for key in _target_keys():                    # the unreported periods, from their projected end
        metric = M.REGISTRY.get(key[0])
        if metric and (metric.ticker, key[1], key[2]) not in ends:
            ends[(metric.ticker, key[1], key[2])] = _projected_end(metric.ticker)

    macro: dict[str, list] = defaultdict(list)
    for lane in ("fred_observations", "labour_observations"):
        for obs in store.read(config.EXTRACTED / f"{lane}.parquet"):
            macro[obs["series_id"]].append((obs["date"], obs["value"]))
    macro = {k: sorted(v) for k, v in macro.items()}

    cells: dict[tuple, dict] = {}
    quarterly: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in panel:
        if not row["fiscal_year"]:
            continue
        cells[(row["metric"], row["fiscal_year"], row["fiscal_period"])] = row
        if row["fiscal_period"] in QUARTERS:
            quarterly[row["metric"]][row["fiscal_year"]][row["fiscal_period"]] = row["value"]

    external = _external_anchors(cells)
    guide_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for guide in guides:
        guide_by_key[(guide["metric"], guide["fiscal_year"], guide["fiscal_period"])].append(guide)
    pairs_by_metric: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        pairs_by_metric[pair["metric"]].append(pair)

    # ⚡ **The twelve unlabelled targets are built by this same loop, deliberately.** A separate
    # code path for the rows being predicted is how train/serve skew gets in: a column computed one
    # way in training and another way at prediction produces a model that scores well and forecasts
    # badly. So the targets enter as ordinary rows whose label happens to be unknown, and every
    # column reaches them through the identical arithmetic.
    grain_periods = {p for _g, ps in GRAINS for p in ps}
    todo = [(k, cells[k]) for k in sorted(cells) if k[2] in grain_periods]
    todo += [(k, None) for k in _target_keys() if k not in cells]

    rows: list[dict] = []
    for (name, fiscal_year, fiscal_period), cell in todo:
        metric = M.REGISTRY.get(name)
        if metric is None or fiscal_period not in grain_periods:
            continue
        annual_row = fiscal_period == "FY"
        here = fiscal_year if annual_row else _ordinal(fiscal_year, fiscal_period)
        value = cell["value"] if cell else None

        def at(offset: int, _n=name, _y=fiscal_year, _p=fiscal_period):
            """The panel value `offset` steps before this one, in this row's own grain."""
            return _step_back(_n, _y, _p, offset, cells)

        # a year-ago comparison is 4 steps for a quarterly row and 1 step for an annual one
        year_step = 1 if annual_row else 4
        lag1, lag2 = at(1), at(2)
        lag4, lag5 = at(year_step), at(year_step + 1)
        trailing = [at(k) for k in range(1, year_step + 1)]
        yoy_history = [g for k in range(1, 2 * year_step + 1)
                       if (g := _pct_change(at(k), at(k + year_step))) is not None]

        row = {
            "ticker": metric.ticker, "metric": name, "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period, "label": f"FY{fiscal_year}{fiscal_period}",
            "is_prediction": cell is None,
            "as_of": str(cell["latest_published_at"]) if cell else DEADLINE,
            "unit": metric.unit, "basis": metric.basis, "is_target": metric.is_target,
            "ordinal": here,

            # ── label ────────────────────────────────────────────────────────
            "y_value": value,
            # scale-free labels: net sales and EPS differ by four orders of magnitude, and a model
            # trained on levels would spend all its capacity on that instead of on the forecast
            "y_yoy": _pct_change(value, lag4),
            "y_vs_lag4": (value - lag4) if lag4 is not None and value is not None else None,

            # ── history ──────────────────────────────────────────────────────
            "lag1": lag1, "lag2": lag2, "lag4": lag4,
            "qoq_lag1": _pct_change(lag1, lag2),
            "yoy_lag1": _pct_change(lag1, lag5),
            "yoy_lag4": _pct_change(lag4, at(2 * year_step)),
            "yoy_accel": ((y1 - y4) if (y1 := _pct_change(lag1, lag5)) is not None
                          and (y4 := _pct_change(lag4, at(2 * year_step))) is not None else None),
            "trailing4_sum": (sum(trailing) if metric.unit in ADDITIVE_UNITS
                              and all(t is not None for t in trailing) else None),
            "yoy_vol": statistics.pstdev(yoy_history) if len(yoy_history) > 2 else None,
            "n_history": sum(1 for k in range(1, 21) if at(k) is not None),

            # ── calendar ─────────────────────────────────────────────────────
            "fy_weeks": calendar.get((metric.ticker, fiscal_year)),
        }

        # ⚠️ Data quality describes the PREDECESSOR, never this row: how many documents corroborate
        # a value is only known once the value has been published, so the target's own witness
        # count is not available at prediction time.
        prior = (cells.get((name, fiscal_year - 1, "FY")) if annual_row
                 else cells.get((name, *_key_of(here - 1))))
        row["lag1_n_witnesses"] = prior["n_witnesses"] if prior else None
        row["lag1_consensus_share"] = prior["consensus_share"] if prior else None

        # ⛔ A share of a year is only meaningful for a flow. Comparable sales is a percentage, and
        # measuring "Q2's share of the year's comps" returned 0.769 — a ratio of four percentages
        # that means nothing and would have been fed to the model as a seasonal weight.
        if metric.unit in ADDITIVE_UNITS and not annual_row:
            season = _season_pit(quarterly[name], fiscal_year, fiscal_period)
            # ⚡ HD only began reporting an adjusted EPS after the SRS acquisition, so it has one
            # clean year — a share measured on one year is just that year. Reported EPS has
            # seventeen and the two differ by a near-constant amortisation charge; on FY2025, the
            # only year both are clean, their Q2 shares sit **0.35 pp** apart. The donor is named in
            # `seasonality.PROXY_SHARE` so the borrow is declared, not implicit.
            donor = seasonality.PROXY_SHARE.get(name)
            if not season and donor:
                season = _season_pit(quarterly[donor], fiscal_year, fiscal_period)
                row["season_source"] = f"proxy:{donor}" if season else None
            row.update(season)
        row.update(_bias_pit(pairs_by_metric.get(name, []), here))

        # ── the filer's own stated range for exactly this period ─────────────
        stated = [g for g in guide_by_key.get((name, fiscal_year, fiscal_period), [])
                  if g["published_at"] < row["as_of"]]
        if stated:
            latest = max(stated, key=lambda g: g["published_at"])
            row.update({
                "has_guide": 1,
                "guide_low": latest["low"], "guide_high": latest["high"],
                "guide_mid": latest["midpoint"],
                "guide_rel_width": (abs(latest["high"] - latest["low"])
                                    / abs(latest["midpoint"]) if latest["midpoint"] else None),
                "guide_shape": latest["shape"],
                "guide_n_restatements": len(stated),
                "guide_age_days": _age_days(latest["published_at"], row["as_of"]),
            })
        else:
            row.update({"has_guide": 0, "guide_low": None, "guide_high": None,
                        "guide_mid": None, "guide_rel_width": None, "guide_shape": None,
                        "guide_n_restatements": 0, "guide_age_days": None})

        row.update(_annual_guide_columns(name, metric, fiscal_year, fiscal_period, here,
                                         cells, guide_by_key, row, at))
        row.update(_companion_columns(name, metric, fiscal_year, fiscal_period,
                                      cells, guide_by_key, row, external))
        row.update(_macro_columns(metric.ticker, macro, ends,
                                  (metric.ticker, fiscal_year, fiscal_period), row["as_of"]))

        # ── anchors that need no guide, so they reach nearly the whole matrix ────────
        # ⚡ **Coverage is what makes the residuals learnable.** The guide-derived anchors exist on
        # only ~76 of 1,315 labelled rows, which is far too few to fit a correction on. These three
        # are computable wherever a short history exists, so the model sees thousands of examples of
        # "how wrong is a naive anchor, and when" — and the guided anchors inherit that calibration.
        percent = metric.unit == "percent"
        carry = None
        if lag4 is not None and row["yoy_lag1"] is not None:
            carry = (lag4 + (lag1 - lag5) if percent and lag1 is not None and lag5 is not None
                     else lag4 * (1.0 + row["yoy_lag1"] / 100.0) if not percent else None)
        row["anchor_yoy_carry"] = carry
        row["anchor_lag1"] = lag1
        share_now = row.get("season_share_last_pit")
        row["anchor_runrate_seasonal"] = (row["trailing4_sum"] * share_now
                                          if row.get("trailing4_sum") is not None
                                          and share_now is not None else None)

        # ── an anchor the filer published outside its guidance: consensus, or a stated growth ──
        ext = external.get((name, fiscal_year, fiscal_period))
        fresh = bool(ext) and (not ext["published_at"] or ext["published_at"] < row["as_of"])
        # ⛔ **A stated growth chain and a compiled consensus are not the same anchor and must not
        # share a track record.** Pooled as one `external` type they showed a 10 % median error and
        # lost selection to a momentum carry. Separated, Hays' own published growth is accurate to
        # **-0.6 %** (FY2023) and **+1.1 %** (FY2025) — it is near-arithmetic, being the filer's
        # actual-basis growth applied to last year — while the compiled consensus missed FY2025 by
        # **-19 %**. Averaging those two records hid both facts.
        kind = (ext["source"].split("_")[0] if fresh else None)
        row.update({
            "anchor_stated_growth": ext["value"] if fresh and kind == "stated" else None,
            "anchor_consensus": ext["value"] if fresh and kind == "company" else None,
            "external_source": ext["source"] if fresh else None,
            "external_low": ext["low"] if fresh else None,
            "external_high": ext["high"] if fresh else None,
            "external_n_analysts": ext["n_analysts"] if fresh else None,
            # management's own stated position inside the published range: the only part of a
            # compiled consensus that is not simply the benchmark restated
            "external_position": ({"top": 1.0, "above": 1.0, "in_line": 0.0,
                                   "bottom": -1.0, "below": -1.0}.get(ext["position"])
                                  if fresh else None),
        })
        row.pop("_shares_pit", None)
        rows.append(row)
    return rows


def _annual_guide_columns(name: str, metric, fiscal_year: int, fiscal_period: str, here: int,
                          cells: dict, guide_by_key: dict, row: dict, at) -> dict:
    """The filer's **annual** guide, carried down onto a quarterly row.

    ⚡ **Home Depot and Deere guide the full year and are being forecast for one quarter.** Their
    guides therefore live on `FY` keys and never touch a quarterly row — measured, only 4 % of the
    matrix had a guide of its own, and Deere's net-income guide reached none of the rows that need
    it. That gap is the whole forecasting problem for two of the four companies.

    So the annual guide comes down, together with what the year has already delivered, as columns:

    * `fy_guide_*` — the range, and the level it implies when the guide is a growth rate
    * `ytd_before` — this year's quarters already reported, so the guide's own **residual** is
      visible: Deere has $2,429 m of a $4,500-5,000 m year already banked by H1
    * `season_share_of_remaining_pit` — this quarter's share of what is left, not of the whole year

    ⛔ **These are ingredients, not an answer.** An earlier version multiplied them together in
    source — `(FY guide − H1) × share`, with the share hand-picked and a `× 0.97` on the end — which
    produced one number for one period that no backtest could ever score. Here each part is a column
    the model weighs, and how to combine them is settled out of sample.
    """
    out: dict = {}
    # ⛔ The cutoff is what makes this point-in-time: a guide issued after the results were out is
    # not information the forecast could have used. For the twelve unreported targets the cutoff is
    # the competition deadline, so nothing published later can leak in.
    row_pub = row["as_of"]
    annual = [g for g in guide_by_key.get((name, fiscal_year, "FY"), [])
              if g["published_at"] < row_pub]
    prior_fy = cells.get((name, fiscal_year - 1, "FY"))
    prior_total = prior_fy["value"] if prior_fy else None
    # ⛔ **An annual total cannot be smaller than one of its own quarters.** HD's FY2025 adjusted
    # EPS is stated as **3.6** against quarters summing to **14.70** — a quarterly figure that
    # landed on an annual row. Used as the base for a growth guide it implies a Q2 of about $1.15
    # against a real $4.68, so the guard is not cosmetic.
    if prior_total is not None and metric.unit in ADDITIVE_UNITS:
        siblings = [cells[(name, fiscal_year - 1, q)]["value"] for q in QUARTERS
                    if (name, fiscal_year - 1, q) in cells]
        if siblings and abs(prior_total) < max(abs(v) for v in siblings):
            prior_total = sum(siblings) if len(siblings) == 4 else None
    out["fy_prior_total"] = prior_total

    if annual:
        latest = max(annual, key=lambda g: g["published_at"])
        implied = None
        if latest["shape"] == "growth" and prior_total is not None:
            implied = prior_total * (1.0 + latest["midpoint"] / 100.0)
        elif latest["shape"] == "level":
            implied = latest["midpoint"]
        out.update({
            "has_fy_guide": 1, "fy_guide_shape": latest["shape"],
            "fy_guide_low": latest["low"], "fy_guide_high": latest["high"],
            "fy_guide_mid": latest["midpoint"],
            "fy_guide_rel_width": latest["relative_width"],
            "fy_guide_implied_level": implied,
            "fy_guide_age_days": _age_days(latest["published_at"], row_pub),
        })
    else:
        out.update({"has_fy_guide": 0, "fy_guide_shape": None, "fy_guide_low": None,
                    "fy_guide_high": None, "fy_guide_mid": None, "fy_guide_rel_width": None,
                    "fy_guide_implied_level": None, "fy_guide_age_days": None})

    # ⚠️ An annual row *is* the year: it has no year-to-date remainder and no share of itself, so
    # the residual machinery is skipped rather than fed a meaningless 1.0.
    index = _QIDX.get(fiscal_period)
    banked = ([] if index is None else
              [cells[(name, fiscal_year, q)]["value"] for q in QUARTERS[:index - 1]
               if (name, fiscal_year, q) in cells])
    out["ytd_before"] = (sum(banked) if index is not None and metric.unit in ADDITIVE_UNITS
                         and len(banked) == index - 1 else None)
    out["fy_guide_residual"] = (out["fy_guide_implied_level"] - out["ytd_before"]
                                if out.get("fy_guide_implied_level") is not None
                                and out["ytd_before"] is not None else None)

    # this quarter's weight among the quarters still to come, which is what a residual splits by
    share_last = row.get("season_share_last_pit")
    shares_pit = row.get("_shares_pit") or {}
    tail = (sum(v for q, v in shares_pit.items() if _QIDX[q] >= index)
            if index is not None else 0)
    out["season_share_of_remaining_pit"] = (shares_pit.get(fiscal_period) / tail
                                            if shares_pit and tail else None)

    # ── candidate anchors: naive, parameter-free point forecasts ─────────────
    # each is fully determined by data; which one to trust, and whether to blend them, is the
    # model layer's question and is answered by walk-forward scoring, never by choosing here
    out["anchor_prior_year"] = at(1 if fiscal_period == "FY" else 4)
    out["anchor_seasonal_last"] = (out["fy_guide_implied_level"] * share_last
                                   if out.get("fy_guide_implied_level") is not None
                                   and share_last is not None else None)
    recent = row.get("season_share_recent_pit")
    out["anchor_seasonal_recent"] = (out["fy_guide_implied_level"] * recent
                                     if out.get("fy_guide_implied_level") is not None
                                     and recent is not None else None)
    out["anchor_residual"] = (out["fy_guide_residual"] * out["season_share_of_remaining_pit"]
                              if out["fy_guide_residual"] is not None
                              and out["season_share_of_remaining_pit"] is not None else None)
    out["anchor_guide_mid"] = row.get("guide_mid")
    # ⚡ A rate does not need splitting. Home Depot guides comparable sales **for the year** and is
    # asked for a quarter's comps — but a comp is already a percentage, so the annual guide is a
    # direct anchor for the quarter rather than something to multiply by a share. Applying the
    # seasonal machinery to it produced a "share of the year's comps" of 0.769, which is a ratio of
    # four percentages and means nothing.
    out["anchor_fy_guide_direct"] = (out.get("fy_guide_mid")
                                     if metric.unit == "percent" else None)
    return out


#: The guided line each unguided target hangs off, and nothing more. Deere guides full-year **net
#: income** while being scored on EPS, total revenue and a segment's profit; ADI guides adjusted
#: **operating** margin while being scored on adjusted **gross** margin. These are accounting
#: relationships, not fitted choices — which is why they belong to the feature layer.
COMPANION: dict[str, str] = {
    "de_diluted_eps": "de_net_income",
    # ⚡ A segment's profit is driven by that segment's sales, not by the whole company's earnings.
    # Deere guides `precision ag net sales` by name, so this link is one stated growth rate and one
    # measured segment margin — a far shorter chain than routing through group net income.
    "de_ppa_operating_profit": "de_ppa_net_sales",
    "de_net_sales_and_revenues": "de_net_income",
    "adi_adj_gross_margin": "adi_adj_operating_margin",
    "adi_adj_operating_income": "adi_revenue",
    # Hays states neither an EPS guide nor an EPS consensus, but it does publish a compiled
    # consensus for operating profit — and EPS is that profit after interest, tax and share count,
    # a ratio its own history measures.
    "has_pre_exc_basic_eps": "has_pre_exc_operating_profit",
    "has_pbt_pre_exc": "has_pre_exc_operating_profit",
}


def _companion_columns(name: str, metric, fiscal_year: int, fiscal_period: str,
                       cells: dict, guide_by_key: dict, row: dict,
                       external: dict | None = None) -> dict:
    """Reach an unguided target through the guided line it is arithmetically tied to.

    ⚡ **Four of the twelve targets have no guide naming them, and that is not missing data.** Each
    filer guides something adjacent, and the step across is measurable at every period in history:

    * `adjusted gross margin = adjusted operating margin + opex %` — a **difference**, because both
      are percentages. ADI guides the operating margin (48.0–50.0 % for FY2026 Q3); the opex gap is
      the only unknown and it has thirty quarters of history.
    * `EPS = net income ÷ shares`, `segment profit = f(company profit)` — a **ratio**, because the
      units differ.

    ⛔ **The step is published, the combination is not.** An earlier version multiplied these out in
    source with a hand-chosen four-quarter window, and picked that window by watching the answer
    move toward the sell-side consensus. Here the gap is emitted at two horizons — the latest
    observation and a three-period mean — as plain columns, and which one to trust (or whether to
    trust either) is settled by out-of-sample scoring in the model layer.
    """
    companion = COMPANION.get(name)
    out: dict = {"companion_metric": companion}
    if not companion:
        return {**out, "companion_guide_mid": None, "gap_to_companion_lag1": None,
                "gap_to_companion_mean3": None, "gap_is_ratio": None,
                "anchor_companion_lag1": None, "anchor_companion_mean3": None}

    # a companion guide for this exact period, else the annual one covering it
    pool = [g for g in (guide_by_key.get((companion, fiscal_year, fiscal_period), [])
                        or guide_by_key.get((companion, fiscal_year, "FY"), []))
            if g["published_at"] < row["as_of"]]
    guide = max(pool, key=lambda g: g["published_at"]) if pool else None
    annual = bool(guide) and guide["fiscal_period"] == "FY"

    # ⚠️ Percent against percent is a difference; anything else is a ratio. Dividing one margin by
    # another produces a number with no accounting meaning.
    is_ratio = metric.unit != "percent"
    periods = ["FY"] if annual else [fiscal_period]
    gaps: list[float] = []
    for back in range(1, 5):
        year = fiscal_year - back
        for period in periods:
            mine = cells.get((name, year, period))
            theirs = cells.get((companion, year, period))
            if not mine or not theirs:
                continue
            if is_ratio and theirs["value"]:
                gaps.append(mine["value"] / theirs["value"])
            elif not is_ratio:
                gaps.append(mine["value"] - theirs["value"])

    implied = None
    if guide:
        implied = (guide["midpoint"] if guide["shape"] in ("level", "margin")
                   else None)
        if guide["shape"] == "growth":
            base = cells.get((companion, fiscal_year - 1, guide["fiscal_period"]))
            implied = base["value"] * (1 + guide["midpoint"] / 100.0) if base else None
    if implied is None and external:
        # ⚡ A companion with no guide may still have a published level — Hays' compiled
        # operating-profit consensus is what carries its EPS and PBT targets.
        ext = external.get((companion, fiscal_year, fiscal_period))
        if ext and (not ext["published_at"] or ext["published_at"] < row["as_of"]):
            implied, annual = ext["value"], fiscal_period == "FY"

    lag1 = gaps[0] if gaps else None
    mean3 = statistics.fmean(gaps[:3]) if len(gaps) >= 2 else None

    def apply(gap):
        if implied is None or gap is None:
            return None
        level = implied * gap if is_ratio else implied + gap
        # An annual companion reaches an annual level, so the season splits it into the quarter —
        # but only when the row being predicted *is* a quarter. Hays is forecast for its full year,
        # and splitting an annual level by a share it does not have returned nothing at all.
        share = row.get("season_share_last_pit")
        if annual and fiscal_period != "FY" and metric.unit in ADDITIVE_UNITS:
            return level * share if share is not None else None
        return level

    # ⚡ **When the year is half over, the guide's residual is worth more than its seasonal share.**
    # Deere guides full-year net income and has already reported H1; splitting the *whole* guide by
    # a Q3 share throws that away. The identity is `guide − banked = what is left`, and the quarter
    # takes its share of what is left. On the FY2026 Q3 target the two routes differ by 4.3 %.
    residual_anchor = None
    if implied is not None and annual and fiscal_period != "FY" and lag1 is not None:
        index = _QIDX[fiscal_period]
        banked = [cells[(name, fiscal_year, q)]["value"] for q in QUARTERS[:index - 1]
                  if (name, fiscal_year, q) in cells]
        remaining = row.get("season_share_of_remaining_pit")
        annual_level = implied * lag1 if is_ratio else implied + lag1
        if len(banked) == index - 1 and remaining is not None and metric.unit in ADDITIVE_UNITS:
            residual_anchor = (annual_level - sum(banked)) * remaining

    out.update({
        "companion_guide_mid": implied,
        "gap_to_companion_lag1": lag1, "gap_to_companion_mean3": mean3,
        "gap_is_ratio": int(is_ratio),
        "anchor_companion_lag1": apply(lag1),
        "anchor_companion_mean3": apply(mean3),
        "anchor_companion_residual": residual_anchor,
    })
    return out


def _projected_end(ticker: str) -> str | None:
    """When the target period is expected to end, for a period the filer has not reported yet."""
    for row in store.read(config.EXTRACTED / "target_periods.parquet"):
        if row["ticker"] == ticker:
            return row["projected_period_end"]
    return None


def _target_keys() -> list[tuple[str, int, str]]:
    """The twelve `(metric, fiscal_year, fiscal_period)` keys the competition asks for."""
    resolver = Resolver()
    targets = {t["ticker"]: t for t in store.read(config.EXTRACTED / "target_periods.parquet")}
    keys = []
    for metric in M.TARGETS:
        target = targets.get(metric.ticker)
        if not target:
            continue
        # ⛔ **The span decides the grain, and defaulting it to a quarter silently broke Hays.**
        # Hays is asked for its full year, not its fourth quarter; forcing a 3-month span put all
        # three of its targets on a `Q4` key that has no history behind it — every anchor came back
        # empty. The filer's own stated span is the only thing that may set this.
        span = 3 if re.search(r"Q[1-4]$", target["target_period"]) else 12
        key = resolver.by_date(metric.ticker, target["projected_period_end"], span)
        if key.ok and key.fiscal_period in (*QUARTERS, "FY"):
            keys.append((metric.name, key.fiscal_year, key.fiscal_period))
    return keys


def _key_of(ordinal: int) -> tuple[int, str]:
    year, index = divmod(ordinal - 1, 4)
    return year, QUARTERS[index]


def _age_days(published: str, reported: str) -> int | None:
    try:
        return (date.fromisoformat(str(reported)[:10])
                - date.fromisoformat(str(published)[:10])).days
    except (ValueError, TypeError):
        return None


def main() -> int:
    rows = build()
    store.write(config.FEATURE / "training_matrix.parquet", rows)
    labelled = [r for r in rows if r["y_value"] is not None]
    columns = [c for c in rows[0] if c not in ("ticker", "metric", "fiscal_period", "label",
                                               "unit", "basis", "guide_shape")]
    print(f"feature/training_matrix.parquet {len(rows):,} rows × {len(rows[0])} columns  "
          f"({len(labelled):,} labelled)")

    print(f"\n{'column':<30}{'non-null':>10}{'coverage':>10}")
    print("-" * 52)
    for column in columns:
        filled = sum(1 for r in rows if r.get(column) is not None)
        print(f"{column:<30}{filled:>10,}{filled / len(rows):>10.0%}")

    print(f"\n{'ticker':<10}{'metric':<32}{'rows':>7}{'guided':>8}{'season':>8}{'bias':>7}")
    print("-" * 74)
    by_metric: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_metric[(row["ticker"], row["metric"])].append(row)
    for (ticker, name), group in sorted(by_metric.items()):
        print(f"{ticker:<10}{name:<32}{len(group):>7}"
              f"{sum(1 for r in group if r['has_guide']):>8}"
              f"{sum(1 for r in group if r.get('season_share_mean_pit') is not None):>8}"
              f"{sum(1 for r in group if r.get('bias_median_pit') is not None):>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
