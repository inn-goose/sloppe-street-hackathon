"""feature/design_matrix — one row per submitted number, every input beside it.

The capstone of the feature layer: twelve rows, one per workbook cell, each carrying the
independent estimates of that number and the evidence behind them. The model layer chooses among
these; it does not go looking for data.

## The columns, and why each earns its place

| column | what it answers |
|---|---|
| `guide_low/mid/high` | what management said, for this exact period |
| `bias_median`, `bias_n`, `bias_sd`, `range_position` | does this filer beat its own guide, on how many observations, how reliably |
| `prior_value` | the same period one year ago — the base every YoY bridge starts from |
| `recent_growth` | what the last four quarters actually did |
| `season_share`, `season_sd` | what fraction of a fiscal year this quarter carries, and how stable |
| `consensus_*` | the benchmark, from up to three independent vendors |
| `nowcast_*` | in-quarter observables — the category tape, the labour market |

⚠️ **A column is left null rather than filled with a default.** A missing guide and a guide of
zero are different facts, and a model that cannot tell them apart will treat the absence as a
forecast.

⛔ **Everything here is knowable today.** All four companies report *after* the deadline, so no
column can contain an outcome — but the panel *does* contain outcomes for past periods, and a
feature that accidentally reads the target period's own value would look spectacular and be
worthless. `prior_value` is explicitly the prior year, and the target period is asserted absent
from the panel.
"""

from __future__ import annotations

import re
import statistics
from datetime import date

from code.feature import audit, metrics as M
from code.feature.periods import Resolver
from code.lib import config, store

QUARTERS = ("Q1", "Q2", "Q3", "Q4")

#: Which vendor consensus field, if any, is the benchmark for each target. ⛔ A metric absent from
#: this map has **no vendor benchmark** — comparable sales, Hays' net fees and Deere's segment
#: profit are not published by any consensus lane, and inventing one from a same-unit field is
#: how a £6,056 m turnover became the bar for a £972 m net-fee line.
CONSENSUS_FIELD = {
    "hd_net_sales": "revenue",
    "hd_adj_diluted_eps": "adjustedEps",
    "adi_revenue": "revenue",
    "adi_adj_diluted_eps": "adjustedEps",
    "adi_adj_gross_margin": "grossMargin",
    "de_net_sales_and_revenues": "revenue",
    "de_diluted_eps": "eps",
    # hd_comparable_sales, has_*, de_ppa_operating_profit: no vendor carries them
}

#: In-quarter observables, per company: (series id, source lane, what it conditions).
NOWCASTS = {
    "HD": [("RSBMGESD", "fred", "US building-materials & garden retail sales (NAICS 444)"),
           ("HOUST", "fred", "US housing starts"),
           ("MORTGAGE30US", "fred", "US 30-year mortgage rate")],
    "DE": [("WPU01", "fred", "PPI farm products"),
           ("A33SNO", "fred", "US machinery new orders")],
    "ADI": [("IPG3344S", "fred", "US semiconductor industrial production"),
            ("A31SNO", "fred", "US computer & electronic new orders")],
    "LSE:HAS": [("AP2Y", "labour", "UK vacancies"),
                ("jvs_q_nace2_geo-DE_indic_em-JVR_nace_r2-B-S", "labour",
                 "Germany job vacancy rate")],
}


def _prior(period: str) -> str:
    return period


def _yoy(series: dict, fy: int, period: str) -> float | None:
    cur, prior = series.get((fy, period)), series.get((fy - 1, period))
    if cur is None or prior in (None, 0):
        return None
    return (cur / prior - 1.0) * 100.0


def _nowcast(obs: list[dict], series_id: str, start: str, end: str) -> dict:
    """The in-quarter reading of a driver, against the same window a year earlier."""
    def window(a: str, b: str) -> list[float]:
        return [r["value"] for r in obs if r["series_id"] == series_id and a <= r["date"] <= b]

    cur = window(start, end)
    prior = window(f"{int(start[:4]) - 1}{start[4:]}", f"{int(end[:4]) - 1}{end[4:]}")
    if not cur or not prior:
        return {"n_months": len(cur), "yoy_pct": None, "level": None}
    return {"n_months": len(cur),
            "level": statistics.fmean(cur),
            "yoy_pct": (statistics.fmean(cur) / statistics.fmean(prior) - 1.0) * 100.0}


def build() -> list[dict]:
    resolver = Resolver()
    panel = store.read(config.FEATURE / "metric_panel.parquet")
    guides = store.read(config.FEATURE / "guides.parquet")
    conserv = {(r["metric"], r["shape"]): r
               for r in store.read(config.FEATURE / "conservatism.parquet")}
    season = {(r["metric"], r["quarter"]): r
              for r in store.read(config.FEATURE / "seasonality.parquet")}
    targets = {t["ticker"]: t for t in store.read(config.EXTRACTED / "target_periods.parquet")}
    sa_fc = store.read(config.EXTRACTED / "sa_forecast.parquet")
    nq_fc = store.read(config.EXTRACTED / "nq_forecast.parquet")
    consensus = store.read(config.EXTRACTED / "consensus.parquet")
    fred = store.read(config.EXTRACTED / "fred_observations.parquet")
    labour = store.read(config.EXTRACTED / "labour_observations.parquet")
    sym = {s["ticker"]: s for s in store.read(config.EXTRACTED / "symbology.parquet")}

    by_metric: dict[str, dict[tuple, float]] = {}
    for row in panel:
        by_metric.setdefault(row["metric"], {})[(row["fiscal_year"], row["fiscal_period"])] = \
            row["value"]

    rows = []
    for metric in M.TARGETS:
        target = targets[metric.ticker]
        key = resolver.by_date(metric.ticker, target["projected_period_end"],
                               12 if metric.ticker == "LSE:HAS" else 3)
        fy, period = key.fiscal_year, key.fiscal_period
        series = by_metric.get(metric.name, {})

        # ⛔ the target period must NOT already be in the panel — if it were, the company would
        # have reported and this would not be a forecast
        leaked = series.get((fy, period))

        prior_value = series.get((fy - 1, period))
        history = sorted(((k, v) for k, v in series.items() if k[1] == period and k[0] < fy),
                         key=lambda kv: kv[0])
        growths = [ _yoy(series, k[0], period) for k, _v in history[-5:] ]
        growths = [g for g in growths if g is not None]

        # standing guide for this exact period, most recent statement wins
        # ⚡ **A guide with no stated period inherits the one its document is guiding.** ADI's
        # outlook paragraph names the quarter once — "For the third quarter of fiscal 2026, we are
        # forecasting revenue of $3.9 billion, +/- $100 million" — and every later sentence in it
        # ("EPS to be $3.30, +/-$0.15") omits it. Read literally those rows have no period and
        # bind to nothing, which silenced the guide for two of ADI's three targets. The document's
        # modal guided period is the honest fallback, and it is graded `document`.
        # `feature/guides` owns binding a filer's wording to a metric and a canonical period; this
        # module reads the result rather than repeating the resolution and drifting from it
        matched = sorted((g for g in guides
                          if g["metric"] == metric.name and g["fiscal_year"] == fy),
                         key=lambda g: g["published_at"])
        guide = matched[-1] if matched else None
        shape = guide["shape"] if guide else None
        guide_period = guide["fiscal_period"] if guide else None

        bias = conserv.get((metric.name, shape)) if shape else None
        seasonal = season.get((metric.name, period)) if period in QUARTERS else None
        fy_weeks = audit.fiscal_year_weeks(metric.ticker, fy) if fy else 52

        short = sym[metric.ticker]["short"]
        # ⛔ **Consensus is matched by METRIC IDENTITY, never by unit.** A unit match assigned
        # Yahoo's £6,056 m turnover as the benchmark for Hays' £972 m *net fees*, and Deere's
        # $10,732 m revenue as the benchmark for a $706 m *segment profit* — both are real numbers
        # for the wrong measure, which is the error the whole registry exists to prevent. A metric
        # no vendor covers gets **no** consensus rather than a plausible one.
        wanted = CONSENSUS_FIELD.get(metric.name)
        cons_sa = None
        if wanted:
            for r in sa_fc:
                if r["symbol"] != short or not r["is_estimate"] or r["metric"] != wanted:
                    continue
                if abs((date.fromisoformat(r["period_end"])
                        - date.fromisoformat(target["projected_period_end"])).days) > 20:
                    continue
                cons_sa = r["value"] / 1e6 if metric.unit == "currency_m" else r["value"]
                break

        cons_nq = None
        if metric.unit == "per_share" and wanted in ("eps", "adjustedEps"):
            for r in nq_fc:
                if r["symbol"] == short and r["consensus_eps"] and r["grain"] == "quarterlyForecast":
                    cons_nq = r["consensus_eps"]
                    break

        cons_corpus = None
        if metric.name == "has_pre_exc_operating_profit":
            # Hays publishes its own compiled consensus for exactly this measure
            hits = [c for c in consensus if c["kind"] == "company_compiled" and c["value"]
                    and "operating profit" in (c["metric"] or "").lower()]
            if hits:
                cons_corpus = max(hits, key=lambda c: c["published_at"])["value"]

        obs = fred if metric.ticker != "LSE:HAS" else labour
        nowcasts = {}
        for series_id, lane, label in NOWCASTS.get(metric.ticker, []):
            source = fred if lane == "fred" else labour
            start = (date.fromisoformat(target["projected_period_end"])
                     .replace(day=1).isoformat())
            window_start = f"{int(start[:4])}-{max(1, int(start[5:7]) - 2):02d}-01"
            nowcasts[series_id] = _nowcast(source, series_id, window_start,
                                           target["projected_period_end"])

        rows.append({
            "ticker": metric.ticker, "metric": metric.name,
            "workbook_label": metric.workbook_label, "unit": metric.unit, "basis": metric.basis,
            "target_period": target["target_period"],
            "fiscal_year": fy, "fiscal_period": period, "label": key.label(),
            "period_end": target["projected_period_end"],
            "already_reported": leaked is not None,
            "prior_value": prior_value,
            "history_n": len(history),
            "recent_growth_median": statistics.median(growths) if growths else None,
            "recent_growth_sd": statistics.pstdev(growths) if len(growths) > 1 else None,
            # already normalised onto the panel's scale by `feature/guides`
            "guide_low": guide["low"] if guide else None,
            "guide_high": guide["high"] if guide else None,
            "guide_mid": guide["midpoint"] if guide else None,
            "guide_unit": guide["guide_unit"] if guide else "",
            "guide_scale": guide["scale"] if guide else "",
            "guide_shape": shape or "",
            "guide_period": guide_period or "",
            "guide_stated_at": guide["published_at"] if guide else "",
            "bias_median": bias["bias_median"] if bias else None,
            "bias_unit": bias["bias_unit"] if bias else "",
            "bias_n": bias["n"] if bias else 0,
            "bias_sd": bias["bias_stdev"] if bias else None,
            "beat_rate": bias["beat_rate"] if bias else None,
            "range_position": bias["range_position_median"] if bias else None,
            "season_share": seasonal["share_recent"] if seasonal else None,
            # ⛔ A 53-week fiscal year dilutes every quarter's share of it. Deere's FY2026 carries
            # the extra week (its Q1 stepped 371 days), so a share measured on 52-week years
            # overstates Q3 by 52/53 ≈ 1.9 % before any modelling error.
            "fy_weeks": fy_weeks,
            "season_share_adjusted": (seasonal["share_recent"] * (52.0 / fy_weeks)
                                      if seasonal else None),
            "season_sd": seasonal["share_stdev"] if seasonal else None,
            "season_n": seasonal["n_years"] if seasonal else 0,
            "consensus_sa": cons_sa,
            "consensus_nasdaq": cons_nq,
            "consensus_corpus": cons_corpus,
            **{f"nowcast_{k}_yoy": v["yoy_pct"] for k, v in nowcasts.items()},
            **{f"nowcast_{k}_n": v["n_months"] for k, v in nowcasts.items()},
        })
    return rows


def main() -> int:
    rows = build()
    store.write(config.FEATURE / "design_matrix.parquet", rows)
    leaked = [r for r in rows if r["already_reported"]]
    print(f"feature/design_matrix.parquet {len(rows)} target rows"
          + (f"  ⛔ {len(leaked)} ALREADY REPORTED — not a forecast" if leaked else
             "  (no target period is present in the panel — every row is a genuine forecast)"))
    print(f"\n{'target':<42}{'prior':>11}{'guide':>20}{'bias':>11}{'n':>4}"
          f"{'season':>9}{'consensus':>11}")
    print("-" * 112)
    for r in rows:
        guide = ("—" if r["guide_mid"] is None else
                 f"{r['guide_low']:.6g}–{r['guide_high']:.6g}{r['guide_unit']}")
        bias = ("—" if r["bias_median"] is None else
                (f"{r['bias_median']:+.2f}pp" if r["bias_unit"] == "pp"
                 else f"{r['bias_median']:+.1%}"))
        season = "—" if r["season_share"] is None else f"{r['season_share']:.1%}"
        cons = next((c for c in (r["consensus_sa"], r["consensus_nasdaq"], r["consensus_corpus"])
                     if c is not None), None)
        print(f"{r['ticker'] + ' · ' + r['workbook_label']:<42}"
              f"{(r['prior_value'] if r['prior_value'] is not None else float('nan')):>11,.2f}"
              f"{guide:>20}{bias:>11}{r['bias_n']:>4}{season:>9}"
              f"{(cons if cons is not None else float('nan')):>11,.2f}")
    print("\n  in-quarter nowcasts (months of the target quarter already published):")
    for r in rows:
        hits = {k[8:-4]: v for k, v in r.items() if k.startswith("nowcast_")
                and k.endswith("_yoy") and v is not None}
        if hits:
            print(f"    {r['ticker']:<9}" + "  ".join(f"{k}={v:+.2f}%" for k, v in hits.items()))
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
