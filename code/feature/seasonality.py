"""feature/seasonality — what share of a fiscal year each quarter carries.

## Why this is the load-bearing feature for two of the four companies

Home Depot and Deere both guide the **full year** and are being forecast for **one quarter**:

* HD guides FY2026 total sales growth 2.5–4.5 % and adjusted EPS growth flat–4.0 %. The target is
  Q2, its largest quarter.
* Deere guides FY2026 net income of $4.5–5.0 bn. The target is Q3, and $2.429 bn of it is already
  reported in H1.

Neither guide is usable until the year is split. That split is this module, and it is measured
from the filer's own history rather than assumed uniform — a flat 25 % would put HD's Q2 about
**12 % below** its true seasonal weight, which on a $45 bn quarter is a $5 bn error.

## How the split is measured

For each `(ticker, metric)` with four quarters in a fiscal year, the share is
`Q / Σ(Q1..Q4)`. Only **complete** years count: a year missing a quarter would inflate the
remaining three, which is the obvious way to get this quietly wrong.

Three estimates are published per quarter and they answer different questions:

* `share_mean` / `share_median` — the long-run seasonal weight
* `share_recent` — the mean of the last `RECENT_YEARS` complete years, which is what a business
  whose mix is shifting actually looks like now
* `share_trend` — the per-year drift in the share, so a consumer can see whether the season is
  moving rather than inferring it from two numbers

⚠️ **A share is only meaningful for a flow.** Summing four quarters of a *margin* or an *EPS*
is arithmetic nonsense for the margin and merely conventional for EPS, so only `currency_m`
metrics get a share; per-share metrics get one too because EPS is additive across quarters, and
percent metrics are refused outright.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from code.feature import metrics as M
from code.lib import config, store

QUARTERS = ("Q1", "Q2", "Q3", "Q4")
#: How many recent complete years feed `share_recent`.
RECENT_YEARS = 4
#: A share is only computed for metrics that are additive across quarters.
ADDITIVE_UNITS = {"currency_m", "per_share"}


def _ols_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def build() -> tuple[list[dict], list[dict]]:
    panel = store.read(config.FEATURE / "metric_panel.parquet")

    by_metric: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    stated_fy: dict[tuple, float] = {}
    for row in panel:
        if row["fiscal_period"] in QUARTERS and row["fiscal_year"]:
            by_metric[row["metric"]][row["fiscal_year"]][row["fiscal_period"]] = row["value"]
        elif row["fiscal_period"] == "FY" and row["fiscal_year"] and row["lanes"] != "derived_fy":
            stated_fy[(row["metric"], row["fiscal_year"])] = row["value"]

    shares: list[dict] = []
    years_used: list[dict] = []
    rejected: dict[str, list[int]] = {}
    for name, years in by_metric.items():
        metric = M.REGISTRY.get(name)
        if metric is None or metric.unit not in ADDITIVE_UNITS:
            continue
        complete = {fy: q for fy, q in years.items()
                    if len(q) == 4 and sum(abs(v) for v in q.values()) > 0}
        # ⛔ **Four quarters that do not reach the filer's own annual total are four wrong quarters,
        # and a share measured from them is wrong in a way nothing downstream can see.** Deere's
        # pre-2018 quarters are contaminated with year-to-date columns — FY2013's summed to
        # **$6,267 m** against a reported **$3,537 m** — which drags the denominator up and pushes
        # every quarter's share down. Deere's Q3 net income share is a direct input to a submitted
        # number, so the years that fail their own arithmetic are dropped rather than de-weighted.
        tol = 0.08 if metric.unit == "per_share" else 0.005   # annual EPS uses annual average shares
        refused = set()
        for fy, q in complete.items():
            # ⛔ A year whose quarters cancel out has no meaningful share — the denominator is
            # noise and the ratio explodes. Measured, ADI's adjusted operating income produced a
            # **−6,213 %** Q1 share this way, which would silently poison the mean for every year.
            if abs(sum(q.values())) < 0.5 * sum(abs(v) for v in q.values()):
                refused.add(fy)
                continue
            total = stated_fy.get((name, fy))
            if total is None or abs(sum(q.values()) / total - 1.0) <= tol:
                continue
            # ⚡ **When the quarters and the annual figure disagree, ask which one is impossible.**
            # An annual total cannot be smaller than a single quarter of itself. Home Depot's
            # FY2025 adjusted EPS is stated as **3.6** against quarters summing to **14.7** — 3.6 is
            # one quarter's figure that landed on an annual row, and dropping the year on its word
            # would have cost the seasonal share for a submitted metric. The quarters win that one.
            if abs(total) >= max(abs(v) for v in q.values()):
                refused.add(fy)
        complete = {fy: q for fy, q in complete.items() if fy not in refused}
        rejected[name] = sorted(refused)
        if len(complete) < 3:
            continue
        per_quarter: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for fy, q in sorted(complete.items()):
            total = sum(q.values())
            if total == 0:
                continue
            years_used.append({"metric": name, "ticker": metric.ticker, "fiscal_year": fy,
                               "fy_total": total,
                               **{f"share_{k}": q[k] / total for k in QUARTERS}})
            for quarter in QUARTERS:
                per_quarter[quarter].append((fy, q[quarter] / total))

        for quarter, points in per_quarter.items():
            points.sort()
            values = [v for _fy, v in points]
            recent = [v for _fy, v in points[-RECENT_YEARS:]]
            shares.append({
                "ticker": metric.ticker, "metric": name, "quarter": quarter,
                "unit": metric.unit,
                "n_years": len(values),
                "first_year": points[0][0], "last_year": points[-1][0],
                "share_mean": statistics.fmean(values),
                "share_median": statistics.median(values),
                "share_recent": statistics.fmean(recent),
                # ⚡ **The most recent complete year, on its own.** A multi-year mean assumes the
                # seasonal mix is stationary, and for Home Depot it is not: the SRS acquisition
                # (2024) added a trade-distribution business with a different profile, so HD's
                # Q2 share was **28.38 %** last year against a four-year mean of 27.59 %. On a
                # $165 bn year that 0.8 pp is **$1.3 bn** — larger than any edge being chased.
                "share_last": points[-1][1],
                "share_last2": statistics.fmean([v for _fy, v in points[-2:]]),
                "share_stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "share_trend": _ols_slope([float(fy) for fy, _v in points], values),
                "share_min": min(values), "share_max": max(values),
            })
    shares.extend(_borrow_shares(shares, by_metric))
    shares.sort(key=lambda r: (r["ticker"], r["metric"], r["quarter"]))
    return shares, years_used, {k: v for k, v in rejected.items() if v}


#: A metric with too little history of its own, and the sibling whose season it shares.
PROXY_SHARE: dict[str, str] = {"hd_adj_diluted_eps": "hd_diluted_eps"}


def _borrow_shares(shares: list[dict], by_metric: dict) -> list[dict]:
    """Give a target its seasonal split from a sibling, and measure whether that is allowed.

    ⚡ **Home Depot's adjusted EPS is a submitted number with no usable season of its own.** HD only
    began reporting an adjusted figure after the SRS acquisition, so it has one clean year — and a
    seasonal share measured on one year is just that year. Reported EPS has seventeen.

    The two are the same earnings apart from a near-constant quarterly amortisation charge, so the
    split should transfer. That claim is not assumed, it is **checked on the overlap**: in FY2025,
    the only year both are clean, adjusted EPS put **31.84 %** in Q2 and reported EPS **32.18 %** —
    0.34 pp apart, against a Q2 share of roughly 31 %. The measured gap is published on every
    borrowed row, so a consumer can see what the borrow cost rather than take it on trust.
    """
    have = {r["metric"] for r in shares}
    donors = {r["metric"]: {x["quarter"]: x for x in shares if x["metric"] == r["metric"]}
              for r in shares}
    out = []
    for name, donor_name in PROXY_SHARE.items():
        if name in have or donor_name not in donors:
            continue
        metric = M.REGISTRY.get(name)
        if metric is None:
            continue
        gaps = []
        for fy, quarters in by_metric.get(name, {}).items():
            other = by_metric.get(donor_name, {}).get(fy, {})
            if len(quarters) != 4 or len(other) != 4:
                continue
            mine, theirs = sum(quarters.values()), sum(other.values())
            if mine <= 0 or theirs <= 0:
                continue
            # the gap is only evidence if both years are clean — HD's FY2024 adjusted EPS carries a
            # −6.80 third quarter, and measuring against that reported a 170 pp gap for a borrow
            # that is actually good to a third of a point
            if any(abs(sum(v.values())) < 0.5 * sum(abs(x) for x in v.values())
                   for v in (quarters, other)):
                continue
            gaps.append(max(abs(quarters[q] / mine - other[q] / theirs) for q in QUARTERS))
        for quarter, row in donors[donor_name].items():
            out.append({**row, "ticker": metric.ticker, "metric": name, "unit": metric.unit,
                        "source": f"proxy:{donor_name}",
                        "proxy_max_gap": max(gaps) if gaps else None,
                        "proxy_overlap_years": len(gaps)})
    return out


def main() -> int:
    shares, years, rejected = build()
    store.write(config.FEATURE / "seasonality.parquet", shares)
    store.write(config.FEATURE / "seasonality_years.parquet", years)
    print(f"feature/seasonality.parquet {len(shares)} (metric, quarter) shares over "
          f"{len({(r['metric']) for r in shares})} metrics")
    print(f"\n{'metric':<32}{'Q1':>9}{'Q2':>9}{'Q3':>9}{'Q4':>9}{'yrs':>5}  "
          f"recent-4y in brackets")
    print("-" * 96)
    for name in sorted({r["metric"] for r in shares}):
        mine = {r["quarter"]: r for r in shares if r["metric"] == name}
        if len(mine) != 4:
            continue
        row = "".join(f"{mine[q]['share_mean']:>9.1%}" for q in QUARTERS)
        rec = " ".join(f"{mine[q]['share_recent']:.1%}" for q in QUARTERS)
        print(f"{name:<32}{row}{mine['Q1']['n_years']:>5}  [{rec}]")
    if rejected:
        print("\n  years dropped for failing Σ4Q == stated annual total:")
        for name, fys in sorted(rejected.items()):
            print(f"    {name:<32}{', '.join(f'FY{f}' for f in fys)}")
    print("\n  the two splits the forecast turns on:")
    for name, quarter in (("hd_net_sales", "Q2"), ("de_net_income", "Q3")):
        hit = next((r for r in shares if r["metric"] == name and r["quarter"] == quarter), None)
        if hit:
            drift = f"{hit['share_trend']:+.2%}/yr" if hit["share_trend"] is not None else "n/a"
            print(f"    {name} {quarter}: mean {hit['share_mean']:.2%}  "
                  f"recent {hit['share_recent']:.2%}  sd {hit['share_stdev']:.2%}  "
                  f"drift {drift}  (n={hit['n_years']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
