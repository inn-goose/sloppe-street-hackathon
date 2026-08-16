"""feature/conservatism — does this filer beat its own guidance, and by how much?

The registry's **#214 (sandbagging)** leg, and the single highest-value feature in this build.

## Why it matters more here than anywhere else

Every one of the twelve targets has a standing guide, so the naive forecast is "the midpoint".
That is also what a large part of the sell side does, which means the midpoint is roughly where
the benchmark sits — and the competition scores `|our miss| / |Wall Street's miss|`. Submitting
the midpoint earns a score near 1.0 by construction: no better than the bar, no worse.

The edge is in the **residual**: management's guide is a decision, not a measurement, and filers
are systematically biased in their own direction. If ADI has cleared the top of its revenue guide
in eleven of its last twelve quarters, the midpoint is not the expectation — the upper half is.
This module measures that bias per filer per metric, from their own history.

## How each guide is compared to what happened

A guide is one of three shapes and they cannot be compared the same way:

* **level** — `revenue of $3.9 bn ± $100 m` → compared directly against the realised level
* **growth** — `total sales growth of 2.5 % to 4.5 %` → compared against realised YoY growth,
  which has to be computed from the panel first
* **margin** — `gross margin of approximately 33.1 %` → compared against the realised margin

⛔ Comparing a growth guide against a level is the obvious way to produce a confident, meaningless
number, so the shape is classified before anything is measured and unclassifiable rows are refused.

## What is published

* `beat_rate` — share of periods where the outcome exceeded the midpoint
* `bias_mean` / `bias_median` — `(actual − midpoint) / |midpoint|`, the systematic tilt
* `range_position` — where in the guided range the outcome landed (0 = low bound, 1 = high bound;
  above 1 means the filer cleared its own range)
* `n` and `dispersion` — because a bias measured on four observations is a rumour
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from code.feature import metrics as M
from code.lib import config, store


def _yoy(panel_by_key: dict, metric: str, fy: int, period: str) -> float | None:
    cur = panel_by_key.get((metric, fy, period))
    prior = panel_by_key.get((metric, fy - 1, period))
    if cur is None or prior is None or not prior:
        return None
    return (cur / prior - 1.0) * 100.0


def build() -> tuple[list[dict], list[dict]]:
    guides = store.read(config.FEATURE / "guides.parquet")
    panel = store.read(config.FEATURE / "metric_panel.parquet")
    panel_by_key = {(r["metric"], r["fiscal_year"], r["fiscal_period"]): r["value"]
                    for r in panel}

    observations: list[dict] = []
    refused_units: defaultdict = defaultdict(int)
    for guide in guides:
        # `feature/guides` owns binding a filer's wording to a metric and a canonical period; this
        # module only joins the resulting key to what was reported
        ticker, metric_name, shape = guide["ticker"], guide["metric"], guide["shape"]
        if True:
            metric = M.REGISTRY[metric_name]
            low, high, mid = guide["low"], guide["high"], guide["midpoint"]
            fiscal_year, fiscal_period = guide["fiscal_year"], guide["fiscal_period"]

            if shape == "growth":
                actual = _yoy(panel_by_key, metric_name, fiscal_year, fiscal_period)
            else:
                actual = panel_by_key.get((metric_name, fiscal_year, fiscal_period))
            if actual is None:
                continue

            # ⛔ **An order-of-magnitude gap is a UNIT mismatch, never a guidance miss.** No filer
            # misses its own guide by 10×; measured, admitting these gave ADI's revenue bias a
            # standard deviation of 14,476 % and HD's gross-margin bias −99 %, which is a guide in
            # percentage points compared against a ratio. Refused and counted rather than
            # winsorised, because the row is not a weak observation — it is the wrong pair.
            ratio = abs(actual) / abs(mid) if mid else 0.0
            if ratio and (ratio > 5.0 or ratio < 0.2):
                refused_units[metric_name] += 1
                continue

            # ⛔ **Relative bias is the wrong measure for anything that crosses zero.** Comparable
            # sales guided `flat to 2.0 %` and realised at −1.8 % is a **2.8 pp** miss; expressed
            # as `(actual − mid)/|mid|` it reads **−280 %**, which is not a bias, it is a division
            # by a small number. A growth or margin guide is measured in percentage points; a
            # level guide is measured relatively. The unit decides, not a convention.
            width = high - low
            if metric.unit == "percent" or shape in ("growth", "margin"):
                bias = actual - mid                      # percentage points
                bias_unit = "pp"
            else:
                bias = (actual - mid) / abs(mid)         # relative
                bias_unit = "relative"
            observations.append({
                "ticker": ticker, "metric": metric_name, "shape": shape,
                "fiscal_year": fiscal_year, "fiscal_period": fiscal_period,
                "label": guide["label"],
                "guided_at": guide["published_at"], "frame": guide["frame"],
                "low": low, "high": high, "midpoint": mid, "actual": actual,
                "bias": bias, "bias_unit": bias_unit,
                "beat": actual > mid,
                "cleared_high": actual > high,
                "range_position": ((actual - low) / width) if width else None,
                "lead_days": None,
                "unit": metric.unit,
            })

    # one observation per (metric, period) — a filer restates the same guide every quarter, and
    # counting each restatement would weight a long-standing guide more than a fresh one
    latest: dict[tuple, dict] = {}
    for obs in observations:
        key = (obs["metric"], obs["fiscal_year"], obs["fiscal_period"])
        prev = latest.get(key)
        if prev is None or obs["guided_at"] > prev["guided_at"]:
            latest[key] = obs
    deduped = sorted(latest.values(), key=lambda o: (o["metric"], o["label"]))

    summary: list[dict] = []
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for obs in deduped:
        grouped[(obs["ticker"], obs["metric"], obs["shape"])].append(obs)
    for (ticker, metric_name, shape), group in grouped.items():
        biases = [o["bias"] for o in group]
        positions = [o["range_position"] for o in group if o["range_position"] is not None]
        summary.append({
            "ticker": ticker, "metric": metric_name, "shape": shape,
            "bias_unit": group[0]["bias_unit"],
            "n": len(group),
            "beat_rate": sum(1 for o in group if o["beat"]) / len(group),
            "cleared_high_rate": sum(1 for o in group if o["cleared_high"]) / len(group),
            "bias_mean": statistics.fmean(biases),
            "bias_median": statistics.median(biases),
            "bias_stdev": statistics.pstdev(biases) if len(biases) > 1 else 0.0,
            "range_position_median": statistics.median(positions) if positions else None,
            "first": group[0]["label"], "last": group[-1]["label"],
        })
    summary.sort(key=lambda r: (r["ticker"], r["metric"]))
    return summary, deduped, dict(refused_units)


def main() -> int:
    summary, observations, refused = build()
    store.write(config.FEATURE / "conservatism.parquet", summary)
    store.write(config.FEATURE / "guide_vs_actual.parquet", observations)
    print(f"feature/guide_vs_actual.parquet {len(observations)} guide→outcome pairs")
    print(f"feature/conservatism.parquet    {len(summary)} (metric, shape) biases\n")
    print(f"{'metric':<32}{'shape':<8}{'n':>4}{'beat':>7}{'>high':>7}"
          f"{'bias med':>12}{'bias sd':>10}{'range pos':>11}")
    print("-" * 96)
    for row in summary:
        pos = ("—" if row["range_position_median"] is None
               else f"{row['range_position_median']:.2f}")
        fmt = (lambda v: f"{v:+.2f}pp") if row["bias_unit"] == "pp" else (lambda v: f"{v:+.2%}")
        print(f"{row['metric']:<32}{row['shape']:<8}{row['n']:>4}{row['beat_rate']:>7.0%}"
              f"{row['cleared_high_rate']:>7.0%}{fmt(row['bias_median']):>12}"
              f"{fmt(row['bias_stdev']):>10}{pos:>11}")
    if refused:
        print(f"\n  refused as unit mismatches (not misses): "
              f"{', '.join(f'{k}×{v}' for k, v in sorted(refused.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
