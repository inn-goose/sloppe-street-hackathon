"""Does the conservatism adjustment actually help? Walk-forward, on the competition's own loss.

    PYTHONPATH=. .venv/bin/python -m code.feature.validate

## Why this exists

`conservatism.py` measures that ADI has beaten its own revenue guide 88 % of the time by a median
2.54 %. That is an **in-sample description**. Using it as a forecast is a different claim, and an
untested one: a bias fitted on all of history and then applied to that same history will always
look good.

So each estimator is re-fitted at every period using **only the periods before it**, applied to
that period's guide, and scored against what the company actually reported. An expanding window,
no peeking.

## The loss is the competition's, not a convenience

The prize is decided on `|our miss| / |benchmark miss|`, so the number that matters is the **ratio
of mean absolute errors against the midpoint baseline**:

    skill = mean|error of estimator| / mean|error of guide midpoint|

`skill < 1` means the adjustment earns its place. `skill ≥ 1` means submitting the midpoint is
better and the feature must be dropped, however good its in-sample story looked.

## Estimators tested

* `midpoint` — the guide's own midpoint. The baseline, and roughly where the sell side sits.
* `bias_median` — midpoint shifted by the median historical bias.
* `bias_mean` — the same with the mean, which a few large beats will drag.
* `range_position` — `low + p·(high − low)` with `p` the median historical landing point. Scale-
  free, so it survives a filer whose guide width changes.

⚠️ **A minimum training length is enforced.** A bias fitted on two observations is a coin flip
wearing a decimal point, and scoring it would flatter the method.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from code.lib import config, store

MIN_TRAIN = 5
ESTIMATORS = ("midpoint", "bias_median", "bias_mean", "range_position")


def _predict(name: str, train: list[dict], row: dict) -> float | None:
    mid, low, high = row["midpoint"], row["low"], row["high"]
    if name == "midpoint":
        return mid
    if name in ("bias_median", "bias_mean"):
        biases = [t["bias"] for t in train]
        shift = (statistics.median(biases) if name == "bias_median"
                 else statistics.fmean(biases))
        # `bias` is relative for a level guide and in points for a growth/margin one
        return mid + shift if row["bias_unit"] == "pp" else mid * (1.0 + shift)
    if name == "range_position":
        positions = [t["range_position"] for t in train if t["range_position"] is not None]
        if not positions or high == low:
            return None
        return low + statistics.median(positions) * (high - low)
    return None


def build() -> tuple[list[dict], list[dict]]:
    pairs = store.read(config.FEATURE / "guide_vs_actual.parquet")
    by_metric: dict[tuple, list[dict]] = defaultdict(list)
    for row in pairs:
        by_metric[(row["metric"], row["shape"])].append(row)

    folds: list[dict] = []
    for (metric, shape), rows in by_metric.items():
        rows.sort(key=lambda r: (r["fiscal_year"], r["fiscal_period"]))
        if len(rows) < MIN_TRAIN + 2:
            continue
        for i in range(MIN_TRAIN, len(rows)):
            train, test = rows[:i], rows[i]
            for name in ESTIMATORS:
                pred = _predict(name, train, test)
                if pred is None:
                    continue
                folds.append({
                    "metric": metric, "shape": shape, "estimator": name,
                    "label": test["label"], "n_train": len(train),
                    "actual": test["actual"], "pred": pred,
                    "abs_error": abs(pred - test["actual"]),
                })

    summary: list[dict] = []
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for fold in folds:
        grouped[(fold["metric"], fold["estimator"])].append(fold)
    baseline = {k[0]: statistics.fmean([f["abs_error"] for f in v])
                for k, v in grouped.items() if k[1] == "midpoint"}
    for (metric, estimator), group in grouped.items():
        mae = statistics.fmean([f["abs_error"] for f in group])
        base = baseline.get(metric)
        summary.append({
            "metric": metric, "estimator": estimator, "n_folds": len(group),
            "mae": mae,
            "skill_vs_midpoint": (mae / base) if base else None,
            "median_abs_error": statistics.median([f["abs_error"] for f in group]),
            "win_rate_vs_midpoint": None,
        })
    # per-fold head-to-head, which is what the prize actually pays on
    for row in summary:
        mids = {f["label"]: f["abs_error"]
                for f in grouped.get((row["metric"], "midpoint"), [])}
        mine = grouped.get((row["metric"], row["estimator"]), [])
        contested = [(f["abs_error"], mids[f["label"]]) for f in mine if f["label"] in mids]
        if contested:
            row["win_rate_vs_midpoint"] = sum(1 for a, b in contested if a < b) / len(contested)
    summary.sort(key=lambda r: (r["metric"], r["skill_vs_midpoint"] or 9))
    return summary, folds


def main() -> int:
    summary, folds = build()
    store.write(config.FEATURE / "estimator_validation.parquet", summary)
    store.write(config.FEATURE / "estimator_folds.parquet", folds)
    if not summary:
        print("no metric has enough guide→outcome pairs to walk forward "
              f"(need {MIN_TRAIN + 2})")
        return 0
    print(f"walk-forward, expanding window, min {MIN_TRAIN} training observations\n")
    print(f"{'metric':<26}{'estimator':<17}{'folds':>6}{'MAE':>12}"
          f"{'skill':>8}{'win vs mid':>12}  verdict")
    print("-" * 92)
    for row in summary:
        skill = row["skill_vs_midpoint"]
        win = row["win_rate_vs_midpoint"]
        verdict = ("baseline" if row["estimator"] == "midpoint"
                   else ("USE" if skill is not None and skill < 0.98 else "drop"))
        print(f"{row['metric']:<26}{row['estimator']:<17}{row['n_folds']:>6}"
              f"{row['mae']:>12,.3f}{(skill if skill else float('nan')):>8.3f}"
              f"{(win if win is not None else float('nan')):>12.0%}  {verdict}")
    print("\n  skill = MAE / MAE(midpoint). Below 1.00 the adjustment earns its place;")
    print("  at or above 1.00 the guide midpoint is the better submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
