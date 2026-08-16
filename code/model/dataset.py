"""model/dataset — reshape the feature matrix into what is actually being learned.

    PYTHONPATH=. .venv/bin/python -m code.model.dataset

## What the model is asked to learn, and why it is not the level

Predicting `$46,861 m of net sales` and `4.68 of EPS` with one model is a scale problem, not a
forecasting problem: the two differ by four orders of magnitude and a squared-error learner would
spend all of its capacity on Home Depot's revenue. There are also only ~1,300 labelled rows across
26 metrics and four companies, which is nowhere near enough to learn each metric's level on its own.

So the target is the **residual of an anchor**. Every row already carries several parameter-free
point forecasts — last year's same quarter, the guide midpoint, an implied annual level times a
seasonal share, a companion line times its historical ratio. Each is a legitimate forecast on its
own. What varies systematically, and what a filer's history can actually teach, is *how wrong each
kind of anchor tends to be, and when*.

One training example is therefore one `(row, anchor)` pair:

```text
    features:  the row's history, guide, season, calendar, macro  +  which anchor this is
    label:     y − anchor        (percentage metrics, in points)
               y / anchor − 1    (money and per-share metrics, relative)
```

⚡ **Pooling across metrics and companies is what makes the sample size workable.** "Filers land
above the midpoint of their own guide" and "a seasonal share is more reliable than a momentum
carry" are cross-sectional regularities. Reshaping this way turns 1,315 rows into roughly 5,000
examples and lets a Deere row inform a Home Depot one.

## The refusals

⛔ **An anchor near zero cannot carry a relative residual.** Home Depot's comparable sales ran
`+0.6 %`; `y/anchor − 1` against that is a division by noise, and a single such example can dominate
an MAE objective. Percentage metrics use a difference, and any other anchor whose magnitude falls
below a floor derived from the metric's own history is dropped rather than winsorised.

⛔ **`y_value`, `y_yoy` and `y_vs_lag4` never become features.** They are the label in three
different costumes.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from code.lib import config, store

#: Columns that are the answer, or a restatement of it. Never features.
LEAKING = frozenset({"y_value", "y_yoy", "y_vs_lag4"})
#: Bookkeeping columns the learner must not see as signal.
IDENTITY = frozenset({"label", "as_of", "is_prediction", "ordinal", "companion_metric",
                      "external_source", "season_source", "guide_shape", "fy_guide_shape"})
#: Treated as levels to divide into; everything else is compared in points.
RELATIVE_UNITS = frozenset({"currency_m", "per_share", "shares_m"})
#: An anchor smaller than this share of the metric's own typical magnitude is too close to zero
#: for a relative residual to mean anything.
NEAR_ZERO = 0.05
#: ⛔ **Beyond this, the example is a data fault rather than a forecasting lesson.** The worst
#: relative residuals in this table reached **509,000** — every one of them `hd_diluted_shares`,
#: where the panel holds cells of `0.001` against real values near 500. That is a scale error in a
#: metric nothing is forecast from, but pooled training does not know it: an L1 objective is robust
#: to a heavy tail, not to a label six orders of magnitude out. Capping also matches the incentive,
#: because a competition score is capped at 5.0 while the floor is 0 — the model must never learn
#: to make a 200 % correction, since being that wrong costs eight times what being right gains.
MAX_RELATIVE_RESIDUAL = 2.0
MAX_POINTS_RESIDUAL = 25.0


def anchor_columns(row: dict) -> list[str]:
    return [c for c in row if c.startswith("anchor_")]


def build() -> tuple[list[dict], dict]:
    matrix = store.read(config.FEATURE / "training_matrix.parquet")
    anchors = anchor_columns(matrix[0])

    typical: dict[str, float] = {}
    for name, group in store.group_by(
            [r for r in matrix if r["y_value"] is not None], "metric").items():
        values = [abs(r["y_value"]) for r in group if r["y_value"] is not None]
        typical[name[0]] = statistics.median(values) if values else 0.0

    feature_names = [c for c in matrix[0]
                     if c not in LEAKING and c not in IDENTITY and c not in anchors]

    examples: list[dict] = []
    refused: defaultdict = defaultdict(int)
    for row in matrix:
        available = {a: row[a] for a in anchors if row.get(a) is not None}
        if not available:
            refused["no_anchor"] += 1
            continue

        # ⚡ **Disagreement between anchors is itself a feature, and a strong one.** When the guide,
        # the seasonal split and the momentum carry all land together the number is nearly settled;
        # when they scatter, the row is genuinely uncertain and the model should shrink toward the
        # centre. Both facts are only visible across anchors, so they are computed once per row.
        values = list(available.values())
        centre = statistics.median(values)
        scale = abs(centre) if abs(centre) > 1e-9 else 1.0
        spread = (max(values) - min(values)) / scale if len(values) > 1 else 0.0

        shared = {c: row[c] for c in feature_names}
        shared.update({
            "n_anchors": len(available),
            "anchor_spread": spread,
            "anchor_median": centre,
        })

        relative = row["unit"] in RELATIVE_UNITS
        for name, value in available.items():
            if relative and abs(value) < NEAR_ZERO * max(typical.get(row["metric"], 0.0), 1e-9):
                refused[f"near_zero:{name}"] += 1
                continue
            example = dict(shared)
            example.update({
                "ticker": row["ticker"], "metric": row["metric"],
                "fiscal_year": row["fiscal_year"], "fiscal_period": row["fiscal_period"],
                "label": row["label"], "as_of": row["as_of"],
                "is_prediction": row["is_prediction"],
                "anchor_type": name.replace("anchor_", ""),
                "anchor_value": value,
                "anchor_is_relative": int(relative),
                # scale-free views of where this anchor sits against the row's own history, which
                # is how a Deere example becomes comparable to a Home Depot one
                "anchor_over_lag1": (value / row["lag1"]
                                     if row.get("lag1") not in (None, 0) else None),
                "anchor_over_lag4": (value / row["lag4"]
                                     if row.get("lag4") not in (None, 0) else None),
                "anchor_vs_median": (value / centre - 1.0) if abs(centre) > 1e-9 else None,
                "y_value": row["y_value"],
            })
            if row["y_value"] is None:
                example["residual"] = None
            else:
                residual = (row["y_value"] / value - 1.0) if relative else row["y_value"] - value
                cap = MAX_RELATIVE_RESIDUAL if relative else MAX_POINTS_RESIDUAL
                if abs(residual) > cap:
                    refused[f"residual_beyond_cap:{row['metric']}"] += 1
                    continue
                example["residual"] = residual
            examples.append(example)

    stats = {"rows": len(matrix), "examples": len(examples),
             "labelled": sum(1 for e in examples if e["residual"] is not None),
             "refused": dict(refused), "features": feature_names}
    return examples, stats


def main() -> int:
    examples, stats = build()
    store.write(config.FEATURE.parent / "model" / "anchor_examples.parquet", examples)
    print(f"model/anchor_examples.parquet {stats['examples']:,} examples "
          f"({stats['labelled']:,} labelled) from {stats['rows']:,} matrix rows")

    # ⚠️ Relative and points residuals are different quantities and are never averaged together —
    # a 0.06 relative miss and a 2.4 pp miss are not comparable, and one table of both is a lie.
    for family, keep in (("relative  (y/anchor − 1)", True), ("points    (y − anchor, pp)", False)):
        subset = [e for e in examples if bool(e["anchor_is_relative"]) is keep
                  and e["residual"] is not None]
        if not subset:
            continue
        print(f"\n{family}   {len(subset):,} labelled examples")
        print(f"  {'anchor':<22}{'n':>7}{'median |r|':>12}{'p90 |r|':>10}{'bias':>10}")
        print("  " + "-" * 61)
        for name, group in sorted(store.group_by(subset, "anchor_type").items()):
            errs = sorted(abs(e["residual"]) for e in group)
            signed = [e["residual"] for e in group]
            p90 = errs[int(0.9 * (len(errs) - 1))]
            print(f"  {name[0]:<22}{len(group):>7,}{statistics.median(errs):>12.4f}"
                  f"{p90:>10.4f}{statistics.median(signed):>+10.4f}")
    if stats["refused"]:
        print("\n  refused:")
        for key, count in sorted(stats["refused"].items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {key:<52}{count:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
