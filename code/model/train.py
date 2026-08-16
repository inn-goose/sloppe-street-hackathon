"""model/train — walk-forward training of the anchor-correction model.

    PYTHONPATH=. .venv/bin/python -m code.model.train

LightGBM, gradient-boosted trees, `objective="l1"`, trained on the `(row, anchor)` residuals from
`model/dataset`. Two models, never one: a relative residual (`y/anchor − 1`) and a points residual
(`y − anchor`, in percentage points) are different quantities and averaging them is meaningless.

## The scoring rule, which is the competition's own

Absolute error is not comparable across a $47 bn revenue line and a 4.68 EPS, so every fold is
scored on the metric the prize actually uses, with Wall Street's miss set to zero — its worst case
for us, and the only part of it computable before the companies report:

```text
    floor  = 0.5 % of |reported|   (money, per-share)   |   0.5 pp (percentage)
    score  = min(5.0, |prediction − reported| / floor)
```

This is scale-free, so one number ranks a Deere segment profit against a Hays EPS, and it is
directly the quantity being minimised on the day.

## What is being decided here, rather than assumed

⛔ **The model is not assumed to help.** Each fold scores three predictors side by side and the
choice between them is made on out-of-sample evidence only:

* `raw` — use the anchor untouched. The null hypothesis.
* `bias` — shift the anchor by the expanding-window **median** residual of its own anchor type.
  A median, not a mean, because a handful of large beats otherwise sets the correction.
* `gbm` — the trained correction.

⚡ **Levels are deliberately withheld from the model.** With ~3,900 examples spread over 26 metrics,
raw magnitudes are a fingerprint: given `lag1 = 45,277` a tree can identify Home Depot's Q2 and
memorise its answer instead of learning anything transferable. Only scale-free columns are exposed —
ratios, growth rates, shares, dispersions — so a Deere example genuinely informs a Home Depot one.
`metric` is withheld as a categorical for the same reason; `ticker`, `unit` and `anchor_type` are not.

⚠️ **Folds are cut on `as_of`, the date the label became public** — not on fiscal period. Filers
report on different calendars, and cutting by fiscal quarter would train on a Deere result published
after the Home Depot row being tested.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

import numpy as np

from code.lib import config, store

MODEL = config.DATA / "model"
#: Withheld: the label in any costume, bookkeeping, and every raw level (a metric fingerprint).
DROP = frozenset({
    "residual", "y_value", "label", "as_of", "is_prediction", "fiscal_year", "ordinal",
    "metric", "anchor_value", "anchor_median", "lag1", "lag2", "lag4", "trailing4_sum",
    "fy_prior_total", "guide_low", "guide_high", "guide_mid", "fy_guide_low", "fy_guide_high",
    "fy_guide_mid", "fy_guide_implied_level", "fy_guide_residual", "ytd_before",
    "companion_guide_mid", "external_low", "external_high", "companion_metric",
    "external_source", "season_source", "guide_shape", "fy_guide_shape", "workbook_label",
})
CATEGORICAL = ("ticker", "fiscal_period", "anchor_type", "unit", "basis")
#: Competition denominators when Wall Street is perfect.
PERCENT_FLOOR = 0.5
MONEY_FLOOR = 0.005
SCORE_CAP = 5.0
#: Fewest training examples before a fold is scored at all.
MIN_TRAIN = 400
#: Fewest observations before a metric may pick its anchor on its own record.
MIN_METRIC_OBS = 6
N_FOLDS = 8
#: How far to shrink the chosen anchor toward the median of all anchors; 1.00 = no shrinkage.
SHRINK_GRID = (1.00, 0.90, 0.80, 0.70, 0.55, 0.40)


def floor_of(unit: str, actual: float) -> float:
    if unit == "percent":
        return PERCENT_FLOOR
    return max(MONEY_FLOOR * abs(actual), 1e-6)


def competition_score(pred: float, actual: float, unit: str) -> float:
    """The prize's own rule with Wall Street's miss set to zero — realistic, but not selectable on.

    ⛔ **Reported, never optimised against.** Scored this way **74 %** of predictions sit at the
    5.0 cap, because clearing it needs an error inside 2.5 % of the reported figure and a good
    quarterly forecast is not that sharp. A metric that saturates has no gradient: every estimator
    ties at 5.00 and the comparison decides nothing. It stays as the honest worst case — what the
    score becomes if the benchmark is perfect — while selection uses the numerator below.
    """
    return min(SCORE_CAP, abs(pred - actual) / floor_of(unit, actual))


def normalised_error(pred: float, actual: float, unit: str) -> float:
    """The competition's numerator, made comparable across metrics. This is what is minimised.

    ⚡ The denominator of the real score is `max(Wall Street's miss, floor)` — unknown, and outside
    our control. It is also roughly fixed per metric, so ranking on the numerator alone ranks on the
    real thing. Expressed relatively for money and per-share lines and in points for rates, one
    number compares a Deere segment profit with a Hays EPS.
    """
    if unit == "percent":
        return abs(pred - actual)
    return abs(pred - actual) / max(abs(actual), 1e-9)


def to_level(anchor: float, residual: float, relative: bool) -> float:
    return anchor * (1.0 + residual) if relative else anchor + residual


def _matrix(rows: list[dict], columns: list[str], codes: dict) -> np.ndarray:
    out = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
    for j, column in enumerate(columns):
        if column in CATEGORICAL:
            table = codes[column]
            for i, row in enumerate(rows):
                out[i, j] = table.get(row.get(column), -1)
        else:
            for i, row in enumerate(rows):
                value = row.get(column)
                if isinstance(value, bool):
                    out[i, j] = float(value)
                elif isinstance(value, (int, float)) and value is not None:
                    out[i, j] = float(value)
    return out


def _fit(train: list[dict], columns: list[str], codes: dict, seed: int = 0):
    import lightgbm as lgb

    x = _matrix(train, columns, codes)
    y = np.array([r["residual"] for r in train], dtype=np.float64)
    cat_index = [columns.index(c) for c in CATEGORICAL if c in columns]
    # ⚠️ Deliberately small and heavily regularised. With a few thousand examples over 26 metrics a
    # default LightGBM memorises the panel; shallow trees with a high leaf minimum and column/row
    # subsampling force it to use the transferable columns instead.
    params = dict(objective="l1", num_leaves=7, max_depth=4, learning_rate=0.05,
                  n_estimators=300, min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=-1,
                  random_state=seed)
    model = lgb.LGBMRegressor(**params)
    model.fit(x, y, categorical_feature=cat_index)
    return model


def select_estimators(train: list[dict], columns: list[str], codes: dict) -> dict[str, str]:
    """Per anchor type, decide whether to correct it at all — on held-out data inside the window.

    ⛔ **A correction that helps on average can still ruin the anchors that matter.** Pooled over
    every anchor the GBM is clearly better (median normalised error **0.125** against **0.151**),
    but applied to the anchor a row would actually be forecast from it made things *worse*
    (**0.122** against **0.095**). The reason is visible per anchor type: the model earns its lift
    repairing weak anchors — `lag1` 1.47×, `prior_year` 1.45× — while degrading the already
    well-calibrated ones, turning the seasonal split's 0.0625 into 0.0990.

    ⚠️ **The choice cannot be made on the training residuals.** The GBM has seen them, so it wins by
    construction and every anchor would be "corrected". The window is therefore split again by date:
    fit on the earlier part, compare `raw`, `bias` and `gbm` on the later part, and keep the winner
    per anchor type. The outer fold stays untouched, so the reported score still measures a decision
    that was made without it.
    """
    dates = sorted({e["as_of"] for e in train})
    if len(dates) < 4:
        return {}
    edge = dates[int(len(dates) * 0.7)]
    inner_train = [e for e in train if e["as_of"] < edge]
    inner_test = [e for e in train if e["as_of"] >= edge]
    if len(inner_train) < MIN_TRAIN // 2 or not inner_test:
        return {}

    shift: dict[tuple, float] = {}
    for key, group in store.group_by(inner_train, "anchor_type", "anchor_is_relative").items():
        shift[key] = statistics.median([g["residual"] for g in group])
    model = _fit(inner_train, columns, codes)
    predicted = model.predict(_matrix(inner_test, columns, codes))

    scores: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for example, gbm_residual in zip(inner_test, predicted):
        relative = bool(example["anchor_is_relative"])
        anchor, actual, unit = example["anchor_value"], example["y_value"], example["unit"]
        options = (("raw", 0.0),
                   ("bias", shift.get((example["anchor_type"], example["anchor_is_relative"]), 0.0)),
                   ("gbm", float(gbm_residual)))
        for name, residual in options:
            scores[example["anchor_type"]][name].append(
                normalised_error(to_level(anchor, residual, relative), actual, unit))

    chosen = {}
    for anchor_type, options in scores.items():
        ranked = {k: statistics.median(v) for k, v in options.items() if v}
        chosen[anchor_type] = min(ranked, key=lambda k: ranked[k]) if ranked else "raw"
    return chosen


def walk_forward(examples: list[dict], columns: list[str], codes: dict) -> list[dict]:
    """Expanding-window folds cut on publication date, scored on the competition's own rule."""
    labelled = sorted((e for e in examples if e["residual"] is not None),
                      key=lambda e: (e["as_of"], e["label"]))
    dates = sorted({e["as_of"] for e in labelled})
    if len(dates) < N_FOLDS + 1:
        return []
    edges = [dates[int(len(dates) * (i + 1) / (N_FOLDS + 1))] for i in range(N_FOLDS)]

    folds: list[dict] = []
    # ⚡ **Which correction to trust is itself learned forward in time.** Picking the per-anchor
    # winner from the finished backtest would be selection on the very data the score is reported
    # on. Instead each fold decides using only folds already completed — the first fold has no
    # evidence and therefore corrects nothing, which is the right prior anyway.
    earned: dict[tuple, list[float]] = defaultdict(list)

    for cut, nxt in zip(edges, edges[1:] + [None]):
        train = [e for e in labelled if e["as_of"] < cut]
        test = [e for e in labelled
                if cut <= e["as_of"] and (nxt is None or e["as_of"] < nxt)]
        if len(train) < MIN_TRAIN or not test:
            continue

        progressive: dict[str, str] = {}
        for anchor_type in {e["anchor_type"] for e in test}:
            scored = {name: earned.get((anchor_type, name), [])
                      for name in ("raw", "bias", "gbm")}
            usable = {k: statistics.median(v) for k, v in scored.items() if len(v) >= 5}
            progressive[anchor_type] = min(usable, key=lambda k: usable[k]) if usable else "raw"

        # the bias baseline: this anchor type's own median residual, as known at the cut
        bias: dict[tuple, float] = {}
        for key, group in store.group_by(train, "anchor_type", "anchor_is_relative").items():
            bias[key] = statistics.median([g["residual"] for g in group])

        # ⚡ **Where in its own guided range does this filer land?** A median-bias shift ignores the
        # width the filer chose: applied to ADI's Q3 midpoint it lands at 4,008 against a stated
        # 3,800–4,000, i.e. above the top of a range ADI has cleared only 29 % of the time. The
        # landing *position* is scale-free and respects the filer's own stated uncertainty.
        positions: dict[bool, float] = {}
        for key, group in store.group_by(train, "anchor_is_relative").items():
            landed = [(g["y_value"] - g["guide_low"]) / (g["guide_high"] - g["guide_low"])
                      for g in group
                      if g["anchor_type"] == "guide_mid" and g.get("guide_low") is not None
                      and g.get("guide_high") is not None
                      and g["guide_high"] - g["guide_low"] > 1e-9]
            if len(landed) >= 5:
                positions[key[0]] = statistics.median(landed)

        # how reliable each anchor type has been, measured on the training window only
        dispersion: dict[tuple, float] = {}
        for key, group in store.group_by(train, "anchor_type", "anchor_is_relative").items():
            spread = statistics.median([abs(g["residual"]) for g in group])
            dispersion[key] = max(spread, 1e-4)
        # an anchor type first seen in the test window has no track record; fall back to its
        # residual family's typical spread rather than dropping the anchor or trusting it blindly
        family_default = {}
        for key, group in store.group_by(train, "anchor_is_relative").items():
            family_default[key[0]] = max(
                statistics.median([abs(g["residual"]) for g in group]), 1e-4)

        # ⚡ **The same anchor is not equally good everywhere.** A seasonal split is excellent for
        # Home Depot's Q2 and useless for Deere's segment profit. Where a metric has enough of its
        # own history the anchor is chosen on that metric's record; otherwise it falls back to the
        # pooled one, so a thin metric borrows strength instead of trusting four observations.
        per_metric: dict[tuple, float] = {}
        for key, group in store.group_by(train, "metric", "anchor_type").items():
            if len(group) >= MIN_METRIC_OBS:
                per_metric[key] = max(
                    statistics.median([abs(g["residual"]) for g in group]), 1e-4)

        def spread_of(example: dict) -> float:
            return dispersion.get((example["anchor_type"], example["anchor_is_relative"]),
                                  family_default.get(example["anchor_is_relative"], 1.0))

        def metric_spread_of(example: dict) -> float:
            return per_metric.get((example["metric"], example["anchor_type"]),
                                  spread_of(example))

        chosen = select_estimators(train, columns, codes)
        model = _fit(train, columns, codes)
        predicted = model.predict(_matrix(test, columns, codes))
        combos: dict[tuple, list] = defaultdict(list)

        for example, gbm_residual in zip(test, predicted):
            relative = bool(example["anchor_is_relative"])
            anchor, actual, unit = example["anchor_value"], example["y_value"], example["unit"]
            shifted = bias.get((example["anchor_type"], example["anchor_is_relative"]), 0.0)
            for name, residual in (("raw", 0.0), ("bias", shifted), ("gbm", float(gbm_residual))):
                prediction = to_level(anchor, residual, relative)
                earned[(example["anchor_type"], name)].append(
                    normalised_error(prediction, actual, unit))
                folds.append({
                    "cut": cut, "estimator": name,
                    "ticker": example["ticker"], "metric": example["metric"],
                    "label": example["label"], "unit": unit,
                    "anchor_type": example["anchor_type"],
                    "anchor_is_relative": example["anchor_is_relative"],
                    "n_train": len(train),
                    "actual": actual, "prediction": prediction,
                    "residual_pred": residual,
                    "abs_error": abs(prediction - actual),
                    "nae": normalised_error(prediction, actual, unit),
                    "score": competition_score(prediction, actual, unit),
                })
            key = (example["ticker"], example["metric"], example["label"])
            combos[key].append({
                "anchor_type": example["anchor_type"], "unit": unit, "actual": actual,
                "weight": 1.0 / spread_of(example) ** 2,
                "raw": anchor,
                "gbm": to_level(anchor, float(gbm_residual), relative),
                "bias": to_level(anchor, shifted, relative),
                "dispersion": spread_of(example),
                "metric_dispersion": metric_spread_of(example),
                "chosen": chosen.get(example["anchor_type"], "raw"),
                "selective": to_level(
                    anchor,
                    {"raw": 0.0, "bias": shifted,
                     "gbm": float(gbm_residual)}[chosen.get(example["anchor_type"], "raw")],
                    relative),
                "range_pos": (
                    example["guide_low"] + positions[example["anchor_is_relative"]]
                    * (example["guide_high"] - example["guide_low"])
                    if example["anchor_type"] == "guide_mid"
                    and example["anchor_is_relative"] in positions
                    and example.get("guide_low") is not None
                    and example.get("guide_high") is not None
                    and example["guide_high"] - example["guide_low"] > 1e-9
                    else None),
                "progressive_choice": progressive.get(example["anchor_type"], "raw"),
                "progressive": to_level(
                    anchor,
                    {"raw": 0.0, "bias": shifted,
                     "gbm": float(gbm_residual)}[progressive.get(example["anchor_type"], "raw")],
                    relative),
            })

        folds.extend(_combine(combos, cut, len(train)))
    return folds


def _combine(combos: dict, cut: str, n_train: int) -> list[dict]:
    """Blend the anchors for one row into a single number, and score that.

    ⚡ **Anchors are independent readings of the same quantity, so their disagreement is
    information.** Weighting by `1 / dispersion²` — the inverse variance of that anchor type's own
    residuals on the training window — gives the guide midpoint and the seasonal split far more say
    than a momentum carry, automatically and without a hand-set preference.

    ⛔ **A single wild anchor must not drag the blend.** The competition caps a metric at 5.0 while
    the best attainable is 0, so a large miss costs roughly eight times what an equivalent gain
    earns. Any anchor further than three median-absolute-deviations from the weighted centre is
    dropped before averaging — Deere's momentum carry put its segment profit at **357** against a
    prior year of **580**, and one such reading should not move the submitted number.
    """
    out = []
    for (ticker, metric, label), parts in combos.items():
        actual, unit = parts[0]["actual"], parts[0]["unit"]
        for name in ("raw", "gbm", "bias"):
            values = [(p[name], p["weight"]) for p in parts]
            centre = statistics.median([v for v, _w in values])
            deviations = [abs(v - centre) for v, _w in values]
            mad = statistics.median(deviations) or 0.0
            kept = [(v, w) for (v, w), d in zip(values, deviations)
                    if mad == 0.0 or d <= 3.0 * mad]
            if not kept:
                kept = values
            total = sum(w for _v, w in kept)
            blended = sum(v * w for v, w in kept) / total if total else centre
            out.append({
                "cut": cut, "estimator": f"combo_{name}", "ticker": ticker, "metric": metric,
                "label": label, "unit": unit, "anchor_type": "*combined*",
                "anchor_is_relative": None, "n_train": n_train,
                "actual": actual, "prediction": blended, "residual_pred": None,
                "abs_error": abs(blended - actual),
                "nae": normalised_error(blended, actual, unit),
                "score": competition_score(blended, actual, unit),
                "n_anchors_used": len(kept),
            })
        # ── selection rather than averaging ──────────────────────────────────────
        # The blends above lose to simply picking the most reliable anchor, because anchor quality
        # varies far more than blending can absorb. These variants sharpen that choice.
        pooled = min(parts, key=lambda p: p["dispersion"])
        specific = min(parts, key=lambda p: p["metric_dispersion"])
        ranked = sorted(parts, key=lambda p: p["metric_dispersion"])[:2]
        mass = sum(1.0 / p["metric_dispersion"] ** 2 for p in ranked)
        top2 = sum(p["raw"] / p["metric_dispersion"] ** 2 for p in ranked) / mass
        top2_gbm = sum(p["gbm"] / p["metric_dispersion"] ** 2 for p in ranked) / mass

        top2_sel = sum(p["selective"] / p["metric_dispersion"] ** 2 for p in ranked) / mass
        # ⚡ **Shrinking the chosen anchor toward the consensus of all anchors trades median for
        # tail.** Picking one anchor gives the sharpest typical forecast but inherits that anchor's
        # bad days; the blend is duller but its p90 is materially lower. The competition caps a
        # metric at 5.0 and averages twelve of them, so a fat tail is expensive — the mixing weight
        # is therefore measured rather than assumed.
        centre_all = statistics.median([p["raw"] for p in parts])
        shrunk = {a: a * specific["raw"] + (1.0 - a) * centre_all for a in SHRINK_GRID}
        for name, value, used, chosen in (
                ("best_single", pooled["raw"], 1, pooled["anchor_type"]),
                ("best_metric", specific["raw"], 1, specific["anchor_type"]),
                ("best_metric_gbm", specific["gbm"], 1, specific["anchor_type"]),
                ("best_metric_bias", specific["bias"], 1, specific["anchor_type"]),
                ("best_selective", specific["selective"], 1,
                 f"{specific['anchor_type']}:{specific['chosen']}"),
                ("best_progressive", specific["progressive"], 1,
                 f"{specific['anchor_type']}:{specific['progressive_choice']}"),
                # ⚡ **One targeted hypothesis, not a blanket correction.** Shifting every anchor by
                # its median residual is not significantly better (t = −1.18); but the *guide
                # midpoint* specifically has 123 guide→outcome pairs behind it and a filer's habit
                # of clearing its own guide is the single best-evidenced effect in this data — ADI
                # has beaten its revenue guide 88 % of the time by a median 2.5 %. Submitting the
                # midpoint scores ~1.0 by construction, because that is where the sell side sits,
                # so this is the one place a deliberate deviation can earn anything.
                ("best_guidebias",
                 specific["bias"] if specific["anchor_type"] == "guide_mid" else specific["raw"],
                 1, f"{specific['anchor_type']}"
                    f"{':bias' if specific['anchor_type'] == 'guide_mid' else ''}"),
                ("best_guiderange",
                 (specific["range_pos"] if specific["anchor_type"] == "guide_mid"
                  and specific.get("range_pos") is not None else specific["raw"]),
                 1, f"{specific['anchor_type']}"
                    f"{':range' if specific['anchor_type'] == 'guide_mid' else ''}"),
                ("top2", top2, len(ranked), "+".join(p["anchor_type"] for p in ranked)),
                ("top2_gbm", top2_gbm, len(ranked), "+".join(p["anchor_type"] for p in ranked)),
                ("top2_selective", top2_sel, len(ranked),
                 "+".join(f"{p['anchor_type']}:{p['chosen']}" for p in ranked)),
                *((f"shrunk_{a:.2f}", v, len(parts), specific["anchor_type"])
                  for a, v in shrunk.items())):
            out.append({
                "cut": cut, "estimator": name, "ticker": ticker, "metric": metric,
                "label": label, "unit": unit, "anchor_type": chosen,
                "anchor_is_relative": None, "n_train": n_train,
                "actual": actual, "prediction": value, "residual_pred": None,
                "abs_error": abs(value - actual),
                "nae": normalised_error(value, actual, unit),
                "score": competition_score(value, actual, unit),
                "n_anchors_used": used,
            })
    return out


def main() -> int:
    examples = store.read(MODEL / "anchor_examples.parquet")
    columns = [c for c in examples[0] if c not in DROP]
    codes = {c: {v: i for i, v in enumerate(sorted({str(e.get(c)) for e in examples}))}
             for c in CATEGORICAL}
    for example in examples:
        for column in CATEGORICAL:
            example[column] = str(example.get(column))

    folds = walk_forward(examples, columns, codes)
    store.write(MODEL / "cv_folds.parquet", folds)
    if not folds:
        print("not enough history to walk forward")
        return 1

    print(f"walk-forward: {len({f['cut'] for f in folds})} folds, "
          f"{len(columns)} features, {len(examples):,} examples\n")
    by_estimator = store.group_by(folds, "estimator")
    raw = {(f["label"], f["anchor_type"]): f["nae"] for f in by_estimator[("raw",)]}
    print("per-anchor estimators (one prediction per anchor)")
    print(f"{'estimator':<14}{'n':>8}{'median NAE':>12}{'p75':>9}{'p90':>9}"
          f"{'beat raw':>10}{'capped @5':>11}")
    print("-" * 73)

    for name in ("raw", "bias", "gbm"):
        group = by_estimator[(name,)]
        errors = sorted(f["nae"] for f in group)
        pick = lambda p: errors[int(p * (len(errors) - 1))]
        wins = sum(1 for f in group if f["nae"] < raw.get((f["label"], f["anchor_type"]), 9e9))
        capped = sum(1 for f in group if f["score"] >= SCORE_CAP)
        print(f"{name:<14}{len(group):>8,}{statistics.median(errors):>12.4f}"
              f"{pick(0.75):>9.4f}{pick(0.90):>9.4f}{wins / len(group):>10.0%}"
              f"{capped / len(group):>11.0%}")

    print("\nblended — one prediction per row, which is what gets submitted")
    print(f"{'estimator':<14}{'rows':>8}{'median NAE':>12}{'p75':>9}{'p90':>9}"
          f"{'mean score':>12}{'capped @5':>11}")
    print("-" * 75)
    for name in ("best_single", "best_metric", "best_metric_gbm", "best_metric_bias",
                 "best_selective", "best_progressive", "best_guidebias", "best_guiderange", "top2", "top2_gbm", "top2_selective",
                 "combo_raw", "combo_bias", "combo_gbm",
                 *(f"shrunk_{a:.2f}" for a in SHRINK_GRID)):
        group = by_estimator.get((name,), [])
        if not group:
            continue
        errors = sorted(f["nae"] for f in group)
        pick = lambda p: errors[int(p * (len(errors) - 1))]
        capped = sum(1 for f in group if f["score"] >= SCORE_CAP)
        print(f"{name:<14}{len(group):>8,}{statistics.median(errors):>12.4f}"
              f"{pick(0.75):>9.4f}{pick(0.90):>9.4f}"
              f"{statistics.fmean([f['score'] for f in group]):>12.3f}"
              f"{capped / len(group):>11.0%}")

    print("\nper anchor type — median normalised error out of sample, lower is better")
    print(f"  {'anchor':<22}{'n':>6}{'raw':>9}{'bias':>9}{'gbm':>9}  best   lift")
    print("  " + "-" * 68)
    skill: list[dict] = []
    for anchor in sorted({f["anchor_type"] for f in folds
                          if f["estimator"] in ("raw", "bias", "gbm")}):
        cell = {}
        for name in ("raw", "bias", "gbm"):
            errors = [f["nae"] for f in folds
                      if f["anchor_type"] == anchor and f["estimator"] == name]
            cell[name] = statistics.median(errors) if errors else float("nan")
        n = sum(1 for f in folds if f["anchor_type"] == anchor and f["estimator"] == "raw")
        best = min(cell, key=lambda k: cell[k])
        lift = cell["raw"] / cell[best] if cell[best] else 1.0
        skill.append({"anchor_type": anchor, "n": n,
                      **{f"nae_{k}": v for k, v in cell.items()},
                      "best": best, "lift_vs_raw": lift})
        print(f"  {anchor:<22}{n:>6}{cell['raw']:>9.4f}{cell['bias']:>9.4f}"
              f"{cell['gbm']:>9.4f}  {best:<6}{lift:>6.2f}×")
    store.write(MODEL / "anchor_skill.parquet", skill)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
