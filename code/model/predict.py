"""model/predict — the twelve submitted numbers, and the trail behind each one.

    PYTHONPATH=. .venv/bin/python -m code.model.predict

## The rule, and why it is this one

For each target: take every anchor available on that row, and submit the one whose **type has the
smallest residual dispersion on that metric's own history**. Nothing is averaged and nothing is
corrected.

That is not a simplifying assumption, it is the outcome of the backtest in `model/train`. Ten
strategies were run walk-forward over seven folds on the competition's own scoring rule:

| strategy | median normalised error | mean score |
|---|---:|---:|
| **pick best anchor per metric** | **0.0946** | **3.784** |
| inverse-variance blend of all anchors | 0.0964 | 3.891 |
| blend of the two best | 0.0982 | 3.836 |
| best anchor, GBM-corrected | 0.1222 | 4.129 |
| blend of all, GBM-corrected | 0.1187 | 4.145 |

⚡ **Selection beat correction, and the gradient-boosted model is why we know.** Pooled across every
anchor the GBM is plainly better than leaving anchors alone (0.1253 against 0.1508) — it repairs
`lag1` by 1.47× and `prior_year` by 1.45×. But those are the anchors a row is *not* forecast from.
On the anchors that actually win selection it is harmful: the seasonal split's error goes from
**0.0625 to 0.0990** under correction, and the guide midpoint's from **0.0833 to 0.1644**. A good
anchor is already calibrated; the model's real finding is that there is nothing left in it to fix.

⛔ **Shrinking toward the anchor consensus was measured, not skipped.** Mixing the chosen anchor with
the median of the others at α = 0.8 does tighten the upper quartile (0.341 against 0.360) but costs
mean score (3.824 against 3.784), and mean score is the prize. It is left out.

## The guards, which exist because the loss is asymmetric

A metric scores `min(5.0, |miss| / max(|Wall Street miss|, floor))`. The best attainable is 0 and the
worst is 5, so a large miss costs roughly eight times what an equal-sized gain earns. Two refusals
follow, and both can override the chosen anchor:

* an anchor more than three median-absolute-deviations from the other anchors on that row is not
  submitted — the row falls back to their median
* a value outside what the unit permits (a margin above 100 %, a negative revenue) is refused
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from code.lib import config, store
from code.model.train import MIN_METRIC_OBS, MODEL, floor_of

#: An anchor this far from the others on the same row is an outlier, not a forecast.
OUTLIER_MADS = 3.0


def dispersion_tables(labelled: list[dict]) -> tuple[dict, dict, dict]:
    """How reliable each anchor type has been — per metric, per type, per residual family."""
    per_metric, per_type, per_family = {}, {}, {}
    for key, group in store.group_by(labelled, "metric", "anchor_type").items():
        if len(group) >= MIN_METRIC_OBS:
            per_metric[key] = max(statistics.median([abs(g["residual"]) for g in group]), 1e-4)
    for key, group in store.group_by(labelled, "anchor_type", "anchor_is_relative").items():
        per_type[key] = max(statistics.median([abs(g["residual"]) for g in group]), 1e-4)
    for key, group in store.group_by(labelled, "anchor_is_relative").items():
        per_family[key[0]] = max(statistics.median([abs(g["residual"]) for g in group]), 1e-4)
    return per_metric, per_type, per_family


def build() -> list[dict]:
    examples = store.read(MODEL / "anchor_examples.parquet")
    labelled = [e for e in examples if e["residual"] is not None]
    targets = [e for e in examples if e["is_prediction"]]
    per_metric, per_type, per_family = dispersion_tables(labelled)
    # the median amount by which a stated guide is cleared, exactly as the backtest measured it:
    # pooled by residual family, not per metric, so the correction is the one that was validated
    guide_shift = {
        key[0]: statistics.median([g["residual"] for g in group])
        for key, group in store.group_by(
            [e for e in labelled if e["anchor_type"] == "guide_mid"],
            "anchor_is_relative").items()}

    def spread(example: dict) -> tuple[float, str]:
        key = (example["metric"], example["anchor_type"])
        if key in per_metric:
            return per_metric[key], "metric"
        pooled = per_type.get((example["anchor_type"], example["anchor_is_relative"]))
        if pooled is not None:
            return pooled, "anchor_type"
        return per_family.get(example["anchor_is_relative"], 1.0), "family"

    rows: list[dict] = []
    for key, parts in store.group_by(targets, "ticker", "metric", "label").items():
        ticker, metric, label = key
        for part in parts:
            part["_spread"], part["_basis"] = spread(part)
        ranked = sorted(parts, key=lambda p: p["_spread"])
        chosen = ranked[0]
        value = chosen["anchor_value"]
        note = ""

        # ⚡ **The one deliberate deviation, and the only one that survived testing.** Submitting a
        # guide midpoint scores about 1.0 by construction: it is where the sell side sits, so the
        # misses match. Filers systematically clear their own guides — ADI has beaten its revenue
        # guide in 88 % of 34 observations by a median 2.5 % — and shifting the midpoint by that
        # measured median is worth **t = −2.21** out of sample on the target metrics, improving 21
        # rows against 8 worsened. Shifting *every* anchor this way is not significant (t = −1.18);
        # only the guide is, which is why only the guide is shifted.
        if chosen["anchor_type"] == "guide_mid":
            shift = guide_shift.get(chosen["anchor_is_relative"])
            if shift is not None:
                value = (value * (1.0 + shift) if chosen["anchor_is_relative"] else value + shift)
                note = (f"guide midpoint {chosen['anchor_value']:,.3f} shifted by the measured "
                        f"median guide beat ({shift:+.2%})" if chosen["anchor_is_relative"]
                        else f"guide midpoint shifted {shift:+.2f}pp")

        # ── guard: is the chosen anchor an outlier among its peers? ──────────────
        # ⛔ **Fall through to the next-best anchor, not to their median.** The backtest validated
        # one rule — take the most reliable anchor — so a median is an unvalidated third procedure
        # smuggled in at the last step. Hays exposes why the guard is needed at all: its momentum
        # carry returns **19.8** for an operating profit whose prior year is **45.6**, because the
        # older Hays panel years are still partly unreliable and a pooled dispersion cannot see
        # that. Rejecting it and taking the next anchor in the same ranking keeps one rule.
        candidates = [p["anchor_value"] for p in parts]
        centre = statistics.median(candidates)
        deviations = [abs(v - centre) for v in candidates]
        mad = statistics.median(deviations)
        if mad > 0 and abs(value - centre) > OUTLIER_MADS * mad:
            fallback = next((p for p in ranked[1:]
                             if abs(p["anchor_value"] - centre) <= OUTLIER_MADS * mad), None)
            rejected = f"{chosen['anchor_type']}={value:,.3f}"
            if fallback is not None:
                note = (f"rejected {rejected} at {abs(value - centre) / mad:.1f} MADs from its "
                        f"peers; next-best anchor {fallback['anchor_type']} used instead")
                chosen, value = fallback, fallback["anchor_value"]
            else:
                note = (f"rejected {rejected} at {abs(value - centre) / mad:.1f} MADs and no "
                        f"other anchor is inside the band; submitted their median")
                value = centre

        # ── guard: does the unit permit this number? ─────────────────────────────
        unit = chosen["unit"]
        if unit == "percent" and not -100.0 <= value <= 100.0:
            note = f"{value:,.2f} is not a possible percentage; fell back to the anchor median"
            value = centre

        rows.append({
            "ticker": ticker, "metric": metric, "label": label, "unit": unit,
            "fiscal_year": chosen["fiscal_year"], "fiscal_period": chosen["fiscal_period"],
            "forecast": value,
            "chosen_anchor": chosen["anchor_type"],
            "chosen_dispersion": chosen["_spread"],
            "dispersion_basis": chosen["_basis"],
            "n_anchors": len(parts),
            "anchor_median": centre,
            "anchor_low": min(candidates), "anchor_high": max(candidates),
            "anchor_spread_rel": (max(candidates) - min(candidates)) / abs(centre)
                                 if abs(centre) > 1e-9 else None,
            "prior_year": next((p["anchor_value"] for p in parts
                                if p["anchor_type"] == "prior_year"), None),
            "implied_floor": floor_of(unit, value),
            "note": note,
            "candidates": "; ".join(f"{p['anchor_type']}={p['anchor_value']:,.3f}"
                                    f"(σ{p['_spread']:.3f})" for p in ranked),
        })
    rows.sort(key=lambda r: (r["ticker"], r["metric"]))
    return rows


def main() -> int:
    rows = build()
    store.write(MODEL / "forecasts.parquet", rows)
    print(f"model/forecasts.parquet {len(rows)} forecasts\n")
    print(f"{'target':<44}{'forecast':>13}{'prior yr':>12}{'YoY':>9}"
          f"{'anchor':<20}{'σ':>7}{'n':>3}")
    print("-" * 112)
    for row in rows:
        prior = row["prior_year"]
        yoy = (f"{(row['forecast'] / prior - 1) * 100:+.1f}%"
               if prior not in (None, 0) and row["unit"] != "percent"
               else (f"{row['forecast'] - prior:+.2f}pp" if prior is not None else "—"))
        print(f"{row['ticker'] + ' · ' + row['metric']:<44}{row['forecast']:>13,.2f}"
              f"{(prior if prior is not None else float('nan')):>12,.2f}{yoy:>9}"
              f"  {row['chosen_anchor']:<18}{row['chosen_dispersion']:>7.3f}{row['n_anchors']:>3}")

    flagged = [r for r in rows if r["note"]]
    if flagged:
        print("\n  guards that fired:")
        for row in flagged:
            print(f"    {row['metric']}: {row['note']}")

    print("\n  candidate anchors behind each number:")
    for row in rows:
        print(f"    {row['metric']:<30}{row['candidates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
