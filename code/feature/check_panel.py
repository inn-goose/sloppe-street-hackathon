"""Does the panel return the RIGHT value for a known (metric, period)? The gate before modelling.

    PYTHONPATH=. .venv/bin/python -m code.feature.check_panel

`metric_panel` reconciles several witnesses into one number. That reconciliation is a *decision*,
so it needs the same treatment the extraction got: ground truths read by hand from the filings,
asserted against the panel's chosen value — not against "a value exists somewhere in the key".

Every case below is a **prior-year comparable for one of the twelve targets**, i.e. exactly the
base each forecast bridges off. An error here propagates straight into a submitted number.
"""

from __future__ import annotations

from code.lib import config, store

#: (metric, FY label, expected value in the registry's unit, tolerance)
CASES = (
    ("hd_net_sales", "FY2025Q2", 45277.0, 1.0),
    ("hd_net_sales", "FY2026Q1", 41765.0, 1.0),
    ("hd_adj_diluted_eps", "FY2025Q2", 4.68, 0.005),
    ("hd_adj_diluted_eps", "FY2026Q1", 3.43, 0.005),
    ("hd_comparable_sales", "FY2025Q2", 1.0, 0.05),
    ("hd_comparable_sales", "FY2026Q1", 0.6, 0.05),
    ("adi_revenue", "FY2026Q2", 3623.0, 1.0),
    ("adi_revenue", "FY2025Q3", 2880.0, 2.0),
    ("adi_adj_diluted_eps", "FY2026Q2", 3.09, 0.005),
    ("adi_adj_gross_margin", "FY2026Q2", 73.0, 0.05),
    ("de_net_sales_and_revenues", "FY2026Q2", 13369.0, 1.0),
    ("de_diluted_eps", "FY2026Q2", 6.55, 0.005),
    ("de_ppa_operating_profit", "FY2026Q2", 706.0, 1.0),
    ("has_net_fees", "FY2025FY", 972.4, 0.1),
    ("has_pre_exc_operating_profit", "FY2025FY", 45.6, 0.1),
    ("has_pre_exc_basic_eps", "FY2025FY", 1.31, 0.005),
)


def main() -> int:
    rows = {(r["metric"], r["label"]): r
            for r in store.read(config.FEATURE / "metric_panel.parquet")}

    print(f"{'metric':<34}{'period':<11}{'expected':>11}{'panel':>12}"
          f"{'wit':>5}{'agree':>7}{'lanes':<18}verdict")
    print("-" * 112)
    passed = failed = 0
    misses = []
    for metric, label, expected, tol in CASES:
        row = rows.get((metric, label))
        if row is None:
            failed += 1
            misses.append((metric, label, "key absent from the panel", None))
            print(f"{metric:<34}{label:<11}{expected:>11,.2f}{'—':>12}{'':>5}{'':>7}{'':<18}MISS")
            continue
        got = row["value"]
        ok = abs(got - expected) <= tol
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        if not ok:
            misses.append((metric, label, f"panel says {got:,.4g}", row))
        print(f"{metric:<34}{label:<11}{expected:>11,.2f}{got:>12,.3f}"
              f"{row['n_witnesses']:>5}{row['consensus_share']:>7.0%}"
              f"  {row['lanes']:<16}{'PASS' if ok else 'MISS'}")
    print("-" * 112)
    print(f"{passed}/{passed + failed} panel values match the filings")
    if misses:
        print("\nmisses:")
        for metric, label, why, row in misses:
            extra = (f"  distinct={row['n_distinct']} spread={row['spread']:.1%} "
                     f"min={row['min']:,.4g} max={row['max']:,.4g}") if row else ""
            print(f"  {metric:<32}{label:<10}{why}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
