"""extracted/nq_* — the revision panel, the realised surprise table, and quarterly statements.

## The leg nothing else in the store carries

`quarterlyForecast.rows` gives, per fiscal quarter: `consensusEPSForecast`, `highEPSForecast`,
`lowEPSForecast`, `noOfEstimates`, and — uniquely — **`up` and `down`, the number of revisions in
the last four weeks**. stockanalysis and Yahoo both give a consensus *level*; only this gives its
**drift**, which is the §C revision-diffusion family and the best-documented pre-print predictor
of the surprise's sign.

⚠️ **Money arrives as a formatted string** (`"$41,765,000"`, `"(1,234)"`), and a thousands
separator inside quotes is exactly what a naive `float()` throws on. Parsed with an explicit
reader that keeps accounting negatives.

⚠️ **`fiscalEnd` is a month label** (`"Jul 2026"`), not a date, and it is the filer's fiscal
quarter — HD's "Jul 2026" quarter actually ends 2026-08-02. It is stored verbatim; matching it to
a period is the feature layer's job, on the calendar rather than on the label.
"""

from __future__ import annotations

import re
from collections import Counter

from code.lib import config, rawstore, store

_MONEY = re.compile(r"^\(?\s*-?\$?\s*[\d,]+(?:\.\d+)?\s*\)?%?$")


def _num(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in ("--", "N/A", "NA", ""):
        return None
    if not _MONEY.match(text):
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^\d.\-]", "", text)
    if not cleaned or cleaned in ("-", "."):
        return None
    try:
        out = float(cleaned)
    except ValueError:
        return None
    return -out if negative else out


def build() -> dict[str, list[dict]]:
    lanes: dict[str, list[dict]] = {"forecast": [], "surprise": [], "statements": []}
    for meta, envelope in rawstore.iter_captures("nasdaq"):
        # ⚠️ The payload nests under `data`; the envelope's siblings are `message` and `status`,
        # and `status` is where this API hides absence inside a 200.
        body = (envelope or {}).get("data") or {}
        base = {"symbol": meta["symbol"], "for_ticker": meta.get("for_ticker") or "",
                "captured_at": meta.get("fetched_at") or ""}
        product = meta["product"]

        if product == "earnings_forecast":
            for grain in ("quarterlyForecast", "annualForecast", "yearlyForecast"):
                block = body.get(grain) or {}
                for row in block.get("rows") or []:
                    if not isinstance(row, dict):
                        continue
                    lanes["forecast"].append({
                        **base, "grain": grain, "fiscal_end": row.get("fiscalEnd"),
                        "consensus_eps": _num(row.get("consensusEPSForecast")),
                        "high_eps": _num(row.get("highEPSForecast")),
                        "low_eps": _num(row.get("lowEPSForecast")),
                        "n_estimates": _num(row.get("noOfEstimates")),
                        "revisions_up_4w": _num(row.get("up")),
                        "revisions_down_4w": _num(row.get("down")),
                        "as_of": block.get("asOf")})

        elif product == "earnings_surprise":
            for row in ((body.get("earningsSurpriseTable") or {}).get("rows") or []):
                if isinstance(row, dict):
                    lanes["surprise"].append({
                        **base, "fiscal_quarter_end": row.get("fiscalQtrEnd"),
                        "date_reported": row.get("dateReported"),
                        "eps_actual": _num(row.get("eps")),
                        "eps_consensus": _num(row.get("consensusForecast")),
                        "surprise_pct": _num(row.get("percentageSurprise"))})

        elif product == "financials_quarterly":
            for table in ("incomeStatementTable", "balanceSheetTable", "cashFlowTable",
                          "financialRatiosTable"):
                block = body.get(table) or {}
                headers = block.get("headers") or {}
                for row in block.get("rows") or []:
                    if not isinstance(row, dict):
                        continue
                    label = str(row.get("value1") or "").strip()
                    if not label:
                        continue
                    for key, value in row.items():
                        if key == "value1":
                            continue
                        num = _num(value)
                        if num is None:
                            continue
                        lanes["statements"].append({
                            **base, "table": table, "label": label,
                            "period_end": headers.get(key), "value": num})
    return lanes


def main() -> int:
    lanes = build()
    for name, rows in lanes.items():
        store.write(config.EXTRACTED / f"nq_{name}.parquet", rows)
        print(f"extracted/nq_{name + '.parquet':<22}{len(rows):>7,} rows")
    print("\n  revision panel for the target quarters:")
    for row in sorted(lanes["forecast"], key=lambda r: (r["symbol"], str(r["fiscal_end"]))):
        if row["for_ticker"] == row["symbol"] and row["grain"] == "quarterlyForecast":
            print(f"    {row['symbol']:<6}{str(row['fiscal_end']):<10}"
                  f"cons {row['consensus_eps']}  "
                  f"[{row['low_eps']}–{row['high_eps']}]  n={row['n_estimates']:.0f}  "
                  f"up {row['revisions_up_4w']:.0f} / down {row['revisions_down_4w']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
