"""The metric registry — one canonical name per measure, and how each lane spells it.

The second half of the join key. `(ticker, fiscal_year, fiscal_period)` says *when*; this says
*what*, and without it four lanes carrying the same number never meet: the corpus writes
`Net sales`, stockanalysis writes `revenue`, SEC writes
`RevenueFromContractWithCustomerExcludingAssessedTax`, Yahoo writes `totalRevenue`.

## Scope: only what the twelve targets need

This is deliberately not a chart of accounts. Every entry is either **a submitted number** or a
term in the bridge that produces one — the EPS denominator, the margin that backs into adjusted
gross margin, the conversion rate that turns Hays' net fees into operating profit. A registry that
tried to canonicalise all 1,819 distinct Deere labels would be a week's work and would not move a
single forecast.

## Three rules that make an entry correct

⛔ **BASIS is part of the identity, never a variant.** `adj_diluted_eps` and `diluted_eps` are
different metrics, not two spellings of one. Five of the twelve targets are non-GAAP, vendors
publish GAAP, and reading one as the other is invisible because both numbers are real. Every entry
declares its basis and refuses lanes that cannot supply it — which is why `sa` and `sec` are empty
on every adjusted row.

⛔ **SCOPE is part of the identity.** Deere states `Operating profit` for three segments in one
release. `de_ppa_operating_profit` therefore carries a required `scope` pattern, and a fact whose
table caption does not match it is not this metric.

⛔ **A pattern is anchored.** `^operating profit$` — not a substring test. `Operating profit` and
`Financial services operating profit` are different lines, and a loose pattern silently prefers
whichever appears first.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Metric:
    name: str
    ticker: str
    unit: str                    # currency_m | per_share | percent | shares_m | ratio
    basis: str                   # gaap | adjusted | kpi | segment
    is_target: bool = False
    workbook_label: str = ""
    #: per-lane anchored patterns; "" means the lane cannot supply this metric
    corpus: str = ""
    prose: str = ""
    sa: str = ""
    sec: str = ""
    yahoo: str = ""
    #: required table caption for a segment measure, and the section it must not sit under
    scope: str = ""
    section_veto: str = (r"outside\s+the\s+u\.?s|united\s+states|canada|europe|asia"
                         r"|latin\s+america|geograph|by\s+(?:region|country|product|market)")
    notes: str = ""


_M: list[Metric] = [
    # ───────────────────────────────── Home Depot ─────────────────────────────────
    Metric("hd_net_sales", "HD", "currency_m", "gaap", True, "Net sales",
           corpus=r"^net sales$", prose=r"^(?:total\s+)?sales$|^net sales$",
           sa=r"^revenue$",
           sec=r"^(?:Revenues|RevenueFromContractWithCustomerExcludingAssessedTax)$",
           yahoo=r"^totalRevenue$"),
    Metric("hd_adj_diluted_eps", "HD", "per_share", "adjusted", True, "Adjusted diluted EPS",
           corpus=r"^adjusted diluted earnings per share(?: \(non-gaap\))?$",
           prose=r"^adjusted diluted earnings per share$",
           notes="Table form exists only from FY2024; the prose states it every quarter and is "
                 "the series. No vendor publishes it."),
    Metric("hd_comparable_sales", "HD", "percent", "kpi", True,
           "Comparable sales, total company",
           corpus=r"^comparable sales(?: \(% change\))?$",
           prose=r"^(?:total\s+)?comparable sales$",
           notes="The table row mis-binds when its change columns read N/A; the prose lane is the "
                 "reliable one and covers 25 releases."),
    Metric("hd_diluted_eps", "HD", "per_share", "gaap", corpus=r"^diluted earnings per share$",
           prose=r"^diluted earnings per share$", sa=r"^epsdil$",
           sec=r"^EarningsPerShareDiluted$"),
    Metric("hd_diluted_shares", "HD", "shares_m", "gaap",
           corpus=r"^diluted weighted average common shares$", sa=r"^sharesDiluted$",
           sec=r"^WeightedAverageNumberOfDilutedSharesOutstanding$",
           notes="The EPS denominator — buybacks move adjusted EPS independently of profit."),
    Metric("hd_gross_margin", "HD", "percent", "gaap", corpus=r"^gross margin$", sa=r"^grossMargin$"),
    Metric("hd_operating_income", "HD", "currency_m", "gaap", corpus=r"^operating income$",
           sa=r"^opinc$", sec=r"^OperatingIncomeLoss$"),

    # ───────────────────────────────── Analog Devices ─────────────────────────────
    Metric("adi_revenue", "ADI", "currency_m", "gaap", True, "Revenue",
           corpus=r"^revenue$", sa=r"^revenue$",
           sec=r"^(?:Revenues|RevenueFromContractWithCustomerExcludingAssessedTax)$",
           yahoo=r"^totalRevenue$"),
    Metric("adi_adj_diluted_eps", "ADI", "per_share", "adjusted", True, "Adjusted diluted EPS",
           corpus=r"^adjusted diluted earnings per share$",
           prose=r"^adjusted diluted earnings per share$"),
    # ⛔ ADI states BOTH `Adjusted gross margin` ($2,645 m) and `Adjusted gross margin percentage`
    # (73.0 %) in the same table. A pattern admitting the first put a dollar figure into a percent
    # metric and the panel returned 2,645,262 for a target whose true value is 73.0.
    Metric("adi_adj_gross_margin", "ADI", "percent", "adjusted", True, "Adjusted gross margin",
           corpus=r"^adjusted gross margin percentage$",
           notes="⛔ NOT guided. ADI guides adjusted OPERATING margin, so this target is reached "
                 "through the opex bridge below — the single derived number in the submission."),
    Metric("adi_adj_operating_margin", "ADI", "percent", "adjusted",
           corpus=r"^adjusted operating margin(?: percentage)?$",
           notes="The guided quantity that backs into adjusted gross margin."),
    Metric("adi_adj_gross_profit", "ADI", "currency_m", "adjusted",
           corpus=r"^adjusted gross margin$"),
    Metric("adi_adj_operating_income", "ADI", "currency_m", "adjusted",
           corpus=r"^adjusted operating income$"),
    Metric("adi_diluted_eps", "ADI", "per_share", "gaap", corpus=r"^diluted earnings per share$",
           sa=r"^epsdil$", sec=r"^EarningsPerShareDiluted$"),
    Metric("adi_gross_margin", "ADI", "percent", "gaap",
           corpus=r"^gross margin percentage$", sa=r"^grossMargin$"),
    Metric("adi_diluted_shares", "ADI", "shares_m", "gaap", sa=r"^sharesDiluted$",
           sec=r"^WeightedAverageNumberOfDilutedSharesOutstanding$"),

    # ───────────────────────────────── Hays plc ───────────────────────────────────
    # ⚠️ Hays writes the same line four ways — `Net fees`, `Net Fees`, `Net fees (£m)` and
    # `Net fees {(1)}` (a footnote marker that survives the converter). An exact pattern on one
    # spelling split the series into fragments; measured, the panel saw a single period.
    Metric("has_net_fees", "LSE:HAS", "currency_m", "kpi", True, "Net fees",
           corpus=r"^net fees(?:\s*\(£m\))?$", prose=r"^(?:group\s+)?net fees$",
           notes="Annual in the accounts; the quarterly trading updates state GROWTH, not level, "
                 "which is why the growth chain below exists."),
    Metric("has_net_fee_growth_actual", "LSE:HAS", "percent", "kpi",
           corpus=r"^total$",
           notes="The Total row of a quarterly update's YoY-growth table, Actual basis. Actual "
                 "carries FX and the six-country disposal; LFL strips both, and the level bridge "
                 "needs actual."),
    # ⛔ **`(before exceptional items)` and `(after exceptional items)` are opposite facts** and
    # Hays states both, one line apart, plus a bare `Operating profit` and a
    # `Pre-exceptional operating profit`. The pattern must admit the three pre-exceptional
    # spellings and refuse the post-exceptional one — reading the wrong one is a ~£30 m error on
    # a ~£45 m number, and both are real.
    Metric("has_pre_exc_operating_profit", "LSE:HAS", "currency_m", "adjusted", True,
           "Pre-exceptional operating profit",
           corpus=r"^(?:pre-exceptional operating profit"
                  r"|operating profit \(before exceptional items\)"
                  r"|operating profit)$",
           prose=r"pre-exceptional operating profit"),
    Metric("has_pre_exc_basic_eps", "LSE:HAS", "per_share", "adjusted", True,
           "Pre-exceptional basic EPS",
           corpus=r"^(?:pre-exceptional basic earnings per share"
                  r"|basic earnings per share \(before exceptional items\))$"),
    Metric("has_conversion_rate", "LSE:HAS", "percent", "kpi", corpus=r"^conversion rate$",
           notes="Net fees → pre-exceptional operating profit. The bridge between two targets."),
    Metric("has_pbt_pre_exc", "LSE:HAS", "currency_m", "adjusted",
           corpus=r"^profit before tax \(before exceptional items\)$"),
    Metric("has_headcount", "LSE:HAS", "shares_m", "kpi",
           corpus=r"^period-end consultant headcount$",
           notes="Consultants are the capacity that generates fees; stated every quarter."),

    # ───────────────────────────────── Deere & Company ────────────────────────────
    Metric("de_net_sales_and_revenues", "DE", "currency_m", "gaap", True,
           "Worldwide net sales and revenues",
           corpus=r"^(?:total )?net sales and revenues$", prose=r"^worldwide net sales and revenues$",
           sa=r"^revenue$", sec=r"^Revenues$", yahoo=r"^totalRevenue$"),
    Metric("de_diluted_eps", "DE", "per_share", "gaap", True, "Diluted EPS (GAAP)",
           corpus=r"^(?:fully )?diluted eps$|^diluted earnings per share$",
           prose=r"^diluted earnings per share$", sa=r"^epsdil$",
           sec=r"^EarningsPerShareDiluted$"),
    Metric("de_ppa_operating_profit", "DE", "currency_m", "segment", True,
           "Production & Precision Ag operating profit",
           corpus=r"^operating profit$",
           scope=r"production\s*&?\s*(?:and\s*)?precision\s+ag",
           notes="⛔ Deere states `Operating profit` for three segments in one release — 706, 719 "
                 "and 561 in Q2 FY2026. Only the table caption separates them."),
    Metric("de_ppa_net_sales", "DE", "currency_m", "segment",
           corpus=r"^net sales$", scope=r"production\s*&?\s*(?:and\s*)?precision\s+ag",
           notes="The segment's own top line — operating profit is forecast through its margin."),
    Metric("de_net_income", "DE", "currency_m", "gaap",
           corpus=r"^net income attributable to deere & company$|^net income$",
           sa=r"^netinc$", sec=r"^NetIncomeLoss$",
           notes="Deere guides FY net income, so the EPS target is reached through this."),
    Metric("de_diluted_shares", "DE", "shares_m", "gaap", sa=r"^sharesDiluted$",
           sec=r"^WeightedAverageNumberOfDilutedSharesOutstanding$"),
]

#: ⛔ **A consolidated metric must refuse a row that belongs to an operating segment**, and the
#: segment is named in the table's caption or its section heading rather than in the row label.
#: Measured: Hays' `Net fees` returned **308.9** against a true group figure of **972.4** because
#: the divisional table repeats the same row label under Germany, UK & Ireland, ANZ and Rest of
#: World; its operating profit returned 52.1 against 45.6 the same way. These are the filers' own
#: segment names, taken from their own captions.
SEGMENT_NAMES = {
    "LSE:HAS": (r"germany|united\s+kingdom|uk\s*&\s*i|ireland|australia|new\s+zealand"
                r"|rest\s+of\s+world|\banz\b|\brow\b|temp|perm|contracting|enterprise"
                r"|france|spain|switzerland|japan|china|poland|americas|asia|emea"),
    "DE": (r"production\s*&?\s*(?:and\s*)?precision\s+ag|small\s+ag|agriculture\s*&\s*turf"
           r"|construction\s*&\s*forestry|financial\s+services|equipment\s+operations"),
    "HD": r"\bsrs\b|home\s+depot\s+segment|supply\b",
    "ADI": r"industrial|automotive|communications|consumer",
}

REGISTRY: dict[str, Metric] = {m.name: m for m in _M}
TARGETS: list[Metric] = [m for m in _M if m.is_target]
BY_TICKER: dict[str, list[Metric]] = {}
for _m in _M:
    BY_TICKER.setdefault(_m.ticker, []).append(_m)

#: What a value is divided by to reach the registry's unit, per lane.
#: Corpus `value_scaled`, SA, SEC and Yahoo are all in base units; the workbook wants millions.
TO_MILLIONS = 1e6


def target_for(ticker: str, workbook_label: str) -> Metric | None:
    for metric in TARGETS:
        if metric.ticker == ticker and metric.workbook_label == workbook_label:
            return metric
    return None
