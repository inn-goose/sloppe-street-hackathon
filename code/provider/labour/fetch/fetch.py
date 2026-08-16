"""raw/labour — UK (ONS), EU (Eurostat) and euro-area (ECB) labour series. Keyless.

    data/raw/labour/series/<cadence>/<as-of>/<source>/<series id>.json.gz

## Why this provider exists

Because **FRED's vacancy series are dead**: the OECD UK series stops at 2023-06 and the German at
2024-01. A discontinued macro series does not error — it freezes a feature at its last value.

That matters most here of anywhere: **Hays is 25 % of the competition score**, its revenue line
*is* recruitment activity, Germany is 32 % of its net fees and UK & Ireland 19 %. Job vacancies
are the exogenous driver of the exact number being forecast, published monthly through a quarter
the company reports once.

⚡ **Vacancies by NACE sector, not just the aggregate.** Hays' four largest specialisms are
Technology (25 % of group net fees), Accountancy & Finance (15 %), Engineering (11 %) and
Construction & Property (11 %) — so NACE J (ICT), M (professional/scientific), F (construction)
and C (manufacturing) track the book far more closely than the all-industry rate.

## Two failure modes this fetcher was corrected for

⛔ **Eurostat answers HTTP 200 with zero observations for an unknown dimension value.** A first
attempt used `indic_em=JOBRATE` and every request "succeeded" while returning nothing; reading the
dataset's own `dimension.category.index` showed the code is `JVR` and that `sizeclas` is a
required dimension that was omitted entirely. A 200-with-nothing is the failure mode to fear here,
so a zero-observation body is refused rather than banked.

⛔ **The ECB `JVS` flow 404s on every positional key shape tried.** Its dimension order is not what
the flow name suggests, and a wrong guess is indistinguishable from "no vacancies". Only the
labour-force flow is taken from ECB; the vacancy statistic comes from Eurostat's named-parameter
API where a wrong filter is visible as a wrong filter.
"""

from __future__ import annotations

import json
import time

from code.lib import config, rawstore

PROVIDER = "labour"
_MIN_INTERVAL = 0.4
_TIMEOUT = 45

ONS_SERIES = (
    ("AP2Y", "UNEM", "month", "UK vacancies, total (thousands)"),
    ("JP9Z", "UNEM", "month", "UK vacancies, ratio per 100 employee jobs"),
    ("MGSX", "LMS", "month", "UK unemployment rate, aged 16+"),
    ("LF24", "LMS", "month", "UK employment level, aged 16+"),
    ("MGRZ", "LMS", "month", "UK claimant count"),
)
_ONS_URLS = (
    "https://www.ons.gov.uk/employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/{cdid}/{ds}/data",
    "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/timeseries/{cdid}/{ds}/data",
    "https://api.ons.gov.uk/timeseries/{cdid}/dataset/{ds}/data",
)

_EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}"
_JVR = {"indic_em": "JVR", "sizeclas": "TOTAL", "s_adj": "SA"}
EUROSTAT_SERIES = (
    ("jvs_q_nace2", {"geo": "DE", "nace_r2": "B-S", **_JVR}, "quarter",
     "Germany job vacancy rate, all industry"),
    ("jvs_q_nace2", {"geo": "DE", "nace_r2": "J", **_JVR}, "quarter",
     "Germany job vacancy rate — ICT (NACE J)"),
    ("jvs_q_nace2", {"geo": "DE", "nace_r2": "M", **_JVR}, "quarter",
     "Germany job vacancy rate — professional/scientific (M)"),
    ("jvs_q_nace2", {"geo": "DE", "nace_r2": "F", **_JVR}, "quarter",
     "Germany job vacancy rate — construction (NACE F)"),
    ("jvs_q_nace2", {"geo": "DE", "nace_r2": "C", **_JVR}, "quarter",
     "Germany job vacancy rate — manufacturing (NACE C)"),
    ("jvs_q_nace2", {"geo": "DE", "nace_r2": "B-S", "indic_em": "JOBVAC", "sizeclas": "TOTAL",
                     "s_adj": "SA"}, "quarter", "Germany job vacancies — level"),
    ("jvs_q_nace2", {"geo": "EU27_2020", "nace_r2": "B-S", **_JVR}, "quarter",
     "EU27 job vacancy rate"),
    ("jvs_q_nace2", {"geo": "FR", "nace_r2": "B-S", **_JVR}, "quarter",
     "France job vacancy rate"),
    ("jvs_q_nace2", {"geo": "NL", "nace_r2": "B-S", **_JVR}, "quarter",
     "Netherlands job vacancy rate"),
    ("jvs_q_nace2", {"geo": "ES", "nace_r2": "B-S", **_JVR}, "quarter",
     "Spain job vacancy rate"),
    ("jvs_q_nace2", {"geo": "PL", "nace_r2": "B-S", **_JVR}, "quarter",
     "Poland job vacancy rate"),
    ("une_rt_m", {"geo": "DE", "s_adj": "SA", "age": "TOTAL", "sex": "T", "unit": "PC_ACT"},
     "month", "Germany unemployment rate"),
    ("sts_copr_q", {"geo": "DE", "s_adj": "SCA", "nace_r2": "F", "unit": "I21"}, "quarter",
     "Germany construction production index"),
    ("sts_inpr_m", {"geo": "DE", "s_adj": "SCA", "nace_r2": "B-D", "unit": "I21"}, "month",
     "Germany industrial production index"),
    ("namq_10_gdp", {"geo": "DE", "s_adj": "SCA", "unit": "CLV_PCH_PRE", "na_item": "B1GQ"},
     "quarter", "Germany real GDP, QoQ %"),
    ("namq_10_gdp", {"geo": "EU27_2020", "s_adj": "SCA", "unit": "CLV_PCH_PRE",
                     "na_item": "B1GQ"}, "quarter", "EU27 real GDP, QoQ %"),
)

ECB_SERIES = (
    ("LFSI/M.DE.S.UNEHRT.TOTAL0.15_74.T", "month", "Germany unemployment rate (ECB)"),
    ("LFSI/M.U2.S.UNEHRT.TOTAL0.15_74.T", "month", "Euro area unemployment rate (ECB)"),
)


def _session():
    import curl_cffi.requests as cr

    return cr.Session(impersonate="chrome")


def _pace(last: list[float]) -> None:
    wait = _MIN_INTERVAL - (time.monotonic() - last[0])
    if wait > 0:
        time.sleep(wait)
    last[0] = time.monotonic()


def _slug(dataset: str, filters: dict) -> str:
    """A series id carrying EVERY distinguishing filter.

    ⚠️ Five German series differ only by `nace_r2`/`indic_em`. Keying on `dataset:geo` alone gave
    all five one id, which any dedup downstream would collapse to whichever landed last.
    """
    keep = {k: v for k, v in filters.items() if k not in ("s_adj", "sizeclas")}
    return "_".join([dataset] + [f"{k}-{v}" for k, v in sorted(keep.items())])


def fetch() -> dict:
    config.ensure_dirs()
    session = _session()
    as_of = rawstore.now()
    ledger = rawstore.Ledger(PROVIDER)
    last = [0.0]
    banked = refused = disk = 0

    def _bank(source, series_id, cadence, description, text, n, url):
        nonlocal banked, disk
        directory = rawstore.capture_dir(PROVIDER, "series", cadence, as_of=as_of)
        leaf = rawstore.leaf_name(source, series_id, ext="json", compression="gz")
        body = rawstore.envelope(text, fetched_at=as_of, request={"url": url})
        written = rawstore.bank(directory / leaf, body)
        disk += written
        banked += 1
        ledger.add(provider=PROVIDER, source=source, product="series", series_id=series_id,
                   for_ticker="LSE:HAS", description=description, cadence=cadence,
                   state="ok", n_observations=n,
                   path=str((directory / leaf).relative_to(config.RAW)),
                   source_bytes=len(text), disk_bytes=written,
                   fetched_at=as_of.isoformat(timespec="seconds"))
        print(f"  ok {source:<9}{series_id[:40]:<42}{n} obs   {description[:40]}")

    def _refuse(source, series_id, description, note):
        nonlocal refused
        refused += 1
        ledger.add(provider=PROVIDER, source=source, product="series", series_id=series_id,
                   description=description, state="refused", note=note,
                   fetched_at=as_of.isoformat(timespec="seconds"))
        print(f"  -- {source:<9}{series_id[:40]:<42}{note[:44]}")

    print(f"labour: {len(ONS_SERIES)} ONS + {len(EUROSTAT_SERIES)} Eurostat + {len(ECB_SERIES)} "
          f"ECB -> raw/labour/series/<cadence>/{rawstore.segment_of(as_of)}")

    for cdid, dataset, cadence, description in ONS_SERIES:
        tried = []
        for template in _ONS_URLS:
            url = template.format(cdid=cdid, ds=dataset)
            _pace(last)
            try:
                resp = session.get(url, timeout=_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                tried.append(type(exc).__name__)
                continue
            if resp.status_code != 200:
                tried.append(f"HTTP {resp.status_code}")
                continue
            try:
                body = json.loads(resp.text)
            except ValueError:
                tried.append("not JSON")
                continue
            n = sum(len(body.get(f) or []) for f in ("months", "quarters", "years"))
            if not n:
                tried.append("no observations")
                continue
            _bank("ons", cdid, cadence, description, resp.text, n, url)
            break
        else:
            _refuse("ons", cdid, description, "; ".join(tried))

    for dataset, filters, cadence, description in EUROSTAT_SERIES:
        series_id = _slug(dataset, filters)
        url = _EUROSTAT.format(dataset=dataset)
        _pace(last)
        try:
            resp = session.get(url, params={"format": "JSON", "lang": "EN", **filters},
                               timeout=_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            _refuse("eurostat", series_id, description, type(exc).__name__)
            continue
        if resp.status_code != 200:
            _refuse("eurostat", series_id, description, f"HTTP {resp.status_code}")
            continue
        try:
            n = len(json.loads(resp.text).get("value") or {})
        except ValueError:
            _refuse("eurostat", series_id, description, "not JSON")
            continue
        # ⛔ a 200 with no observations means a wrong dimension value, not an empty statistic
        if not n:
            _refuse("eurostat", series_id, description, "200 with no observations (bad filter)")
            continue
        _bank("eurostat", series_id, cadence, description, resp.text, n, url)

    for key, cadence, description in ECB_SERIES:
        url = f"https://data-api.ecb.europa.eu/service/data/{key}"
        series_id = key.replace("/", "_").replace(".", "-")
        _pace(last)
        try:
            resp = session.get(url, params={"format": "jsondata"}, timeout=_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            _refuse("ecb", series_id, description, type(exc).__name__)
            continue
        if resp.status_code != 200 or not resp.text.strip():
            _refuse("ecb", series_id, description, f"HTTP {resp.status_code}")
            continue
        try:
            series = ((json.loads(resp.text).get("dataSets") or [{}])[0].get("series") or {})
        except ValueError:
            _refuse("ecb", series_id, description, "not JSON")
            continue
        n = sum(len(s.get("observations") or {}) for s in series.values())
        if not n:
            _refuse("ecb", series_id, description, "no observations")
            continue
        _bank("ecb", series_id, cadence, description, resp.text, n, url)

    ledger.write()
    return {"banked": banked, "refused": refused, "disk": disk}


def main() -> int:
    stats = fetch()
    print(f"\nraw/labour  {stats['banked']} series  {stats['disk'] / 1e6:.2f} MB on disk  "
          f"({stats['refused']} refused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
