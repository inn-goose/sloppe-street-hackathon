"""Paths, the company registry and the target spec every extractor reads from.

Layout — one laptop, no server, everything under the repository root:

    code/provider/<name>/fetch/     banks data/raw/, verbatim and as-of
    code/provider/<name>/extract/   derives data/extracted/, faithful
    code/feature/                   joins the lanes into the training matrix
    code/model/                     trains, validates, predicts, writes the workbooks
    code/lib/                       shared substrate (store, text grammar, settings)
    code/vendor/                    declared pre-existing components
    settings/                       provider settings, laptop-local
    data/raw|extracted|feature/     the store  (regenerated; never committed)
    challenge/templates/            the supplied workbook templates
    submission/                     the four completed workbooks

⚠️ **The document corpus is the one input this repository does not contain.** It is third-party
material licensed to the competition organisers and explicitly not redistributable, so it is read
from a sibling checkout of the challenge repository and never copied into version control. Point
`CHALLENGE_CORPUS` at it if you keep it somewhere else; `README.md` says how to obtain it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SETTINGS_DIR = WORKSPACE / "settings"

#: The supplied corpus, which lives outside this repository for licensing reasons.
CORPUS_ROOT = Path(os.environ.get("CHALLENGE_CORPUS")
                   or WORKSPACE / "starter" / "challenge" / "offline-data")

COMPANIES_JSON = WORKSPACE / "challenge" / "companies.json"
TEMPLATE_DIR = TEMPLATES = WORKSPACE / "challenge" / "templates"
SUBMISSION_DIR = SUBMISSION = WORKSPACE / "submission"
LOG_DIR = WORKSPACE / "logs"

DATA = WORKSPACE / "data"
RAW = DATA / "raw"
EXTRACTED = DATA / "extracted"
FEATURE = DATA / "feature"
STORE = DATA
MODEL_OUT = DATA / "model"

CORPUS_DIRS = {"HD": "home-depot", "ADI": "analog-devices", "LSE:HAS": "hays", "DE": "deere"}

# The fiscal-year-end month each issuer states. HD/ADI/DE run 52/53-week years, so a quarter end
# is a date NEAR a month end, never on one; the panel keys on the fiscal label the filer itself
# writes and uses these only to sanity-check a resolved label against its publication date.
FISCAL_YEAR_END_MONTH = {"HD": 1, "ADI": 11, "LSE:HAS": 6, "DE": 10}

# Reporting currency, and the unit the workbook wants. GBp is 1/100 of GBP, which is live here
# because Hays' EPS is submitted in pence while its profit lines are in millions of pounds.
REPORTING_CCY = {"HD": "USD", "ADI": "USD", "LSE:HAS": "GBP", "DE": "USD"}


@dataclass(frozen=True)
class Metric:
    label: str
    units: str


@dataclass(frozen=True)
class Company:
    company: str
    ticker: str
    period: str
    output_file: str
    metrics: tuple[Metric, ...]

    @property
    def slug(self) -> str:
        return CORPUS_DIRS[self.ticker]

    @property
    def short(self) -> str:
        return self.ticker.rsplit(":", 1)[-1]

    @property
    def corpus_dir(self) -> Path:
        return CORPUS_ROOT / self.slug

    @property
    def fiscal_year(self) -> int:
        return int(self.period[2:6])

    @property
    def fiscal_quarter(self) -> int | None:
        return int(self.period[-1]) if "Q" in self.period else None

    @property
    def currency(self) -> str:
        return REPORTING_CCY[self.ticker]


def load_companies() -> list[Company]:
    payload = json.loads(COMPANIES_JSON.read_text(encoding="utf-8"))
    return [
        Company(
            company=item["company"],
            ticker=item["ticker"],
            period=item["period"],
            output_file=item["outputFile"],
            metrics=tuple(Metric(m["label"], m["units"]) for m in item["metrics"]),
        )
        for item in payload["companies"]
    ]


COMPANIES = {c.short: c for c in load_companies()}


def ensure_dirs() -> None:
    for path in (RAW, EXTRACTED, FEATURE, MODEL_OUT, SUBMISSION_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ settings

@lru_cache(maxsize=None)
def _settings(provider: str) -> dict:
    """Provider settings, laptop-local.

    A file on disk plus an environment override is the whole mechanism; there is no settings
    service. A missing file is not an error — every reader states its own default.
    """
    path = SETTINGS_DIR / "provider" / f"{provider}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def setting(provider: str, *keys: str, default=None):
    """`setting("yahoo", "http", "max_rps", default=2.0)`, overridable by environment.

    ⚠️ The env override is read but never echoed: a credential belongs in the process, not in a
    log line, a commit or an argv.
    """
    env_key = "_".join(("FORECAST", provider, *keys)).upper()
    if env_key in os.environ:
        raw = os.environ[env_key]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    node = _settings(provider)
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node if node is not None else default


def secret(name: str) -> str | None:
    """A credential, by environment variable name only.

    ⛔ Never read from a settings file and never returned into a log. The caller passes the value
    straight to the client and nothing else touches it.
    """
    value = os.environ.get(name)
    return value or None
