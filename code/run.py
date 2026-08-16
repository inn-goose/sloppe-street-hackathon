"""The pipeline. `fetch` banks raw/, `extract` derives extracted/, `all` does both.

    PYTHONPATH=. .venv/bin/python -m code.run fetch      # every provider, in order
    PYTHONPATH=. .venv/bin/python -m code.run extract    # every extractor, in dependency order
    PYTHONPATH=. .venv/bin/python -m code.run all
    PYTHONPATH=. .venv/bin/python -m code.run status     # what is on disk right now

Order matters in one place only: `corpus.extract.columns` and `calendar.earnings` read
`corpus.extract.lanes`, so the corpus chain is sequential. Everything else is independent.

⚠️ **`fetch` reaches the network; `extract` never does.** That split is what makes a rerun of the
model a pure function of banked bytes, and it is why the final run can be executed with the
network unplugged.
"""

from __future__ import annotations

import argparse
import importlib
import time

from code.lib import config, store

#: Ordered cheapest-first so a broken credential surfaces before a long sweep runs.
#: `sec.*` and `alpaca` need environment credentials; every other one is keyless.
FETCHERS = (
    ("corpus", "code.provider.corpus.fetch.fetch"),
    ("fred", "code.provider.fred.fetch.fetch"),
    ("labour", "code.provider.labour.fetch.fetch"),
    ("nasdaq", "code.provider.nasdaq.fetch.fetch"),
    ("stockanalysis", "code.provider.stockanalysis.fetch.fetch"),
    ("sec.facts", "code.provider.sec.fetch.fetch"),
    ("sec.documents", "code.provider.sec.fetch.documents"),
    ("yahoo", "code.provider.yahoo.fetch.fetch"),
    ("alpaca", "code.provider.alpaca.fetch.fetch"),
)

#: Every extractor walks `data/raw/<provider>/_ledger.jsonl` and opens the banked files. None of
#: them reaches the network, so a rerun is a pure function of hashed bytes.
#:
#: Order matters only inside the corpus chain: `columns` and `statements` read what `documents`
#: and `tables` wrote, and `calendar` reads `columns`.
EXTRACTORS = (
    ("symbology", "code.provider.symbology.resolve"),
    ("corpus.documents", "code.provider.corpus.extract.documents"),
    ("corpus.tables", "code.provider.corpus.extract.tables"),
    ("corpus.columns", "code.provider.corpus.extract.columns"),
    ("corpus.statements", "code.provider.corpus.extract.statements"),
    ("corpus.prose_facts", "code.provider.corpus.extract.prose_facts"),
    ("corpus.guidance", "code.provider.corpus.extract.guidance"),
    ("corpus.consensus", "code.provider.corpus.extract.consensus"),
    ("calendar", "code.provider.calendar.earnings"),
    ("yahoo.panel", "code.provider.yahoo.extract.panel"),
    ("stockanalysis.financials", "code.provider.stockanalysis.extract.financials"),
    ("sec.facts", "code.provider.sec.extract.facts"),
    ("sec.peer_guidance", "code.provider.sec.extract.peer_guidance"),
    ("fred.observations", "code.provider.fred.extract.observations"),
    ("labour.observations", "code.provider.labour.extract.observations"),
    ("nasdaq.analyst", "code.provider.nasdaq.extract.analyst"),
    ("alpaca.tape", "code.provider.alpaca.extract.tape"),
)


#: The feature layer, in dependency order: periods → panel → guides → (seasonality, conservatism)
#: → matrix → design. Nothing here reaches the network either.
#:
#: ⛔ **The feature layer emits series, never forecasts.** Every column of `training_matrix` is
#: computable at every period from data published before it, which is what lets the model layer
#: score a rule out of sample. A module that produces a value only for the target period cannot be
#: validated at all, and its parameters end up being set by eye — which is how a hand-tuned bridge
#: got fitted to the sell-side consensus it was supposed to be competing against.
FEATURES = (
    ("periods", "code.feature.periods"),
    ("panel", "code.feature.panel"),
    ("check_panel", "code.feature.check_panel"),
    ("guides", "code.feature.guides"),
    ("seasonality", "code.feature.seasonality"),
    ("conservatism", "code.feature.conservatism"),
    ("matrix", "code.feature.matrix"),
    ("design", "code.feature.design"),
    ("audit", "code.feature.audit"),
)

#: The model layer. `dataset` reshapes the matrix into (row, anchor) residuals, `train` runs the
#: walk-forward that decides how to forecast, `predict` applies it, `workbooks` writes the four
#: `.xlsx`. Every parameter is settled by `train`; none is written here.
MODELS = (
    ("dataset", "code.model.dataset"),
    ("train", "code.model.train"),
    ("predict", "code.model.predict"),
    ("workbooks", "code.model.workbooks"),
)


def _run(stages, label: str) -> int:
    failures = 0
    for name, module in stages:
        started = time.time()
        print(f"\n=== {label}: {name} " + "=" * max(0, 60 - len(name)))
        try:
            importlib.import_module(module).main()
        except Exception as exc:  # noqa: BLE001 — one stage failing must not hide the rest
            failures += 1
            print(f"  FAILED {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ({time.time() - started:.1f}s)")
    return failures


def status() -> None:
    total = 0
    for path in sorted(config.DATA.rglob("*.parquet")):
        print(f"  {store.describe(path)}")
        total += path.stat().st_size
    print(f"\n  {total / 1e6:.1f} MB on disk")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["fetch", "extract", "feature", "model",
                                          "all", "status"])
    parser.add_argument("--only", help="run one named stage")
    args = parser.parse_args()

    if args.stage == "status":
        status()
        return 0

    failures = 0
    if args.stage in ("fetch", "all"):
        stages = [s for s in FETCHERS if not args.only or s[0] == args.only]
        failures += _run(stages, "fetch")
    if args.stage in ("extract", "all"):
        stages = [s for s in EXTRACTORS if not args.only or s[0] == args.only]
        failures += _run(stages, "extract")
    if args.stage in ("feature", "all"):
        stages = [s for s in FEATURES if not args.only or s[0] == args.only]
        failures += _run(stages, "feature")
    if args.stage in ("model", "all"):
        stages = [s for s in MODELS if not args.only or s[0] == args.only]
        failures += _run(stages, "model")

    print("\n" + "=" * 68)
    status()
    if failures:
        print(f"\n{failures} stage(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
