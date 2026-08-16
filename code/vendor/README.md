# Vendored pre-existing components — DECLARED

Everything in this directory is **pre-existing generic library code** written before the
hackathon for a different model (an earnings-*move* model over the whole US listed universe).
It is vendored here verbatim-in-substance, unmodified in behaviour, and is declared in
`entry.json` under `existingComponents`.

Per `RULES.md`: *"Off-the-shelf models, public libraries, agent frameworks, generic utilities and
your normal unmodified coding harness are allowed. Declare any existing components you use in
`entry.json`."*

| File | What it is | Why it is generic, not challenge-specific |
|---|---|---|
| `sentence_grammar.py` | Abbreviation-aware sentence index, prose-vs-deck detector, evidence quoting | Reads any SEC/RNS document. Knows nothing about these four companies |
| `guidance_grammar.py` | The stated-forward-range reader — four binding frames, measured veto classes | Mined from a 2,600-document stratified sample of SEC earnings releases 2004→2026, across ~3,000 filers |

| `devalue.py` | Hydrator for SvelteKit's `devalue` wire format | A general JSON-graph decoder for a public site's payloads; no company logic |

**What is NOT vendored and was built during the event:** every company reader and extractor, the
period resolver, the reconciled panel, every feature, the anchor set, the training matrix, the
model, the validators, the workbook writer and the orchestration. Those are in `code/provider/`,
`code/feature/` and `code/model/`.

The vendored grammar contributes one lane (`extracted/guidance`). It is used as a *cross-check and
prior* on the numbers the company readers extract, never as the sole source of a submitted figure —
`code/model/train.py` records which anchors a forecast may be built from and how each was scored.
