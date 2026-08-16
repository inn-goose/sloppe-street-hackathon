"""extracted/peer_guidance — what the peers guided, from their own EX-99 exhibits.

The competition corpus is four companies with **no cross-section at all**, so peer read-through
(§R) is dead in it. But it is real and it is dated: Texas Instruments and Microchip both guide
their September quarter **before ADI reports its own**, Lowe's guides against Home Depot's, AGCO
and CNH against Deere's.

344 peer earnings exhibits are banked. This runs the same two guidance grammars over them that
run over the corpus, so a peer's stated forward range lands on the same shape of row as a
target's, and the two are directly comparable.

⛔ **HTML is stripped, never interpreted.** Tags are removed, entities decoded, whitespace
collapsed. That is a byte transform with no inference in it — the same trust class as reading a
markdown table.

⚠️ **A peer's guide informs a target only where the PERIODS overlap**, and they usually do not
line up: TXN guides calendar quarters, ADI fiscal ones ending in late July. The overlap is the
feature layer's decision; this lane records the guide and the peer's filing date and leaves it.
"""

from __future__ import annotations

import html
import re
from collections import Counter

from code.lib import config, rawstore, store
from code.provider.corpus.extract import _guidance_forms
from code.vendor import guidance_grammar

_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v\xa0]+")


def to_text(markup: str) -> str:
    """HTML → the text it renders. A byte transform, no inference."""
    text = _SCRIPT.sub(" ", markup)
    text = re.sub(r"<(br|/p|/div|/tr|/h\d)[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = _WS.sub(" ", text)
    return re.sub(r"\n{2,}", "\n", text)


def build() -> list[dict]:
    rows: list[dict] = []
    for meta, markup in rawstore.iter_captures("sec_documents", parse=False):
        if meta.get("product") != "earnings_exhibit":
            continue
        body = to_text(markup)
        hits = list(guidance_grammar.extract(body))
        seen = {(h.metric.lower(), h.period.lower(), h.low, h.high) for h in hits}
        for h in _guidance_forms.extract(body):
            if (h.metric.lower(), h.period.lower(), h.low, h.high) not in seen:
                hits.append(h)
        for h in hits:
            if _guidance_forms._NOT_GUIDANCE.search(h.metric) or \
                    _guidance_forms._NOT_GUIDANCE.search(h.evidence):
                continue
            rows.append({
                "peer": meta["symbol"], "informs": meta.get("for_ticker") or "",
                "cik": meta.get("cik"), "accession": meta.get("accession"),
                "form": meta.get("form"), "filing_date": meta.get("filing_date"),
                "document": meta.get("document"),
                "metric": h.metric, "period": h.period,
                "low": h.low, "high": h.high, "midpoint": (h.low + h.high) / 2.0,
                "unit": h.unit, "scale": h.scale, "frame": h.frame,
                "confidence": h.confidence, "evidence": h.evidence[:320],
            })
    return rows


def main() -> int:
    rows = build()
    store.write(config.EXTRACTED / "peer_guidance.parquet", rows)
    stated = [r for r in rows if r["confidence"] == "stated"]
    print(f"extracted/peer_guidance.parquet {len(rows):,} peer forward ranges "
          f"({len(stated):,} stated) from "
          f"{len({r['accession'] for r in rows})} exhibits")
    print(f"  frames: {dict(Counter(r['frame'] for r in rows).most_common(6))}")
    for target in ("HD", "ADI", "DE"):
        mine = [r for r in stated if r["informs"] == target]
        peers = Counter(r["peer"] for r in mine)
        recent = [r for r in mine if (r["filing_date"] or "") >= "2026-04-01"]
        print(f"  informs {target:<4}{len(mine):>5} stated ranges over {len(peers)} peers, "
              f"{len(recent)} filed since 2026-04  {dict(peers.most_common(4))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
