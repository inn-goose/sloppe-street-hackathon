"""VENDORED PRE-EXISTING COMPONENT — see README.md.

SvelteKit `devalue` payload hydration, carried over from a pre-existing `stockanalysis` extractor.

⛔ **The payload is a flattened graph and must be hydrated before anything can be read.** A node's
`data` is one array whose element 0 is the root; every member of a container is an **index** into
that same array, so a reader that walks it as JSON gets integers where the values are. Values are
stored once and referenced, which is why `hydrate` memoises — a shared subtree would otherwise be
rebuilt once per reference.

⛔ **`"[PRO]"` is a paywall marker sitting inside numeric arrays, never a value.** It marks whole
periods, not single cells: the period's date and fiscal labels are still stated while every figure
in it reads `"[PRO]"`. Such a period is counted and not stored — a row of nulls would claim the
source said nothing about it, which is the opposite of what it says.
"""

from __future__ import annotations

import json

#: Sentinels devalue encodes as negative indices. Anything else negative is unknown and reads null.
_SENTINELS = {-1: None, -2: None, -3: float("nan"), -4: float("inf"), -5: float("-inf"), -6: -0.0}

PRO = "[PRO]"


def hydrate(flat, index, memo=None):
    """One flattened node → the object it encodes."""
    if isinstance(index, bool) or not isinstance(index, int):
        return index
    if index < 0:
        return _SENTINELS.get(index)
    memo = {} if memo is None else memo
    if index in memo:
        return memo[index]
    node = flat[index]
    if isinstance(node, list):
        out = [hydrate(flat, i, memo) for i in node]
    elif isinstance(node, dict):
        out = {k: hydrate(flat, i, memo) for k, i in node.items()}
    else:
        out = node
    memo[index] = out
    return out


def document(payload) -> dict:
    """A banked capture → the merged page objects, or `{}` when it carries none.

    Accepts the raw text or the already-parsed envelope body, because the raw store hands back a
    parsed `response` member while a direct read hands back text.
    """
    body = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    out: dict = {}
    for node in body.get("nodes") or ():
        if node and node.get("type") == "data":
            out.update(hydrate(node["data"], 0))
    return out


def num(value):
    """A figure, or `None` for anything that is not one — the paywall marker included."""
    if isinstance(value, bool) or value is None or value == PRO:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").replace("%", "").strip())
        except ValueError:
            return None
    return None
