"""Live data fetchers.

Every fetcher is optional: a failure becomes a warning, and the checklist items that
needed it fall back to manual inputs or UNKNOWN — never a crash, never a guess. The
rest of the article's sources (short interest, ETF flows, AAII, NAAIM, breadth, the
gamma flip level) are paywalled or have no stable free API, so they're manual inputs.
"""

from __future__ import annotations

from datetime import date
from typing import Callable

from ..store import SeriesStore
from . import cboe, cnn, prices, squeeze
from .http import get

Fetch = Callable[[str], bytes]


def price_symbols(cfg: dict) -> list[str]:
    symbols = (cfg["volatility"]["semis_symbol"], cfg["short_interest"]["symbol"], cfg["gamma"]["underlying"])
    return list(dict.fromkeys(s.upper() for s in symbols))


def refresh(store: SeriesStore, cfg: dict, as_of: date, fetch: Fetch = get) -> list[str]:
    jobs = [(f"Cboe {cboe.VOL_INDICES[m]}", lambda m=m: cboe.fetch_vol_index(m, fetch)) for m in cboe.VOL_INDICES]
    jobs += [
        ("Cboe put/call ratios", lambda: cboe.fetch_put_call(as_of, fetch)),
        ("SqueezeMetrics DIX/GEX", lambda: squeeze.fetch(fetch)),
        ("CNN Fear & Greed", lambda: cnn.fetch(fetch)),
    ]
    jobs += [(f"{s} closes", lambda s=s: prices.fetch_closes(s, fetch)) for s in price_symbols(cfg)]

    warnings = []
    for name, job in jobs:
        try:
            points = job()
        except Exception as exc:  # any fetcher failure degrades to missing data
            warnings.append(f"{name}: unavailable ({exc})")
            continue
        if not points:
            warnings.append(f"{name}: returned no data")
        for metric, day, value, source in points:
            store.add(metric, day, value, source)
    return warnings
