"""Headline sentiment: a small finance lexicon, headline files and RSS/Atom feeds.

The word lists are hand-picked for market headlines (beats and misses, upgrades and downgrades,
guidance, lawsuits). A negator in the three words before a hit flips it. Lexicon NLP is fast and
transparent, and crude: it can't read sarcasm, context or numbers.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

NEW_YORK = ZoneInfo("America/New_York")

POSITIVE = frozenset(
    """beat beats beating record records surge surges surged soar soars soared jump jumps jumped rally rallies
    rallied gain gains gained rise rises rising upgrade upgrades upgraded outperform outperforms outperformed
    bullish strong stronger strongest growth profit profits profitable raises raised boost boosts boosted
    expand expands expanded win wins approval approves approved breakthrough partnership buyback exceeds
    exceeded optimistic tops topped rebound rebounds rebounded recovery accelerates accelerating""".split()
)
NEGATIVE = frozenset(
    """miss misses missed plunge plunges plunged slump slumps slumped fall falls fell drop drops dropped decline
    declines declined downgrade downgrades downgraded underperform underperforms underperformed bearish weak
    weaker weakest loss losses lawsuit lawsuits sues sued probe probes investigation recall recalls warning
    warns warned layoffs bankruptcy fraud default defaults delay delays delayed slowdown slowing halt halts
    halted fined penalty pessimistic tumble tumbles tumbled crash crashes crashed shortfall""".split()
)
NEGATORS = frozenset("not no never without fails failed fail lacks lack".split())


def score(text: str) -> float:
    """Headline tone from -1 (all negative words) to +1 (all positive); 0 when no word in the lexicon.

    A negator flips only the next lexicon word, within three words and the same clause.
    """
    total = hits = 0
    for clause in re.split(r"[;:,.!?]", text.lower()):
        negate_until = -1
        for i, token in enumerate(re.findall(r"[a-z']+", clause)):
            if token in NEGATORS:
                negate_until = i + 3
                continue
            value = (token in POSITIVE) - (token in NEGATIVE)
            if not value:
                continue
            if i <= negate_until:
                value, negate_until = -value, -1
            total += value
            hits += 1
    return total / hits if hits else 0.0


def load_headlines(path: Path) -> list[tuple[np.datetime64, str, str]]:
    """CSV with columns date, symbol, headline (extra columns are ignored)."""
    with open(path, newline="") as f:
        return [(np.datetime64(r["date"][:10], "D"), r["symbol"].upper(), r["headline"]) for r in csv.DictReader(f)]


def parse_feed(xml: bytes) -> list[tuple[str, str]]:
    """(YYYY-MM-DD in New York, title) for each item of an RSS 2.0 or Atom feed."""
    root = ET.fromstring(xml)
    out = []
    for item in root.iter():
        tag = item.tag.rsplit("}", 1)[-1]
        if tag not in ("item", "entry"):
            continue
        fields = {child.tag.rsplit("}", 1)[-1]: (child.text or "").strip() for child in item}
        title = fields.get("title", "")
        stamp = fields.get("pubDate") or fields.get("published") or fields.get("updated")
        if not title or not stamp:
            continue
        try:
            when = parsedate_to_datetime(stamp) if "," in stamp else datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if when.tzinfo is not None:
            when = when.astimezone(NEW_YORK)  # date it on the market's calendar
        out.append((when.date().isoformat(), title))
    return out


def append_headlines(path: Path, symbol: str, items: list[tuple[str, str]]) -> int:
    """Add (date, title) items for a symbol to a headlines CSV, skipping ones already there."""
    path = Path(path)
    seen = set()
    if path.exists():
        with open(path, newline="") as f:
            seen = {(r["date"], r["symbol"], r["headline"]) for r in csv.DictReader(f)}
    new = [(d, symbol.upper(), t) for d, t in items if (d, symbol.upper(), t) not in seen]
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["date", "symbol", "headline"])
        w.writerows(new)
    return len(new)
