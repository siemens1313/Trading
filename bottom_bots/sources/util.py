from __future__ import annotations

from datetime import date, datetime

_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%m/%d/%y")


def parse_day(text: str) -> date:
    text = text.strip()
    for fmt in _FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {text!r}")
