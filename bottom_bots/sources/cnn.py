"""CNN Fear & Greed index."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Callable

FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"


def parse_fear_greed(payload: dict) -> list[tuple[date, float]]:
    out: dict[date, float] = {}
    for point in payload.get("fear_and_greed_historical", {}).get("data", []):
        try:
            day = datetime.fromtimestamp(float(point["x"]) / 1000, tz=timezone.utc).date()
            out[day] = float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
    current = payload.get("fear_and_greed", {})
    if "score" in current and "timestamp" in current:
        try:
            stamp = datetime.fromisoformat(str(current["timestamp"]).replace("Z", "+00:00"))
            out[stamp.date()] = float(current["score"])
        except (TypeError, ValueError):
            pass
    if not out:
        raise ValueError("no Fear & Greed values in the response")
    return sorted(out.items())


def fetch(get: Callable[[str], bytes]) -> list[tuple[str, date, float, str]]:
    return [("fear_greed", day, value, "cnn") for day, value in parse_fear_greed(json.loads(get(FEAR_GREED_URL)))]
