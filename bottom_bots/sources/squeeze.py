"""SqueezeMetrics' free DIX / GEX history."""

from __future__ import annotations

import csv
import io
from datetime import date
from typing import Callable

from .util import parse_day

DIX_URL = "https://squeezemetrics.com/monitor/static/DIX.csv"


def parse_dix_csv(text: str) -> list[tuple[str, date, float, str]]:
    reader = csv.DictReader(io.StringIO(text))
    columns = {name.strip().lower(): name for name in reader.fieldnames or []}
    if not {"date", "dix", "gex"} <= set(columns):
        raise ValueError(f"unexpected DIX.csv columns: {reader.fieldnames}")
    points = []
    for row in reader:
        try:
            day = parse_day(row[columns["date"]])
            dix = float(row[columns["dix"]])
            gex = float(row[columns["gex"]])
        except (TypeError, ValueError):
            continue
        # Published as a fraction and in dollars; stored as % and $bn.
        points.append(("dix", day, dix * 100 if dix <= 1.5 else dix, "squeezemetrics"))
        points.append(("gex", day, gex / 1e9 if abs(gex) > 1e6 else gex, "squeezemetrics"))
        if "price" in columns:
            try:
                points.append(("close:SPX", day, float(row[columns["price"]]), "squeezemetrics"))
            except (TypeError, ValueError):
                pass
    return points


def fetch(get: Callable[[str], bytes]) -> list[tuple[str, date, float, str]]:
    return parse_dix_csv(get(DIX_URL).decode("utf-8-sig"))
