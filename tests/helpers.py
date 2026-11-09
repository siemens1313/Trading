from __future__ import annotations

from datetime import date, timedelta

from bottom_bots import config
from bottom_bots.store import SeriesStore

AS_OF = date(2026, 10, 23)  # a Friday


def cfg(**sections: dict) -> dict:
    loaded = config.load()
    for section, values in sections.items():
        loaded[section].update(values)
    return loaded


def sessions(n: int, end: date = AS_OF) -> list[date]:
    """The last n weekdays ending on `end`, oldest first."""
    days, day = [], end
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    return days[::-1]


def weeks(n: int, end: date = AS_OF) -> list[date]:
    return [end - timedelta(weeks=n - 1 - i) for i in range(n)]


def make_store(daily: dict | None = None, weekly: dict | None = None, end: date = AS_OF) -> SeriesStore:
    """metric -> number (dated `end`) or list of numbers (consecutive sessions / weeks ending `end`)."""
    store = SeriesStore()
    for spacing, data in ((sessions, daily or {}), (weeks, weekly or {})):
        for metric, values in data.items():
            values = values if isinstance(values, list) else [values]
            for day, value in zip(spacing(len(values), end), values):
                store.add(metric, day, value, "test")
    return store
