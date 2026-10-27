"""Per-metric daily time series, merged from live fetches, manual entries and the on-disk cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

MANUAL = "manual"


@dataclass(frozen=True)
class Point:
    day: date
    value: float
    source: str


def _priority(source: str) -> int:
    # A number you typed in beats a fetched number for the same day.
    return 2 if source.startswith(MANUAL) else 1


class SeriesStore:
    def __init__(self) -> None:
        self._series: dict[str, dict[date, Point]] = {}

    def add(self, metric: str, day: date, value: float, source: str) -> bool:
        series = self._series.setdefault(metric, {})
        current = series.get(day)
        if current is not None and _priority(current.source) > _priority(source):
            return False
        series[day] = Point(day, float(value), source)
        return True

    def series(self, metric: str, until: date | None = None) -> list[Point]:
        points = self._series.get(metric, {})
        return [points[d] for d in sorted(points) if until is None or d <= until]

    def latest(self, metric: str, until: date | None = None) -> Point | None:
        points = self.series(metric, until)
        return points[-1] if points else None

    def metrics(self) -> list[str]:
        return sorted(self._series)

    def to_dict(self, *, include_manual: bool = True) -> dict:
        out = {}
        for metric in self.metrics():
            rows = [
                [p.day.isoformat(), p.value, p.source]
                for p in self.series(metric)
                if include_manual or not p.source.startswith(MANUAL)
            ]
            if rows:
                out[metric] = rows
        return out

    @classmethod
    def from_dict(cls, data: dict) -> SeriesStore:
        store = cls()
        for metric, rows in data.items():
            for day, value, source in rows:
                store.add(metric, date.fromisoformat(day), value, source)
        return store

    def save(self, path: Path) -> None:
        """Cache fetched data only: the manual TOML stays the one place typed-in numbers live."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(include_manual=False), indent=1))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> SeriesStore:
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text()))
