"""Shared plumbing for the five sensor bots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Iterable

from ..models import BotResult, Reading, Status
from ..store import Point, SeriesStore


class SensorBot(ABC):
    key: str
    item: int
    title: str
    rule: str

    def __init__(self, cfg: dict, store: SeriesStore, as_of: date) -> None:
        self.cfg = cfg
        self.store = store
        self.as_of = as_of
        self._warnings: list[str] = []

    @abstractmethod
    def evaluate(self) -> BotResult: ...

    def warn(self, message: str) -> None:
        if message not in self._warnings:
            self._warnings.append(message)

    def max_age(self, metric: str) -> int:
        return int(self.cfg["staleness_days"][metric.split(":")[0]])

    def is_fresh(self, point: Point, metric: str) -> bool:
        return (self.as_of - point.day).days <= self.max_age(metric)

    def _stale(self, point: Point, metric: str) -> None:
        self.warn(
            f"{metric} is stale (last value {point.day.isoformat()}, limit "
            f"{self.max_age(metric)} days) — treated as missing"
        )

    def fresh(self, metric: str) -> Point | None:
        """Latest reading on or before the run date; None if missing or stale."""
        point = self.store.latest(metric, until=self.as_of)
        if point is None:
            return None
        if not self.is_fresh(point, metric):
            self._stale(point, metric)
            return None
        return point

    def fresh_history(self, metric: str) -> list[Point]:
        """Every reading up to the run date, or [] when the latest one is stale."""
        points = self.store.series(metric, until=self.as_of)
        if points and not self.is_fresh(points[-1], metric):
            self._stale(points[-1], metric)
            return []
        return points

    def aligned(self, a: str, b: str) -> tuple[date, float, float] | None:
        """Latest day on which both series have a value, if that day is fresh."""
        left = {p.day: p.value for p in self.store.series(a, until=self.as_of)}
        right = {p.day: p.value for p in self.store.series(b, until=self.as_of)}
        common = sorted(set(left) & set(right))
        if not common:
            return None
        day = common[-1]
        if (self.as_of - day).days > min(self.max_age(a), self.max_age(b)):
            return None
        return day, left[day], right[day]

    @staticmethod
    def reading(label: str, point: Point | None, fmt: str = ".2f", unit: str = "") -> Reading:
        if point is None:
            return Reading(label, None, fmt=fmt, unit=unit)
        return Reading(label, point.value, point.day, point.source, fmt, unit)

    def result(
        self,
        status: Status,
        summary: str,
        readings: Iterable[Reading],
        flags: Iterable[str] = (),
        notes: Iterable[str] = (),
        extremes: Iterable[str] = (),
    ) -> BotResult:
        return BotResult(
            key=self.key,
            item=self.item,
            title=self.title,
            rule=self.rule,
            status=status,
            summary=summary,
            readings=list(readings),
            flags=list(flags),
            warnings=list(self._warnings),
            notes=list(notes),
            extremes=list(extremes),
        )
