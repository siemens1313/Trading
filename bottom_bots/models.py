"""Result types shared by the bots."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class Status(str, Enum):
    MET = "MET"
    NOT_MET = "NOT_MET"
    UNKNOWN = "UNKNOWN"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ")


@dataclass(frozen=True)
class Reading:
    label: str
    value: float | None
    day: date | None = None
    source: str | None = None
    fmt: str = ".2f"
    unit: str = ""

    def text(self, as_of: date | None = None) -> str:
        if self.value is None:
            return f"{self.label} —"
        text = f"{self.label} {format(self.value, self.fmt)}{self.unit}"
        if self.day is not None and as_of is not None and self.day != as_of:
            text += f" ({self.day:%m-%d})"
        return text

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "value": self.value,
            "date": self.day.isoformat() if self.day else None,
            "source": self.source,
        }


@dataclass
class BotResult:
    key: str
    item: int
    title: str
    rule: str
    status: Status
    summary: str
    readings: list[Reading] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)  # context signals, not scored
    warnings: list[str] = field(default_factory=list)  # stale data, divergences
    notes: list[str] = field(default_factory=list)  # the article's own caveats
    extremes: list[str] = field(default_factory=list)  # article-defined extreme zones reached

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "item": self.item,
            "title": self.title,
            "rule": self.rule,
            "status": self.status.value,
            "summary": self.summary,
            "readings": [r.to_dict() for r in self.readings],
            "flags": self.flags,
            "warnings": self.warnings,
            "notes": self.notes,
            "extremes": self.extremes,
        }
