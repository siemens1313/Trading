"""Day-keyed record of past runs: drives score persistence and the tranche ratchet."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class RunRecord:
    as_of: date
    score: int
    unknown: int
    persistent: int
    target: float
    increment: float
    statuses: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["as_of"] = self.as_of.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> RunRecord:
        return cls(
            as_of=date.fromisoformat(data["as_of"]),
            score=int(data["score"]),
            unknown=int(data["unknown"]),
            persistent=int(data["persistent"]),
            target=float(data["target"]),
            increment=float(data["increment"]),
            statuses=dict(data.get("statuses", {})),
        )


class History:
    def __init__(self, records: list[RunRecord] | None = None) -> None:
        self._records = {r.as_of: r for r in records or []}

    def before(self, as_of: date) -> list[RunRecord]:
        """Runs strictly before `as_of`, oldest first — re-running a day never counts itself."""
        return [self._records[d] for d in sorted(self._records) if d < as_of]

    def put(self, record: RunRecord) -> None:
        self._records[record.as_of] = record

    def all(self) -> list[RunRecord]:
        return [self._records[d] for d in sorted(self._records)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"runs": [r.to_dict() for r in self.all()]}, indent=1))

    @classmethod
    def load(cls, path: Path) -> History:
        if not path.exists():
            return cls()
        return cls([RunRecord.from_dict(r) for r in json.loads(path.read_text())["runs"]])
